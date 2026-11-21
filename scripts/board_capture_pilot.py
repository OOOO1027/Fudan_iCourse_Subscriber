"""One-lecture trial of src/ai/board_capture.py on a real replay.

Samples one frame every BOARD_INTERVAL seconds from the lecture named by
BOARD_COURSE_ID / BOARD_SUB_ID, extracts board pages and writes
$BOARD_OUTPUT/bundle.bin: a tar of the pages, a contact sheet, a time-lapse
strip and report.json, encrypted to scripts/board_review_key.pub.  The private
key is not in this repository.  Nothing is written to the database or to any
branch.
"""
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from Crypto.Cipher import AES, PKCS1_OAEP
from Crypto.Hash import SHA256
from Crypto.PublicKey import RSA
from Crypto.Random import get_random_bytes
from PIL import Image, ImageDraw

from main import login_with_retry
from src.ai.board_capture import BoardExtractor, contact_sheet
from src.api.icourse import ICourseClient

COURSE = os.environ.get("BOARD_COURSE_ID", "38985")
SUB = os.environ.get("BOARD_SUB_ID", "693743")
INTERVAL = float(os.environ.get("BOARD_INTERVAL", "5"))


def clock(ms):
    s = int(ms) // 1000
    return f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}"


def stream(client):
    """A freshly signed stream; the signature is short-lived."""
    url = client.get_video_url(COURSE, SUB)
    if not url:
        raise RuntimeError("Replay unavailable")
    return client.get_stream_params(url)


def describe_sources(client):
    """Replay sources the platform lists: field names for the public log, short
    plain values (never URLs) for the encrypted report."""
    def plain(value):
        return isinstance(value, (int, float)) or (
            isinstance(value, str) and len(value) < 40 and "/" not in value)

    info = client.get_sub_info(COURSE, SUB)
    names, values = {}, {}
    for field in ("video_list", "playurl"):
        entries = info.get(field)
        if not isinstance(entries, dict):
            continue
        for key, entry in entries.items():
            if key == "now":
                continue
            fields = entry if isinstance(entry, dict) else {}
            names[f"{field}.{key}"] = sorted(fields)
            values[f"{field}.{key}"] = {a: b for a, b in fields.items() if plain(b)}
    return names, values


RECONNECT = ["-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5"]


def quiet(cmd, timeout):
    """Run ffmpeg/ffprobe; their errors can hold the signed URL and cookies, so never print them."""
    result = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"{cmd[0]} failed (details withheld from public logs)")
    return result.stdout


def probe(url, headers):
    data = json.loads(quiet(
        ["ffprobe", "-v", "error", "-headers", headers, "-i", url, "-select_streams", "v:0",
         "-show_entries", "format=duration,bit_rate:stream=codec_name,width,height,avg_frame_rate",
         "-of", "json"], 180))
    return {**data["format"], **data["streams"][0]}


def survey_sources(client, directory):
    """What each listed replay source is: its format and three stills."""
    info = client.get_sub_info(COURSE, SUB)
    now = info.get("now")
    now = int(now) if isinstance(now, str) else now
    found = {}
    for key, entry in (info.get("video_list") or {}).items():
        base = entry.get("preview_url") if isinstance(entry, dict) else None
        if not (isinstance(base, str) and base.endswith(".mp4")):
            continue
        try:
            url, headers = client.get_stream_params(client.sign_video_url(base, now=now))
            found[key] = probe(url, headers)
            for share in (0.2, 0.5, 0.8):
                at = int(float(found[key]["duration"]) * share)
                quiet(["ffmpeg", "-nostdin", "-y", "-headers", headers, *RECONNECT, "-ss", str(at),
                       "-i", url, "-frames:v", "1", "-q:v", "3",
                       str(directory / f"source-{key}-{at:05d}s.jpg")], 180)
        except Exception as exc:                  # a survey must not stop the trial
            found.setdefault(key, {})["error"] = type(exc).__name__
        print(f"Source {key}:", {k: found[key].get(k) for k in
              ("codec_name", "width", "height", "avg_frame_rate", "bit_rate", "duration", "error")},
              flush=True)
    return found


def sample_from(client, directory, start, done):
    """Sample from `start` seconds on; returns the times of the frames written."""
    url, headers = stream(client)
    seek = ["-ss", f"{start:.3f}"] if start else []
    cmd = ["ffmpeg", "-nostdin", "-nostats", "-y", "-headers", headers, *RECONNECT, *seek,
           "-i", url, "-map", "0:v:0", "-vf",
           f"setpts=PTS-STARTPTS,select='isnan(prev_selected_t)+gte(t-prev_selected_t,{INTERVAL})',showinfo",
           "-fps_mode", "vfr", "-start_number", str(done + 1), str(directory / "sample-%05d.png")]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    timer = threading.Timer(3600, proc.kill)
    timer.start()
    times = []
    try:
        for line in proc.stderr:
            match = re.search(rb"Parsed_showinfo.*\bn:\s*\d+.*pts_time:([\d.]+)", line)
            if match:
                times.append(start + float(match.group(1)))
                if (done + len(times)) % 200 == 0:
                    print(f"  sampled {done + len(times)} frames, at {clock(times[-1] * 1000)}", flush=True)
                    if shutil.disk_usage(directory).free < 2 * 1024 ** 3:
                        raise RuntimeError("Scratch disk nearly full")
        code = proc.wait()
    finally:
        timer.cancel()
        if proc.poll() is None:
            proc.kill()
        proc.wait()
    return times, code


def sample_frames(client, directory, duration):
    """The gateway can close the stream early and ffmpeg then exits cleanly, so
    resume after the last frame until the whole lecture is covered."""
    times, restarts = [], 0
    while not times or times[-1] < duration - 2 * INTERVAL - 10:
        start = times[-1] + INTERVAL if times else 0.0
        got, code = sample_from(client, directory, start, len(times))
        times += got
        if not got or code:
            restarts += 1
            print(f"  stream stopped (exit {code}, {len(got)} new frames); attempt {restarts}", flush=True)
            if restarts > 6:
                raise RuntimeError("Frame sampling incomplete (details withheld from public logs)")
            time.sleep(10)
        elif times[-1] < duration - 2 * INTERVAL - 10:
            print(f"  stream ended early at {clock(times[-1] * 1000)}; resuming", flush=True)
    paths = sorted(directory.glob("sample-*.png"))[:len(times)]
    if len(paths) != len(times):
        raise RuntimeError(f"Frame files and timestamps disagree: {len(paths)} vs {len(times)}")
    return times, paths, restarts


def strips(times, paths, every=120, cols=6, per_sheet=42):
    """Time-lapse of the whole room, one thumbnail every two minutes."""
    step = max(1, round(every / INTERVAL))
    picks = list(range(0, len(paths), step))
    sheets = []
    for start in range(0, len(picks), per_sheet):
        chunk = picks[start:start + per_sheet]
        sheet = Image.new("RGB", (cols * 480, -(-len(chunk) // cols) * 290), "white")
        draw = ImageDraw.Draw(sheet)
        for k, i in enumerate(chunk):
            x, y = k % cols * 480, k // cols * 290
            with Image.open(paths[i]) as frame:
                sheet.paste(frame.convert("RGB").resize((480, 270)), (x, y + 18))
            draw.text((x + 4, y + 3), clock(times[i] * 1000), fill="black")
        sheets.append(sheet)
    return sheets


def encrypt_bundle(directory, target):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        tar.add(directory, arcname="review")
    key_file = Path(__file__).with_name("board_review_key.pub")
    session = get_random_bytes(32)
    wrapped = PKCS1_OAEP.new(RSA.import_key(key_file.read_bytes()), hashAlgo=SHA256).encrypt(session)
    cipher = AES.new(session, AES.MODE_GCM, nonce=get_random_bytes(12))
    body, tag = cipher.encrypt_and_digest(buffer.getvalue())
    target.write_bytes(b"BRD1" + len(wrapped).to_bytes(2, "big") + wrapped + cipher.nonce + tag + body)
    return len(body)


def main():
    output = Path(os.environ["BOARD_OUTPUT"])
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    client = ICourseClient(login_with_retry())
    source_names, sources = describe_sources(client)
    print("Replay sources listed by the platform (field names only):",
          json.dumps(source_names, ensure_ascii=False), flush=True)
    video = probe(*stream(client))
    duration = float(video["duration"])
    print("Video:", {k: video.get(k) for k in ("codec_name", "width", "height", "avg_frame_rate",
                                               "bit_rate", "duration")}, flush=True)

    with tempfile.TemporaryDirectory(prefix="board-frames-") as scratch, \
            tempfile.TemporaryDirectory(prefix="board-review-") as review:
        review = Path(review)
        try:
            (review / "sources").mkdir()
            survey = survey_sources(client, review / "sources")
            t0 = time.time()
            times, paths, restarts = sample_frames(client, Path(scratch), duration)
            t_sample = time.time() - t0
            scratch_gb = sum(p.stat().st_size for p in paths) / 1024 ** 3
            print(f"Sampled {len(paths)} frames in {t_sample:.0f}s, {scratch_gb:.1f} GB, "
                  f"{restarts} restarts", flush=True)
            for k, sheet in enumerate(strips(times, paths)):
                sheet.save(review / f"strip-{k + 1}.jpg", quality=80)

            def load(i):
                with Image.open(paths[i]) as frame:
                    return np.asarray(frame.convert("RGB"))

            t0 = time.time()
            extractor = BoardExtractor(load)
            for i, t in enumerate(times):
                extractor.add(load(i), round(t * 1000))
                if (i + 1) % 400 == 0:
                    print(f"  analysed {i + 1} frames in {time.time() - t0:.0f}s", flush=True)
            t_track = time.time() - t0
            t0 = time.time()
            pages, report = extractor.finish()
            t_finish = time.time() - t0

            (review / "pages").mkdir()
            for k, page in enumerate(pages):
                page.image.save(review / "pages" / f"page-{k + 1:02d}.png")
            if pages:
                contact_sheet(pages).save(review / "sheet.jpg", quality=90)
            report.update({
                "course_id": COURSE, "sub_id": SUB, "sources": sources, "source_survey": survey,
                "stream_restarts": restarts,
                "video": {k: video.get(k) for k in ("codec_name", "width", "height",
                                                    "avg_frame_rate", "bit_rate", "duration")},
                "seconds": {"sampling": round(t_sample), "tracking": round(t_track),
                            "composing": round(t_finish), "total": round(time.time() - started)},
                "scratch_gb": round(scratch_gb, 2), "page_list": [p.meta() for p in pages]})
            (review / "report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=1, default=float))
        finally:
            # Whatever was produced before a failure is still worth reviewing.
            size = encrypt_bundle(review, output / "bundle.bin")

    print(f"Pages: {len(pages)} (from {report['candidates']} candidates, {len(report['wipes'])} wipes); "
          f"coverage {report['coverage']['share']}; dropped blocks {report['dropped_blocks']}", flush=True)
    for k, page in enumerate(pages):
        end = "end" if page.lost_ms is None else clock(page.lost_ms)
        print(f"  #{k + 1:02d} {clock(page.first_ms)}-{clock(page.last_ms)} gone {end} "
              f"rect {page.rect} blocks {page.ink_blocks} uncaptured {page.uncaptured:.3f} "
              f"absorbed {len(page.absorbed)}", flush=True)
    print(f"Seconds: sampling {t_sample:.0f}, tracking {t_track:.0f}, composing {t_finish:.0f}; "
          f"bundle {size / 1024 ** 2:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
