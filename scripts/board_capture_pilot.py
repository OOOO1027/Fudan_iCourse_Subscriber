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


def probe(client):
    url, headers = stream(client)
    # ffmpeg and ffprobe errors can contain the signed URL and cookies: never print them.
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-headers", headers, "-i", url, "-select_streams", "v:0",
         "-show_entries", "format=duration,bit_rate:stream=codec_name,width,height,avg_frame_rate",
         "-of", "json"], capture_output=True, timeout=180)
    if result.returncode:
        raise RuntimeError("ffprobe failed (details withheld from public logs)")
    data = json.loads(result.stdout)
    return {**data["format"], **data["streams"][0]}


def sample_frames(client, directory, duration):
    url, headers = stream(client)
    cmd = ["ffmpeg", "-nostdin", "-nostats", "-y", "-headers", headers, "-i", url, "-map", "0:v:0", "-vf",
           f"setpts=PTS-STARTPTS,select='isnan(prev_selected_t)+gte(t-prev_selected_t,{INTERVAL})',showinfo",
           "-fps_mode", "vfr", str(directory / "sample-%05d.png")]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    timer = threading.Timer(3600, proc.kill)
    timer.start()
    times = []
    try:
        for line in proc.stderr:
            match = re.search(rb"Parsed_showinfo.*\bn:\s*\d+.*pts_time:([\d.]+)", line)
            if match:
                times.append(float(match.group(1)))
                if len(times) % 200 == 0:
                    print(f"  sampled {len(times)} frames, at {clock(times[-1] * 1000)}", flush=True)
                    if shutil.disk_usage(directory).free < 2 * 1024 ** 3:
                        raise RuntimeError("Scratch disk nearly full")
        code = proc.wait()
    finally:
        timer.cancel()
        if proc.poll() is None:
            proc.kill()
        proc.wait()
    paths = sorted(directory.glob("sample-*.png"))
    if code or len(paths) != len(times) or not times or times[-1] < duration - 2 * INTERVAL - 10:
        raise RuntimeError(f"Frame sampling incomplete: exit {code}, {len(paths)} files, "
                           f"{len(times)} timestamps (details withheld from public logs)")
    return times, paths


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
    video = probe(client)
    duration = float(video["duration"])
    print("Video:", {k: video.get(k) for k in ("codec_name", "width", "height", "avg_frame_rate",
                                               "bit_rate", "duration")}, flush=True)

    with tempfile.TemporaryDirectory(prefix="board-frames-") as scratch, \
            tempfile.TemporaryDirectory(prefix="board-review-") as review:
        t0 = time.time()
        times, paths = sample_frames(client, Path(scratch), duration)
        t_sample = time.time() - t0
        scratch_gb = sum(p.stat().st_size for p in paths) / 1024 ** 3
        print(f"Sampled {len(paths)} frames in {t_sample:.0f}s, {scratch_gb:.1f} GB", flush=True)

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

        review = Path(review)
        (review / "pages").mkdir()
        for k, page in enumerate(pages):
            page.image.save(review / "pages" / f"page-{k + 1:02d}.png")
        if pages:
            contact_sheet(pages).save(review / "sheet.jpg", quality=90)
        for k, sheet in enumerate(strips(times, paths)):
            sheet.save(review / f"strip-{k + 1}.jpg", quality=80)
        report.update({
            "course_id": COURSE, "sub_id": SUB, "sources": sources,
            "video": {k: video.get(k) for k in ("codec_name", "width", "height",
                                                "avg_frame_rate", "bit_rate", "duration")},
            "seconds": {"sampling": round(t_sample), "tracking": round(t_track),
                        "composing": round(t_finish), "total": round(time.time() - started)},
            "scratch_gb": round(scratch_gb, 2), "page_list": [p.meta() for p in pages]})
        (review / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=float))
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
