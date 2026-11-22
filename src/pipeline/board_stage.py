"""Board pages for the lectures of chosen courses: extract, read, keep, and
hand the readings to the summary.

Off unless BOARD_COURSE_IDS names courses.  For a lecture of those courses the
replay is sampled every few seconds, src/ai/board_capture.py turns the samples
into one image per board-full, src/ai/board_reader.py reads each image into
text, and src/pipeline/board_store.py keeps both.  The readings join the
summary prompt (prompt_with_boards) and follow the summary as a record
(summary_with_boards).

Lectures summarised before their boards were ready are handled by backfill():
at most BOARD_BACKFILL_PER_RUN of them per run, newest first, are captured and
summarised again from the stored transcript.

Nothing here may stop a lecture from being summarised.  notes_for(),
mark_in_summary() and backfill() catch their own failures; a lecture whose
boards failed gets its summary without them and is tried again by backfill().
"""
from __future__ import annotations

import io
import json
import re
import shutil
import subprocess
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from src.ai.board_capture import BoardExtractor
from src.ai.board_reader import EMPTY, BoardReader, clock
from src.data import crypto_box
from src.pipeline.board_store import BoardStore
from src.runtime import config

VERSION = 1          # of what is stored per lecture; raise it to redo every lecture
MAX_ATTEMPTS = 3     # failed captures of one lecture before it is left alone
INTERVAL = 5.0       # seconds between sampled frames
RECORD_HEADING = "### 板书记录"

GUIDE = (
    "【板书说明】材料末尾的【板书识读】是老师课上手写在黑板上的内容：程序从回放画面中逐块提取，"
    "再由模型识读成文字，每块标有书写的起止时间。这门课老师主要靠板书讲解，板书最能代表实际讲授的内容。\n"
    "1. 板书上的每个公式、推导步骤、定义、图示和例子，都必须写进笔记里对应的位置，不得遗漏；"
    "按时间把它和同一时段的讲解合并，用讲解把板书上的推导讲清楚。\n"
    "2. 板书识读可能有误，尤其是手写符号、上下标和分式；与录音明显矛盾时结合上下文修正，"
    "无法确定的照录并注明「板书此处识别不清」。标为 [?] 的地方是没认出来的字。\n"
    "3. 不要单独列一节「板书内容」，把它自然地并入正文；笔记末尾会由系统另附板书记录。\n"
    "4. 板书上只有只言片语的地方（一个数、一个箭头、半行式子），含义以同一时段的讲解为准；"
    "讲解里找不到依据时照录原文，不要自行解释它的含义。\n"
    "5. 标题层级的规定不变：只用 ###、####、#####，不要用 # 和 ##。\n\n"
)

RECONNECT = ["-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5"]


class ReplayUnavailable(RuntimeError):
    pass


# ── Frames from the replay ────────────────────────────────────────────────

def quiet(cmd, timeout):
    """Run ffmpeg/ffprobe.  Their errors can hold the signed URL and cookies,
    so only say which known kind of failure it was."""
    result = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if result.returncode:
        text = result.stderr.decode(errors="replace").lower()
        kinds = [k for k in (
            "403 forbidden", "404 not found", "server returned 4", "server returned 5",
            "connection timed out", "connection reset", "end of file", "invalid data",
            "invalid argument", "no such file", "error while decoding") if k in text]
        raise RuntimeError(f"{cmd[0]} failed with exit {result.returncode}: {kinds}")
    return result.stdout


def probe(url, headers):
    data = json.loads(quiet(
        ["ffprobe", "-v", "error", "-headers", headers, "-i", url, "-select_streams", "v:0",
         "-show_entries", "format=duration,bit_rate:stream=codec_name,width,height,avg_frame_rate",
         "-of", "json"], 180))
    return {**data["format"], **data["streams"][0]}


def _sample_from(stream, directory, start, done, interval, log):
    """Sample from `start` seconds on; returns the times of the frames written."""
    url, headers = stream()
    seek = ["-ss", f"{start:.3f}"] if start else []
    cmd = ["ffmpeg", "-nostdin", "-nostats", "-y", "-headers", headers, *RECONNECT, *seek,
           "-i", url, "-map", "0:v:0", "-vf",
           f"setpts=PTS-STARTPTS,select='isnan(prev_selected_t)+gte(t-prev_selected_t,{interval})',showinfo",
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
                if (done + len(times)) % 400 == 0:
                    log(f"    [Boards] sampled {done + len(times)} frames, at {clock(times[-1] * 1000)}")
                    if shutil.disk_usage(directory).free < 2 * 1024 ** 3:
                        raise RuntimeError("Scratch disk nearly full")
        code = proc.wait()
    finally:
        timer.cancel()
        if proc.poll() is None:
            proc.kill()
        proc.wait()
    return times, code


def sample_frames(stream, directory, duration, interval=INTERVAL, log=print):
    """One PNG every `interval` seconds of the replay.  `stream()` returns a
    freshly signed (url, headers).  The gateway can close the stream early and
    ffmpeg then exits cleanly, so resume after the last frame until the whole
    lecture is covered."""
    times, restarts = [], 0
    while not times or times[-1] < duration - 2 * interval - 10:
        start = times[-1] + interval if times else 0.0
        got, code = _sample_from(stream, directory, start, len(times), interval, log)
        times += got
        if not got or code:
            restarts += 1
            log(f"    [Boards] stream stopped (exit {code}, {len(got)} new frames); attempt {restarts}")
            if restarts > 6:
                raise RuntimeError("Frame sampling incomplete")
            time.sleep(10)
    paths = sorted(directory.glob("sample-*.png"))[:len(times)]
    if len(paths) != len(times):
        raise RuntimeError(f"Frame files and timestamps disagree: {len(paths)} vs {len(times)}")
    return times, paths, restarts


def capture(client, course_id, sub_id, interval=INTERVAL, log=print, inspect=None):
    """Board pages of one lecture: (pages, report).  `inspect(times, paths,
    extractor)` is called while the sampled frames still exist."""
    def stream():
        url = client.get_video_url(course_id, sub_id)
        if not url:
            raise ReplayUnavailable("no replay")
        return client.get_stream_params(url)          # signed afresh: a signature works once

    started = time.time()
    video = probe(*stream())
    duration = float(video["duration"])
    with tempfile.TemporaryDirectory(prefix="board-frames-") as scratch:
        times, paths, restarts = sample_frames(stream, Path(scratch), duration, interval, log)
        sampled = time.time()

        def load(i):
            with Image.open(paths[i]) as frame:
                return np.asarray(frame.convert("RGB"))

        extractor = BoardExtractor(load)
        for i, t in enumerate(times):
            extractor.add(load(i), round(t * 1000))
        pages, report = extractor.finish()
        if inspect:
            inspect(times, paths, extractor)
    report.update({
        "duration_ms": round(duration * 1000), "stream_restarts": restarts,
        "video": {k: video.get(k) for k in ("codec_name", "width", "height", "avg_frame_rate",
                                            "bit_rate", "duration")},
        "seconds": {"sampling": round(sampled - started), "extracting": round(time.time() - sampled)}})
    return pages, report


# ── Readings into the summary ─────────────────────────────────────────────

def speech_during(transcript, segments, first_ms, last_ms, duration_ms, limit=1600):
    """What was said while a board was being written, to help read it.  With no
    timed segments the transcript is cut by position, assuming even speech."""
    if segments:
        text = " ".join(s.get("text", "") for s in segments
                        if s.get("end_ms", 0) >= first_ms - 30000 and s.get("start_ms", 0) <= last_ms + 30000)
    elif transcript and duration_ms:
        slack = 0.03 * len(transcript)
        text = transcript[max(0, int(len(transcript) * first_ms / duration_ms - slack)):
                          int(len(transcript) * last_ms / duration_ms + slack)]
    else:
        text = ""
    text = text.strip()
    if len(text) > limit:
        text = text[:limit // 2] + " …… " + text[-limit // 2:]
    return text


def prompt_with_boards(prompt_text: str, notes: list[dict] | None) -> str:
    if not notes:
        return prompt_text
    boards = "\n\n".join(f"[板书 {n['n']} @ {clock(n['first_ms'])}–{clock(n['last_ms'])}]\n{n['text']}"
                         for n in notes)
    return f"{GUIDE}{prompt_text}\n\n【板书识读（按书写时间排列）】\n{boards}"


def _third_level(summary: str) -> str:
    """Headings above ### brought down to ###.  The summariser is told not to
    use them, and did when it was given boards (dry run of 2026-11-22)."""
    lines, fenced = [], False
    for line in summary.split("\n"):
        if line.lstrip().startswith("```"):
            fenced = not fenced
        lines.append(line if fenced else re.sub(r"^#{1,2}(?=\s)", "###", line))
    return "\n".join(lines)


def summary_with_boards(summary: str, notes: list[dict] | None) -> str:
    """The summary followed by every board's reading, so that nothing on a
    board depends on the summarising model having kept it."""
    if not notes or RECORD_HEADING in summary:
        return summary
    summary = _third_level(summary)
    parts = [f"{RECORD_HEADING}\n\n以下是程序从回放中提取的 {len(notes)} 块板书，由模型识读成文字，"
             "按书写时间排列，供核对。识读可能有误；原图已加密留存。"]
    parts += [f"#### 板书 {n['n']}（{clock(n['first_ms'])}–{clock(n['last_ms'])}）\n\n{n['text']}"
              for n in notes]
    return summary.rstrip() + "\n\n" + "\n\n".join(parts) + "\n"


# ── The stage ─────────────────────────────────────────────────────────────

class BoardStage:
    def __init__(self, store: BoardStore, reader: BoardReader, course_ids, log=print,
                 backfill_per_run: int = 0, deadline_s: float = 4 * 3600, capture_fn=capture):
        self.store = store
        self.reader = reader
        self.course_ids = [str(c) for c in course_ids]
        self.log = log
        self.backfill_per_run = backfill_per_run
        self.deadline_s = deadline_s      # no capture starts later than this into the run
        self.capture_fn = capture_fn
        self.started = time.monotonic()

    @classmethod
    def from_env(cls, log=print) -> "BoardStage | None":
        if not config.BOARD_COURSE_IDS:
            return None
        reader = BoardReader()
        if not reader.available():
            log("[Boards] Off for this run: GEMINI_API_KEY is not set.")
            return None
        try:
            key = Path(__file__).resolve().parents[2] / "scripts" / "board_review_key.pub"
            store = BoardStore(
                config.BOARD_STORE_DIR,
                crypto_box.derive_new_password(config.STUDENT_ID, config.PASSWORD),
                key.read_bytes() if key.exists() else None)
        except Exception as exc:
            log(f"[Boards] Off for this run: {type(exc).__name__}: {exc}")
            return None
        log(f"[Boards] On for {len(config.BOARD_COURSE_IDS)} course(s); "
            f"{len(store.lectures)} lecture(s) stored.")
        return cls(store, reader, config.BOARD_COURSE_IDS, log, config.BOARD_BACKFILL_PER_RUN)

    def wants(self, course_id) -> bool:
        return str(course_id) in self.course_ids

    def _usable(self, sub_id) -> dict | None:
        entry = self.store.get(sub_id)
        return entry if entry and entry.get("version") == VERSION else None

    def notes_for(self, client, course_id, sub_id, title, transcript="", segments=None):
        """The readings of a lecture's boards, [{"n", "first_ms", "last_ms",
        "lost_ms", "text"}], captured now if they are not stored yet.  None
        when there are none to use: course not chosen, capture failed or given
        up on, replay not out, or no time left in this run."""
        if not self.wants(course_id):
            return None
        try:
            return self._notes_for(client, str(course_id), str(sub_id), title, transcript, segments)
        except ReplayUnavailable:
            self.log(f"    [Boards] {sub_id}: replay is not available yet.")
        except Exception as exc:
            self.log(f"    [Boards] {sub_id} failed: {type(exc).__name__}"
                     f"{f': {exc}' if isinstance(exc, RuntimeError) else ''}")
        return None

    def _notes_for(self, client, course_id, sub_id, title, transcript, segments):
        entry = self._usable(sub_id)
        if entry and entry["status"] == "ok":
            return entry["notes"]
        attempts = entry["attempts"] if entry else 0
        if attempts >= MAX_ATTEMPTS:
            return None
        if time.monotonic() - self.started > self.deadline_s:
            self.log(f"    [Boards] {sub_id}: left for a later run, this one has gone on too long.")
            return None
        if not self.reader.preflight():
            self.log(f"    [Boards] {sub_id}: left for a later run, no vision model is answering.")
            return None

        self.log(f"    [Boards] {sub_id}: sampling the replay...")
        try:
            pages, report = self.capture_fn(client, course_id, sub_id, log=self.log)
            pages = sorted(pages, key=lambda p: p.first_ms)      # numbered in the order written
            texts, model = self.reader.read(title, [{
                "image": p.image, "first_ms": p.first_ms, "last_ms": p.last_ms,
                "context": speech_during(transcript, segments, p.first_ms, p.last_ms,
                                         report.get("duration_ms"))} for p in pages]) if pages else ([], "")
        except ReplayUnavailable:
            raise
        except Exception as exc:
            self.store.set(sub_id, {
                "course_id": course_id, "status": "failed", "attempts": attempts + 1,
                "version": VERSION, "updated": datetime.now().isoformat(timespec="seconds"),
                "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
            self.store.save()
            raise

        images, listed, notes = [], [], []
        for k, (page, text) in enumerate(zip(pages, texts)):
            buffer = io.BytesIO()
            page.image.save(buffer, "PNG")
            images.append(buffer.getvalue())
            listed.append({**page.meta(), "n": k + 1, "text": text})
            if text.strip() and text.strip() != EMPTY:
                notes.append({"n": k + 1, "first_ms": page.first_ms, "last_ms": page.last_ms,
                              "lost_ms": page.lost_ms, "text": text})
        file = self.store.put_pages(sub_id, {
            "sub_id": sub_id, "course_id": course_id, "title": title, "reader": model,
            "pages": listed, "report": report}, images)
        self.store.set(sub_id, {
            "course_id": course_id, "status": "ok", "attempts": attempts + 1, "version": VERSION,
            "updated": datetime.now().isoformat(timespec="seconds"), "file": file,
            "pages": len(pages), "reader": model, "in_summary": False, "notes": notes,
            "coverage": report.get("coverage", {}).get("share")})
        self.store.save()
        self.log(f"    [Boards] {sub_id}: {len(pages)} page(s), {len(notes)} with writing, read by "
                 f"{model or 'nobody'}; sampling {report['seconds']['sampling']}s, "
                 f"extracting {report['seconds']['extracting']}s.")
        return notes

    def mark_in_summary(self, sub_id) -> None:
        try:
            entry = self._usable(sub_id)
            if entry and entry["status"] == "ok" and not entry.get("in_summary"):
                self.store.set(sub_id, {**entry, "in_summary": True})
                self.store.save()
        except Exception as exc:
            self.log(f"    [Boards] could not record {sub_id} as summarised: {type(exc).__name__}")

    def backfill(self, client, db, summarizer, check_session=None) -> None:
        """Give board readings to lectures that were summarised without them."""
        if self.backfill_per_run <= 0 or summarizer is None:
            return
        try:
            self._backfill(client, db, summarizer, check_session)
        except Exception as exc:
            self.log(f"[Boards] Backfill stopped: {type(exc).__name__}"
                     f"{f': {exc}' if isinstance(exc, RuntimeError) else ''}")

    def _backfill(self, client, db, summarizer, check_session):
        from src.ai import bucketer
        from src.pipeline.lecture_selection import has_usable_summary

        handled = 0
        for course_id in self.course_ids:
            if check_session:
                check_session(client)
            detail = client.get_course_detail(course_id)
            for lecture in sorted(detail["lectures"], key=lambda l: str(l.get("date", "")), reverse=True):
                if handled >= self.backfill_per_run:
                    return
                sub_id = str(lecture["sub_id"])
                row = db.get_lecture(sub_id)
                if not has_usable_summary(row) or not (row.get("transcript") or "").strip():
                    continue
                entry = self._usable(sub_id)
                if entry and (entry.get("in_summary") or
                              (entry["status"] == "failed" and entry["attempts"] >= MAX_ATTEMPTS)):
                    continue
                if time.monotonic() - self.started > self.deadline_s:
                    return
                if check_session:
                    check_session(client)
                self.log(f"[Boards] Backfill: {lecture.get('sub_title', sub_id)}")
                handled += 1
                notes = self.notes_for(client, course_id, sub_id, detail["title"], row["transcript"])
                if notes is None:
                    continue
                if notes:
                    prompt, _ = bucketer.assemble(row["transcript"], None, db.get_done_ppt_pages(sub_id))
                    summary, model = summarizer.summarize(
                        detail["title"], prompt_with_boards(prompt, notes))
                    db.update_summary(sub_id, summary_with_boards(summary, notes), model)
                    self.log(f"    [Boards] {sub_id}: summary rewritten with {len(notes)} board(s) by {model}.")
                self.mark_in_summary(sub_id)
