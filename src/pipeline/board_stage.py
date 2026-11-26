"""Board pages for the lectures of chosen courses: extract, read, keep, and
hand the readings to the summary.

Off unless BOARD_COURSE_IDS names courses.  For a lecture of those courses the
replay is sampled every few seconds, src/ai/board_capture.py turns the samples
into one image per board-full, src/ai/board_reader.py reads each image into
text, and src/pipeline/board_store.py keeps both.  The readings join the
summary prompt (prompt_with_boards) and follow the summary as a record
(summary_with_boards).

A lecture whose boards could not be done in time (no vision model answering,
a quota spent halfway through the reading, a failed capture) is summarised
without them and marked in the store; backfill() does it in a later run,
reading on from the pages and readings already stored, and summarises it again
from the stored transcript.  The summary it replaces is kept in the store.  backfill() also takes up to BOARD_BACKFILL_PER_RUN lectures per
run, newest first, that were summarised before boards were switched on.

Nothing here may stop a lecture from being summarised.  notes_for(),
mark_in_summary() and backfill() catch their own failures; a lecture whose
boards failed gets its summary without them and is tried again by backfill().
"""
from __future__ import annotations

import difflib
import io
import json
import re
import shutil
import subprocess
import tempfile
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from src.ai.board_capture import BoardExtractor
from src.ai.board_reader import EMPTY, BoardReader, clock
from src.data import crypto_box
from src.pipeline.board_store import BoardStore
from src.runtime import config

# Of what is stored per lecture; raise it to redo every lecture.  2: the lecture
# of 2026-11-18 was given up after three failures that were bugs here.
VERSION = 2
MAX_ATTEMPTS = 3     # failed captures of one lecture before it is left alone
INTERVAL = 5.0       # seconds between sampled frames
STALL = 180          # seconds without a new frame before a stream counts as stuck
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
    "5. 标题层级的规定不变：只用 ###、####、#####，不要用 # 和 ##。\n"
    "6. 板书上的数值结果，如果能用同一块板上的公式和给出的数值算出来，自己验算一遍；"
    "验算结果和识读的数字不一致时，以验算为准，并注明板书上识读成了另一个数。\n\n"
)

KEEP = (
    "【已有笔记说明】这节课此前已有一份没有用到板书的笔记，附在材料最后。新笔记是它的增订版："
    "已有笔记里的每个知识点、推导、例子、数字和课程事项都必须保留，不得删减或一笔带过；"
    "在此基础上并入板书内容，并订正已有笔记中与板书或录音不符的地方。篇幅应当比已有笔记长。"
    "只输出一份完整的笔记，按课堂顺序把两份材料合并在一起：不要先写一遍新笔记、再把已有笔记接在后面，"
    "同一个主题只出现一次。\n\n"
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
    beat = [time.monotonic()]

    def watch():                                   # the stream can hang without closing
        while proc.poll() is None:
            if time.monotonic() - beat[0] > STALL:
                proc.kill()
                return
            time.sleep(5)

    threading.Thread(target=watch, daemon=True).start()
    times = []
    try:
        for line in proc.stderr:
            match = re.search(rb"Parsed_showinfo.*\bn:\s*\d+.*pts_time:([\d.]+)", line)
            if match:
                beat[0] = time.monotonic()
                times.append(start + float(match.group(1)))
                if (done + len(times)) % 400 == 0:
                    log(f"    [Boards] sampled {done + len(times)} frames, at {clock(times[-1] * 1000)}")
                    if shutil.disk_usage(directory).free < 2 * 1024 ** 3:
                        raise RuntimeError("Scratch disk nearly full")
        code = proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
    return times, code


def _whole(path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def sample_frames(stream, directory, duration, interval=INTERVAL, log=print):
    """One PNG every `interval` seconds of the replay.  `stream()` returns a
    freshly signed (url, headers).  The gateway can close the stream early, and
    ffmpeg then exits cleanly, or leave it hanging, and ffmpeg is then killed:
    either way sampling resumes after the last frame that was written whole."""
    times, restarts = [], 0
    while not times or times[-1] < duration - 2 * interval - 10:
        start = times[-1] + interval if times else 0.0
        got, code = _sample_from(stream, directory, start, len(times), interval, log)
        times += got
        # A stopped ffmpeg may have announced frames it never finished writing.
        while times and not _whole(directory / f"sample-{len(times):05d}.png"):
            times.pop()
        if not got or code:
            restarts += 1
            log(f"    [Boards] stream stopped (exit {code}, {len(got)} new frames); attempt {restarts}")
            if restarts > 6:
                raise RuntimeError("Frame sampling incomplete")
            time.sleep(10)
    paths = [directory / f"sample-{k + 1:05d}.png" for k in range(len(times))]
    if not all(path.exists() for path in paths):
        raise RuntimeError("Sampled frames are missing")
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
        if inspect:                       # before finish(), which is where a new lecture breaks
            inspect(times, paths, extractor)
        pages, report = extractor.finish()
    report.update({
        "duration_ms": round(duration * 1000), "stream_restarts": restarts,
        "video": {k: video.get(k) for k in ("codec_name", "width", "height", "avg_frame_rate",
                                            "bit_rate", "duration")},
        "seconds": {"sampling": round(sampled - started), "extracting": round(time.time() - sampled)}})
    return pages, report


def _where(exc: Exception) -> str:
    """The innermost place in this repository an exception passed through.  The
    message stays out of the public log (it can hold a signed URL); the store
    keeps it."""
    frames = [f for f in traceback.extract_tb(exc.__traceback__) if "/src/" in f.filename]
    return f" at {Path(frames[-1].filename).name}:{frames[-1].lineno}" if frames else ""


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


def prompt_with_boards(prompt_text: str, notes: list[dict] | None, previous: str = "") -> str:
    """The summary prompt with the boards' readings.  `previous` is the summary
    the lecture already has, when it is being rewritten: a rewrite from the
    same material dropped things the first one had (2026-11-18: the reverse
    process and its training objective), so the first one is given too, to be
    kept."""
    if not notes:
        return prompt_text
    boards = "\n\n".join(f"[板书 {n['n']} @ {clock(n['first_ms'])}–{clock(n['last_ms'])}]\n{n['text']}"
                         for n in notes)
    prompt = f"{GUIDE}{prompt_text}\n\n【板书识读（按书写时间排列）】\n{boards}"
    previous = previous.split(RECORD_HEADING)[0].strip()
    if previous:
        prompt = (KEEP + prompt + "\n\n【已有笔记（没有用到板书）】\n" + previous)
    return prompt


def _heading_keys(summary: str) -> list[str]:
    keys = []
    for line in summary.split(RECORD_HEADING)[0].splitlines():
        match = re.match(r"^#{3,5}\s+(.*)", line)
        if match:
            key = re.sub(r"[\s$`*【】\\{}^_：:、，,（）()\-–—]", "", match.group(1)).lower()
            if len(key) >= 4:
                keys.append(key)
    return keys


def repeats(summary: str, need: int = 4) -> bool:
    """Whether a summary covers the lecture twice.  A rewrite given the old
    summary to keep once wrote the new one and then the old one after it
    (2026-11-18): seven of its 57 headings had a near twin, against at most
    one in every other summary on hand."""
    seen, twins = [], 0
    for key in _heading_keys(summary):
        twins += any(difflib.SequenceMatcher(None, key, other).ratio() >= 0.75 for other in seen)
        seen.append(key)
    return twins >= need


def _read_by(notes: list[dict]) -> str:
    """Which models read these boards, for the record: they differ in how far
    a handwritten digit can be trusted."""
    names = notes[0].get("by", "").replace("gemini/", "")
    return f"识读型号：{names}。" if names else ""


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
             "按书写时间排列，供核对。识读可能有误，数字、上下标和分式以原图为准；[?] 是没认出来的地方。"
             f"原图已加密留存。{_read_by(notes)}"]
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
        self.reader_down = False          # set when reading stops short; no more boards this run

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
            self.log(f"    [Boards] {sub_id} failed{_where(exc)}: {type(exc).__name__}"
                     f"{f': {exc}' if isinstance(exc, RuntimeError) else ''}")
        return None

    def _set(self, sub_id, entry: dict) -> None:
        """Store an entry.  That the lecture's first summary is kept carries
        over from whatever entry it had, of any version: it is kept only once."""
        if (self.store.get(sub_id) or {}).get("kept_summary"):
            entry = {**entry, "kept_summary": True}
        self.store.set(sub_id, entry)

    def _notes_for(self, client, course_id, sub_id, title, transcript, segments):
        entry = self._usable(sub_id)
        if entry and entry["status"] == "ok":
            return entry["notes"]
        attempts = entry["attempts"] if entry else 0
        unread = bool(entry) and entry["status"] == "unread"
        if attempts >= MAX_ATTEMPTS and not unread:
            return None
        late = time.monotonic() - self.started > self.deadline_s
        if late or self.reader_down:
            self.log(f"    [Boards] {sub_id}: left for a later run, "
                     f"{'this one has gone on too long' if late else 'the vision models stopped answering'}.")
            if entry is None:                 # so that backfill() comes back to it
                self._set(sub_id, {
                    "course_id": course_id, "status": "pending", "attempts": 0, "version": VERSION,
                    "updated": datetime.now().isoformat(timespec="seconds")})
                self.store.save()
            return None

        stored = self.store.load_pages(sub_id) if unread else None
        if stored:                            # extracted in an earlier run, not yet read in full
            manifest, images = stored
            pages = [SimpleNamespace(image=Image.open(io.BytesIO(image)).convert("RGB"), **meta)
                     for meta, image in zip(manifest["pages"], images)]
            report, file = manifest["report"], entry["file"]
            have = {int(k): text for k, text in entry.get("texts", {}).items()}
            readers = [r for r in entry.get("reader", "").split(", ") if r]
        elif attempts >= MAX_ATTEMPTS:        # the stored pages are gone and capturing again is used up
            return None
        else:
            self.log(f"    [Boards] {sub_id}: sampling the replay...")
            try:
                pages, report = self.capture_fn(client, course_id, sub_id, log=self.log)
            except ReplayUnavailable:
                raise
            except Exception as exc:
                self._set(sub_id, {
                    "course_id": course_id, "status": "failed", "attempts": attempts + 1,
                    "version": VERSION, "updated": datetime.now().isoformat(timespec="seconds"),
                    "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
                self.store.save()
                raise
            attempts += 1
            pages = sorted(pages, key=lambda p: p.first_ms)      # numbered in the order written
            images = []
            for page in pages:
                buffer = io.BytesIO()
                page.image.save(buffer, "PNG")
                images.append(buffer.getvalue())
            file = self.store.put_pages(sub_id, {
                "sub_id": sub_id, "course_id": course_id, "title": title, "report": report,
                "pages": [{**page.meta(), "n": k + 1} for k, page in enumerate(pages)]}, images)
            have, readers = {}, []
            self.log(f"    [Boards] {sub_id}: {len(pages)} page(s); sampling "
                     f"{report['seconds']['sampling']}s, extracting {report['seconds']['extracting']}s.")

        base = {"course_id": course_id, "attempts": attempts, "version": VERSION, "file": file,
                "pages": len(pages), "coverage": report.get("coverage", {}).get("share"),
                "updated": datetime.now().isoformat(timespec="seconds")}
        texts, model = self.reader.read(title, [{
            "image": p.image, "first_ms": p.first_ms, "last_ms": p.last_ms,
            "context": speech_during(transcript, segments, p.first_ms, p.last_ms,
                                     report.get("duration_ms"))} for p in pages], have) if pages else ({}, "")
        model = ", ".join(dict.fromkeys(readers + [r for r in model.split(", ") if r]))   # over all runs
        if len(texts) < len(pages):           # quota spent or models down: go on from here next run
            self.reader_down = True
            self._set(sub_id, {**base, "status": "unread", "reader": model,
                                    "texts": {str(k): text for k, text in texts.items()}})
            self.store.save()
            self.log(f"    [Boards] {sub_id}: {len(texts)} of {len(pages)} boards read; "
                     "the rest in a later run.")
            return None
        notes = [{"n": k + 1, "first_ms": page.first_ms, "last_ms": page.last_ms,
                  "lost_ms": page.lost_ms, "text": texts[k], "by": model}
                 for k, page in enumerate(pages) if texts[k].strip() and texts[k].strip() != EMPTY]
        self._set(sub_id, {**base, "status": "ok", "reader": model, "in_summary": False,
                                "notes": notes})
        self.store.save()
        self.log(f"    [Boards] {sub_id}: {len(pages)} board(s) read by {model or 'nobody'}, "
                 f"{len(notes)} with writing.")
        return notes

    def mark_in_summary(self, sub_id) -> None:
        try:
            entry = self._usable(sub_id)
            if entry and entry["status"] == "ok" and not entry.get("in_summary"):
                self.store.set(sub_id, {**entry, "in_summary": True})
                self.store.save()
        except Exception as exc:
            self.log(f"    [Boards] could not record {sub_id} as summarised: {type(exc).__name__}")

    def _owed(self, entry: dict) -> bool:
        return not entry.get("in_summary") and entry.get("rewrites", 0) < MAX_ATTEMPTS and not (
            entry["status"] == "failed" and entry["attempts"] >= MAX_ATTEMPTS)

    def backfill(self, client, db, summarizer, check_session=None) -> None:
        """Give board readings to lectures that were summarised without them:
        those this stage left for later, and, up to backfill_per_run per run,
        those summarised before it was switched on."""
        if summarizer is None:
            return
        if self.backfill_per_run <= 0 and not any(
                e.get("version") == VERSION and self._owed(e) for e in self.store.lectures.values()):
            return
        try:
            self._backfill(client, db, summarizer, check_session)
        except Exception as exc:
            self.log(f"[Boards] Backfill stopped: {type(exc).__name__}"
                     f"{f': {exc}' if isinstance(exc, RuntimeError) else ''}")

    def _backfill(self, client, db, summarizer, check_session):
        from src.ai import bucketer
        from src.pipeline.lecture_selection import has_usable_summary

        handled = older = 0
        for course_id in self.course_ids:
            if check_session:
                check_session(client)
            detail = client.get_course_detail(course_id)
            for lecture in sorted(detail["lectures"], key=lambda l: str(l.get("date", "")), reverse=True):
                if handled >= max(1, self.backfill_per_run):
                    return
                sub_id = str(lecture["sub_id"])
                row = db.get_lecture(sub_id)
                if not has_usable_summary(row) or not (row.get("transcript") or "").strip():
                    continue
                entry = self._usable(sub_id)
                if entry is None and older >= self.backfill_per_run:
                    continue                  # from before boards were on: only when asked for
                redo = bool(entry and entry.get("in_summary") and repeats(row["summary"]))
                if entry is not None and not self._owed(entry) and not redo:
                    continue
                if redo and entry.get("rewrites", 0) >= MAX_ATTEMPTS:
                    continue
                if self.reader_down or time.monotonic() - self.started > self.deadline_s:
                    return
                older += entry is None
                if check_session:
                    check_session(client)
                self.log(f"[Boards] Backfill: {lecture.get('sub_title', sub_id)}")
                handled += 1
                notes = self.notes_for(client, course_id, sub_id, detail["title"], row["transcript"])
                if notes is None:
                    continue
                if notes:
                    entry = self.store.get(sub_id)
                    if not entry.get("kept_summary"):    # the one written without boards, once
                        self.store.put_summary(sub_id, row["summary"], row.get("summary_model"))
                        self.store.set(sub_id, {**entry, "kept_summary": True})
                        self.store.save()
                        original = row["summary"]
                    else:                                # rewritten before: keep to the first one
                        original = self.store.load_summary(sub_id)
                        if original is None:
                            self.log(f"    [Boards] {sub_id}: the first summary could not be read back.")
                            continue
                    prompt, _ = bucketer.assemble(row["transcript"], None, db.get_done_ppt_pages(sub_id))
                    prompt = prompt_with_boards(prompt, notes, original)
                    for attempt in (1, 2):
                        summary, model = summarizer.summarize(detail["title"], prompt)
                        if not repeats(summary):
                            break
                        self.log(f"    [Boards] {sub_id}: the rewrite covers the lecture twice "
                                 f"(attempt {attempt}).")
                    entry = self.store.get(sub_id)
                    if repeats(summary):                 # leave the summary as it is; another run tries
                        self._set(sub_id, {**entry, "in_summary": False,
                                           "rewrites": entry.get("rewrites", 0) + 1})
                        self.store.save()
                        continue
                    db.update_summary(sub_id, summary_with_boards(summary, notes), model)
                    self.log(f"    [Boards] {sub_id}: summary rewritten with {len(notes)} board(s) by {model}"
                             f"{' (it had covered the lecture twice)' if redo else ''}.")
                    if redo:
                        self._set(sub_id, {**entry, "in_summary": False})
                        self.store.save()
                self.mark_in_summary(sub_id)
