"""One-lecture dry run of the board stage on a real replay.

Runs src/pipeline/board_stage.py on the lecture named by BOARD_COURSE_ID /
BOARD_SUB_ID exactly as the scheduled check would, but against a throwaway
store and without saving anything: board pages are extracted and read, and,
when BOARD_DB points at a decrypted database, the lecture is summarised again
with them.  $BOARD_OUTPUT/bundle.bin holds the pages, their readings, both
summaries, the store files, a contact sheet and report.json, encrypted to
scripts/board_review_key.pub.  The private key is not in this repository.

BOARD_PAGE_CACHE names a directory where the extracted pages are kept
(encrypted with the database password) so that a later run on the same lecture
skips the twenty minutes of replay sampling, and the login with it.
"""
import io
import json
import os
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image, ImageDraw

from src.ai.board_capture import contact_sheet
from src.ai.board_reader import BoardReader, clock
from src.data import crypto_box
from src.pipeline import board_stage, board_store
from src.pipeline.board_stage import BoardStage, probe
from src.pipeline.board_store import BoardStore
from src.runtime import config

COURSE = os.environ.get("BOARD_COURSE_ID", "38985")
SUB = os.environ.get("BOARD_SUB_ID", "693743")
INTERVAL = board_stage.INTERVAL
SKIP_READING = bool(os.environ.get("BOARD_SKIP_READING"))


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


def survey_sources(client):
    """The format of each listed replay source.

    Stills taken on 2026-11-21 showed that of the two sources of this lecture
    hall the first (type 3) faces the boards and the second (type 4) faces the
    audience, so only the first is read and no stills are kept.
    """
    def listed():
        info = client.get_sub_info(COURSE, SUB)
        now = info.get("now")
        now = int(now) if isinstance(now, str) else now
        return {key: (entry.get("preview_url"), now)
                for key, entry in (info.get("video_list") or {}).items() if isinstance(entry, dict)}

    found = {}
    for key, (base, _) in listed().items():
        if not (isinstance(base, str) and base.endswith(".mp4")):
            continue
        try:
            # A signed URL is refused on its second request: sign again for every one.
            base, now = listed()[key]
            found[key] = probe(*client.get_stream_params(client.sign_video_url(base, now=now)))
        except Exception as exc:                  # a survey must not stop the trial
            found[key] = {"error": str(exc) if isinstance(exc, RuntimeError) else type(exc).__name__}
        print(f"Source {key}:", {k: found[key].get(k) for k in
              ("codec_name", "width", "height", "avg_frame_rate", "bit_rate", "duration", "error")},
              flush=True)
    return found


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
    key = Path(__file__).with_name("board_review_key.pub").read_bytes()
    target.write_bytes(board_store.seal_for_review(buffer.getvalue(), key))
    return len(buffer.getvalue())


class Replay:
    """capture() for the stage: from the page cache when it has this lecture,
    else from the replay, keeping what the review needs along the way."""
    def __init__(self, review, password):
        self.review, self.password = review, password
        cache = os.environ.get("BOARD_PAGE_CACHE")
        self.cache = Path(cache) / f"pages-{SUB}.enc" if cache else None
        self.pages, self.report, self.cached = [], {}, False

    def __call__(self, client, course_id, sub_id, log=print):
        if self.cache and self.cache.exists():
            manifest, images = board_store.unpack(
                crypto_box.decrypt(self.cache.read_bytes(), self.password))
            self.pages = [SimpleNamespace(image=Image.open(io.BytesIO(image)).convert("RGB"),
                                          meta=lambda m=meta: m, **meta)
                          for meta, image in zip(manifest["pages"], images)]
            self.report, self.cached = manifest["report"], True
            print(f"Pages from the cache: {len(self.pages)}", flush=True)
            return self.pages, self.report

        from main import login_with_retry
        from src.api.icourse import ICourseClient
        client = ICourseClient(login_with_retry())
        source_names, sources = describe_sources(client)
        print("Replay sources listed by the platform (field names only):",
              json.dumps(source_names, ensure_ascii=False), flush=True)
        survey = survey_sources(client)

        def inspect(times, paths, extractor):
            for k, sheet in enumerate(strips(times, paths)):
                sheet.save(self.review / f"strip-{k + 1}.jpg", quality=80)
            # Block-level board and chalk masks (not images) for offline work on the tracker.
            np.savez_compressed(self.review / "features.npz", F=np.stack(extractor.feats) & 7,
                                times=np.asarray(times))

        self.pages, self.report = board_stage.capture(client, course_id, sub_id, log=log,
                                                      inspect=inspect)
        self.report.update({"sources": sources, "source_survey": survey})
        print("Video:", self.report["video"], flush=True)
        if self.cache:
            images = []
            for page in self.pages:
                buffer = io.BytesIO()
                page.image.save(buffer, "PNG")
                images.append(buffer.getvalue())
            self.cache.parent.mkdir(parents=True, exist_ok=True)
            self.cache.write_bytes(crypto_box.encrypt(board_store.pack(
                {"pages": [p.meta() for p in self.pages], "report": self.report}, images),
                self.password))
        return self.pages, self.report


def lecture_from_database():
    """(course title, lecture row, PPT pages) from $BOARD_DB, or blanks."""
    path = os.environ.get("BOARD_DB", "")
    if not (path and Path(path).exists()):
        print("No database: boards are read without speech and nothing is summarised.", flush=True)
        return "", {}, []
    from src.data.database import Database
    db = Database(path)
    row = db.get_lecture(SUB) or {}
    course = db.conn.execute("SELECT title FROM courses WHERE course_id = ?", (COURSE,)).fetchone()
    print(f"Database: transcript {len(row.get('transcript') or '')} chars, "
          f"summary {len(row.get('summary') or '')} chars", flush=True)
    return (course["title"] if course else ""), row, db.get_done_ppt_pages(SUB)


def main():
    output = Path(os.environ["BOARD_OUTPUT"])
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    password = crypto_box.derive_new_password(config.STUDENT_ID, config.PASSWORD)
    title, row, ppt_pages = lecture_from_database()
    transcript = row.get("transcript") or ""

    with tempfile.TemporaryDirectory(prefix="board-review-") as review, \
            tempfile.TemporaryDirectory(prefix="board-store-") as store_dir:
        review = Path(review)
        try:
            reader = BoardReader()
            if SKIP_READING:                  # a run about extraction only: spend no vision quota
                reader.models = []

            (Path(store_dir) / board_store.READY).write_text("none")
            store = BoardStore(store_dir, password,
                               Path(__file__).with_name("board_review_key.pub").read_bytes())
            replay = Replay(review, password)
            stage = BoardStage(store, reader, [COURSE], lambda m: print(m, flush=True),
                               capture_fn=replay)
            t0 = time.time()
            notes = stage.notes_for(None, COURSE, SUB, title or "课程", transcript)
            t_boards = time.time() - t0
            pages = sorted(replay.pages, key=lambda p: p.first_ms)   # the stage's numbering
            report = dict(replay.report)
            entry = store.get(SUB) or {}

            (review / "pages").mkdir()
            for k, page in enumerate(pages):
                page.image.save(review / "pages" / f"page-{k + 1:02d}.png")
            if pages and not replay.cached:
                contact_sheet(pages).save(review / "sheet.jpg", quality=90)
            for path in Path(store_dir).glob("*.rev"):       # what the boards branch would get
                (review / "store").mkdir(exist_ok=True)
                (review / "store" / path.name).write_bytes(path.read_bytes())
            read = notes or [{"n": int(k) + 1, "first_ms": pages[int(k)].first_ms,
                              "last_ms": pages[int(k)].last_ms, "text": text}
                             for k, text in sorted((entry.get("texts") or {}).items(), key=lambda i: int(i[0]))]
            (review / "readings.md").write_text("\n\n".join(
                f"## 板书 {n['n']}（{clock(n['first_ms'])}–{clock(n['last_ms'])}）\n\n{n['text']}"
                for n in read) + "\n")

            summary_info = {}
            if notes and transcript.strip():
                from src.ai import bucketer
                from src.ai.summarizer import Summarizer
                prompt, mode = bucketer.assemble(transcript, None, ppt_pages)
                prompt = board_stage.prompt_with_boards(prompt, notes)
                t0 = time.time()
                summary, model = Summarizer().summarize(title, prompt)
                summary = board_stage.summary_with_boards(summary, notes)
                (review / "summary-with-boards.md").write_text(summary)
                (review / "summary-before.md").write_text(row.get("summary") or "")
                (review / "prompt.txt").write_text(prompt)
                summary_info = {"model": model, "mode": mode, "prompt_chars": len(prompt),
                                "chars": len(summary), "before_chars": len(row.get("summary") or ""),
                                "before_model": row.get("summary_model"),
                                "seconds": round(time.time() - t0)}

            report.update({
                "course_id": COURSE, "sub_id": SUB, "from_cache": replay.cached,
                "reader": entry.get("reader"), "status": entry.get("status"),
                "error": entry.get("error"), "boards_with_writing": len(notes or []),
                "summary": summary_info, "page_list": [p.meta() for p in pages],
                "seconds_total": round(time.time() - started), "seconds_boards": round(t_boards)})
            (review / "report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=1, default=float))
        finally:
            # Whatever was produced before a failure is still worth reviewing.
            size = encrypt_bundle(review, output / "bundle.bin")

    print(f"Pages: {len(pages)}; with writing: {len(notes or [])}; read by {entry.get('reader')}; "
          f"status {entry.get('status')}; boards took {t_boards:.0f}s", flush=True)
    for k, page in enumerate(pages):
        end = "end" if page.lost_ms is None else clock(page.lost_ms)
        print(f"  #{k + 1:02d} {clock(page.first_ms)}-{clock(page.last_ms)} gone {end} "
              f"rect {tuple(page.rect)}", flush=True)
    if summary_info:
        print("Summary with boards:", summary_info, flush=True)
    print(f"Bundle {size / 1024 ** 2:.1f} MB", flush=True)
    if entry.get("status") != "ok" and not (SKIP_READING and entry.get("status") == "unread"):
        raise SystemExit("Board stage did not finish; see the review bundle.")


if __name__ == "__main__":
    main()
