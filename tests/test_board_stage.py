"""Board pages from capture to summary, with the replay, the vision model and
the summariser replaced by stand-ins."""
import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from Crypto.Cipher import AES, PKCS1_OAEP
from Crypto.Hash import SHA256
from Crypto.PublicKey import RSA
from PIL import Image

from src.ai.board_reader import BoardReader, parse
from src.data import crypto_box
from src.pipeline import board_stage, board_store
from src.pipeline.board_stage import BoardStage, ReplayUnavailable
from src.pipeline.board_store import BoardStore, StoreUnavailable

PASSWORD = "test-password"
SUMMARY = "### 复习优先级判断\n\n正文。\n\n### 第一部分\n\n更多正文。"


class Page(SimpleNamespace):
    def meta(self):
        return {"first_ms": self.first_ms, "last_ms": self.last_ms, "lost_ms": self.lost_ms}


def pages(count):
    return [Page(image=Image.new("RGB", (56, 28), (40, 60, 50)), first_ms=k * 600_000,
                 last_ms=k * 600_000 + 300_000, lost_ms=None) for k in range(count)]


class QuotaSpent(Exception):
    status_code = 429


class Vision:
    """Stands in for the OpenAI client: answers every board it is shown, until
    `quota` requests that show boards have been answered."""
    def __init__(self, skip_in_batches=(), broken_models=(), blank=(), quota=None):
        self.skip_in_batches, self.broken_models, self.blank = skip_in_batches, broken_models, blank
        self.quota = quota
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, model, messages, timeout):
        self.requests.append((model, messages))
        if model in self.broken_models:
            raise RuntimeError("model not found")
        content = messages[-1]["content"]
        if isinstance(content, str):
            return self._answer("OK")
        if self.quota is not None:
            if self.quota <= 0:
                raise QuotaSpent("quota exceeded")
            self.quota -= 1
        asked = [int(n) for part in content if part["type"] == "text"
                 for n in re.findall(r"^板书 (\d+)，", part["text"])]
        answered = [n for n in asked if len(asked) == 1 or n not in self.skip_in_batches]
        return self._answer("\n".join(
            f"<<<板书 {n}>>>\n" + ("（无可读内容）" if n in self.blank else f"$x_{n} = {n}$ on board {n}")
            for n in answered))

    @staticmethod
    def _answer(text):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def open_store(directory, review_key=None):
    (Path(directory) / board_store.READY).write_text("none")
    return BoardStore(directory, PASSWORD, review_key)


class Scene:
    """A stage over a temporary store, with a replay that yields `count` pages."""
    def __init__(self, test, count=3, vision=None, capture=None, directory=None, **options):
        self.dir = directory or tempfile.TemporaryDirectory()
        if directory is None:
            test.addCleanup(self.dir.cleanup)
        self.vision = vision or Vision()
        self.captures = 0
        self.log = []

        def default_capture(client, course_id, sub_id, log=print):
            self.captures += 1
            return pages(count), {"duration_ms": 3_600_000, "coverage": {"share": 0.98},
                                  "seconds": {"sampling": 1, "extracting": 1}}

        self.stage = BoardStage(
            open_store(self.dir.name),
            BoardReader(self.vision, ["model-a", "model-b"], pause=0, retry_wait=0),
            ["100"], self.log.append, capture_fn=capture or default_capture, **options)

    def reopened(self):
        return BoardStore(self.dir.name, PASSWORD)


class StoreTest(unittest.TestCase):
    def test_bundle_round_trip(self):
        manifest, images = board_store.unpack(board_store.pack({"sub_id": "7"}, [b"abc", b"", b"defg"]))
        self.assertEqual((manifest["sub_id"], images), ("7", [b"abc", b"", b"defg"]))

    def test_refuses_a_directory_that_was_not_fetched(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(StoreUnavailable):
                BoardStore(directory, PASSWORD)

    def test_refuses_an_index_it_cannot_read(self):
        with tempfile.TemporaryDirectory() as directory:
            store = open_store(directory)
            store.set("7", {"status": "ok"})
            store.save()
            with self.assertRaises(Exception):
                BoardStore(directory, "another-password")

    def test_saved_entries_and_pages_come_back(self):
        key = RSA.generate(2048)
        with tempfile.TemporaryDirectory() as directory:
            store = open_store(directory, key.publickey().export_key())
            name = store.put_pages("693743", {"sub_id": "693743"}, [b"png-bytes"])
            store.set("693743", {"status": "ok", "file": name})
            store.save()
            self.assertNotIn("693743", name)
            self.assertEqual(BoardStore(directory, PASSWORD).get("693743")["file"], name)
            blob = crypto_box.decrypt((Path(directory) / f"{name}.enc").read_bytes(), PASSWORD)
            self.assertEqual(board_store.unpack(blob)[1], [b"png-bytes"])
            # the review copy opens with the private key, as work/board-review does
            sealed = (Path(directory) / f"{name}.rev").read_bytes()
            size = int.from_bytes(sealed[4:6], "big")
            session = PKCS1_OAEP.new(key, hashAlgo=SHA256).decrypt(sealed[6:6 + size])
            rest = sealed[6 + size:]
            plain = AES.new(session, AES.MODE_GCM, nonce=rest[:12]).decrypt_and_verify(rest[28:], rest[12:28])
            self.assertEqual(plain, blob)


class ReaderTest(unittest.TestCase):
    def test_parse_keeps_only_the_boards_asked_for(self):
        answer = "preamble\n<<<板书 2>>>\n# Title\ntwo\n<<< 板书 3 >>>\nthree\n<<<板书 9>>>\nnine"
        self.assertEqual(parse(answer, [1, 2]), {1: "**Title**\ntwo", 2: "three"})

    def test_a_board_left_out_of_a_batch_is_asked_for_alone(self):
        vision = Vision(skip_in_batches=(2,))
        texts, model = BoardReader(vision, ["model-a"], pause=0).read("课", [
            {"image": p.image, "first_ms": p.first_ms, "last_ms": p.last_ms, "context": "speech"}
            for p in pages(5)])
        self.assertEqual(model, "gemini/model-a")
        self.assertEqual([texts[k][:5] for k in range(5)], [f"$x_{n} " for n in range(1, 6)])
        self.assertEqual(len(vision.requests), 3)           # boards 1-4, board 5, board 2 again

    def test_falls_back_to_the_next_model(self):
        vision = Vision(broken_models=("model-a",))
        reader = BoardReader(vision, ["model-a", "model-b"], pause=0)
        self.assertEqual(reader.read("课", [{"image": pages(1)[0].image, "first_ms": 0, "last_ms": 1}])[1],
                         "gemini/model-b")


class StageTest(unittest.TestCase):
    def test_captured_boards_are_read_stored_and_reused(self):
        scene = Scene(self, count=3, vision=Vision(blank=(2,)))
        notes = scene.stage.notes_for(object(), "100", "7", "课", "x" * 1000)
        self.assertEqual([n["n"] for n in notes], [1, 3])      # board 2 had nothing readable
        entry = scene.reopened().get("7")
        self.assertEqual((entry["status"], entry["pages"], entry["in_summary"]), ("ok", 3, False))
        manifest, images = scene.reopened().load_pages("7")
        self.assertEqual((len(images), [p["n"] for p in manifest["pages"]]), (3, [1, 2, 3]))
        self.assertEqual(scene.stage.notes_for(object(), "100", "7", "课"), notes)
        self.assertEqual(scene.captures, 1)

    def test_a_reading_cut_short_is_finished_later_without_capturing_again(self):
        scene = Scene(self, count=10, vision=Vision(quota=1))          # boards 1-4, then 429
        self.assertIsNone(scene.stage.notes_for(object(), "100", "7", "课"))
        entry = scene.reopened().get("7")
        self.assertEqual((entry["status"], sorted(entry["texts"], key=int)), ("unread", ["0", "1", "2", "3"]))
        # the same run does not touch boards again
        self.assertIsNone(scene.stage.notes_for(object(), "100", "8", "课"))
        self.assertEqual((scene.captures, scene.reopened().get("8")["status"]), (1, "pending"))

        later = Scene(self, count=10, directory=scene.dir)             # the next run
        notes = later.stage.notes_for(object(), "100", "7", "课")
        self.assertEqual([n["n"] for n in notes], list(range(1, 11)))
        self.assertEqual((later.captures, later.reopened().get("7")["status"]), (0, "ok"))
        shown = [n for _, messages in later.vision.requests if not isinstance(messages[-1]["content"], str)
                 for part in messages[-1]["content"] if part["type"] == "text"
                 for n in re.findall(r"^板书 (\d+)，", part["text"])]
        self.assertEqual(shown, ["5", "6", "7", "8", "9", "10"])      # only what was missing

    def test_other_courses_are_left_alone(self):
        scene = Scene(self)
        self.assertIsNone(scene.stage.notes_for(object(), "200", "7", "课"))
        self.assertEqual((scene.captures, scene.vision.requests), (0, []))

    def test_a_failing_lecture_is_given_up_after_three_attempts(self):
        def capture(client, course_id, sub_id, log=print):
            raise RuntimeError("Frame sampling incomplete")

        scene = Scene(self, capture=capture)
        for _ in range(5):
            self.assertIsNone(scene.stage.notes_for(object(), "100", "7", "课"))
        entry = scene.reopened().get("7")
        self.assertEqual((entry["status"], entry["attempts"]), ("failed", 3))
        self.assertEqual(scene.vision.requests, [])

    def test_a_replay_that_is_not_out_yet_does_not_count_as_a_failure(self):
        def capture(client, course_id, sub_id, log=print):
            raise ReplayUnavailable("no replay")

        scene = Scene(self, capture=capture)
        self.assertIsNone(scene.stage.notes_for(object(), "100", "7", "课"))
        self.assertIsNone(scene.reopened().get("7"))

    def test_pages_are_kept_when_no_model_answers(self):
        scene = Scene(self, vision=Vision(broken_models=("model-a", "model-b")))
        self.assertIsNone(scene.stage.notes_for(object(), "100", "7", "课"))
        entry = scene.reopened().get("7")
        self.assertEqual((scene.captures, entry["status"], entry["texts"]), (1, "unread", {}))
        self.assertEqual(len(scene.reopened().load_pages("7")[1]), 3)

    def test_nothing_is_captured_when_time_is_up(self):
        late = Scene(self, deadline_s=-1)
        self.assertIsNone(late.stage.notes_for(object(), "100", "7", "课"))
        self.assertEqual((late.captures, late.reopened().get("7")["status"]), (0, "pending"))

    def test_speech_during_a_board(self):
        segments = [{"start_ms": k * 60_000, "end_ms": k * 60_000 + 50_000, "text": f"s{k}"} for k in range(10)]
        # half a minute of slack either side takes in the end of s1
        self.assertEqual(board_stage.speech_during("", segments, 120_000, 180_000, None), "s1 s2 s3")
        flat = "".join(str(k) * 100 for k in range(10))
        cut = board_stage.speech_during(flat, None, 300_000, 400_000, 1_000_000)
        self.assertEqual(set(cut), {"2", "3", "4"})             # 3xx with a little slack either side


class SummaryTest(unittest.TestCase):
    NOTES = [{"n": 1, "first_ms": 60_000, "last_ms": 3_700_000, "lost_ms": None, "text": "$a = b$"},
             {"n": 3, "first_ms": 4_000_000, "last_ms": 4_100_000, "lost_ms": None, "text": "REINFORCE"}]

    def test_prompt_and_record(self):
        prompt = board_stage.prompt_with_boards("PROMPT", self.NOTES)
        self.assertIn("PROMPT\n\n【板书识读", prompt)
        self.assertIn("[板书 1 @ 0:01:00–1:01:40]\n$a = b$", prompt)
        summary = board_stage.summary_with_boards(SUMMARY, self.NOTES)
        self.assertIn("### 板书记录", summary)
        self.assertIn("#### 板书 3（1:06:40–1:08:20）\n\nREINFORCE", summary)
        self.assertEqual(board_stage.summary_with_boards(summary, self.NOTES), summary)
        self.assertEqual(board_stage.prompt_with_boards("PROMPT", []), "PROMPT")

    def test_headings_above_the_third_level_are_brought_down(self):
        text = "## 一\n\n### 二\n\n```python\n# comment\n## also code\n```\n\n# 三\n\n#hashtag"
        summary = board_stage.summary_with_boards(text, self.NOTES)
        self.assertTrue(summary.startswith(
            "### 一\n\n### 二\n\n```python\n# comment\n## also code\n```\n\n### 三\n\n#hashtag"))
        self.assertEqual(board_stage.summary_with_boards(SUMMARY, None), SUMMARY)

    def test_the_runner_puts_boards_into_the_summary(self):
        try:
            from src.pipeline.lecture_runner import LectureRunner
        except ImportError as exc:       # needs the OCR and ASR packages of requirements.txt
            self.skipTest(f"lecture runner not importable here: {exc}")
        seen = {}

        def summarize(title, prompt):
            seen["prompt"] = prompt
            return SUMMARY, "provider/model"

        scene = Scene(self)
        notes = scene.stage.notes_for(object(), "100", "7", "课")
        runner = LectureRunner.__new__(LectureRunner)
        runner._db = SimpleNamespace(update_summary=lambda sub, text, model: seen.update(saved=text),
                                     update_error=lambda *a: None)
        runner._summarizer = SimpleNamespace(summarize=summarize)
        runner._reporter = SimpleNamespace(info=lambda message: None)
        runner._boards = scene.stage
        runner._summarize("7", "课", "transcript text", None, [], notes)
        self.assertIn("【板书识读", seen["prompt"])
        self.assertIn("### 板书记录", seen["saved"])
        self.assertTrue(scene.reopened().get("7")["in_summary"])


class BackfillTest(unittest.TestCase):
    def run_backfill(self, scene, lectures, rows):
        saved = {}
        client = SimpleNamespace(get_course_detail=lambda course_id: {"title": "课", "lectures": lectures})
        db = SimpleNamespace(get_lecture=rows.get, get_done_ppt_pages=lambda sub_id: [],
                             update_summary=lambda sub_id, text, model: saved.update({sub_id: text}))
        summarizer = SimpleNamespace(summarize=lambda title, prompt: (SUMMARY, "provider/model"))
        scene.stage.backfill(client, db, summarizer)
        return saved

    def test_newest_lectures_first_and_only_those_already_summarised(self):
        lectures = [{"sub_id": 1, "date": "2026-10-14"}, {"sub_id": 2, "date": "2026-10-21"},
                    {"sub_id": 3, "date": "2026-10-28"}, {"sub_id": 4, "date": "2026-11-04"}]
        rows = {str(k): {"summary": SUMMARY, "transcript": "speech " * 50} for k in (1, 2, 3)}
        rows["4"] = {"summary": None, "transcript": None}       # not processed yet: the runner's job
        scene = Scene(self, backfill_per_run=2)
        saved = self.run_backfill(scene, lectures, rows)
        self.assertEqual(sorted(saved), ["2", "3"])
        self.assertIn("### 板书记录", saved["3"])
        store = scene.reopened()                                # the summary it replaced is kept
        kept = json.loads(crypto_box.decrypt(
            (Path(scene.dir.name) / f"{store.name('3')}.summary.enc").read_bytes(), PASSWORD))
        self.assertEqual((kept["summary"], store.get("3")["kept_summary"]), (SUMMARY, True))
        self.assertEqual(sorted(self.run_backfill(scene, lectures, rows)), ["1"])
        self.assertEqual(self.run_backfill(scene, lectures, rows), {})
        self.assertEqual(scene.captures, 3)

    def test_older_lectures_are_left_alone_by_default(self):
        scene = Scene(self)
        rows = {"1": {"summary": SUMMARY, "transcript": "speech"}}
        self.assertEqual(self.run_backfill(scene, [{"sub_id": 1, "date": "2026-10-14"}], rows), {})

    def test_a_lecture_left_for_later_is_done_in_the_next_run(self):
        lectures = [{"sub_id": 1, "date": "2026-10-14"}, {"sub_id": 2, "date": "2026-10-21"}]
        rows = {str(k): {"summary": SUMMARY, "transcript": "speech " * 50} for k in (1, 2)}
        scene = Scene(self, vision=Vision(broken_models=("model-a", "model-b")))
        self.assertIsNone(scene.stage.notes_for(object(), "100", "1", "课"))     # no model answering
        self.assertEqual(self.run_backfill(scene, lectures, rows), {})           # nor later in that run
        later = Scene(self, directory=scene.dir)                                 # backfill_per_run is 0
        saved = self.run_backfill(later, lectures, rows)
        self.assertEqual(sorted(saved), ["1"])
        self.assertTrue(later.reopened().get("1")["in_summary"])
        self.assertIsNone(later.reopened().get("2"))


if __name__ == "__main__":
    unittest.main()
