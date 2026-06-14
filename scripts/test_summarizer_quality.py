import sys
import types
from pathlib import Path


fake_openai = types.ModuleType("openai")
fake_openai.OpenAI = object
sys.modules.setdefault("openai", fake_openai)

fake_imagehash = types.ModuleType("imagehash")
fake_imagehash.dhash = lambda image: "0" * 16
sys.modules.setdefault("imagehash", fake_imagehash)

fake_rapidocr = types.ModuleType("rapidocr_onnxruntime")
fake_rapidocr.RapidOCR = lambda: None
sys.modules.setdefault("rapidocr_onnxruntime", fake_rapidocr)

fake_sherpa = types.ModuleType("sherpa_onnx")
sys.modules.setdefault("sherpa_onnx", fake_sherpa)

fake_ppt_pipeline = types.ModuleType("src.pipeline.ppt_pipeline")
fake_ppt_pipeline.PPTPipeline = object
sys.modules.setdefault("src.pipeline.ppt_pipeline", fake_ppt_pipeline)

fake_transcriber = types.ModuleType("src.ai.transcriber")
fake_transcriber.IncompleteAudioError = type(
    "IncompleteAudioError", (Exception,), {}
)
fake_transcriber.NoAudioStreamError = type(
    "NoAudioStreamError", (Exception,), {}
)
sys.modules.setdefault("src.ai.transcriber", fake_transcriber)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ai.summarizer import Summarizer
from src.pipeline.lecture_runner import LectureRunner


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


BAD_SUMMARY = (
    "### 复习优先级判断\n"
    "| 级别 | 必修课内容 1."
    + (" " * 250_000)
)

GOOD_SUMMARY = (
    "### 复习优先级判断\n\n"
    "| 级别 | 内容 |\n"
    "|---|---|\n"
    "| 必须掌握 | 细胞周期调控、信号通路、实验方法。 |\n\n"
    + ("这段笔记保留了老师讲解的重点、判断依据和复习层级。\n" * 800)
)


class FakeSummarizer(Summarizer):
    def __init__(self, responses):
        self.providers = [
            {"name": "fake", "models": ["fake-model"]},
        ]
        self._clients = {"fake": object()}
        self._responses = list(responses)
        self.calls = 0

    def _call_llm(self, client, model, title, content):
        self.calls += 1
        return self._responses.pop(0)


def test_pathological_whitespace_summary_is_retried():
    summarizer = FakeSummarizer([BAD_SUMMARY, GOOD_SUMMARY])

    summary, model = summarizer.summarize("细胞生物学", "课堂转录和 PPT OCR")

    assert_true(summary == GOOD_SUMMARY, "pathological whitespace output should be rejected and retried")
    assert_true(model == "fake/fake-model", "successful retry should keep provider/model id")
    assert_true(summarizer.calls == 2, "summarizer should retry after rejecting bad output")


def test_normal_long_markdown_summary_is_accepted():
    summarizer = FakeSummarizer([GOOD_SUMMARY])

    summary, _ = summarizer.summarize("细胞生物学", "课堂转录和 PPT OCR")

    assert_true(summary == GOOD_SUMMARY, "normal long markdown should not be changed or rejected")
    assert_true(summarizer.calls == 1, "valid output should not be retried")


def test_runner_does_not_skip_bad_existing_summary():
    assert_true(
        LectureRunner._has_summary({"summary": BAD_SUMMARY}) is False,
        "bad existing summary should be treated as missing so single_run can rewrite it",
    )
    assert_true(
        LectureRunner._has_summary({"summary": GOOD_SUMMARY}) is True,
        "valid existing summary should still be skipped",
    )


if __name__ == "__main__":
    test_pathological_whitespace_summary_is_retried()
    test_normal_long_markdown_summary_is_accepted()
    test_runner_does_not_skip_bad_existing_summary()
    print("summarizer quality checks ok")
