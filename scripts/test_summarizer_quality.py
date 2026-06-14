import sys
import types
from pathlib import Path


fake_openai = types.ModuleType("openai")
fake_openai.OpenAI = object
sys.modules.setdefault("openai", fake_openai)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ai.summarizer import Summarizer


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


if __name__ == "__main__":
    test_pathological_whitespace_summary_is_retried()
    test_normal_long_markdown_summary_is_accepted()
    print("summarizer quality checks ok")
