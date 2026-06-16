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
from src.pipeline.lecture_selection import (
    lecture_needs_processing,
    promote_hidden_playbacks,
)
from src.pipeline.lecture_runner import LectureRunner


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


BAD_SUMMARY = (
    "### 复习优先级判断\n"
    "| 级别 | 必修课内容 1."
    + (" " * 250_000)
)

BAD_SEPARATOR_SUMMARY = (
    "### 复习优先级判断\n\n"
    "| 必须掌握 | 需要理解 | 可以略读 | 判断依据 |\n"
    + "\n".join("-" * 280 for _ in range(60))
)

BAD_SINGLE_LINE_TABLE_SUMMARY = (
    "### 复习优先级判断\n\n"
    "| 必须掌握 | 需要理解 | 可以略读 | 判断依据 |\n"
    + "| 必须掌握 | "
    + ("细胞结构与功能、膜转运、实验技术。" * 20_000)
)

BAD_TABLE_ONLY_SUMMARY = (
    "### 复习优先级判断\n\n"
    "| 类别 | 知识点 |\n"
    "|:------|:"
    + ("-" * 180)
    + "|\n"
    + "| 必须掌握 | "
    + "显微镜技术、FRET、FRAP、离心技术、SDS-PAGE、免疫沉淀、CRISPR/Cas9、细胞膜结构、跨膜运输。"
    * 35
    + " |\n"
    + "| 需要理解 | "
    + "扫描电子显微镜与冷冻电镜、亲和层析、糖脂和胆固醇、膜蛋白分类、脂筏模型、ATP驱动泵。"
    * 35
    + " |\n"
    + "| 可以略读 | "
    + "诺贝尔奖年份、具体仪器操作、HeLa细胞历史、部分药物耐药机制细节。"
    * 25
    + " |\n"
    + "| 判断依据 | 老师强调的重点集中在实验技术和膜运输，但这里没有展开正文笔记结构。 |\n"
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


def test_pathological_separator_summary_is_rejected():
    try:
        Summarizer._validate_summary_text(BAD_SEPARATOR_SUMMARY)
    except Exception as e:
        assert_true(
            e.__class__.__name__ == "InvalidSummaryError",
            "separator-polluted output should raise InvalidSummaryError",
        )
        return
    raise AssertionError("separator-polluted output should be rejected")


def test_extremely_long_single_line_summary_is_rejected():
    try:
        Summarizer._validate_summary_text(BAD_SINGLE_LINE_TABLE_SUMMARY)
    except Exception as e:
        assert_true(
            e.__class__.__name__ == "InvalidSummaryError",
            "single-line table output should raise InvalidSummaryError",
        )
        return
    raise AssertionError("single-line table output should be rejected")


def test_table_only_priority_summary_is_rejected():
    try:
        Summarizer._validate_summary_text(BAD_TABLE_ONLY_SUMMARY)
    except Exception as e:
        assert_true(
            e.__class__.__name__ == "InvalidSummaryError",
            "table-only output should raise InvalidSummaryError",
        )
        return
    raise AssertionError("table-only output should be rejected")


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


def test_enumerator_requeues_processed_bad_summary_only():
    processed = {"bad", "good", "no_summary"}

    assert_true(
        lecture_needs_processing(
            {"sub_id": "bad", "has_playback": True},
            {"summary": BAD_SUMMARY},
            processed,
        ) is True,
        "processed lecture with malformed summary should be requeued",
    )
    assert_true(
        lecture_needs_processing(
            {"sub_id": "bad-separator", "has_playback": True},
            {"summary": BAD_SEPARATOR_SUMMARY},
            processed | {"bad-separator"},
        ) is True,
        "processed lecture with separator-polluted summary should be requeued",
    )
    assert_true(
        lecture_needs_processing(
            {"sub_id": "bad-single-line", "has_playback": True},
            {"summary": BAD_SINGLE_LINE_TABLE_SUMMARY},
            processed | {"bad-single-line"},
        ) is True,
        "processed lecture with single-line table summary should be requeued",
    )
    assert_true(
        lecture_needs_processing(
            {"sub_id": "bad-table", "has_playback": True},
            {"summary": BAD_TABLE_ONLY_SUMMARY},
            processed | {"bad-table"},
        ) is True,
        "processed lecture with table-only summary should be requeued",
    )
    assert_true(
        lecture_needs_processing(
            {"sub_id": "good", "has_playback": True},
            {"summary": GOOD_SUMMARY},
            processed,
        ) is False,
        "processed lecture with valid summary should still be skipped",
    )
    assert_true(
        lecture_needs_processing(
            {"sub_id": "no_summary", "has_playback": True},
            {"summary": ""},
            processed,
        ) is False,
        "processed lecture without a summary should keep the existing skip behavior",
    )
    assert_true(
        lecture_needs_processing(
            {"sub_id": "hidden_no_summary", "has_playback": True, "hidden_playback": True},
            {"summary": ""},
            processed | {"hidden_no_summary"},
        ) is True,
        "processed hidden playback without a summary should be requeued",
    )
    assert_true(
        lecture_needs_processing(
            {"sub_id": "hidden_ppt_only_old", "has_playback": True, "hidden_playback": True},
            {"summary": GOOD_SUMMARY, "transcript": ""},
            processed | {"hidden_ppt_only_old"},
        ) is True,
        "processed hidden playback with old PPT-only summary should be requeued",
    )
    assert_true(
        lecture_needs_processing(
            {"sub_id": "hidden_ppt_only_fixed", "has_playback": True, "hidden_playback": True},
            {
                "summary": "### 材料边界说明\n本节没有可用语音转写；以下内容仅基于 PPT OCR。\n\n" + GOOD_SUMMARY,
                "transcript": "",
            },
            processed | {"hidden_ppt_only_fixed"},
        ) is False,
        "processed hidden playback with honest PPT-only marker should be kept",
    )
    assert_true(
        lecture_needs_processing(
            {"sub_id": "new", "has_playback": True},
            None,
            processed,
        ) is True,
        "new playback lecture should still be queued",
    )
    assert_true(
        lecture_needs_processing(
            {"sub_id": "no_playback", "has_playback": False},
            None,
            processed,
        ) is False,
        "lecture without playback should not be queued",
    )


def test_hidden_playback_probe_promotes_lecture_for_queueing():
    lectures = [
        {"sub_id": "576352", "sub_title": "2026-03-23第6-8节", "has_playback": False},
        {"sub_id": "585415", "sub_title": "2026-04-06第6-8节", "has_playback": False},
        {"sub_id": "566700", "sub_title": "2026-03-09第6-8节", "has_playback": True},
    ]
    probed = []

    def has_hidden_video(sub_id):
        probed.append(sub_id)
        return sub_id == "576352"

    promoted, promoted_count = promote_hidden_playbacks(lectures, has_hidden_video)

    assert_true(promoted_count == 1, "only the hidden-video lecture should be promoted")
    assert_true(
        promoted[0]["has_playback"] is True and promoted[0]["hidden_playback"] is True,
        "hidden video lecture should become eligible for processing",
    )
    assert_true(
        lecture_needs_processing(promoted[0], None, set()) is True,
        "promoted hidden video lecture should enter the processing queue",
    )
    assert_true(
        promoted[1]["has_playback"] is False,
        "lecture with no deep video URL should remain non-playback",
    )
    assert_true(
        probed == ["576352", "585415"],
        "existing playback lectures should not trigger hidden-video probes",
    )


def test_empty_transcript_can_still_summarize_ppt_ocr():
    assert_true(
        LectureRunner._has_summary_material("", [{"text": "PPT OCR 内容"}]) is True,
        "PPT OCR text should be enough to generate a fallback summary",
    )
    assert_true(
        LectureRunner._has_summary_material("", [{"text": "   "}]) is False,
        "blank PPT OCR should not generate an empty summary request",
    )
    assert_true(
        LectureRunner._has_summary_material("老师讲了重点", []) is True,
        "normal transcript text should still be enough to summarize",
    )


def test_ppt_only_summary_gets_source_boundary():
    prompt = LectureRunner._add_ppt_only_prompt_warning("", "【PPT 文字识别】\n内容")
    assert_true(
        "没有可用音频转录" in prompt and "不要写“老师强调”" in prompt,
        "PPT-only prompts should forbid oral-emphasis claims",
    )
    summary = LectureRunner._apply_ppt_only_summary_notice("", GOOD_SUMMARY)
    assert_true(
        summary.startswith("### 材料边界说明\n本节没有可用语音转写"),
        "PPT-only summaries should be explicitly marked in saved output",
    )
    assert_true(
        LectureRunner._apply_ppt_only_summary_notice("老师讲了内容", GOOD_SUMMARY) == GOOD_SUMMARY,
        "normal transcript summaries should not receive the PPT-only notice",
    )


if __name__ == "__main__":
    test_pathological_whitespace_summary_is_retried()
    test_pathological_separator_summary_is_rejected()
    test_extremely_long_single_line_summary_is_rejected()
    test_table_only_priority_summary_is_rejected()
    test_normal_long_markdown_summary_is_accepted()
    test_runner_does_not_skip_bad_existing_summary()
    test_enumerator_requeues_processed_bad_summary_only()
    test_hidden_playback_probe_promotes_lecture_for_queueing()
    test_empty_transcript_can_still_summarize_ppt_ocr()
    test_ppt_only_summary_gets_source_boundary()
    print("summarizer quality checks ok")
