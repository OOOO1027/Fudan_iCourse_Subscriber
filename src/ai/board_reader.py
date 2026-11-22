"""Read extracted board pages into text with a vision model.

Uses Gemini through its OpenAI-compatible endpoint, with the GEMINI_API_KEY the
summarizer already falls back to.  A request carries a few pages and, for each,
the speech heard while it was being written; the model may use that speech only
to settle what an unclear mark says.
"""
from __future__ import annotations

import base64
import io
import os
import re
import time

from PIL import Image

BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
# Vision models with a free tier as of 2026-11.  On the lecture of 2026-10-28,
# gemini-4-hydrogen read every figure that could be checked correctly and
# marked 31 places as unreadable; gemini-3.6-flash marked one, and turned
# 1,740,032 into 1,760,032, which the summary then repeated.  A weaker model is
# therefore not a fallback: when these do not answer, the lecture waits for a
# later run.  The free tier allows 20 requests a day per model, failed ones
# included; a lecture of 22 boards takes 6.
MODELS = "gemini-4-hydrogen,gemini-3.8-flash"
BATCH = 4
EMPTY = "（无可读内容）"
MARK = re.compile(r"^\s*<<<\s*板书\s*(\d+)\s*>>>\s*$", re.M)

SYSTEM = """你在为一门大学课程转写老师的板书。每张图是一块黑板在被擦掉之前的样子，由课堂回放的画面拼成：分辨率低，可能有拼接痕迹、反光和没擦干净的残痕。

要求：
1. 只转写图上看得见的粉笔字、公式和图示，按板面上的位置顺序写（通常从左到右分栏，每栏从上到下）。
2. 公式用 LaTeX：行内 $...$，单独成行 $$...$$，LaTeX 里不要出现中文。板上写的是英文就保留英文，不要翻译。
3. 图示（坐标图、框图、箭头关系等）用一两句话说明画了什么，以"【图】"开头。
4. 每块板附有同一时段的录音转写。它只用来辨认看不清的字和符号，比如判断一个潦草的词是哪个术语；录音里讲到但板上没写的内容不要写进来。
5. 认不出的地方写 [?]，不要猜。整块板没有可读内容时只写"（无可读内容）"。
6. 不要解释，不要总结，不要用 Markdown 标题，只输出转写本身。"""


def clock(ms) -> str:
    s = int(ms) // 1000
    return f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}"


def _kind(exc: Exception) -> str:
    """What failed, without the response body."""
    status = getattr(exc, "status_code", None)
    return f"{type(exc).__name__}{f' {status}' if status else ''}"


def _quota(exc: Exception) -> str:
    """The body of a 429, which names the quota that ran out.  Long digit runs
    (project numbers) are cut."""
    body = getattr(exc, "body", None) or getattr(getattr(exc, "response", None), "text", "") or exc
    return re.sub(r"\d{8,}", "…", str(body))[:900]


def _transient(exc: Exception) -> bool:
    """Worth one more try.  Not a 429: on the free tier that is the daily quota."""
    if getattr(exc, "status_code", None) in (500, 502, 503, 504):
        return True
    return any(word in type(exc).__name__ for word in ("Timeout", "Connection"))


def encode(image: Image.Image) -> str:
    """Pages are about 560x280; enlarged so the model spends more detail on them."""
    scale = min(3.0, 2048 / max(image.size))
    if scale > 1:
        image = image.resize((round(image.width * scale), round(image.height * scale)),
                             Image.LANCZOS)
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, "PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def tidy(text: str) -> str:
    text = re.sub(r"^```\w*\s*$", "", text.strip(), flags=re.M)
    return re.sub(r"^#{1,6}\s*(.+)$", r"**\1**", text, flags=re.M).strip()


def parse(answer: str, wanted: list[int]) -> dict[int, str]:
    """Readings by page index from an answer split by <<<板书 n>>> lines."""
    parts = MARK.split(answer)
    found = {}
    for number, text in zip(parts[1::2], parts[2::2]):
        k = int(number) - 1
        if k in wanted and text.strip():
            found[k] = tidy(text)
    if len(wanted) == 1 and not found and answer.strip():
        found[wanted[0]] = tidy(answer)
    return found


class BoardReader:
    def __init__(self, client=None, models: list[str] | None = None, pause: float = 4.0,
                 retry_wait: float = 20.0):
        self.models = models or [m.strip() for m in (os.environ.get(
            "BOARD_VISION_MODELS", "").strip() or MODELS).split(",") if m.strip()]
        self.pause = pause                # between requests
        self.retry_wait = retry_wait      # before asking again after a 429 / 5xx / timeout
        self._client = client
        self._key = os.environ.get("GEMINI_API_KEY", "").strip()

    def available(self) -> bool:
        return self._client is not None or bool(self._key)

    def _api(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(                  # retries are done here, visibly
                api_key=self._key, max_retries=0,
                base_url=os.environ.get("GEMINI_BASE_URL", "").strip() or BASE_URL)
        return self._client

    def read(self, title: str, pages: list[dict],
             have: dict[int, str] | None = None) -> tuple[dict[int, str], str]:
        """pages: [{"image": PIL image, "first_ms", "last_ms", "context"}].
        Returns the readings by page index, those in `have` included, and the
        model that produced the last of them.  When no model can go on (a
        spent quota, an outage) the readings so far come back and the caller
        asks again another time with them as `have`."""
        texts, used = dict(have or {}), ""
        for model in self.models:
            if len(texts) == len(pages):
                break
            before = len(texts)
            try:
                self._read_with(model, title, pages, texts)
            except Exception as exc:
                detail = f": {_quota(exc)}" if getattr(exc, "status_code", None) == 429 else ""
                print(f"[Boards] {model} stopped after {len(texts)} of {len(pages)} boards: "
                      f"{_kind(exc)}{detail}", flush=True)
            if len(texts) > before:
                used = model
        return texts, f"gemini/{used}" if used else ""

    def _read_with(self, model: str, title: str, pages: list[dict], texts: dict[int, str]) -> None:
        for size in (BATCH, 1):                     # pages a batch skipped are asked for singly
            todo = [k for k in range(len(pages)) if k not in texts]
            for at in range(0, len(todo), size):
                chunk, t0 = todo[at:at + size], time.time()
                texts.update(self._ask(model, title, pages, chunk))
                print(f"[Boards] {model} was shown boards {chunk[0] + 1}-{chunk[-1] + 1}: "
                      f"{time.time() - t0:.0f}s", flush=True)
                time.sleep(self.pause)
        if len(texts) < len(pages):
            raise RuntimeError("the answer left some boards out")

    def _ask(self, model: str, title: str, pages: list[dict], chunk: list[int]) -> dict[int, str]:
        content = [{"type": "text", "text":
                    f"课程：《{title}》。下面是 {len(chunk)} 块板。"
                    "每块板的转写以单独一行 <<<板书 编号>>> 开头，编号用下面给出的编号。"}]
        for k in chunk:
            page = pages[k]
            content.append({"type": "text", "text":
                            f"板书 {k + 1}，写于 {clock(page['first_ms'])}–{clock(page['last_ms'])}。"
                            f"同一时段的录音转写（可能有识别错误）：\n{page.get('context') or '（无）'}"})
            content.append({"type": "image_url", "image_url": {"url": encode(page["image"])}})
        for attempt in (1, 2):                      # failed requests count against the quota too
            try:
                response = self._api().chat.completions.create(
                    model=model, timeout=180,
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": content}])
                break
            except Exception as exc:
                if attempt == 2 or not _transient(exc):
                    raise
                time.sleep(self.retry_wait)
        return parse(response.choices[0].message.content or "", chunk)
