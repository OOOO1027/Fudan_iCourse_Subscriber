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
# Vision models with a free tier as of 2026-11, most capable first.
MODELS = "gemini-3.8-flash,gemini-4-hydrogen,gemini-3.6-flash,gemini-3-flash-preview"
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


def _transient(exc: Exception) -> bool:
    if getattr(exc, "status_code", None) in (429, 500, 502, 503, 504):
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
    def __init__(self, client=None, models: list[str] | None = None, pause: float = 4.0):
        self.models = models or [m.strip() for m in os.environ.get(
            "BOARD_VISION_MODELS", MODELS).split(",") if m.strip()]
        self.pause = pause
        self._client = client
        self._key = os.environ.get("GEMINI_API_KEY", "").strip()

    def available(self) -> bool:
        return self._client is not None or bool(self._key)

    def _api(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self._key,
                base_url=os.environ.get("GEMINI_BASE_URL", "").strip() or BASE_URL)
        return self._client

    def preflight(self) -> bool:
        """Is a model answering right now?  Asked before the twenty minutes of
        frame sampling; the first model that answers goes to the front."""
        if not self.available():
            return False
        for model in list(self.models):
            try:
                self._api().chat.completions.create(
                    model=model, messages=[{"role": "user", "content": "Reply with OK."}],
                    timeout=60)
            except Exception as exc:
                print(f"[Boards] {model} is not answering: {_kind(exc)}", flush=True)
                continue
            self.models.remove(model)
            self.models.insert(0, model)
            return True
        return False

    def read(self, title: str, pages: list[dict]) -> tuple[list[str], str]:
        """pages: [{"image": PIL image, "first_ms", "last_ms", "context"}].
        Returns one reading per page and the model that produced them."""
        error = None
        for model in self.models:
            try:
                return self._read_with(model, title, pages), f"gemini/{model}"
            except Exception as exc:
                error = exc
                print(f"[Boards] {model} could not read the boards: {_kind(exc)}", flush=True)
        raise RuntimeError(f"No vision model read the boards ({_kind(error)})")

    def _read_with(self, model: str, title: str, pages: list[dict]) -> list[str]:
        texts: dict[int, str] = {}
        for size in (BATCH, 1):                     # pages a batch skipped are asked for singly
            todo = [k for k in range(len(pages)) if k not in texts]
            for at in range(0, len(todo), size):
                texts.update(self._ask(model, title, pages, todo[at:at + size]))
                time.sleep(self.pause)
        if len(texts) < len(pages):
            raise RuntimeError("the answer left some boards out")
        return [texts[k] for k in range(len(pages))]

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
        for attempt in (1, 2, 3):
            try:
                response = self._api().chat.completions.create(
                    model=model, timeout=240,
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": content}])
                break
            except Exception as exc:
                if attempt == 3 or not _transient(exc):
                    raise
                time.sleep(30 * attempt)
        return parse(response.choices[0].message.content or "", chunk)
