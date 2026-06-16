"""Per-lecture state machine: prefetch → ASR → OCR drain → summarize → release.

One ``LectureRunner`` instance drives one lecture from "started" to either
"summary saved" or "deliberately skipped".  The class is single-use — make a
new instance per lecture so error state can't leak across runs.

Phases (named like the original ``main.process_lecture`` for diff-friendly
log greps):

  A  short-circuit ``summary already exists`` → mark processed, return.
  B  ``PPTPipeline.submit``: stages 1-3 inline, OCR jobs submitted to the
     scheduler pool.  Returns a ``PPTAsyncHandle``.
  C  schedule **next** lecture's prefetch (image + audio) so its
     download overlaps with the current lecture's ASR — the audio slot
     for the next lecture starts filling while we still hold ours.
  D  if no cached transcript: ``Scheduler.audio_downloader.get`` (blocks
     for the ffmpeg spawn that AudioDownloader scheduled earlier), then
     ``Transcriber.transcribe_tail`` reads PCM from the disk file with
     tail-f semantics while ffmpeg keeps writing.
  E  ``handle.drain()`` blocks for any remaining OCR jobs.
  F  ``bucketer.assemble`` builds the prompt; ``Summarizer.summarize``
     calls the LLM round-robin until one succeeds.
  G  persist ``update_summary``, ``mark_processed``, ``clear_error``.
  H  ``audio_downloader.release`` kills ffmpeg + deletes scratch file.

The runner never owns the WebVPN session check; the orchestrator
(LectureRunner's caller) calls ``_check_session`` between lectures so all
threads pick up refreshed cookies through the shared ``ICourseClient``.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

from src.ai import bucketer
from src.pipeline.lecture_selection import (
    PPT_ONLY_SUMMARY_MARKER,
    has_usable_summary,
)
from src.pipeline.ppt_pipeline import PPTPipeline
from src.ai.transcriber import IncompleteAudioError, NoAudioStreamError

if TYPE_CHECKING:
    from src.data.database import Database
    from src.api.icourse import ICourseClient
    from src.runtime.reporter import Reporter
    from src.runtime.scheduler import Scheduler
    from src.ai.transcriber import Transcriber


class LectureRunner:
    """Run one lecture end-to-end.  Construct once per orchestration session,
    call ``run`` for each lecture in order."""

    def __init__(self, client: "ICourseClient", db: "Database",
                 scheduler: "Scheduler", transcriber: "Transcriber",
                 summarizer: "Summarizer", reporter: "Reporter"):
        self._client = client
        self._db = db
        self._scheduler = scheduler
        self._transcriber = transcriber
        self._summarizer = summarizer
        self._reporter = reporter
        self._ppt = PPTPipeline(db, scheduler, reporter)

    # ── Public entry point ──────────────────────────────────────────────

    def run(self, course_id: str, course_title: str, lecture: dict,
            next_info: Optional[tuple[str, str]] = None) -> Optional[str]:
        """Process one lecture.  Returns the summary text or None.

        ``next_info``: ``(course_id, sub_id)`` of the next lecture, used to
        kick off its prefetch concurrently.  Pass ``None`` for the last
        lecture in the batch.
        """
        sub_id = str(lecture["sub_id"])
        sub_title = lecture.get("sub_title", sub_id)
        date = lecture.get("date", "")
        t_start = time.time()
        self._reporter.lecture_start(course_title, sub_title, date)

        existing = self._db.get_lecture(sub_id)
        # ── Phase A — short-circuit if a v2 summary already exists ──────
        summary_ready = self._has_summary(existing)
        if existing and existing.get("summary") and not summary_ready:
            self._reporter.info(
                "    Existing summary is malformed; regenerating summary."
            )
        if summary_ready:
            self._reporter.lecture_skip_v2_done(
                sub_title, len(existing["summary"])
            )
            self._schedule_next(next_info)
            self._db.mark_processed(sub_id)
            self._db.clear_error(sub_id)
            return existing["summary"]

        # ── Phase B — submit PPT pipeline (fetch + dedup, no OCR yet) ──
        # OCR is deferred (defer_ocr=True) so ASR in Phase D gets exclusive
        # CPU.  OCR will be submitted in Phase E (handle.drain()).
        ppt_handle = self._ppt.submit(
            self._client, course_id, sub_id, defer_ocr=True,
        )

        # ── Phase C — schedule next lecture's prefetch ─────────────────
        # Done BEFORE ASR so the next audio download can start filling its
        # AudioDownloader slot while we transcribe.  Both audio + image
        # prefetches are idempotent so this is safe to call any time.
        self._schedule_next(next_info)

        # ── Phase D — ASR transcription ────────────────────────────────
        transcript, transcript_segments = self._get_transcript(
            existing, course_id, sub_id,
        )
        if transcript is None:
            # _get_transcript already logged + persisted the skip reason.
            return None

        # ── Phase E — drain remaining OCR work ─────────────────────────
        ppt_stats = ppt_handle.drain()
        _ = ppt_stats  # stats are emitted by PPTAsyncHandle.drain via reporter

        # ── Phase E2 — kick off next lecture's OCR (runs during LLM) ───
        # Images were prefetched in Phase C; switch to OCR phase by
        # submitting pending pages to the OCR pool so they run in the
        # background while this lecture's LLM call waits for the API.
        if next_info:
            next_course, next_sub = next_info
            try:
                self._ppt.prefetch_and_ocr(
                    self._client, next_course, next_sub,
                )
            except Exception as e:
                self._reporter.info(
                    f"    [Prefetch OCR] {next_sub} failed: "
                    f"{type(e).__name__}: {e}"
                )

        # ── Phase F — bucketed-prompt LLM summary ──────────────────────
        kept_pages = self._db.get_done_ppt_pages(sub_id)
        if not self._has_summary_material(transcript, kept_pages):
            self._reporter.info(
                "    Empty transcript and PPT OCR, skipping summary."
            )
            self._release_audio(sub_id)
            self._db.mark_processed(sub_id)
            self._db.clear_error(sub_id)
            return None

        summary = self._summarize(
            sub_id, course_title, transcript, transcript_segments, kept_pages,
        )
        if summary is None:
            self._release_audio(sub_id)
            return None

        # ── Phase G — persist + clear errors ───────────────────────────
        self._db.mark_processed(sub_id)
        self._db.clear_error(sub_id)

        # ── Phase H — release audio resources (ffmpeg + file) ──────────
        self._release_audio(sub_id)

        elapsed = time.time() - t_start
        self._reporter.lecture_done(course_title, sub_title, elapsed)
        return summary

    # ── Internal helpers ────────────────────────────────────────────────

    @staticmethod
    def _has_summary(existing: dict | None) -> bool:
        return has_usable_summary(existing)

    @staticmethod
    def _has_summary_material(transcript: str, kept_pages: list[dict]) -> bool:
        if transcript and transcript.strip():
            return True
        return any((page.get("text") or "").strip() for page in kept_pages or [])

    @staticmethod
    def _add_ppt_only_prompt_warning(transcript: str, prompt_text: str) -> str:
        if transcript and transcript.strip():
            return prompt_text
        warning = (
            "【材料边界】本节没有可用音频转录，下面只有 PPT OCR。"
            "只能按 PPT 页面内容、结构位置、重复出现和推导完整度做复习整理；"
            "不要写“老师强调”“老师提到”“课堂口头提示”“考试提示”等"
            "只有音频转录才能支持的判断。若无法判断口头重点，必须明确写"
            "“无语音转写，无法判断老师口头强调”。\n\n"
        )
        return warning + prompt_text

    @staticmethod
    def _apply_ppt_only_summary_notice(transcript: str, summary: str) -> str:
        if transcript and transcript.strip():
            return summary
        if PPT_ONLY_SUMMARY_MARKER in summary:
            return summary
        notice = (
            "### 材料边界说明\n"
            "本节没有可用语音转写；以下内容仅基于 PPT OCR。"
            "它不能代表老师口头强调、课堂临场讲解、板书补充或考试提示。"
            "复习优先级只能依据 PPT 内容出现频率、结构位置、推导完整度和"
            "跨章节基础性判断。\n\n"
        )
        return notice + summary

    def _schedule_next(self, next_info: Optional[tuple[str, str]]):
        if next_info is None:
            return
        next_course, next_sub = next_info
        # Image + audio prefetch in one call. Both are idempotent so
        # repeated invocations are no-ops; the audio side will block on its
        # semaphore until a download slot frees.
        self._scheduler.prefetch_lecture(self._client, next_course, next_sub)

    def _get_transcript(self, existing: dict | None, course_id: str,
                        sub_id: str) -> tuple[Optional[str], Optional[list]]:
        """Return (transcript, segments) or (None, None) on skip.

        Reuses an existing transcript if present (segments==None — bucketer
        falls back to flat mode).  Otherwise pulls the prefetched audio
        handle, runs ``transcribe_tail``, and writes the transcript.  On
        ``NoAudioStreamError`` returns (None, None) after marking the
        lecture as a deliberate skip.
        """
        if existing and existing.get("transcript"):
            self._reporter.info(
                f"    Transcript exists "
                f"({len(existing['transcript'])} chars), "
                f"skipping transcription."
            )
            return existing["transcript"], None

        # Pull the audio handle.  ``schedule`` is idempotent — usually the
        # previous lecture already kicked it off (Phase C), but for the
        # first lecture in the batch we still need to fire it ourselves.
        downloader = self._scheduler.audio_downloader
        downloader.schedule(self._client, course_id, sub_id)
        try:
            handle = downloader.get(sub_id, timeout=120)
        except TimeoutError as e:
            self._reporter.info(f"    [SKIP] {e}")
            self._db.update_error(sub_id, "transcribe", str(e))
            return None, None
        if handle is None:
            # AudioDownloader returns None when get_video_url() returned
            # None — i.e. the lecture has no playable video.
            self._reporter.lecture_skip_no_video(
                existing.get("sub_title", sub_id) if existing else sub_id
            )
            return None, None

        try:
            transcript, segments = self._transcriber.transcribe_tail(
                handle.path, handle.process, handle.stderr_chunks,
            )
        except NoAudioStreamError as e:
            self._reporter.info(f"    [SKIP] Video-only (no audio stream): {e}")
            self._db.update_error(sub_id, "transcribe", str(e))
            self._db.mark_processed(sub_id)
            self._release_audio(sub_id)
            return None, None
        except IncompleteAudioError as e:
            # tail mode doesn't enforce a 90% check, but if the transcriber
            # ever raises this we save whatever we got so the next run can
            # decide whether to retry.
            self._reporter.info(f"    [WARN] Incomplete audio: {e}")
            transcript = self._transcriber._last_transcript
            segments = self._transcriber._last_segments
        except Exception as e:
            self._reporter.info(
                f"    [FAIL] Transcription error: {type(e).__name__}: {e}"
            )
            self._db.update_error(sub_id, "transcribe", str(e))
            self._release_audio(sub_id)
            raise

        self._db.update_transcript(sub_id, transcript)
        return transcript, segments

    def _summarize(self, sub_id: str, course_title: str, transcript: str,
                   transcript_segments: list[dict] | None,
                   kept_pages: list[dict] | None = None) -> Optional[str]:
        try:
            if kept_pages is None:
                kept_pages = self._db.get_done_ppt_pages(sub_id)
            prompt_text, mode = bucketer.assemble(
                transcript, transcript_segments, kept_pages,
            )
            prompt_text = self._add_ppt_only_prompt_warning(
                transcript, prompt_text,
            )
            self._reporter.info(
                f"    [Time] Generating summary at "
                f"{time.strftime('%H:%M:%S')}"
                f" — mode={mode}, prompt={len(prompt_text)} chars"
            )
            summary, model_used = self._summarizer.summarize(
                course_title, prompt_text,
            )
            summary = self._apply_ppt_only_summary_notice(transcript, summary)
            self._reporter.info(
                f"    [OK] Summary by {model_used}: {len(summary)} chars"
            )
            self._db.update_summary(sub_id, summary, model_used)
            return summary
        except Exception as e:
            self._reporter.info(
                f"    [FAIL] Summarization error: {type(e).__name__}: {e}"
            )
            self._db.update_error(sub_id, "summarize", str(e))
            raise

    def _release_audio(self, sub_id: str):
        try:
            self._scheduler.audio_downloader.release(sub_id)
        except Exception as e:
            self._reporter.info(
                f"    [WARN] audio release failed: {type(e).__name__}: {e}"
            )
