"""Lecture queue selection helpers."""

from src.ai.summarizer import InvalidSummaryError, Summarizer


def has_usable_summary(existing: dict | None) -> bool:
    """Whether an existing lecture row has a summary worth keeping."""
    if not (existing and existing.get("summary")):
        return False
    try:
        Summarizer._validate_summary_text(existing["summary"])
    except InvalidSummaryError:
        return False
    return True


def lecture_needs_processing(
    lecture: dict,
    existing: dict | None,
    processed_sub_ids: set[str],
) -> bool:
    """Return True when a playback lecture should enter this run's queue."""
    if not lecture.get("has_playback"):
        return False

    sub_id = str(lecture["sub_id"])
    if sub_id not in processed_sub_ids:
        return True

    return bool(existing and existing.get("summary") and not has_usable_summary(existing))
