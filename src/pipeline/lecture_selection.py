"""Lecture queue selection helpers."""

from collections.abc import Callable

from src.ai.summarizer import InvalidSummaryError, Summarizer


PPT_ONLY_SUMMARY_MARKER = "本节没有可用语音转写"


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

    if existing and existing.get("summary") and not has_usable_summary(existing):
        return True

    if lecture.get("hidden_playback") and existing is not None:
        if not existing.get("summary"):
            return True
        if not existing.get("transcript") and PPT_ONLY_SUMMARY_MARKER not in existing["summary"]:
            return True

    return False


def promote_hidden_playbacks(
    lectures: list[dict],
    has_video_url: Callable[[str], bool],
) -> tuple[list[dict], int]:
    """Mark non-playback lectures as playable when a deep video URL exists."""
    promoted: list[dict] = []
    promoted_count = 0

    for lecture in lectures:
        if lecture.get("has_playback"):
            promoted.append(lecture)
            continue

        sub_id = str(lecture["sub_id"])
        if has_video_url(sub_id):
            lecture = dict(lecture)
            lecture["has_playback"] = True
            lecture["hidden_playback"] = True
            promoted_count += 1

        promoted.append(lecture)

    return promoted, promoted_count
