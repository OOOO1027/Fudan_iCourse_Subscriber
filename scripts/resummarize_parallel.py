#!/usr/bin/env python3
"""Parallel resummarize: run OCR and LLM concurrently across lectures.

The existing resummarize_old_lectures() processes lectures one-by-one,
serializing PPT OCR (CPU-bound) and LLM API calls (IO-bound).  For
~350 lectures this takes 6-17 hours, usually hitting the 350-min
GitHub Actions timeout.

This script instead uses a thread pool so that while one lecture is
waiting for the LLM API response, another lecture's OCR runs on the
same CPU cores.  The overall wall-clock time is bounded by the max of
(total OCR time, total LLM time) rather than their sum.

Usage:
    python scripts/resummarize_parallel.py              # all courses
    python scripts/resummarize_parallel.py --limit 50   # first 50 only
    python scripts/resummarize_parallel.py --workers 6

The script shares the same DB, Scheduler, and Summarizer as main.py.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.runtime import config
from src.data.database import Database
from src.ai.summarizer import Summarizer
from src.ai.bucketer import assemble
from src.pipeline.ppt_pipeline import PPTPipeline
from src.runtime.scheduler import Scheduler
from src.runtime.reporter import Reporter
from src.api.icourse import ICourseClient
from src.api.webvpn import WebVPNSession

# ── Configurable knobs ────────────────────────────────────────────────────
DEFAULT_WORKERS = 4       # concurrent lectures (OCR + LLM overlapping)
SESSION_LOCK = threading.Lock()


def _check_session(client: ICourseClient) -> None:
    """Thread-safe session check; only one thread re-logs-in at a time."""
    if client.check_alive():
        return
    with SESSION_LOCK:
        # Double-check after acquiring lock — another thread may have
        # already refreshed the session.
        if client.check_alive():
            return
        print("[Session] WebVPN session expired, re-logging in...")
        vpn = WebVPNSession()
        vpn.login()
        vpn.authenticate_icourse()
        client.vpn = vpn
        client._userinfo = None


def process_one(
    row: dict,
    client: ICourseClient,
    db: Database,
    ppt_pipeline: PPTPipeline,
    summarizer: Summarizer,
    reporter: Reporter,
) -> dict | None:
    """OCR + LLM for a single lecture.  Returns email_item dict or None."""
    sub_id = str(row["sub_id"])
    course_id = row["course_id"]
    sub_title = row.get("sub_title", sub_id)
    course_title = row.get("course_title", "Unknown")
    t0 = time.time()

    try:
        reporter.resummarize_one(course_title, sub_title)

        # Session check (serialized, one at a time)
        _check_session(client)

        # Phase 1: PPT OCR
        try:
            ppt_pipeline.run_blocking(client, course_id, sub_id)
        except Exception as e:
            reporter.info(
                f"    [WARN] PPT OCR failed for {sub_id}: "
                f"{type(e).__name__}: {e}"
            )

        # Phase 2: LLM summarization
        transcript = row.get("transcript") or ""
        if not transcript.strip():
            reporter.info("    Empty transcript, skipping.")
            return None

        kept_pages = db.get_done_ppt_pages(sub_id)
        prompt_text, mode = assemble(transcript, None, kept_pages)
        reporter.info(
            f"    Prompt: mode={mode}, {len(prompt_text)} chars, "
            f"{len(kept_pages)} PPT pages"
        )

        summary, model_used = summarizer.summarize(course_title, prompt_text)
        db.update_summary_v2(sub_id, summary, model_used)
        db.reset_emailed(sub_id)
        elapsed = time.time() - t0
        reporter.info(
            f"    [OK] v2 summary by {model_used}: {len(summary)} chars "
            f"({elapsed:.0f}s)"
        )
        return {
            "sub_id": sub_id,
            "course_title": course_title,
            "sub_title": sub_title,
            "date": row.get("date", ""),
            "summary": summary,
            "is_update": True,
        }
    except Exception:
        reporter.info(f"    [FAIL] Resummarize {sub_id}:")
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Parallel resummarize")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help="Concurrent lecture count (default: 4)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max lectures to process (0 = all)")
    args = parser.parse_args()

    db = Database()
    summarizer = Summarizer()

    # Refresh the scheduler's DB connection — Database() opens with
    # check_same_thread=False which can cause issues across threads.
    # We'll rely on the shared db instance with its internal lock.
    scheduler = Scheduler(reporter=Reporter())
    ppt_pipeline = PPTPipeline(db, scheduler)

    # Login
    print("[Login] WebVPN...")
    vpn = WebVPNSession()
    vpn.login()
    print("[Login] iCourse CAS...")
    vpn.authenticate_icourse()
    client = ICourseClient(vpn)

    # Get targets
    targets = db.get_lectures_to_resummarize()
    if not targets:
        print("No lectures need resummarize.")
        return
    if args.limit:
        targets = targets[:args.limit]
    print(f"Resummarizing {len(targets)} lecture(s) with {args.workers} workers...")

    # Process in parallel
    t_start = time.time()
    done = 0
    email_items: list = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                process_one, row, client, db, ppt_pipeline, summarizer,
                Reporter(),
            ): row
            for row in targets
        }
        for fut in as_completed(futures):
            row = futures[fut]
            done += 1
            try:
                result = fut.result()
                if result:
                    email_items.append(result)
            except Exception as e:
                print(f"    [FATAL] {row.get('sub_id', '?')}: {e}")
            # Progress line
            print(f"  [{done}/{len(targets)}] — "
                  f"{time.time() - t_start:.0f}s elapsed")

    elapsed = time.time() - t_start
    print(f"\nDone: {len(email_items)}/{len(targets)} lectures "
          f"resummarized in {elapsed:.0f}s "
          f"({elapsed / 60:.1f} min)")
    scheduler.shutdown()


if __name__ == "__main__":
    main()
