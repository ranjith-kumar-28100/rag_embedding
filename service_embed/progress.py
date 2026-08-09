"""Pure progress / status roll-up. Side-effect free, so trivially unit-testable."""

from __future__ import annotations

from collections.abc import Sequence

from .models import TERMINAL_FILE_STATUSES, FileStatus, JobStatus


def child_percent(processed: int, total: int) -> int:
    """Per-file percent = processed_children / total_children (§7.3).

    A file with no children at all (parents only) is complete the moment its
    parents are stored — reporting 0% there would stall the bar forever.
    """
    if total <= 0:
        return 100
    return max(0, min(100, round(100 * processed / total)))


def compute_job_progress(file_progress: Sequence[int]) -> int:
    """Job percent = mean of per-file percent (0 when there are no files)."""
    if not file_progress:
        return 0
    return round(sum(file_progress) / len(file_progress))


def compute_job_status(file_statuses: Sequence[FileStatus]) -> JobStatus:
    """Roll per-file statuses up into one job status."""
    if not file_statuses:
        return JobStatus.QUEUED

    if not all(s in TERMINAL_FILE_STATUSES for s in file_statuses):
        any_started = any(s is not FileStatus.QUEUED for s in file_statuses)
        return JobStatus.RUNNING if any_started else JobStatus.QUEUED

    if all(s is FileStatus.FAILED for s in file_statuses):
        return JobStatus.FAILED
    if any(s is FileStatus.FAILED for s in file_statuses):
        return JobStatus.PARTIAL
    return JobStatus.DONE
