"""Pure progress / status roll-up (no DB, no Celery)."""

from __future__ import annotations

import pytest

from service_embed.models import FileStatus, JobStatus
from service_embed.progress import child_percent, compute_job_progress, compute_job_status


@pytest.mark.parametrize(
    ("processed", "total", "expected"),
    [(0, 100, 0), (50, 100, 50), (100, 100, 100), (1, 3, 33)],
)
def test_child_percent(processed: int, total: int, expected: int) -> None:
    assert child_percent(processed, total) == expected


def test_parents_only_file_is_complete_not_stalled() -> None:
    """A file with no children would otherwise sit at 0% forever."""
    assert child_percent(0, 0) == 100


def test_child_percent_clamps() -> None:
    assert child_percent(150, 100) == 100
    assert child_percent(-5, 100) == 0


def test_job_progress_is_the_mean() -> None:
    assert compute_job_progress([0, 50, 100]) == 50
    assert compute_job_progress([]) == 0


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        ([], JobStatus.QUEUED),
        ([FileStatus.QUEUED, FileStatus.QUEUED], JobStatus.QUEUED),
        ([FileStatus.QUEUED, FileStatus.RUNNING], JobStatus.RUNNING),
        ([FileStatus.DONE, FileStatus.QUEUED], JobStatus.RUNNING),
        ([FileStatus.DONE, FileStatus.DONE], JobStatus.DONE),
        ([FileStatus.DONE, FileStatus.FAILED], JobStatus.PARTIAL),
        ([FileStatus.FAILED, FileStatus.FAILED], JobStatus.FAILED),
    ],
)
def test_job_status_rollup(statuses: list[FileStatus], expected: JobStatus) -> None:
    assert compute_job_status(statuses) == expected
