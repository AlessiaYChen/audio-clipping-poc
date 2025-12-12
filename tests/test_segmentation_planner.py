from __future__ import annotations

import pytest

from codex_audio.boundary.candidates import BoundaryCandidate
from codex_audio.segmentation.planner import build_segments


def _candidate(time_s: float) -> BoundaryCandidate:
    return BoundaryCandidate(time_s=time_s, score=1.0, reason=f"silence@{time_s}")


def test_build_segments_enforces_min_length() -> None:
    candidates = [_candidate(20.0), _candidate(70.0), _candidate(150.0)]

    segments = build_segments(candidates, duration_s=200.0, min_segment_s=45.0)

    assert len(segments) == 3
    assert segments[0].start_s == pytest.approx(0.0)
    assert segments[0].end_s == pytest.approx(70.0)
    assert segments[1].start_s == pytest.approx(70.0)
    assert segments[1].end_s == pytest.approx(150.0)
    assert segments[2].start_s == pytest.approx(150.0)
    assert segments[2].end_s == pytest.approx(200.0)


def test_build_segments_merges_short_tail() -> None:
    candidates = [_candidate(60.0), _candidate(110.0)]

    segments = build_segments(candidates, duration_s=150.0, min_segment_s=45.0)

    assert len(segments) == 2
    assert segments[-1].end_s == pytest.approx(150.0)
    assert segments[-1].start_s == pytest.approx(60.0)


def test_build_segments_without_candidates_spans_whole_duration() -> None:
    segments = build_segments([], duration_s=75.0, min_segment_s=30.0)
    assert len(segments) == 1
    assert segments[0].start_s == pytest.approx(0.0)
    assert segments[0].end_s == pytest.approx(75.0)


def test_build_segments_validates_parameters() -> None:
    with pytest.raises(ValueError):
        build_segments([], duration_s=0.0)
    with pytest.raises(ValueError):
        build_segments([], duration_s=10.0, min_segment_s=0.0)
