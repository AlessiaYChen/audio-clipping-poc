from __future__ import annotations

from codex_audio.boundary.candidates import BoundaryCandidate
from codex_audio.segmentation.selection import SegmentConstraint, greedy_select_boundaries


def test_greedy_select_boundaries_respects_constraints() -> None:
    candidates = [
        BoundaryCandidate(time_s=30.0, score=0.6, reason="change_peak"),
        BoundaryCandidate(time_s=60.0, score=1.4, reason="change_peak"),
        BoundaryCandidate(time_s=90.0, score=1.6, reason="change_peak"),
        BoundaryCandidate(time_s=130.0, score=2.0, reason="change_peak"),
    ]
    constraint = SegmentConstraint(min_len=25.0, max_len=80.0)
    selected = greedy_select_boundaries(
        candidates,
        chunk_start=0.0,
        chunk_end=180.0,
        constraints=constraint,
        hard_min_score=0.5,
    )
    assert [round(c.time_s) for c in selected] == [60, 130]


def test_greedy_select_boundaries_respects_min_score() -> None:
    candidates = [
        BoundaryCandidate(time_s=40.0, score=0.4, reason="change_peak"),
        BoundaryCandidate(time_s=80.0, score=0.6, reason="change_peak"),
    ]
    constraint = SegmentConstraint(min_len=20.0)
    selected = greedy_select_boundaries(
        candidates,
        chunk_start=0.0,
        chunk_end=150.0,
        constraints=constraint,
        hard_min_score=0.5,
    )
    assert len(selected) == 1
    assert selected[0].time_s == 80.0
