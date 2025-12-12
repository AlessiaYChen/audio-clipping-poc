from __future__ import annotations

from codex_audio.evaluation.io import Segment
from codex_audio.evaluation.matching import match_segments, segments_to_boundaries


def test_segments_to_boundaries() -> None:
    segments = [
        Segment(0.0, 5.0, "a"),
        Segment(5.0, 7.0, "b"),
        Segment(7.0, 10.0, "c"),
    ]
    boundaries = segments_to_boundaries(segments)
    assert boundaries == [5.0, 7.0]


def test_match_segments_counts_tp_fp_fn() -> None:
    refs = [
        Segment(0.0, 5.0, "a"),
        Segment(5.0, 10.0, "b"),
        Segment(10.0, 15.0, "c"),
    ]
    preds = [
        Segment(0.0, 4.8, "a"),
        Segment(4.8, 10.2, "b"),
        Segment(10.2, 15.0, "c"),
        Segment(15.0, 18.0, "d"),
    ]

    counts = match_segments(preds, refs, tolerance_s=0.3)
    assert counts.tp == 2
    assert counts.fp == 1
    assert counts.fn == 0
