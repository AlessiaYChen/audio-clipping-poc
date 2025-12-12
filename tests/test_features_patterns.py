from __future__ import annotations

import pytest

from codex_audio.boundary.candidates import BoundaryCandidate
from codex_audio.features.diarization import DiarizationSegment
from codex_audio.features.patterns import find_anchor_return_candidates


def _seg(speaker: str, start: float, end: float) -> DiarizationSegment:
    return DiarizationSegment(speaker=speaker, start_s=start, end_s=end)


def test_find_anchor_return_candidates_detects_pattern() -> None:
    segments = [
        _seg("anchor", 0.0, 5.0),
        _seg("guest", 5.0, 7.0),
        _seg("anchor", 7.0, 12.0),
        _seg("reporter", 12.0, 14.0),
        _seg("anchor", 14.0, 20.0),
    ]

    candidates = find_anchor_return_candidates(segments)

    assert len(candidates) == 4
    assert all(isinstance(candidate, BoundaryCandidate) for candidate in candidates)
    reasons = {candidate.reason for candidate in candidates}
    assert {"anchor_return_start", "anchor_return_end"}.issubset(reasons)
    assert sorted(candidate.score for candidate in candidates) == [3.0] * 4
    times = sorted(candidate.time_s for candidate in candidates)
    assert times == pytest.approx([5.0, 7.0, 12.0, 14.0])


def test_find_anchor_return_candidates_without_anchor_pattern() -> None:
    segments = [
        _seg("guest", 0.0, 3.0),
        _seg("guest", 3.0, 6.0),
        _seg("anchor", 6.0, 9.0),
    ]
    assert find_anchor_return_candidates(segments) == []


def test_find_anchor_return_candidates_requires_three_segments() -> None:
    assert find_anchor_return_candidates([]) == []
    assert find_anchor_return_candidates([_seg("anchor", 0.0, 1.0)]) == []
