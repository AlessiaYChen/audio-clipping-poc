from __future__ import annotations

import pytest

from codex_audio.boundary.candidates import BoundaryCandidate
from codex_audio.segmentation.scoring import DEFAULT_WEIGHTS, score_candidates


def test_score_candidates_applies_weights() -> None:
    candidates = [
        BoundaryCandidate(time_s=1.0, score=0.0, reason="anchor_return_start"),
        BoundaryCandidate(time_s=2.0, score=0.5, reason="semantic_shift_0.6"),
        BoundaryCandidate(time_s=3.0, score=0.1, reason="audio_shift_0.6"),
        BoundaryCandidate(time_s=4.0, score=0.1, reason="silence_gap_2.0s"),
        BoundaryCandidate(time_s=5.0, score=0.0, reason="jingle_detected"),
    ]

    scored = score_candidates(candidates, threshold=0.5)

    assert len(scored) == 5
    assert pytest.approx(scored[0].score) == DEFAULT_WEIGHTS["anchor_return"]
    assert pytest.approx(scored[1].score) == 0.5 + DEFAULT_WEIGHTS["semantic_shift"]
    assert pytest.approx(scored[2].score) == 0.1 + DEFAULT_WEIGHTS["audio_shift"]
    assert pytest.approx(scored[3].score) == 0.1 + DEFAULT_WEIGHTS["silence_gap"]
    assert pytest.approx(scored[4].score) == DEFAULT_WEIGHTS["jingle"]


def test_score_candidates_respects_threshold_and_overrides() -> None:
    candidates = [
        BoundaryCandidate(time_s=1.0, score=0.0, reason="silence_gap_short"),
        BoundaryCandidate(time_s=2.0, score=0.0, reason="anchor_return_end"),
    ]

    weights = {"silence_gap": 0.25}
    scored = score_candidates(candidates, weights=weights, threshold=0.5)

    assert len(scored) == 1
    assert scored[0].reason == "anchor_return_end"
