from __future__ import annotations

from typing import Dict, Iterable, List, Mapping

from codex_audio.boundary.candidates import BoundaryCandidate

DEFAULT_WEIGHTS: Mapping[str, float] = {
    "anchor_return": 3.0,
    "semantic_shift": 2.0,
    "audio_shift": 2.0,
    "silence_gap": 1.0,
    "jingle": 1.5,
}
DEFAULT_THRESHOLD = 1.0


def score_candidates(
    candidates: Iterable[BoundaryCandidate],
    *,
    weights: Mapping[str, float] | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> List[BoundaryCandidate]:
    """Apply heuristic weights and thresholding to boundary candidates."""

    weight_map = dict(DEFAULT_WEIGHTS)
    if weights:
        for key, value in weights.items():
            try:
                weight_map[key] = float(value)
            except (TypeError, ValueError):
                continue

    scored: List[BoundaryCandidate] = []
    for candidate in candidates:
        weight = _weight_for_reason(candidate.reason, weight_map)
        total = candidate.score + weight
        if total >= threshold:
            scored.append(
                BoundaryCandidate(time_s=candidate.time_s, score=total, reason=candidate.reason)
            )
    return scored


def _weight_for_reason(reason: str, weights: Mapping[str, float]) -> float:
    for prefix, value in weights.items():
        if reason.startswith(prefix):
            return value
    return 0.0
