from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from codex_audio.boundary.candidates import BoundaryCandidate


@dataclass
class SegmentConstraint:
    min_len: float
    max_len: float | None = None


def select_boundaries(
    candidates: Sequence[BoundaryCandidate],
    *,
    chunk_start: float,
    chunk_end: float,
    constraints: SegmentConstraint,
    hard_min_score: float = 0.0,
) -> List[BoundaryCandidate]:
    """Selects the optimal set of cuts via dynamic programming."""

    valid_candidates = [c for c in candidates if c.score >= hard_min_score]

    points: List[Tuple[float, float, BoundaryCandidate | None]] = []
    points.append((chunk_start, 0.0, None))
    for candidate in sorted(valid_candidates, key=lambda c: c.time_s):
        if chunk_start < candidate.time_s < chunk_end:
            points.append((candidate.time_s, candidate.score, candidate))
    points.append((chunk_end, 0.0, None))

    n = len(points)
    if n < 2:
        return []

    min_len = constraints.min_len
    max_len = constraints.max_len if constraints.max_len is not None else float("inf")

    dp = [-1.0] * n
    parent = [-1] * n
    dp[0] = 0.0

    for i in range(1, n):
        current_time = points[i][0]
        current_score = points[i][1]
        for j in range(i - 1, -1, -1):
            prev_time = points[j][0]
            segment_len = current_time - prev_time
            if segment_len > max_len:
                break
            if segment_len < min_len and i != n - 1:
                continue
            if dp[j] == -1.0:
                continue
            score = dp[j] + current_score
            if score > dp[i]:
                dp[i] = score
                parent[i] = j

    if dp[-1] == -1.0:
        return _fallback_greedy(candidates, chunk_start, chunk_end, constraints)

    selected: List[BoundaryCandidate] = []
    idx = n - 1
    while parent[idx] != -1:
        candidate = points[idx][2]
        if candidate is not None:
            selected.append(candidate)
        idx = parent[idx]
    return sorted(selected, key=lambda c: c.time_s)


def _fallback_greedy(
    candidates: Sequence[BoundaryCandidate],
    chunk_start: float,
    chunk_end: float,
    constraints: SegmentConstraint,
) -> List[BoundaryCandidate]:
    sorted_candidates = sorted(candidates, key=lambda c: c.score, reverse=True)
    selected: List[BoundaryCandidate] = []
    placed = [chunk_start, chunk_end]

    for candidate in sorted_candidates:
        if any(abs(candidate.time_s - taken) < constraints.min_len for taken in placed):
            continue
        selected.append(candidate)
        placed.append(candidate.time_s)
    return sorted(selected, key=lambda c: c.time_s)
