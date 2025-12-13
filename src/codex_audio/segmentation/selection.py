from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from codex_audio.boundary.candidates import BoundaryCandidate


@dataclass
class SegmentConstraint:
    min_len: float
    max_len: float | None = None


def greedy_select_boundaries(
    candidates: Sequence[BoundaryCandidate],
    *,
    chunk_start: float,
    chunk_end: float,
    constraints: SegmentConstraint,
    hard_min_score: float = 0.0,
) -> List[BoundaryCandidate]:
    """Select boundary candidates via dynamic programming.

    Computes the globally optimal subset of boundaries that maximizes the total
    score while respecting segment length constraints.
    """
    sorted_candidates = sorted(candidates, key=lambda c: c.time_s)
    points = [chunk_start] + [c.time_s for c in sorted_candidates] + [chunk_end]
    n = len(points)
    if n < 2:
        return []

    min_len = max(0.0, constraints.min_len)
    max_len = constraints.max_len

    dp = [-float("inf")] * n
    parent = [-1] * n
    dp[0] = 0.0

    for i in range(1, n):
        for j in range(i - 1, -1, -1):
            seg_len = points[i] - points[j]
            if seg_len < 0:
                continue
            if i != n - 1 and seg_len < min_len:
                continue
            if max_len is not None and seg_len > max_len:
                continue
            if i == n - 1:
                cut_score = 0.0
            else:
                candidate = sorted_candidates[i - 1]
                if candidate.score < hard_min_score:
                    continue
                cut_score = candidate.score
            best = dp[j] + cut_score
            if best > dp[i]:
                dp[i] = best
                parent[i] = j

    if dp[-1] == -float("inf"):
        return []

    selected: List[BoundaryCandidate] = []
    idx = n - 1
    while parent[idx] != -1:
        prev = parent[idx]
        if idx != n - 1:
            selected.append(sorted_candidates[idx - 1])
        idx = prev
    selected.reverse()
    return selected

