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
    sorted_candidates = sorted(candidates, key=lambda c: c.time_s)
    selected: List[BoundaryCandidate] = []
    min_len = constraints.min_len
    max_len = constraints.max_len
    current_start = chunk_start
    remaining = list(sorted_candidates)

    while True:
        eligible = [
            c
            for c in remaining
            if (c.time_s - current_start) >= min_len and (chunk_end - c.time_s) >= min_len
        ]
        if not eligible:
            break
        eligible.sort(key=lambda c: c.score, reverse=True)
        chosen: BoundaryCandidate | None = None
        for candidate in eligible:
            if candidate.score < hard_min_score:
                break
            if max_len is not None and (candidate.time_s - current_start) > max_len:
                continue
            if max_len is not None and not _future_segment_ok(
                candidate,
                remaining,
                selected,
                chunk_end,
                min_len,
                max_len,
            ):
                continue
            chosen = candidate
            break
        if chosen is None:
            break
        selected.append(chosen)
        current_start = chosen.time_s
        remaining = [c for c in remaining if c.time_s > current_start]
    final_len = chunk_end - current_start
    if max_len is not None and final_len > max_len:
        # attempt to force a cut if possible
        tail_candidates = [
            c
            for c in sorted_candidates
            if c.time_s > current_start + min_len and (chunk_end - c.time_s) >= min_len
        ]
        tail_candidates.sort(key=lambda c: c.score, reverse=True)
        for candidate in tail_candidates:
            if candidate.score < hard_min_score:
                continue
            if max_len is not None and (candidate.time_s - current_start) > max_len:
                continue
            selected.append(candidate)
            break
    return sorted(selected, key=lambda c: c.time_s)


def _future_segment_ok(
    candidate: BoundaryCandidate,
    remaining: List[BoundaryCandidate],
    selected: List[BoundaryCandidate],
    chunk_end: float,
    min_len: float,
    max_len: float,
) -> bool:
    future = [c for c in remaining if c.time_s > candidate.time_s]
    next_cut_time = chunk_end
    for option in future:
        if (option.time_s - candidate.time_s) >= min_len:
            next_cut_time = option.time_s
            break
    next_segment_len = next_cut_time - candidate.time_s
    return next_segment_len <= max_len
