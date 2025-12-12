from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from codex_audio.evaluation.io import Segment


def segments_to_boundaries(segments: Sequence[Segment]) -> List[float]:
    ordered = sorted(segments, key=lambda seg: seg.start_s)
    return [seg.start_s for idx, seg in enumerate(ordered) if idx > 0]


@dataclass
class MatchCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def accumulate(self, other: "MatchCounts") -> "MatchCounts":
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        return self


def match_segments(
    predictions: Sequence[Segment],
    references: Sequence[Segment],
    *,
    tolerance_s: float,
) -> MatchCounts:
    pred_boundaries = segments_to_boundaries(predictions)
    ref_boundaries = segments_to_boundaries(references)

    ref_matched = [False] * len(ref_boundaries)
    tp = 0

    for pred in pred_boundaries:
        match_index = _find_within_tolerance(pred, ref_boundaries, ref_matched, tolerance_s)
        if match_index is not None:
            ref_matched[match_index] = True
            tp += 1

    fp = len(pred_boundaries) - tp
    fn = len(ref_boundaries) - tp
    return MatchCounts(tp=tp, fp=fp, fn=fn)


def _find_within_tolerance(
    candidate: float,
    references: Sequence[float],
    matched: Sequence[bool],
    tolerance: float,
) -> int | None:
    best_idx: int | None = None
    best_delta = tolerance
    for idx, ref in enumerate(references):
        if matched[idx]:
            continue
        delta = abs(candidate - ref)
        if delta <= tolerance and delta <= best_delta:
            best_idx = idx
            best_delta = delta
    return best_idx
