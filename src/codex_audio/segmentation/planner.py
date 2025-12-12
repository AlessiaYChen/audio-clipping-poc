from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from codex_audio.boundary.candidates import BoundaryCandidate

DEFAULT_MIN_SEGMENT_S = 45.0
TAIL_LABEL = "tail"
FULL_LABEL = "full_audio"


@dataclass
class SegmentPlan:
    start_s: float
    end_s: float
    label: str

    def duration(self) -> float:
        return max(0.0, self.end_s - self.start_s)


def build_segments(
    candidates: Sequence[BoundaryCandidate],
    *,
    duration_s: float,
    min_segment_s: float = DEFAULT_MIN_SEGMENT_S,
) -> List[SegmentPlan]:
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    if min_segment_s <= 0:
        raise ValueError("min_segment_s must be positive")

    usable = [
        candidate
        for candidate in candidates
        if 0.0 < candidate.time_s < duration_s
    ]
    usable.sort(key=lambda candidate: candidate.time_s)

    segments: List[SegmentPlan] = []
    last_start = 0.0

    for idx, candidate in enumerate(usable):
        boundary_time = candidate.time_s
        if boundary_time <= last_start:
            continue
        if (boundary_time - last_start) < min_segment_s:
            continue
        label = candidate.reason or f"candidate_{idx}"
        segments.append(SegmentPlan(start_s=last_start, end_s=boundary_time, label=label))
        last_start = boundary_time

    if not segments:
        return [SegmentPlan(0.0, duration_s, FULL_LABEL)]

    tail_duration = duration_s - last_start
    if tail_duration <= 0:
        return segments

    if tail_duration < min_segment_s:
        segments[-1].end_s = duration_s
        return segments

    segments.append(SegmentPlan(last_start, duration_s, TAIL_LABEL))
    return segments
