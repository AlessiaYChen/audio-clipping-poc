from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class SegmentPlan:
    start_s: float
    end_s: float
    label: str


def plan_segments(boundaries: List[float]) -> List[SegmentPlan]:
    return []
