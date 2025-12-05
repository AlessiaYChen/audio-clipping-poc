from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class BoundaryCandidate:
    time_s: float
    score: float
    reason: str


def propose_candidates() -> List[BoundaryCandidate]:
    return []
