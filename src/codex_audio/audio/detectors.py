from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class AcousticChangePoint:
    time_s: float
    label: str = "change"


def detect_changes(embeddings: List[float]) -> List[AcousticChangePoint]:
    return []
