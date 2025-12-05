from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class SemanticShift:
    time_s: float
    score: float
    summary: str = ""


def detect_topic_shifts(chunks: List[str]) -> List[SemanticShift]:
    return []
