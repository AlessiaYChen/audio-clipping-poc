from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class TextChunk:
    start_s: float
    end_s: float
    text: str


def chunk_words(words: List[str], window_s: float = 8.0) -> List[TextChunk]:
    return []
