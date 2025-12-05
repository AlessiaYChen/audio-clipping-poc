from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class AudioEmbedding:
    start_s: float
    end_s: float
    vector: List[float]


def extract_embeddings(audio_uri: str, window_s: float = 3.0) -> List[AudioEmbedding]:
    return []
