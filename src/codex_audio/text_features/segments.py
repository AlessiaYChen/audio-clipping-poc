from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from codex_audio.transcription import TranscriptWord

DEFAULT_CHUNK_SIZE_S = 5.0
DEFAULT_OVERLAP_RATIO = 0.5


@dataclass
class TextChunk:
    start_s: float
    end_s: float
    text: str


def build_text_chunks(
    words: Sequence[TranscriptWord],
    *,
    chunk_size_s: float = DEFAULT_CHUNK_SIZE_S,
    overlap_ratio: float = DEFAULT_OVERLAP_RATIO,
) -> List[TextChunk]:
    if chunk_size_s <= 0:
        raise ValueError("chunk_size_s must be positive")
    if not (0.0 <= overlap_ratio < 1.0):
        raise ValueError("overlap_ratio must be in [0, 1)")

    ordered = sorted(words, key=lambda word: word.start_s)
    chunks: List[TextChunk] = []
    if not ordered:
        return chunks

    step = chunk_size_s * (1 - overlap_ratio)
    current_start = ordered[0].start_s
    while current_start < ordered[-1].end_s:
        current_end = current_start + chunk_size_s
        tokens = [word.text for word in ordered if current_start <= word.start_s < current_end]
        if tokens:
            chunks.append(TextChunk(start_s=current_start, end_s=current_end, text=" ".join(tokens)))
        current_start += step if step > 0 else chunk_size_s
    return chunks
