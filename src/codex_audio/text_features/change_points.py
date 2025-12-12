from __future__ import annotations

import math
from typing import List, Sequence

from codex_audio.boundary.candidates import BoundaryCandidate
from codex_audio.text_features.embeddings import ChunkEmbedding

DEFAULT_SIMILARITY_THRESHOLD = 0.7
SEMANTIC_SHIFT_REASON = "semantic_shift"
SEMANTIC_SHIFT_SCORE = 2.0


def find_text_change_candidates(
    chunks: Sequence[ChunkEmbedding], *, threshold: float = DEFAULT_SIMILARITY_THRESHOLD
) -> List[BoundaryCandidate]:
    if threshold <= -1.0 or threshold > 1.0:
        raise ValueError("threshold must be within (-1, 1]")

    candidates: List[BoundaryCandidate] = []
    for left, right in zip(chunks, chunks[1:]):
        similarity = _cosine_similarity(left.vector, right.vector)
        if similarity < threshold:
            boundary_time = (left.text.end_s + right.text.start_s) / 2
            candidates.append(
                BoundaryCandidate(
                    time_s=boundary_time,
                    score=SEMANTIC_SHIFT_SCORE,
                    reason=f"{SEMANTIC_SHIFT_REASON}_{similarity:.2f}",
                )
            )
    return candidates


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Embedding vectors must have the same length")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
