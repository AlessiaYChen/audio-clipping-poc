from __future__ import annotations

import pytest

from codex_audio.boundary.candidates import BoundaryCandidate
from codex_audio.text_features import TextChunk
from codex_audio.text_features.embeddings import ChunkEmbedding
from codex_audio.text_features.change_points import find_text_change_candidates


def _embedding(start: float, end: float, vector: list[float]) -> ChunkEmbedding:
    return ChunkEmbedding(text=TextChunk(start_s=start, end_s=end, text=""), vector=vector)


def test_find_text_change_candidates_emits_boundary() -> None:
    chunks = [
        _embedding(0.0, 10.0, [1.0, 0.0]),
        _embedding(10.0, 20.0, [0.1, 0.9]),
        _embedding(20.0, 30.0, [0.09, 0.91]),
    ]

    candidates = find_text_change_candidates(chunks, threshold=0.7)

    assert len(candidates) == 1
    assert isinstance(candidates[0], BoundaryCandidate)
    assert candidates[0].time_s == pytest.approx(10.0)
    assert candidates[0].score == 2.0
    assert candidates[0].reason.startswith("semantic_shift")


def test_find_text_change_candidates_threshold_validation() -> None:
    chunks = [_embedding(0.0, 10.0, [0.0, 0.0])]
    with pytest.raises(ValueError):
        find_text_change_candidates(chunks, threshold=1.5)


def test_find_text_change_candidates_handles_no_changes() -> None:
    chunks = [
        _embedding(0.0, 10.0, [1.0, 0.0]),
        _embedding(10.0, 20.0, [0.9, 0.1]),
    ]
    assert find_text_change_candidates(chunks, threshold=0.3) == []
