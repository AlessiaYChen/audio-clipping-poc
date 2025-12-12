from __future__ import annotations

from codex_audio.text_features import build_text_chunks
from codex_audio.transcription import TranscriptWord
import pytest


def _word(text: str, start: float, end: float) -> TranscriptWord:
    return TranscriptWord(text=text, start_s=start, end_s=end)


def test_build_text_chunks_overlapping_windows() -> None:
    words = [
        _word("hello", 0.0, 0.5),
        _word("there", 1.0, 1.5),
        _word("friend", 2.0, 2.5),
        _word("news", 9.9, 10.4),
        _word("story", 10.5, 11.0),
        _word("continues", 18.0, 19.0),
    ]

    chunks = build_text_chunks(words, chunk_size_s=10.0, overlap_ratio=0.5)

    assert len(chunks) == 4
    texts = [chunk.text for chunk in chunks]
    assert texts[0] == "hello there friend news"
    assert texts[1] == "news story"
    assert texts[2] == "story continues"
    assert texts[3] == "continues"


def test_build_text_chunks_handles_short_lists_and_validation() -> None:
    words: list[TranscriptWord] = []
    assert build_text_chunks(words) == []

    with pytest.raises(ValueError):
        build_text_chunks([_word("a", 0.0, 0.2)], chunk_size_s=0.0)
    with pytest.raises(ValueError):
        build_text_chunks([_word("a", 0.0, 0.2)], overlap_ratio=1.0)
