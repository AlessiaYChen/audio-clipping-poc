
from __future__ import annotations

import pytest

from codex_audio.boundary.candidates import BoundaryCandidate
from codex_audio.features.diarization import DiarizationSegment
from codex_audio.features.embeddings import AudioEmbedding
from codex_audio.features.vad import SILENCE_LABEL, SPEECH_LABEL, VadSegment
from codex_audio.segmentation.candidates import build_boundary_candidates, from_vad
from codex_audio.text_features import TextChunk
from codex_audio.text_features.embeddings import ChunkEmbedding


def test_from_vad_creates_candidates_for_long_silence() -> None:
    segments = [
        VadSegment(0.0, 2.0, SPEECH_LABEL),
        VadSegment(2.0, 5.5, SILENCE_LABEL),
        VadSegment(5.5, 8.0, SPEECH_LABEL),
        VadSegment(8.0, 11.2, SILENCE_LABEL),
    ]

    candidates = from_vad(segments, min_silence_s=2.5)

    assert len(candidates) == 2
    assert candidates[0].time_s == pytest.approx(3.75)
    assert candidates[1].time_s == pytest.approx(9.6)
    assert all(candidate.score > 0 for candidate in candidates)
    assert all(candidate.reason.startswith("silence_gap") for candidate in candidates)


def test_from_vad_skips_short_silence() -> None:
    segments = [
        VadSegment(0.0, 1.0, SILENCE_LABEL),
        VadSegment(1.0, 2.0, SPEECH_LABEL),
    ]

    assert from_vad(segments, min_silence_s=1.5) == []


def test_from_vad_rejects_invalid_threshold() -> None:
    with pytest.raises(ValueError):
        from_vad([], min_silence_s=0.0)


def test_build_boundary_candidates_merges_sources() -> None:
    vad_segments = [
        VadSegment(0.0, 1.0, SPEECH_LABEL),
        VadSegment(1.0, 4.0, SILENCE_LABEL),
    ]
    text_embeddings = [
        ChunkEmbedding(TextChunk(0.0, 5.0, "a"), [1.0, 0.0]),
        ChunkEmbedding(TextChunk(5.0, 10.0, "b"), [0.0, 1.0]),
    ]
    audio_embeddings = [
        AudioEmbedding(start_s=0.0, end_s=3.0, vector=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        AudioEmbedding(start_s=4.0, end_s=7.0, vector=[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ]
    diarization_segments = [
        DiarizationSegment("anchor", 0.0, 4.0),
        DiarizationSegment("guest", 4.0, 6.0),
        DiarizationSegment("anchor", 6.0, 8.0),
    ]
    extra = [BoundaryCandidate(time_s=12.0, score=0.4, reason="manual")]

    candidates = build_boundary_candidates(
        vad_segments=vad_segments,
        text_embeddings=text_embeddings,
        audio_embeddings=audio_embeddings,
        diarization_segments=diarization_segments,
        extra_candidates=extra,
        min_silence_s=2.5,
        text_threshold=0.9,
        audio_threshold=0.8,
        merge_threshold=0.25,
    )

    reasons = {candidate.reason for candidate in candidates}
    assert any(reason.startswith("silence_gap") for reason in reasons)
    assert any(reason.startswith("semantic_shift") for reason in reasons)
    assert any(reason.startswith("audio_shift") for reason in reasons)
    assert "anchor_return_start" in reasons
    assert "anchor_return_end" in reasons
    assert "manual" in reasons
    times = [candidate.time_s for candidate in candidates]
    assert times == sorted(times)
