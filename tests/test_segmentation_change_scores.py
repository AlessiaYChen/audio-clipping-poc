from __future__ import annotations

import math

from codex_audio.boundary.candidates import BoundaryCandidate
from codex_audio.features.diarization import DiarizationSegment
from codex_audio.features.embeddings import AudioEmbedding
from codex_audio.features.vad import SILENCE_LABEL, VadSegment
from codex_audio.segmentation.change_scores import (
    ChangePoint,
    compute_change_points,
    find_peak_candidates,
    smooth_scores,
)
from codex_audio.text_features import TextChunk
from codex_audio.text_features.embeddings import ChunkEmbedding


def _audio_vec(x: float, y: float) -> list[float]:
    return [x, y]


def _text_chunk(text: str, start: float, end: float) -> ChunkEmbedding:
    chunk = TextChunk(start_s=start, end_s=end, text=text)
    return ChunkEmbedding(text=chunk, vector=[math.cos(start), math.sin(start)])


def test_compute_change_points_combines_modalities() -> None:
    audio_embeddings = [
        AudioEmbedding(start_s=0.0, end_s=5.0, vector=_audio_vec(1.0, 0.0)),
        AudioEmbedding(start_s=2.5, end_s=7.5, vector=_audio_vec(0.0, 1.0)),
        AudioEmbedding(start_s=5.0, end_s=10.0, vector=_audio_vec(0.0, 1.0)),
    ]
    text_embeddings = [
        _text_chunk("hello", 0.0, 5.0),
        _text_chunk("world", 2.5, 7.5),
        _text_chunk("again", 5.0, 10.0),
    ]
    vad_segments = [
        VadSegment(start_s=2.0, end_s=3.5, label=SILENCE_LABEL),
        VadSegment(start_s=6.0, end_s=6.8, label=SILENCE_LABEL),
    ]
    diarization_segments = [
        DiarizationSegment(speaker="anchor", start_s=0.0, end_s=4.0),
        DiarizationSegment(speaker="guest", start_s=4.0, end_s=6.0),
        DiarizationSegment(speaker="anchor", start_s=6.0, end_s=9.0),
    ]

    points = compute_change_points(
        audio_embeddings=audio_embeddings,
        text_embeddings=text_embeddings,
        vad_segments=vad_segments,
        diarization_segments=diarization_segments,
        silence_window_s=1.0,
        silence_norm_s=1.0,
        anchor_tolerance_s=0.5,
    )

    assert len(points) == 2
    assert points[0].audio_change > 0.9
    assert points[0].text_change >= 0.0
    assert points[0].silence_change <= 1.0
    assert points[1].anchor_flag in (0.0, 1.0)


def test_smooth_and_peak_detection() -> None:
    times = [0.0, 2.5, 5.0, 7.5]
    scores = [0.1, 0.9, 0.2, 1.1]
    smoothed = smooth_scores(scores, window_size=1)
    assert len(smoothed) == len(scores)
    candidates = find_peak_candidates(times, smoothed, min_score=0.5)
    assert len(candidates) == 1
    assert isinstance(candidates[0], BoundaryCandidate)
