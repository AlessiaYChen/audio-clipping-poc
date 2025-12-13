from __future__ import annotations

import pytest

from codex_audio.features.vad import SILENCE_LABEL, VadSegment
from codex_audio.segmentation.change_scores import ChangePoint
from codex_audio.segmentation.refinement import RefinementParams, refine_chunk_segments
from codex_audio.segmentation.selection import SegmentConstraint
from codex_audio.transcription import TranscriptWord


def _params(**overrides):
    base = RefinementParams(
        constraints=SegmentConstraint(min_len=25.0, max_len=90.0),
        weights={"audio": 1.0, "text": 0.0, "silence": 0.0, "anchor": 0.0},
        candidate_min_score=0.2,
        hard_min_cut_score=0.0,
        smoothing_window=0,
        snap_window_s=0.0,
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def test_refine_chunk_segments_selects_strongest_peaks() -> None:
    points = [
        ChangePoint(time_s=30.0, audio_change=0.1),
        ChangePoint(time_s=60.0, audio_change=1.1),
        ChangePoint(time_s=75.0, audio_change=0.2),
        ChangePoint(time_s=120.0, audio_change=0.3),
        ChangePoint(time_s=150.0, audio_change=1.2),
        ChangePoint(time_s=170.0, audio_change=0.1),
    ]

    segments = refine_chunk_segments(
        0.0,
        220.0,
        change_points=points,
        params=_params(),
    )

    assert len(segments) == 3
    assert pytest.approx(segments[0].end_s) == 60.0
    assert pytest.approx(segments[1].end_s) == 150.0
    assert segments[-1].end_s == pytest.approx(220.0)


def test_refine_chunk_segments_snaps_to_silence_and_words() -> None:
    points = [
        ChangePoint(time_s=49.2, audio_change=1.0),
        ChangePoint(time_s=108.9, audio_change=1.0),
    ]
    vad_segments = [VadSegment(start_s=48.0, end_s=52.0, label=SILENCE_LABEL)]
    words = [TranscriptWord(text="hello", start_s=109.0, end_s=109.4)]

    params = _params(snap_window_s=1.0)
    segments = refine_chunk_segments(
        0.0,
        150.0,
        change_points=points,
        params=params,
        vad_segments=vad_segments,
        transcript_words=words,
    )

    assert pytest.approx(segments[0].end_s) == 50.0
    assert len(segments) == 2
    assert segments[1].start_s == pytest.approx(50.0)
    assert segments[1].end_s == pytest.approx(150.0)

def test_refine_chunk_segments_enters_panic_mode_on_long_chunk() -> None:
    points = [
        ChangePoint(time_s=40.0, audio_change=0.5),
        ChangePoint(time_s=80.0, audio_change=0.6),
    ]
    params = RefinementParams(
        constraints=SegmentConstraint(min_len=20.0, max_len=45.0),
        candidate_min_score=0.1,
        hard_min_cut_score=5.0,
        weights={"audio": 1.0, "text": 0.0, "silence": 0.0, "anchor": 0.0, "keyword": 0.0},
        snap_window_s=0.0,
    )

    segments = refine_chunk_segments(
        0.0,
        150.0,
        change_points=points,
        params=params,
    )

    assert pytest.approx(segments[0].end_s) == 40.0

def test_refine_chunk_segments_forces_cut_when_chunk_long() -> None:
    points = [ChangePoint(time_s=100.0, audio_change=0.2)]
    params = RefinementParams(
        constraints=SegmentConstraint(min_len=20.0, max_len=None),
        candidate_min_score=0.0,
        hard_min_cut_score=0.0,
        weights={"audio": 1.0, "text": 0.0, "silence": 0.0, "anchor": 0.0, "keyword": 0.0},
        snap_window_s=0.0,
    )

    segments = refine_chunk_segments(
        0.0,
        200.0,
        change_points=points,
        params=params,
    )

    assert pytest.approx(segments[0].end_s) == 100.0



