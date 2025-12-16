from __future__ import annotations

from pathlib import Path

import pytest

from codex_audio.boundary.candidates import BoundaryCandidate
from codex_audio.pipeline import PipelineConfig, StorySegmentationPipeline
from codex_audio.transcription import TranscriptWord


def _words() -> list[TranscriptWord]:
    return [
        TranscriptWord(text="hello", start_s=0.0, end_s=0.5),
        TranscriptWord(text="world", start_s=0.5, end_s=1.0),
    ]


def test_generate_llm_candidates_aligns_with_quote(monkeypatch, tmp_path: Path) -> None:
    config = PipelineConfig(station="TEST", transcription_enabled=False)
    pipeline = StorySegmentationPipeline(config)
    pipeline._llm_segmentation_enabled = True

    def fake_detect(words, **kwargs):
        return [BoundaryCandidate(time_s=5.0, score=1.0, reason="llm_topic_change", quote="hello world")]

    monkeypatch.setattr("codex_audio.pipeline.detect_topic_boundaries", fake_detect)
    monkeypatch.setattr(
        "codex_audio.pipeline.match_quote_to_timestamps",
        lambda quote, words, **kwargs: (1500, 2500),
    )
    monkeypatch.setattr(
        "codex_audio.pipeline.refine_range_with_silence",
        lambda path, match_range, **kwargs: (2000, 3000),
    )

    audio_path = tmp_path / "clip.wav"
    audio_path.write_bytes(b"fake")

    candidates = pipeline._generate_llm_candidates(_words(), audio_path=audio_path)
    assert len(candidates) == 1
    assert candidates[0].time_s == pytest.approx(2.0)
    assert candidates[0].quote == "hello world"


def test_generate_llm_candidates_handles_missing_quote(monkeypatch) -> None:
    config = PipelineConfig(station="TEST", transcription_enabled=False)
    pipeline = StorySegmentationPipeline(config)
    pipeline._llm_segmentation_enabled = True

    def fake_detect(words, **kwargs):
        return [BoundaryCandidate(time_s=5.0, score=1.0, reason="llm_topic_change")]

    monkeypatch.setattr("codex_audio.pipeline.detect_topic_boundaries", fake_detect)

    candidates = pipeline._generate_llm_candidates(_words())
    assert candidates[0].time_s == 5.0
