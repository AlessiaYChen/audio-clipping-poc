from __future__ import annotations

import json
from pathlib import Path

from codex_audio.ingest import AudioMetadata
from codex_audio.pipeline import PipelineConfig, StorySegmentationPipeline
from codex_audio.segmentation.change_scores import ChangePoint
from codex_audio.segmentation.planner import SegmentPlan


class FakeTranscription:
    def __init__(self) -> None:
        self.payload = {"model": "azure", "words": []}
        self.words: list = []

    def to_payload(self) -> dict[str, object]:
        return self.payload


def test_pipeline_run_creates_manifest_and_clips(monkeypatch, tmp_path: Path) -> None:
    audio_path = tmp_path / "source.wav"
    audio_path.write_bytes(b"audio")
    normalized_path = tmp_path / "norm.wav"
    normalized_path.write_bytes(b"normalized")

    metadata = AudioMetadata(
        source_path=audio_path,
        duration_s=120.0,
        sample_rate=16000,
        channels=1,
        normalized_sample_rate=16000,
        sample_width=2,
    )

    def fake_load_and_normalize(source_path: Path, work_dir: Path, target_sample_rate: int):
        work_dir.mkdir(parents=True, exist_ok=True)
        return metadata, normalized_path

    def fake_run_vad(path: Path, aggressiveness: int, frame_duration_ms: int):
        return []

    def fake_from_vad(segments, min_silence_s: float):
        return []

    def fake_build_segments(boundaries, duration_s: float, min_segment_s: float):
        return [
            SegmentPlan(0.0, 60.0, "chunk0"),
            SegmentPlan(60.0, 120.0, "chunk1"),
        ]

    def fake_compute_change_points(**kwargs):  # type: ignore[no-untyped-def]
        return [
            ChangePoint(time_s=30.0, audio_change=1.0),
            ChangePoint(time_s=90.0, audio_change=1.0),
        ]

    refine_calls: list[tuple[float, float]] = []

    def fake_refine_chunk_segments(chunk_start: float, chunk_end: float, **kwargs):  # type: ignore[no-untyped-def]
        refine_calls.append((chunk_start, chunk_end))
        if chunk_start < 60.0:
            return [
                SegmentPlan(0.0, 30.0, "inner_a"),
                SegmentPlan(30.0, 60.0, "inner_tail"),
            ]
        return [
            SegmentPlan(60.0, 90.0, "inner_b"),
            SegmentPlan(90.0, 120.0, "inner_tail"),
        ]

    clip_outputs = []

    def fake_clip_segments(source: Path, segments, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for idx, _ in enumerate(segments, start=1):
            clip_path = out_dir / f"clip_{idx}.wav"
            clip_path.write_bytes(b"clip")
            paths.append(clip_path)
        clip_outputs.extend(paths)
        return paths

    monkeypatch.setattr("codex_audio.pipeline.load_and_normalize_audio", fake_load_and_normalize)
    monkeypatch.setattr("codex_audio.pipeline.run_vad", fake_run_vad)
    monkeypatch.setattr("codex_audio.pipeline.from_vad", fake_from_vad)
    monkeypatch.setattr("codex_audio.pipeline.build_segments", fake_build_segments)
    monkeypatch.setattr("codex_audio.pipeline.compute_change_points", fake_compute_change_points)
    monkeypatch.setattr("codex_audio.pipeline.refine_chunk_segments", fake_refine_chunk_segments)
    monkeypatch.setattr("codex_audio.pipeline.clip_segments", fake_clip_segments)
    monkeypatch.setattr(
        "codex_audio.pipeline.StorySegmentationPipeline._run_transcription",
        lambda self, path: FakeTranscription(),
    )
    monkeypatch.setattr(
        "codex_audio.pipeline.StorySegmentationPipeline._compute_audio_embeddings",
        lambda self, path: [],
    )
    monkeypatch.setattr(
        "codex_audio.pipeline.StorySegmentationPipeline._build_text_chunks",
        lambda self, words: [],
    )
    monkeypatch.setattr(
        "codex_audio.pipeline.StorySegmentationPipeline._build_text_embeddings",
        lambda self, chunks: [],
    )

    pipeline = StorySegmentationPipeline(
        PipelineConfig(station="CKNW", working_dir=tmp_path / "work")
    )
    output_dir = tmp_path / "out"
    result = pipeline.run(audio_path=audio_path, output_dir=output_dir)

    assert result.manifest_path and result.manifest_path.exists()
    assert len(result.segments) == 4
    assert [round(seg["start"]) for seg in result.segments] == [0, 30, 60, 90]
    assert len(clip_outputs) == 4
    assert len(refine_calls) == 2
    assert result.transcript_path and result.transcript_path.exists()
    with result.manifest_path.open() as fh:
        payload = json.load(fh)
    assert payload["segments"][0]["clip_path"].endswith("clip_1.wav")
    assert Path(payload["segments"][0]["clip_path"]).exists()
    assert (output_dir / "clips").exists()
    assert payload["transcript"]["model"] == "azure"
