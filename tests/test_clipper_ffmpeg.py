from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

from codex_audio.clipper.ffmpeg import clip_segments


def test_clip_segments_invokes_ffmpeg(tmp_path, monkeypatch):
    source = tmp_path / "source.wav"
    source.write_bytes(b"fake")

    commands: List[list[str]] = []

    def fake_run(cmd, check):  # type: ignore[no-untyped-def]
        commands.append(cmd)
        out_path = Path(cmd[-1])
        out_path.write_bytes(b"clip")

    monkeypatch.setattr("codex_audio.clipper.ffmpeg.subprocess.run", fake_run)

    out_dir = tmp_path / "clips"
    segments = [(0.0, 5.0), (5.0, 12.25)]
    outputs = clip_segments(source, segments, out_dir)

    assert len(outputs) == 2
    assert outputs[0].name == "source_segment_001.wav"
    assert commands[0][0] == "ffmpeg"
    assert "-ss" in commands[0]
    assert outputs[1].exists()


def test_clip_segments_rejects_invalid_args(tmp_path):
    source = tmp_path / "source.wav"
    source.write_bytes(b"fake")

    with pytest.raises(ValueError):
        clip_segments(source, [(5.0, 5.0)], tmp_path / "clips")

    with pytest.raises(FileNotFoundError):
        clip_segments(tmp_path / "missing.wav", [(0.0, 1.0)], tmp_path / "clips")
