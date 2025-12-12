from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from pydub import AudioSegment, effects


@dataclass
class AudioMetadata:
    """Basic details about the ingested audio asset."""

    source_path: Path
    duration_s: float
    sample_rate: int
    channels: int
    normalized_sample_rate: int
    sample_width: int


def load_and_normalize_audio(
    source_path: Path,
    work_dir: Optional[Path] = None,
    target_sample_rate: int = 16_000,
) -> Tuple[AudioMetadata, Path]:
    """Create a normalized 16 kHz mono WAV copy of ``source_path``.

    Returns tuple of (metadata, normalized_path).
    """

    source_path = source_path.expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Audio file not found: {source_path}")

    if work_dir is None:
        work_dir = source_path.parent / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    normalized_path = work_dir / f"{source_path.stem}_normalized.wav"

    audio = AudioSegment.from_file(source_path)
    metadata = AudioMetadata(
        source_path=source_path,
        duration_s=len(audio) / 1000.0,
        sample_rate=audio.frame_rate,
        channels=audio.channels,
        normalized_sample_rate=target_sample_rate,
        sample_width=audio.sample_width,
    )

    normalized = audio.set_frame_rate(target_sample_rate).set_channels(1)
    normalized = effects.normalize(normalized)
    normalized.export(normalized_path, format="wav")

    return metadata, normalized_path
