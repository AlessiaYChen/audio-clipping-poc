from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Sequence, Tuple

from codex_audio.utils import get_logger

logger = get_logger(__name__)


FFMPEG_CMD = "ffmpeg"


def clip_segments(
    source: Path, segments: Sequence[Tuple[float, float]], out_dir: Path
) -> List[Path]:
    source = source.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Source audio not found: {source}")

    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    clipped_paths: List[Path] = []
    for idx, (start, end) in enumerate(segments, start=1):
        start_s = max(0.0, float(start))
        end_s = max(0.0, float(end))
        if end_s <= start_s:
            raise ValueError("Segment end must be greater than start")

        duration = end_s - start_s
        output_path = out_dir / f"{source.stem}_segment_{idx:03}.wav"
        cmd = [
            FFMPEG_CMD,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start_s:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(source),
            "-acodec",
            "pcm_s16le",
            "-ac",
            "1",
            str(output_path),
        ]
        logger.debug("Running ffmpeg clip", extra={"cmd": cmd, "output": str(output_path)})
        try:
            subprocess.run(cmd, check=True)
        except FileNotFoundError as exc:
            raise RuntimeError("ffmpeg executable not found in PATH") from exc
        clipped_paths.append(output_path)

    return clipped_paths
