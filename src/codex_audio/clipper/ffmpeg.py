from __future__ import annotations

from pathlib import Path
from typing import List


def clip_segments(source: Path, segments: List[tuple[float, float]], out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return []
