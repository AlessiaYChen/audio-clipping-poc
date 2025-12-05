from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class StationConfig:
    name: str
    sample_rate: int = 16_000
    vad: Dict[str, Any] = field(default_factory=dict)
    diarization: Dict[str, Any] = field(default_factory=dict)
    heuristics: Dict[str, Any] = field(default_factory=dict)
    text: Dict[str, Any] = field(default_factory=dict)


def load_station_config(path: Path) -> StationConfig:
    data = yaml.safe_load(path.read_text()) if path and path.exists() else {}
    if data is None:
        data = {}
    name = data.get("name", path.stem if path else "default")
    return StationConfig(
        name=name,
        sample_rate=data.get("sample_rate", 16_000),
        vad=data.get("vad", {}),
        diarization=data.get("diarization", {}),
        heuristics=data.get("heuristics", {}),
        text=data.get("text", {}),
    )
