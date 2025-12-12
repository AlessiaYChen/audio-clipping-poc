from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml

DEFAULT_BOUNDARY_WEIGHTS = {
    "anchor_return": 3.0,
    "semantic_shift": 2.0,
    "audio_shift": 2.0,
    "silence_gap": 1.0,
    "jingle": 1.5,
}
DEFAULT_BOUNDARY_THRESHOLD = 1.0


@dataclass
class StationConfig:
    name: str
    sample_rate: int = 16_000
    vad: Dict[str, Any] = field(default_factory=dict)
    diarization: Dict[str, Any] = field(default_factory=dict)
    heuristics: Dict[str, Any] = field(default_factory=dict)
    text: Dict[str, Any] = field(default_factory=dict)

    def boundary_weights(self) -> Dict[str, float]:
        weights = DEFAULT_BOUNDARY_WEIGHTS.copy()
        user_weights = self.heuristics.get("boundary_weights") or {}
        for key, value in user_weights.items():
            try:
                weights[key] = float(value)
            except (TypeError, ValueError):
                continue
        return weights

    def boundary_threshold(self) -> float:
        threshold = self.heuristics.get("boundary_threshold")
        try:
            return float(threshold) if threshold is not None else DEFAULT_BOUNDARY_THRESHOLD
        except (TypeError, ValueError):
            return DEFAULT_BOUNDARY_THRESHOLD


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
