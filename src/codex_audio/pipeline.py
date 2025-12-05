from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

from codex_audio.config.station import StationConfig, load_station_config
from codex_audio.utils import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineConfig:
    station: str
    config_path: Optional[Path] = None
    sample_rate: int = 16_000
    working_dir: Optional[Path] = None

    def resolve_station_config(self) -> StationConfig:
        if self.config_path:
            return load_station_config(self.config_path)
        return StationConfig(name=self.station, sample_rate=self.sample_rate)


@dataclass
class PipelineResult:
    segments: List[dict[str, Any]] = field(default_factory=list)
    output_dir: Path = Path("out")
    manifest_path: Optional[Path] = None


class StorySegmentationPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.station_config = config.resolve_station_config()
        logger.debug("Initialized pipeline", extra={"station": self.station_config.name})

    def run(self, audio_path: Path, output_dir: Path) -> PipelineResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "segments.json"
        placeholder_segments = [
            {
                "start_s": 0.0,
                "end_s": 30.0,
                "label": "TBD",
                "confidence": 0.0,
            }
        ]
        manifest_path.write_text(
            json.dumps({"segments": placeholder_segments, "audio": str(audio_path)}, indent=2)
        )
        logger.info(
            "Pipeline placeholder executed",
            extra={"audio": str(audio_path), "out": str(manifest_path)},
        )
        return PipelineResult(
            segments=placeholder_segments,
            output_dir=output_dir,
            manifest_path=manifest_path,
        )
