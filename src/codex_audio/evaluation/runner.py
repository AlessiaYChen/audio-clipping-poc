from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

from codex_audio.config import load_station_config
from codex_audio.evaluation.metrics import compute_precision_recall


class EvaluationRunner:
    def __init__(self, config_path: Path, tolerance_seconds: float = 3.0) -> None:
        self.config_path = config_path
        self.tolerance_seconds = tolerance_seconds
        self.station_config = load_station_config(config_path)

    def run(self, manifest_path: Path) -> Dict[str, float]:
        _ = self._load_manifest(manifest_path)
        return compute_precision_recall(tp=0, fp=0, fn=0)

    def _load_manifest(self, manifest_path: Path) -> List[Dict[str, str]]:
        with manifest_path.open() as handle:
            reader = csv.DictReader(handle)
            return list(reader)
