from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

from codex_audio.evaluation.runner import EvaluationRunner


@dataclass
class SweepResult:
    params: Dict[str, Any]
    metrics: Dict[str, float]


class SweepRunner:
    def __init__(self, manifest_path: Path, config_path: Path | None = None) -> None:
        self.manifest_path = manifest_path
        self.config_path = config_path

    def run(self, param_grid: Iterable[Dict[str, Any]]) -> List[SweepResult]:
        results: List[SweepResult] = []
        for params in param_grid:
            tolerance = float(params.get("tolerance_s", 3.0))
            runner = EvaluationRunner(config_path=self.config_path, tolerance_seconds=tolerance)
            metrics = runner.run(self.manifest_path)
            results.append(SweepResult(params=params, metrics=metrics))
        results.sort(key=lambda item: item.metrics.get("f1", 0.0), reverse=True)
        return results
