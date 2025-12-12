from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from codex_audio.config import StationConfig, load_station_config
from codex_audio.evaluation import io
from codex_audio.evaluation.io import EvaluationExample
from codex_audio.evaluation.matching import MatchCounts, match_segments
from codex_audio.evaluation.metrics import compute_precision_recall


class EvaluationRunner:
    def __init__(
        self,
        config_path: Optional[Path] = None,
        tolerance_seconds: float = 3.0,
    ) -> None:
        self.config_path = config_path
        self.tolerance_seconds = tolerance_seconds
        self.station_config = (
            load_station_config(config_path) if config_path else StationConfig(name="default")
        )

    def run(self, manifest_path: Path) -> Dict[str, float]:
        examples = io.load_manifest(manifest_path)
        counts = MatchCounts()
        for example in examples:
            example_counts = self._evaluate_example(example)
            counts.accumulate(example_counts)
        metrics = compute_precision_recall(counts.tp, counts.fp, counts.fn)
        metrics.update({"tp": counts.tp, "fp": counts.fp, "fn": counts.fn})
        return metrics

    def _evaluate_example(self, example: EvaluationExample) -> MatchCounts:
        references = io.load_reference_segments(example.annotation_path)
        predictions = io.load_prediction_segments(example.prediction_path)
        return match_segments(
            predictions,
            references,
            tolerance_s=self.tolerance_seconds,
        )
