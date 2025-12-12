"""Evaluation harness exports."""

from .io import EvaluationExample, Segment, load_manifest, load_prediction_segments, load_reference_segments
from .matching import MatchCounts, match_segments
from .metrics import compute_precision_recall
from .runner import EvaluationRunner
from .sweep import SweepResult, SweepRunner

__all__ = [
    "EvaluationExample",
    "Segment",
    "load_manifest",
    "load_prediction_segments",
    "load_reference_segments",
    "MatchCounts",
    "match_segments",
    "compute_precision_recall",
    "EvaluationRunner",
    "SweepRunner",
    "SweepResult",
]
