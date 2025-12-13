"""Segmentation utilities."""

from .candidates import build_boundary_candidates, from_vad
from .change_scores import ChangePoint, compute_change_points, find_peak_candidates, smooth_scores
from .planner import SegmentPlan, build_segments
from .refinement import DEFAULT_CHANGE_WEIGHTS, RefinementParams, refine_chunk_segments
from .selection import SegmentConstraint, select_boundaries

__all__ = [
    "from_vad",
    "build_boundary_candidates",
    "SegmentPlan",
    "build_segments",
    "ChangePoint",
    "compute_change_points",
    "find_peak_candidates",
    "smooth_scores",
    "SegmentConstraint",
    "select_boundaries",
    "RefinementParams",
    "DEFAULT_CHANGE_WEIGHTS",
    "refine_chunk_segments",
]

