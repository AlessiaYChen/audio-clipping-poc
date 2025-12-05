from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from codex_audio.sweeps.parameters import parse_parameter


class SweepRunner:
    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path

    def run(self, manifest_path: Path, param_defs: List[str]) -> Dict[str, List[str]]:
        parsed = [parse_parameter(item) for item in param_defs]
        return {
            "manifest": str(manifest_path),
            "config": str(self.config_path),
            "grid": {p.name: p.values for p in parsed},
        }
