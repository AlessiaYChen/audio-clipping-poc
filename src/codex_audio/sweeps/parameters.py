from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class SweepParameter:
    name: str
    values: List[str]


def parse_parameter(definition: str) -> SweepParameter:
    parts = definition.split()
    if len(parts) < 2:
        raise ValueError("Parameter definition must include a name and at least one value")
    return SweepParameter(name=parts[0], values=parts[1:])
