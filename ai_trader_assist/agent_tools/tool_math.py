"""Math utilities shared across tools."""
from __future__ import annotations

from typing import Iterable, List


def normalize(values: Iterable[float]) -> List[float]:
    values = list(values)
    total = sum(values)
    if total == 0:
        return [0.0 for _ in values]
    return [v / total for v in values]


def clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
