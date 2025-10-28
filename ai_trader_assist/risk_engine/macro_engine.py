"""Macro risk engine producing daily exposure targets."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict

from ..agent_tools.tool_math import clip


@dataclass
class MacroRiskEngine:
    weights: Dict[str, float] = field(
        default_factory=lambda: {
            "RS_SPY": 1.2,
            "RS_QQQ": 1.0,
            "VIX_Z": -1.1,
            "PUTCALL_Z": -0.9,
            "BREADTH": 0.8,
        }
    )
    base_exposure: float = 0.4
    max_increment: float = 0.4

    def _sigmoid(self, x: float) -> float:
        return 1 / (1 + math.exp(-x))

    def evaluate(self, features: Dict[str, float]) -> Dict:
        """Compute the risk score and exposure target."""
        drivers = {}
        weighted_sum = 0.0
        for key, weight in self.weights.items():
            value = float(features.get(key, 0.0))
            contribution = weight * value
            weighted_sum += contribution
            drivers[key] = {
                "value": value,
                "weight": weight,
                "contribution": contribution,
            }

        macro_score = self._sigmoid(weighted_sum)
        target_exposure = clip(
            self.base_exposure + self.max_increment * macro_score,
            self.base_exposure,
            self.base_exposure + self.max_increment,
        )

        if macro_score < 0.33:
            risk_level = "high"
            bias = "bearish"
        elif macro_score < 0.66:
            risk_level = "medium"
            bias = "neutral"
        else:
            risk_level = "low"
            bias = "bullish"

        return {
            "score": macro_score,
            "target_exposure": target_exposure,
            "risk_level": risk_level,
            "bias": bias,
            "drivers": drivers,
        }
