"""Sector and stock scoring heuristics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class StockDecisionEngine:
    buy_threshold: float = 0.6
    reduce_threshold: float = 0.45

    def score_sectors(self, sector_features: Dict[str, Dict]) -> List[Dict]:
        results: List[Dict] = []
        for symbol, feats in sector_features.items():
            momentum5 = float(feats.get("momentum_5d", 0.0))
            momentum20 = float(feats.get("momentum_20d", 0.0))
            relative_strength = float(feats.get("rs", 0.0))
            volume_trend = float(feats.get("volume_trend", 0.0))
            news = float(feats.get("news_score", 0.0))
            score = (
                0.25 * momentum5
                + 0.25 * momentum20
                + 0.2 * relative_strength
                + 0.2 * volume_trend
                + 0.1 * news
            )
            results.append({"symbol": symbol, "score": score})
        results.sort(key=lambda r: r["score"], reverse=True)
        return results

    def score_stocks(
        self,
        stock_features: Dict[str, Dict],
        premarket_flags: Dict[str, Dict],
    ) -> List[Dict]:
        scored: List[Dict] = []
        for symbol, feats in stock_features.items():
            rsi_norm = float(feats.get("rsi_norm", 0.5))
            macd_signal = float(feats.get("macd_signal", 0.0))
            trend_slope = float(feats.get("trend_slope", 0.0))
            volume_score = float(feats.get("volume_score", 0.0))
            structure_score = float(feats.get("structure_score", 0.0))
            risk_modifier = float(feats.get("risk_modifier", 0.0))

            base_score = (
                0.25 * rsi_norm
                + 0.2 * macd_signal
                + 0.2 * trend_slope
                + 0.2 * volume_score
                + 0.15 * structure_score
            )
            base_score += risk_modifier

            premarket = premarket_flags.get(symbol, {})
            pre_penalty = float(premarket.get("score", 0.0)) * 0.25
            adjusted_score = max(0.0, min(1.0, base_score * (1 - pre_penalty)))

            if adjusted_score >= self.buy_threshold:
                action = "buy"
            elif adjusted_score < self.reduce_threshold:
                action = "reduce"
            else:
                action = "hold"

            scored.append(
                {
                    "symbol": symbol,
                    "score": adjusted_score,
                    "action": action,
                    "confidence": adjusted_score,
                    "atr_pct": float(feats.get("atr_pct", 0.02)),
                    "price": float(feats.get("price", 0.0)),
                    "premarket": float(premarket.get("score", 0.0)),
                }
            )

        scored.sort(key=lambda r: r["score"], reverse=True)
        return scored
