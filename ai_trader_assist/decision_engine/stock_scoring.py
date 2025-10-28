"""Sector and stock scoring heuristics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


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
            trend_strength = float(feats.get("trend_strength", 0.0))
            score = (
                0.25 * momentum5
                + 0.25 * momentum20
                + 0.2 * relative_strength
                + 0.15 * volume_trend
                + 0.1 * news
                + 0.05 * trend_strength
            )
            results.append(
                {
                    "symbol": symbol,
                    "score": score,
                    "features": {
                        "momentum_5d": momentum5,
                        "momentum_20d": momentum20,
                        "rs": relative_strength,
                        "volume_trend": volume_trend,
                        "news_score": news,
                        "news": feats.get("news", []),
                        "trend_strength": trend_strength,
                        "trend_state": feats.get("trend_state", "flat"),
                    },
                }
            )
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
            news_score = float(feats.get("news_score", 0.0))
            trend_strength = float(feats.get("trend_strength", 0.0))
            trend_state = feats.get("trend_state", "flat")
            momentum_10d = float(feats.get("momentum_10d", 0.0))
            volatility_trend = float(feats.get("volatility_trend", 1.0))
            position_shares = float(feats.get("position_shares", 0.0))

            trend_component = _clip01(0.5 + 0.5 * trend_strength)
            if trend_state == "uptrend":
                trend_component = _clip01(trend_component + 0.1)
            elif trend_state == "downtrend":
                trend_component = _clip01(trend_component - 0.1)

            momentum_component = _clip01(0.5 + momentum_10d / 0.2)
            macd_component = _clip01(0.5 + macd_signal * 4)
            slope_component = _clip01(0.5 + trend_slope * 10)
            volume_component = _clip01(0.5 + volume_score / 2)
            structure_component = _clip01(0.5 + structure_score / 2)

            vol_penalty = max(0.0, volatility_trend - 1.0) * 0.1

            base_score = (
                0.2 * rsi_norm
                + 0.15 * macd_component
                + 0.15 * slope_component
                + 0.15 * volume_component
                + 0.1 * structure_component
                + 0.15 * momentum_component
                + 0.1 * trend_component
            )
            base_score += risk_modifier
            base_score += 0.08 * news_score
            base_score -= vol_penalty
            base_score = _clip01(base_score)

            premarket = premarket_flags.get(symbol, {})
            pre_penalty = float(premarket.get("score", 0.0)) * 0.25
            adjusted_score = max(0.0, min(1.0, base_score * (1 - pre_penalty)))

            if adjusted_score >= self.buy_threshold:
                action = "buy"
            elif adjusted_score < self.reduce_threshold:
                action = "reduce" if position_shares > 0 else "avoid"
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
                    "trend_score": trend_component,
                    "position_shares": position_shares,
                    "features": {
                        "rsi_norm": rsi_norm,
                        "macd_signal": macd_signal,
                        "trend_slope": trend_slope,
                        "volume_score": volume_score,
                        "structure_score": structure_score,
                        "risk_modifier": risk_modifier,
                        "news_score": news_score,
                        "recent_news": feats.get("recent_news", []),
                        "trend_slope_5d": float(feats.get("trend_slope_5d", 0.0)),
                        "trend_slope_20d": float(feats.get("trend_slope_20d", 0.0)),
                        "momentum_10d": momentum_10d,
                        "volatility_trend": volatility_trend,
                        "moving_avg_cross": int(feats.get("moving_avg_cross", 0)),
                        "trend_strength": trend_strength,
                        "trend_state": trend_state,
                        "momentum_state": feats.get("momentum_state", "stable"),
                        "trend_score": trend_component,
                        "position_shares": position_shares,
                        "position_value": float(feats.get("position_value", 0.0)),
                    },
                }
            )

        scored.sort(key=lambda r: r["score"], reverse=True)
        return scored
