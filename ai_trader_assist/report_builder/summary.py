"""Helpers for rendering report summaries."""
from __future__ import annotations

from typing import Dict, List


def build_trend_rows(stock_scores: List[Dict], limit: int = 12) -> List[Dict]:
    rows: List[Dict] = []
    for item in stock_scores[:limit]:
        symbol = item.get("symbol")
        features = item.get("features", {})
        if not symbol or not features:
            continue
        trend_state = features.get("trend_state", "flat")
        momentum_state = features.get("momentum_state", "stable")
        momentum_10d = float(features.get("momentum_10d", 0.0) or 0.0)
        volatility_trend = float(features.get("volatility_trend", 1.0) or 1.0)
        trend_strength = float(features.get("trend_strength", 0.0) or 0.0)
        trend_score = float(features.get("trend_score", item.get("trend_score", 0.0)))
        comment = (
            f"{trend_state} / {momentum_state}, 强度 {trend_strength:+.2f}, "
            f"评分 {trend_score:.2f}"
        )
        rows.append(
            {
                "symbol": symbol,
                "trend": trend_state,
                "momentum_10d": momentum_10d,
                "volatility_trend": volatility_trend,
                "momentum_state": momentum_state,
                "trend_strength": trend_strength,
                "trend_score": trend_score,
                "comment": comment,
            }
        )
    return rows


def render_trend_table(rows: List[Dict]) -> List[str]:
    if not rows:
        return []
    lines = [
        "## 📈 趋势追踪 (Trend Overview)",
        "| Symbol | Trend | Momentum (10d) | Volatility Trend | Comment |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {symbol} | {trend} | {momentum:+.2%} | {vol:.2f}x | {comment} |".format(
                symbol=row["symbol"],
                trend=row["trend"],
                momentum=row["momentum_10d"],
                vol=row["volatility_trend"],
                comment=row["comment"],
            )
        )
    return lines
