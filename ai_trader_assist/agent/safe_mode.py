"""Safe-mode fallbacks used when LLM orchestration fails."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping


@dataclass
class SafeModeConfig:
    on_llm_failure: str = "no_new_risk"
    max_exposure_cap: float = 0.4


def build_safe_outputs(
    payload: Mapping,
    reason: str,
    safe_config: SafeModeConfig,
) -> Dict[str, Dict]:
    """Generate deterministic fallback payloads when the LLM fails."""

    watchlist = list(payload.get("universe", {}).get("watchlist", []))
    summary_note = (
        "LLM orchestration failed; entered safe mode. "
        "No new risk-taking instructions are generated."
    )

    market = {
        "risk_level": "high",
        "bias": "neutral",
        "drivers": ["LLM failure"],
        "summary": summary_note,
        "premarket_flags": [],
        "news_sentiment": 0.0,
        "data_gaps": [reason],
    }

    sectors = {
        "leading": [],
        "lagging": [],
        "focus_points": ["LLM pipeline failure; manual review required."],
        "data_gaps": [reason],
    }

    categories = {bucket: [] for bucket in ("Buy", "Hold", "Reduce", "Avoid")}
    # Enforce no-new-risk: default existing watchlist to Hold.
    for symbol in watchlist:
        categories["Hold"].append(
            {
                "symbol": symbol,
                "premarket_score": 0.0,
                "trend_change": "unknown",
                "momentum_strength": "neutral",
                "trend_explanation": "LLM fallback",
                "drivers": ["LLM fallback"],
                "risks": ["Review manually"],
                "news_highlights": [],
            }
        )

    stocks = {
        "categories": categories,
        "notes": [summary_note],
        "data_gaps": [reason],
    }

    exposure = {
        "target_exposure": min(safe_config.max_exposure_cap, 0.4),
        "allocation_plan": [
            {"symbol": symbol, "weight": 0.0, "rationale": "Safe mode"}
            for symbol in watchlist
        ],
        "constraints": [
            "Safe mode active: maintain or reduce risk.",
            f"Exposure capped at {safe_config.max_exposure_cap:.0%}",
        ],
        "data_gaps": [reason],
    }

    report = {
        "markdown": (
            "⚠️ **Safe Mode Enabled**\n\n"
            f"- Reason: {reason}\n"
            f"- Exposure capped at {safe_config.max_exposure_cap:.0%}\n"
            "- Review positions manually before market open."
        ),
        "sections": {
            "market": market,
            "sectors": sectors,
            "stocks": stocks,
            "exposure": exposure,
        },
        "data_gaps": [reason],
    }

    return {
        "market_analyzer": market,
        "sector_analyzer": sectors,
        "stock_classifier": stocks,
        "exposure_planner": exposure,
        "report_composer": report,
        "safe_mode": {
            "reason": reason,
            "policy": safe_config.on_llm_failure,
        },
    }
