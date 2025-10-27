"""Structured DeepSeek analysis orchestrator.

This module simulates the step-wise DeepSeek prompt workflow so the pipeline
can persist machine-readable LLM outputs even when the actual API is
unavailable.  Each stage consumes the latest quantitative features and produces
objective JSON fields that mirror the prompt specifications.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from ..portfolio_manager.state import PortfolioState


@dataclass
class DeepSeekAnalyzer:
    """Generates structured DeepSeek-style analysis artefacts."""

    prompt_files: Dict[str, Path]

    def run(
        self,
        trading_day: date,
        risk: Dict,
        sector_scores: List[Dict],
        stock_scores: List[Dict],
        orders: Dict[str, List[Dict]],
        portfolio_state: PortfolioState,
        market_features: Dict,
        premarket_flags: Dict[str, Dict],
    ) -> Dict:
        market_view = self._market_overview(trading_day, risk, market_features, premarket_flags)
        sector_view = self._sector_analysis(sector_scores, market_features)
        stock_view = self._stock_actions(stock_scores, premarket_flags)
        exposure_view = self._exposure_check(risk, portfolio_state, orders)
        final_view = self._compose_report(
            trading_day,
            market_view,
            sector_view,
            stock_view,
            exposure_view,
        )
        return {
            "prompts": {key: str(path) for key, path in self.prompt_files.items()},
            "market_overview": market_view,
            "sector_analysis": sector_view,
            "stock_actions": stock_view,
            "exposure_check": exposure_view,
            "report_compose": final_view,
        }

    @staticmethod
    def _market_overview(
        trading_day: date,
        risk: Dict,
        market_features: Dict,
        premarket_flags: Dict[str, Dict],
    ) -> Dict:
        drivers: List[Dict[str, float]] = []
        for metric, value in market_features.items():
            if value is None:
                continue
            drivers.append({
                "metric": metric,
                "value": float(value),
            })
        drivers.sort(key=lambda item: abs(item["value"]), reverse=True)

        flag_entries: List[Dict] = []
        for symbol, flag in premarket_flags.items():
            score = float(flag.get("score", 0.0))
            if score <= 0:
                continue
            flag_entries.append(
                {
                    "symbol": symbol,
                    "score": score,
                    "deviation": float(flag.get("dev", 0.0)),
                    "volume_ratio": float(flag.get("vol_ratio", 0.0)),
                }
            )
        flag_entries.sort(key=lambda item: item["score"], reverse=True)

        summary_parts: List[str] = []
        risk_level = risk.get("risk_level", "unknown")
        bias = risk.get("bias", "neutral")
        target_exp = risk.get("target_exposure")
        summary_parts.append(
            f"Risk={risk_level}, bias={bias}, target_exposure={target_exp:.2f}" if target_exp is not None else f"Risk={risk_level}, bias={bias}"
        )
        if drivers:
            top_driver = drivers[0]
            summary_parts.append(
                f"Primary driver {top_driver['metric']}={top_driver['value']:.3f}"
            )
        summary = ". ".join(summary_parts)

        missing = [metric for metric, value in market_features.items() if value is None]

        return {
            "date": trading_day.isoformat(),
            "risk_level": risk_level,
            "bias": bias,
            "target_exposure": target_exp,
            "summary": summary,
            "drivers": drivers,
            "premarket_flags": flag_entries,
            "data_gaps": missing,
        }

    @staticmethod
    def _sector_analysis(sector_scores: List[Dict], market_features: Dict) -> Dict:
        leading = sector_scores[:3]
        lagging = sector_scores[-3:] if sector_scores else []

        def _format_entries(entries: Iterable[Dict]) -> List[Dict]:
            formatted: List[Dict] = []
            for item in entries:
                formatted.append(
                    {
                        "symbol": item.get("symbol"),
                        "score": float(item.get("score", 0.0)),
                    }
                )
            return formatted

        focus_points: List[str] = []
        if leading:
            focus_points.append(
                f"Leaders tilt toward {', '.join(item['symbol'] for item in leading)} with scores >= {min(item['score'] for item in leading):.2f}"
            )
        if lagging:
            focus_points.append(
                f"Lagging groups: {', '.join(item['symbol'] for item in lagging)}"
            )
        breadth = market_features.get("BREADTH")
        if breadth is not None:
            focus_points.append(f"Market breadth gauge={breadth:.2f}")

        return {
            "leading": _format_entries(leading),
            "lagging": _format_entries(lagging),
            "focus_points": focus_points,
            "data_gaps": [],
        }

    @staticmethod
    def _stock_actions(stock_scores: List[Dict], premarket_flags: Dict[str, Dict]) -> Dict:
        categories = {"buy": [], "hold": [], "reduce": [], "avoid": []}
        for stock in stock_scores:
            entry = {
                "symbol": stock["symbol"],
                "score": float(stock.get("score", 0.0)),
                "confidence": float(stock.get("confidence", 0.0)),
                "premarket_score": float(premarket_flags.get(stock["symbol"], {}).get("score", 0.0)),
                "atr_pct": float(stock.get("atr_pct", 0.0)),
            }
            action = stock.get("action", "hold")
            if action == "buy":
                categories["buy"].append(entry)
            elif action == "reduce":
                categories["reduce"].append(entry)
            elif action == "hold":
                categories["hold"].append(entry)
            else:
                categories["avoid"].append(entry)

        risks: List[str] = []
        for symbol, flag in premarket_flags.items():
            if float(flag.get("score", 0.0)) >= 0.6:
                risks.append(f"Premarket risk flag on {symbol} score={flag['score']:.2f}")

        return {
            "categories": categories,
            "risks": risks,
            "data_gaps": [],
        }

    @staticmethod
    def _exposure_check(risk: Dict, portfolio_state: PortfolioState, orders: Dict[str, List[Dict]]) -> Dict:
        target = float(risk.get("target_exposure") or 0.0)
        current = float(portfolio_state.current_exposure)
        diff = target - current

        allocation_plan: List[Dict] = []
        for order in orders.get("buy", []):
            allocation_plan.append(
                {
                    "symbol": order["symbol"],
                    "shares": order["shares"],
                    "notional": float(order.get("notional", order["price"] * order["shares"])),
                }
            )
        for order in orders.get("sell", []):
            allocation_plan.append(
                {
                    "symbol": order["symbol"],
                    "shares": -order["shares"],
                    "notional": -float(order.get("notional", order["price"] * order["shares"])),
                }
            )

        constraints = []
        if target > 0.85:
            constraints.append("Exposure cap at 85%")

        return {
            "target_exposure": target,
            "current_exposure": current,
            "difference": diff,
            "allocation_plan": allocation_plan,
            "constraints": constraints,
        }

    @staticmethod
    def _compose_report(
        trading_day: date,
        market: Dict,
        sectors: Dict,
        stocks: Dict,
        exposure: Dict,
    ) -> Dict:
        markdown_lines = [
            f"📋 {trading_day.isoformat()} 盘前 DeepSeek 摘要",
            f"市场风险 {market.get('risk_level')} / 倾向 {market.get('bias')} / 目标敞口 {market.get('target_exposure', 0):.0%}",
        ]
        leaders = ", ".join(item["symbol"] for item in sectors.get("leading", [])) or "-"
        laggards = ", ".join(item["symbol"] for item in sectors.get("lagging", [])) or "-"
        markdown_lines.append(f"领先板块：{leaders}；落后板块：{laggards}")
        markdown_lines.append(
            f"当前敞口 {exposure.get('current_exposure', 0):.0%} vs 目标 {exposure.get('target_exposure', 0):.0%}"
        )
        buy_list = stocks.get("categories", {}).get("buy", [])
        if buy_list:
            markdown_lines.append(
                "买入关注：" + ", ".join(f"{item['symbol']}({item['score']:.2f})" for item in buy_list)
            )
        reduce_list = stocks.get("categories", {}).get("reduce", [])
        if reduce_list:
            markdown_lines.append(
                "减仓候选：" + ", ".join(f"{item['symbol']}({item['score']:.2f})" for item in reduce_list)
            )
        risks = stocks.get("risks", [])
        if risks:
            markdown_lines.append("风险提示：" + "; ".join(risks))
        markdown = "\n".join(markdown_lines)

        data_gaps = list({*market.get("data_gaps", []), *stocks.get("data_gaps", []), *sectors.get("data_gaps", [])})

        return {
            "markdown": markdown,
            "sections": {
                "market": market,
                "sectors": sectors,
                "stocks": stocks,
                "exposure": exposure,
            },
            "data_gaps": data_gaps,
        }
