"""Structured DeepSeek analysis orchestrator.

This module simulates the step-wise DeepSeek prompt workflow so the pipeline
can persist machine-readable LLM outputs even when the actual API is
unavailable.  Each stage consumes the latest quantitative features and produces
objective JSON fields that mirror the prompt specifications.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from ..portfolio_manager.state import PortfolioState


def _round(value: float, digits: int = 2) -> float:
    return float(round(value, digits))


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
        news: Optional[Dict] = None,
    ) -> Dict:
        market_view = self._market_overview(
            trading_day, risk, market_features, premarket_flags, news or {}
        )
        sector_view = self._sector_analysis(
            sector_scores, market_features, (news or {}).get("sectors", {})
        )
        stock_view = self._stock_actions(
            stock_scores, premarket_flags, (news or {}).get("stocks", {})
        )
        exposure_view = self._exposure_check(risk, portfolio_state, orders)
        final_view = self._compose_report(
            trading_day,
            market_view,
            sector_view,
            stock_view,
            exposure_view,
            news or {},
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
        news: Dict,
    ) -> Dict:
        drivers: List[Dict[str, object]] = []
        missing = []
        driver_entries: List[tuple] = []
        for metric, value in market_features.items():
            if value is None:
                missing.append(metric)
                continue
            if isinstance(value, (int, float)):
                numeric_value = float(value)
            else:
                # Nested dictionaries (e.g. trend snapshots) are provided for
                # completeness but they do not translate into scalar drivers.
                # Record them as unavailable for this stage so downstream
                # consumers understand they were intentionally omitted.
                missing.append(metric)
                continue
            direction = DeepSeekAnalyzer._driver_direction(metric, numeric_value)
            driver_entries.append(
                (
                    abs(numeric_value),
                    {
                        "factor": metric,
                        "evidence": f"{metric}={numeric_value:.3f}",
                        "direction": direction,
                    },
                )
            )
        driver_entries.sort(key=lambda item: item[0], reverse=True)
        drivers = [entry for _, entry in driver_entries]

        significant_flags: List[Dict[str, object]] = []
        for symbol, flag in premarket_flags.items():
            deviation = float(flag.get("dev", 0.0))
            vol_ratio = float(flag.get("vol_ratio", 0.0))
            if deviation < 0.03 and vol_ratio < 1.5:
                continue
            comment_bits = []
            if deviation >= 0.03:
                comment_bits.append(f"偏离 {deviation:.2%}")
            if vol_ratio >= 1.5:
                comment_bits.append(f"量能 {vol_ratio:.2f}x")
            score = float(flag.get("score", 0.0))
            significant_flags.append(
                {
                    "symbol": symbol,
                    "deviation": deviation,
                    "volume_ratio": vol_ratio,
                    "comment": "，".join(comment_bits) or "轻微波动",
                    "score": score,
                }
            )
        significant_flags.sort(key=lambda item: item["score"], reverse=True)

        risk_level = risk.get("risk_level", "unknown")
        bias = risk.get("bias", "neutral")
        target_exp = float(risk.get("target_exposure", 0.0) or 0.0)

        news_headlines = news.get("market", {}).get("headlines", [])
        news_sentiment = float(news.get("market", {}).get("sentiment", 0.0))

        top_driver_texts = [item["evidence"] for item in drivers[:2]]
        summary = (
            f"当前市场风险评估为{risk_level}，倾向{bias}，目标敞口约{target_exp:.0%}。"
        )
        if top_driver_texts:
            summary += "主要驱动包括" + "、".join(top_driver_texts) + "。"
        if news_headlines:
            summary += f"新闻情绪偏{('正面' if news_sentiment >= 0 else '谨慎')}，重点事件如{news_headlines[0]['title']}。"
        if len(summary) < 60:
            summary += "宏观指标总体有利于趋势延续。"

        return {
            "date": trading_day.isoformat(),
            "risk_level": risk_level,
            "bias": bias,
            "summary": summary,
            "drivers": drivers,
            "premarket_flags": significant_flags,
            "data_gaps": missing,
            "target_exposure": target_exp,
            "news_highlights": news_headlines[:5],
            "news_sentiment": news_sentiment,
        }

    @staticmethod
    def _driver_direction(metric: str, value: float) -> str:
        if math.isclose(value, 0.0, abs_tol=1e-6):
            return "mixed"
        upper = metric.upper()
        negative_bias = any(keyword in upper for keyword in ("VIX", "PUTCALL", "SPREAD"))
        if negative_bias:
            return "supports_risk_down" if value < 0 else "supports_risk_up"
        return "supports_risk_down" if value > 0 else "supports_risk_up"

    @staticmethod
    def _sector_analysis(
        sector_scores: List[Dict],
        market_features: Dict,
        news: Dict[str, Dict],
    ) -> Dict:
        leading = sector_scores[:3]
        lagging = sector_scores[-3:] if sector_scores else []

        def _format_entries(entries: Iterable[Dict]) -> List[Dict]:
            formatted: List[Dict] = []
            for item in entries:
                features = item.get("features", {})
                sector_symbol = item.get("symbol")
                news_meta = news.get(sector_symbol, {}) if news else {}
                formatted.append(
                    {
                        "sector": item.get("symbol"),
                        "composite_score": float(item.get("score", 0.0)),
                        "evidence": {
                            "mom5": float(features.get("momentum_5d", 0.0)),
                            "mom20": float(features.get("momentum_20d", 0.0)),
                            "rs_z": float(features.get("rs", 0.0)),
                            "volume_trend": float(features.get("volume_trend", 0.0)),
                            "news_score": float(features.get("news_score", 0.0)),
                        },
                        "comment": (
                            f"5日动量{features.get('momentum_5d', 0.0):+.2f}、RS{features.get('rs', 0.0):+.2f}"
                        ),
                        "news_sentiment": float(news_meta.get("sentiment", features.get("news_score", 0.0))),
                        "news_highlights": news_meta.get("headlines", [])[:3],
                    }
                )
            return formatted

        focus_points: List[Dict[str, str]] = []
        if leading:
            focus_points.append(
                {
                    "topic": "领先板块聚焦",
                    "rationale": "、".join(
                        f"{item['symbol']} 动量{item['features'].get('momentum_5d', 0.0):+.2f}"
                        for item in leading
                    ),
                }
            )
        if lagging:
            focus_points.append(
                {
                    "topic": "落后板块风险",
                    "rationale": "、".join(
                        f"{item['symbol']} 动量{item['features'].get('momentum_20d', 0.0):+.2f}"
                        for item in lagging
                    ),
                }
            )
        if news:
            highlighted = [
                (symbol, meta)
                for symbol, meta in news.items()
                if meta.get("headlines")
            ]
            if highlighted:
                first_symbol, first_meta = highlighted[0]
                focus_points.append(
                    {
                        "topic": "板块新闻焦点",
                        "rationale": f"{first_symbol} 最新事件：{first_meta['headlines'][0]['title']}",
                    }
                )
        breadth = market_features.get("BREADTH")
        if breadth is not None:
            focus_points.append(
                {
                    "topic": "市场广度",
                    "rationale": f"BREADTH={float(breadth):.2f} 显示上涨家数占比",
                }
            )

        return {
            "leading": _format_entries(leading),
            "lagging": _format_entries(lagging),
            "focus_points": focus_points or [
                {"topic": "数据不足", "rationale": "板块样本不足以提炼轮动信号"}
            ],
            "data_gaps": [],
        }

    @staticmethod
    def _stock_actions(
        stock_scores: List[Dict],
        premarket_flags: Dict[str, Dict],
        news: Dict[str, Dict],
    ) -> Dict:
        categories = {"Buy": [], "Hold": [], "Reduce": [], "Avoid": []}
        unclassified: List[Dict[str, str]] = []
        data_gaps: List[str] = []

        for stock in stock_scores:
            score = stock.get("score")
            symbol = stock.get("symbol")
            if score is None:
                unclassified.append({"symbol": symbol, "reason": "缺少综合得分"})
                continue

            features = stock.get("features", {})
            pre_flag = premarket_flags.get(symbol, {})
            pre_score = pre_flag.get("score")
            news_meta = news.get(symbol, {}) if news else {}
            news_highlights = news_meta.get("headlines", [])[:5]
            trend_strength = float(features.get("trend_strength", 0.0))
            trend_state = features.get("trend_state", "flat")
            momentum_state = features.get("momentum_state", "stable")
            momentum_10d = float(features.get("momentum_10d", 0.0))
            vol_trend = float(features.get("volatility_trend", 1.0))
            trend_score = float(stock.get("trend_score", features.get("trend_score", 0.0)))

            drivers = []
            if "rsi_norm" in features:
                drivers.append(
                    {
                        "metric": "RSI_norm",
                        "value": _round(features["rsi_norm"], 3),
                        "comment": f"RSI_norm={features['rsi_norm']:.2f} 指示趋势强劲",
                    }
                )
            if "trend_slope" in features:
                drivers.append(
                    {
                        "metric": "trend_slope",
                        "value": _round(features["trend_slope"], 4),
                        "comment": f"10日斜率占价比{features['trend_slope']:.4f}",
                    }
                )
            if "macd_signal" in features:
                drivers.append(
                    {
                        "metric": "macd_signal",
                        "value": _round(features["macd_signal"], 4),
                        "comment": f"MACD 相对价差{features['macd_signal']:.4f}",
                    }
                )
            if "trend_strength" in features:
                drivers.append(
                    {
                        "metric": "trend_strength",
                        "value": _round(trend_strength, 3),
                        "comment": f"趋势状态{trend_state}, 动量{momentum_state}",
                    }
                )
            if "momentum_10d" in features:
                drivers.append(
                    {
                        "metric": "momentum_10d",
                        "value": _round(momentum_10d, 3),
                        "comment": f"10日累计涨幅{momentum_10d:.2%}",
                    }
                )
            news_score = features.get("news_score")
            if news_score is not None:
                drivers.append(
                    {
                        "metric": "news_score",
                        "value": _round(float(news_score), 3),
                        "comment": (
                            "新闻情绪偏向利多" if float(news_score) >= 0 else "新闻情绪偏向谨慎"
                        ),
                    }
                )
            if len(drivers) < 2:
                data_gaps.append(f"{symbol} 指标不足")

            risks: List[Dict[str, object]] = []
            atr_pct = float(stock.get("atr_pct", 0.0))
            if atr_pct >= 0.05:
                risks.append(
                    {
                        "metric": "atr_pct",
                        "value": _round(atr_pct, 3),
                        "comment": "波动率较高需控制仓位",
                    }
                )
            if vol_trend >= 1.2:
                risks.append(
                    {
                        "metric": "volatility_trend",
                        "value": _round(vol_trend, 3),
                        "comment": "短期波动率上升，关注回撤风险",
                    }
                )
            if pre_score is not None and pre_score >= 0.6:
                risks.append(
                    {
                        "metric": "premarket_score",
                        "value": _round(pre_score, 3),
                        "comment": "盘前异动较大",
                    }
                )
            if news_score is not None and float(news_score) <= -0.2:
                risks.append(
                    {
                        "metric": "news_sentiment",
                        "value": _round(float(news_score), 3),
                        "comment": "新闻面偏空需防范事件风险",
                    }
                )
            if not risks:
                risks.append(
                    {
                        "metric": "risk_low",
                        "value": 0.0,
                        "comment": "当前无显著风险信号",
                    }
                )

            position_shares = float(
                stock.get("position_shares", features.get("position_shares", 0.0) or 0.0)
            )

            action = stock.get("action", "hold").lower()
            key = {
                "buy": "Buy",
                "hold": "Hold",
                "reduce": "Reduce",
            }.get(action, "Avoid")
            if key == "Reduce" and position_shares <= 0:
                key = "Avoid"

            categories[key].append(
                {
                    "symbol": symbol,
                    "score": _round(float(score), 3),
                    "price": float(stock.get("price", 0.0)),
                    "atr_pct": _round(atr_pct, 4),
                    "drivers": drivers,
                    "risks": risks,
                    "premarket_score": None if pre_score is None else _round(pre_score, 3),
                    "news_highlights": news_highlights,
                    "news_sentiment": _round(
                        float(news_meta.get("sentiment", news_score or 0.0)), 3
                    ),
                    "trend_change": momentum_state,
                    "momentum_strength": _round((trend_strength + 1) / 2, 3),
                    "trend_explanation": (
                        f"趋势{trend_state}, 动量{momentum_state}, 10日涨幅{momentum_10d:.2%}"
                    ),
                    "trend_score": _round(trend_score, 3),
                    "position_shares": _round(position_shares, 3),
                }
            )

        return {
            "categories": categories,
            "unclassified": unclassified,
            "data_gaps": sorted(set(data_gaps)),
        }

    @staticmethod
    def _exposure_check(
        risk: Dict,
        portfolio_state: PortfolioState,
        orders: Dict[str, List[Dict]],
    ) -> Dict:
        target = float(risk.get("target_exposure") or 0.0)
        current = float(portfolio_state.current_exposure)
        diff = target - current
        delta = _round(diff, 2)
        if delta >= 0.02:
            direction = "increase"
        elif delta <= -0.02:
            direction = "decrease"
        else:
            direction = "maintain"

        allocation_plan: List[Dict[str, object]] = []
        for order in orders.get("buy", []):
            allocation_plan.append(
                {
                    "action": "buy",
                    "symbol": order["symbol"],
                    "size_hint": f"买入 ~USD {order['notional']:.0f}",
                    "rationale": f"提升敞口以逼近目标{target:.0%}",
                    "linked_constraint": "",
                }
            )
        for order in orders.get("sell", []):
            allocation_plan.append(
                {
                    "action": "sell",
                    "symbol": order["symbol"],
                    "size_hint": f"减仓 ~USD {order['notional']:.0f}",
                    "rationale": "降低敞口以控制风险",
                    "linked_constraint": "",
                }
            )

        constraints: List[Dict[str, str]] = []
        if target >= 0.85:
            constraints.append(
                {
                    "name": "max_exposure_cap",
                    "status": "warning",
                    "details": "目标敞口接近 85% 限制",
                }
            )

        return {
            "current_exposure": _round(current, 3),
            "target_exposure": _round(target, 3),
            "delta": delta,
            "direction": direction,
            "allocation_plan": allocation_plan,
            "constraints": constraints,
            "data_gaps": [],
        }

    @staticmethod
    def _compose_report(
        trading_day: date,
        market: Dict,
        sectors: Dict,
        stocks: Dict,
        exposure: Dict,
        news: Dict,
    ) -> Dict:
        buy_actions = stocks["categories"].get("Buy", [])
        reduce_actions = stocks["categories"].get("Reduce", [])

        def _format_action(entry: Dict[str, object]) -> Dict[str, object]:
            price = float(entry.get("price", 0.0))
            atr_pct = float(entry.get("atr_pct", 0.02))
            stop = price * (1 - 1.5 * atr_pct)
            target = price * (1 + 2.5 * atr_pct)
            detail = f"价格 {price:.2f}，止损 {stop:.2f}，目标 {target:.2f}"
            if entry.get("drivers"):
                detail += f"，{entry['drivers'][0]['comment']}"
            return {
                "symbol": entry["symbol"],
                "action": "buy",
                "detail": detail,
            }

        actions_section: List[Dict[str, object]] = []
        actions_section.extend(_format_action(entry) for entry in buy_actions)
        for entry in reduce_actions:
            detail = "高风险信号，建议减仓 25%"
            actions_section.append(
                {
                    "symbol": entry["symbol"],
                    "action": "reduce",
                    "detail": detail,
                }
            )

        market_section = (
            f"风险{market.get('risk_level')}，倾向{market.get('bias')}，目标敞口{market.get('target_exposure', 0):.0%}"
        )
        sector_leaders = ", ".join(item["sector"] for item in sectors.get("leading", [])) or "-"
        sector_lagging = ", ".join(item["sector"] for item in sectors.get("lagging", [])) or "-"
        sectors_section = (
            f"领先：{sector_leaders}；落后：{sector_lagging}"
        )
        exposure_section = (
            f"当前敞口{exposure.get('current_exposure', 0):.0%}，目标{exposure.get('target_exposure', 0):.0%}，方向{exposure.get('direction')}"
        )

        alerts = list(stocks.get("data_gaps", []))
        if not alerts:
            alerts = ["暂无异常"]

        news_highlights = news.get("market", {}).get("headlines", [])[:3]
        markdown_lines = [
            f"📆 {trading_day.isoformat()} 盘前报告（PT）",
            f"[市场] 风险={market.get('risk_level')}, 倾向={market.get('bias')}, 目标仓位={market.get('target_exposure', 0):.0%}",
            f"[板块] 领先：{sector_leaders}；偏弱：{sector_lagging}",
            "[操作清单]",
        ]
        if actions_section:
            for action in actions_section:
                if action["action"] == "buy":
                    markdown_lines.append(
                        f"- {action['symbol']}: 买入 {action['detail']}"
                    )
                else:
                    markdown_lines.append(
                        f"- {action['symbol']}: 减仓，{action['detail']}"
                    )
        else:
            markdown_lines.append("- 暂无新指令")
        markdown_lines.append(
            f"[预计仓位] {exposure.get('target_exposure', 0):.0%}"
        )
        if news_highlights:
            markdown_lines.append("[新闻焦点]")
            for item in news_highlights:
                publisher = item.get("publisher", "")
                headline = item.get("title", "")
                markdown_lines.append(f"- {headline} ({publisher})")
        high_risk = [
            entry["symbol"]
            for entry in stocks["categories"].get("Buy", [])
            if any(risk["metric"] == "premarket_score" and risk["value"] >= 0.6 for risk in entry.get("risks", []))
        ]
        markdown_lines.append(
            "[风控] 盘前高风险：-" if not high_risk else "[风控] 盘前高风险：" + ", ".join(high_risk)
        )
        markdown_lines.append("[待办] 盘中执行后请录入 operations.jsonl")
        markdown = "\n".join(markdown_lines) + "\n"

        combined_gaps = set(market.get("data_gaps", []))
        combined_gaps.update(stocks.get("data_gaps", []))
        combined_gaps.update(exposure.get("data_gaps", []))

        news_section = [
            {
                "symbol": item.get("symbol"),
                "title": item.get("title"),
                "publisher": item.get("publisher"),
                "published": item.get("published"),
            }
            for item in news_highlights
        ]

        return {
            "markdown": markdown,
            "sections": {
                "market": market_section,
                "sectors": sectors_section,
                "actions": actions_section,
                "exposure": exposure_section,
                "news": news_section,
                "alerts": alerts,
            },
            "data_gaps": sorted(gap for gap in combined_gaps if gap),
        }
