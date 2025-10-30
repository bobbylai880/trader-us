"""Daily report builder producing JSON + Markdown summaries."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..portfolio_manager.state import PortfolioState
from .markdown_renderer import MarkdownRenderConfig, MarkdownRenderer
from .summary import build_trend_rows


@dataclass
class DailyReportBuilder:
    sizer_config: Dict

    def _calc_price_levels(self, price: float, atr_pct: float) -> Tuple[float, float]:
        atr_value = price * atr_pct
        k1 = self.sizer_config.get("k1_stop", 1.5)
        k2 = self.sizer_config.get("k2_target", 2.5)
        stop = max(0.0, price - k1 * atr_value)
        target = price + k2 * atr_value
        return stop, target

    def build(
        self,
        trading_day: date,
        risk: Dict,
        sectors: List[Dict],
        stock_scores: List[Dict],
        orders: Dict[str, List[Dict]],
        portfolio_state: PortfolioState,
        news: Optional[Dict] = None,
        premarket_flags: Optional[Dict[str, Dict]] = None,
        snapshot_meta: Optional[Dict[str, Any]] = None,
        safe_mode: Optional[Dict[str, Any]] = None,
        llm_summary: Optional[Dict[str, Any]] = None,
        renderer_config: Optional[MarkdownRenderConfig] = None,
        report_meta: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[Dict, str]:
        payload = self.build_payload(
            trading_day=trading_day,
            risk=risk,
            sectors=sectors,
            stock_scores=stock_scores,
            orders=orders,
            portfolio_state=portfolio_state,
            news=news,
            premarket_flags=premarket_flags,
            snapshot_meta=snapshot_meta,
            safe_mode=safe_mode,
            llm_summary=llm_summary,
            report_meta=report_meta,
        )
        renderer = MarkdownRenderer(renderer_config or MarkdownRenderConfig())
        markdown = renderer.render(payload)
        return payload, markdown

    # ------------------------------------------------------------------
    # Payload construction
    # ------------------------------------------------------------------
    def build_payload(
        self,
        *,
        trading_day: date,
        risk: Mapping[str, Any],
        sectors: Sequence[Mapping[str, Any]],
        stock_scores: Sequence[Mapping[str, Any]],
        orders: Mapping[str, Sequence[Mapping[str, Any]]],
        portfolio_state: PortfolioState,
        news: Optional[Mapping[str, Any]] = None,
        premarket_flags: Optional[Mapping[str, Mapping[str, Any]]] = None,
        snapshot_meta: Optional[Mapping[str, Any]] = None,
        safe_mode: Optional[Mapping[str, Any]] = None,
        llm_summary: Optional[Mapping[str, Any]] = None,
        report_meta: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        snapshot_meta = snapshot_meta or {}
        market_section = self._build_market_section(risk, news, premarket_flags)
        exposure_section = self._build_exposure_section(risk, portfolio_state)
        allocation_plan = self._build_allocation_plan(
            orders,
            stock_scores,
            portfolio_state,
        )
        positions_section = self._build_positions_section(portfolio_state)
        sectors_section = self._build_sectors_section(sectors, news)
        stocks_section = self._build_stocks_section(stock_scores, premarket_flags, news)
        data_gaps = self._collect_data_gaps(risk, news)

        payload: Dict[str, Any] = {
            "as_of": snapshot_meta.get("as_of") or f"{trading_day.isoformat()}",
            "snapshot_id": snapshot_meta.get("snapshot_id"),
            "input_hash": snapshot_meta.get("input_hash"),
            "config_profile": snapshot_meta.get("config_profile"),
            "market": market_section,
            "exposure": exposure_section,
            "allocation_plan": allocation_plan,
            "positions": positions_section,
            "sectors": sectors_section,
            "stocks": stocks_section,
            "data_gaps": data_gaps,
        }

        if safe_mode:
            payload["safe_mode"] = safe_mode

        trend_rows = build_trend_rows(list(stock_scores))
        if trend_rows:
            payload["trend_overview"] = trend_rows

        if news and "artefacts" in news:
            payload["artefacts"] = news["artefacts"]

        if report_meta:
            appendix = report_meta.get("appendix") if isinstance(report_meta, Mapping) else None
            if isinstance(appendix, Mapping):
                payload["appendix"] = dict(appendix)

            artefact_summary = report_meta.get("artefact_summary") if isinstance(report_meta, Mapping) else None
            if isinstance(artefact_summary, Sequence):
                payload["artefact_summary"] = [
                    dict(item)
                    for item in artefact_summary
                    if isinstance(item, Mapping)
                ]

        return payload

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------
    def _build_market_section(
        self,
        risk: Mapping[str, Any],
        news: Optional[Mapping[str, Any]],
        premarket_flags: Optional[Mapping[str, Mapping[str, Any]]],
    ) -> Dict[str, Any]:
        market_news = (news or {}).get("market", {}) if isinstance(news, Mapping) else {}
        highlights = market_news.get("headlines")
        if isinstance(highlights, Sequence):
            news_highlights = [self._coerce_news_item(item) for item in highlights]
        else:
            news_highlights = []

        risk_level = risk.get("risk_level") or risk.get("risk")
        bias = risk.get("bias")
        summary = risk.get("summary")
        drivers = risk.get("drivers") if isinstance(risk.get("drivers"), Sequence) else []
        flags: List[str] = []
        if isinstance(premarket_flags, Mapping):
            for symbol, payload in premarket_flags.items():
                symbol_flags = payload.get("flags") if isinstance(payload, Mapping) else None
                if isinstance(symbol_flags, Sequence):
                    flags.extend(f"{symbol}:{flag}" for flag in symbol_flags)

        sentiment = market_news.get("sentiment")
        market_section = {
            "risk_level": risk_level,
            "bias": bias,
            "summary": summary,
            "drivers": list(drivers),
            "premarket_flags": flags,
            "news_sentiment": sentiment,
            "news_highlights": news_highlights,
        }
        return market_section

    def _build_exposure_section(
        self,
        risk: Mapping[str, Any],
        portfolio_state: PortfolioState,
    ) -> Dict[str, Any]:
        current = portfolio_state.current_exposure
        target = risk.get("target_exposure")
        delta = None
        if isinstance(target, (int, float)):
            delta = target - current
        constraints = {
            key: value
            for key, value in self.sizer_config.items()
            if key in {"max_exposure", "max_single_weight"}
        }
        return {
            "current": current,
            "target": target,
            "delta": delta,
            "constraints": constraints,
        }

    def _build_allocation_plan(
        self,
        orders: Mapping[str, Sequence[Mapping[str, Any]]],
        stock_scores: Sequence[Mapping[str, Any]],
        portfolio_state: PortfolioState,
    ) -> List[Dict[str, Any]]:
        score_lookup = {item.get("symbol"): item for item in stock_scores if item.get("symbol")}
        plan: List[Dict[str, Any]] = []
        equity = portfolio_state.total_equity

        for order in orders.get("buy", []):
            symbol = order.get("symbol")
            if not symbol:
                continue
            score = score_lookup.get(symbol, {})
            atr_pct = float(score.get("atr_pct", 0.02) or 0.02)
            price = float(order.get("price", 0.0) or 0.0)
            stop, target = self._calc_price_levels(price, atr_pct)
            notional = float(order.get("notional", 0.0) or 0.0)
            weight = (notional / equity) if equity > 0 else None
            plan.append(
                {
                    "symbol": symbol,
                    "action": "BUY",
                    "weight": weight,
                    "budget": notional,
                    "shares": order.get("shares"),
                    "price_ref": price,
                    "stops": {"atr_k1": stop},
                    "targets": {"atr_k2": target},
                    "reasons": self._build_buy_reasons(score),
                }
            )

        for order in orders.get("sell", []):
            symbol = order.get("symbol")
            if not symbol:
                continue
            notional = float(order.get("notional", 0.0) or 0.0)
            price = float(order.get("price", 0.0) or 0.0)
            weight = (-notional / equity) if equity > 0 else None
            plan.append(
                {
                    "symbol": symbol,
                    "action": "REDUCE",
                    "weight": weight,
                    "budget": -notional,
                    "shares": -abs(order.get("shares", 0)),
                    "price_ref": price,
                    "stops": {},
                    "targets": {},
                    "reasons": [order.get("reason", "risk signal")],
                }
            )

        return plan

    def _build_buy_reasons(self, score: Mapping[str, Any]) -> List[str]:
        reasons: List[str] = []
        confidence = score.get("confidence")
        if isinstance(confidence, (int, float)):
            reasons.append(f"confidence={confidence:.2f}")
        elif confidence:
            reasons.append(f"confidence={confidence}")
        features = score.get("features") if isinstance(score.get("features"), Mapping) else {}
        if features:
            trend_state = features.get("trend_state")
            if trend_state:
                reasons.append(f"trend={trend_state}")
            momentum = features.get("momentum_state")
            if momentum:
                reasons.append(f"momentum={momentum}")
        return reasons

    def _build_positions_section(self, portfolio_state: PortfolioState) -> Dict[str, Any]:
        items = [
            {
                "symbol": position.symbol,
                "shares": position.shares,
                "avg_cost": position.avg_cost,
            }
            for position in portfolio_state.positions
        ]
        return {
            "cash": portfolio_state.cash,
            "equity_value": portfolio_state.total_equity,
            "exposure": portfolio_state.current_exposure,
            "items": items,
        }

    def _build_sectors_section(
        self,
        sectors: Sequence[Mapping[str, Any]],
        news: Optional[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        highlights = (news or {}).get("sectors", {}) if isinstance(news, Mapping) else {}
        formatted: List[Dict[str, Any]] = []
        for item in sectors:
            symbol = item.get("symbol")
            if not symbol:
                continue
            sector_news = highlights.get(symbol, {}) if isinstance(highlights, Mapping) else {}
            headline = None
            if isinstance(sector_news, Mapping):
                headlines = sector_news.get("headlines")
                if isinstance(headlines, Sequence) and headlines:
                    headline = headlines[0].get("title") if isinstance(headlines[0], Mapping) else str(headlines[0])
            formatted.append(
                {
                    "symbol": symbol,
                    "score": item.get("score"),
                    "state": item.get("features", {}).get("trend_state", "—") if isinstance(item.get("features"), Mapping) else item.get("state"),
                    "news_highlight": headline,
                }
            )
        return formatted

    def _build_stocks_section(
        self,
        stock_scores: Sequence[Mapping[str, Any]],
        premarket_flags: Optional[Mapping[str, Mapping[str, Any]]],
        news: Optional[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        stocks: List[Dict[str, Any]] = []
        premarket_flags = premarket_flags or {}
        news_bundle = (news or {}).get("stocks", {}) if isinstance(news, Mapping) else {}
        for item in stock_scores:
            symbol = item.get("symbol")
            if not symbol:
                continue
            features = item.get("features") if isinstance(item.get("features"), Mapping) else {}
            premarket = premarket_flags.get(symbol, {}) if isinstance(premarket_flags, Mapping) else {}
            stock_news = news_bundle.get(symbol, {}) if isinstance(news_bundle, Mapping) else {}
            news_headline = None
            if isinstance(stock_news, Mapping):
                headlines = stock_news.get("headlines")
                if isinstance(headlines, Sequence) and headlines:
                    first = headlines[0]
                    if isinstance(first, Mapping):
                        news_headline = first.get("title")
                    else:
                        news_headline = str(first)
            recent_news = features.get("recent_news") if isinstance(features, Mapping) else []
            if not news_headline and isinstance(recent_news, Sequence) and recent_news:
                first = recent_news[0]
                news_headline = first.get("title") if isinstance(first, Mapping) else str(first)
            risks = features.get("risks") if isinstance(features, Mapping) else []
            flags = premarket.get("flags") if isinstance(premarket, Mapping) else []
            if isinstance(item.get("flags"), Sequence):
                flags = list(flags or []) + [str(flag) for flag in item.get("flags")]
            momentum = features.get("momentum_10d") if isinstance(features, Mapping) else None
            volatility = features.get("volatility_trend") if isinstance(features, Mapping) else None
            trend_strength = features.get("trend_strength") if isinstance(features, Mapping) else None
            trend_state = features.get("trend_state") if isinstance(features, Mapping) else None
            momentum_state = features.get("momentum_state") if isinstance(features, Mapping) else None
            explanation_bits = [bit for bit in [trend_state, momentum_state] if bit]
            explanation = " / ".join(explanation_bits) if explanation_bits else None
            stocks.append(
                {
                    "symbol": symbol,
                    "category": str(item.get("action", "hold")).upper(),
                    "premarket_score": premarket.get("score", item.get("premarket")),
                    "trend_strength": trend_strength,
                    "momentum_10d": momentum,
                    "volatility_trend": volatility,
                    "trend_explanation": explanation,
                    "news_highlight": news_headline,
                    "risks": risks,
                    "flags": flags,
                }
            )
        return stocks

    def _collect_data_gaps(
        self,
        risk: Mapping[str, Any],
        news: Optional[Mapping[str, Any]],
    ) -> List[str]:
        gaps: List[str] = []
        for source in (risk, news):
            if isinstance(source, Mapping):
                values = source.get("data_gaps")
                if isinstance(values, Sequence):
                    gaps.extend(str(item) for item in values if item)
        return gaps

    def _coerce_news_item(self, item: Any) -> Dict[str, Any]:
        if isinstance(item, Mapping):
            return {
                "title": item.get("title") or item.get("headline"),
                "source": item.get("source") or item.get("publisher"),
                "ts": item.get("ts") or item.get("timestamp"),
            }
        return {"title": str(item)}

    @staticmethod
    def dumps_json(payload: Dict) -> str:
        import json

        return json.dumps(payload, indent=2)
