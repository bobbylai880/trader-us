"""Hybrid agent that can run either the legacy rules or the LLM pipeline."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Mapping, Optional

from ..decision_engine.stock_scoring import StockDecisionEngine
from ..llm.analyzer import DeepSeekAnalyzer
from ..position_sizer.sizer import PositionSizer
from ..portfolio_manager.state import PortfolioState
from ..report_builder.builder import DailyReportBuilder
from ..risk_engine.macro_engine import MacroRiskEngine
from ..utils import log_ok, log_result, log_step

try:  # Optional import to avoid circular dependency in tests
    from .orchestrator import LLMOrchestrator
except ImportError:  # pragma: no cover - fallback when orchestrator missing
    LLMOrchestrator = None  # type: ignore


@dataclass
class PipelineContext:
    """Container for pipeline outputs."""

    risk: Dict
    sector_scores: List[Dict]
    stock_scores: List[Dict]
    orders: Dict[str, List[Dict]]
    report_json: Dict
    report_markdown: str
    llm_analysis: Optional[Dict]
    news: Optional[Dict]
    macro_flags: Optional[Dict[str, Dict]]
    stage_metrics: Dict[str, Dict]


class BaseAgent:
    """High level pipeline coordinating either rule-engine or LLM logic."""

    def __init__(
        self,
        config: Dict,
        macro_engine: MacroRiskEngine,
        stock_engine: StockDecisionEngine,
        sizer: PositionSizer,
        portfolio_state: PortfolioState,
        report_builder: DailyReportBuilder,
        analyzer: Optional[DeepSeekAnalyzer] = None,
        llm_orchestrator: Optional[LLMOrchestrator] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config
        self.macro_engine = macro_engine
        self.stock_engine = stock_engine
        self.sizer = sizer
        self.portfolio_state = portfolio_state
        self.report_builder = report_builder
        self.analyzer = analyzer
        self.llm_orchestrator = llm_orchestrator
        self.logger = logger

    def run(
        self,
        trading_day: date,
        market_features: Dict,
        sector_features: Dict[str, Dict],
        stock_features: Dict[str, Dict],
        premarket_flags: Optional[Dict[str, Dict]] = None,
        trend_features: Optional[Dict] = None,
        macro_flags: Optional[Dict[str, Dict]] = None,
        news: Optional[Dict] = None,
        output_dir: Optional[Path] = None,
    ) -> PipelineContext:
        """Execute the pipeline using either legacy rules or LLM orchestration."""

        stage_metrics: Dict[str, Dict[str, object]] = {}

        if self.llm_orchestrator:
            return self._run_llm_pipeline(
                trading_day=trading_day,
                market_features=market_features,
                sector_features=sector_features,
                stock_features=stock_features,
                trend_features=trend_features or {},
                macro_flags=macro_flags or {},
                premarket_flags=premarket_flags or {},
                news=news or {},
                output_dir=output_dir,
                stage_metrics=stage_metrics,
            )

        return self._run_legacy_pipeline(
            trading_day,
            market_features,
            sector_features,
            stock_features,
            premarket_flags,
            macro_flags,
            news,
            output_dir,
            stage_metrics,
        )

    # ------------------------------------------------------------------
    # Legacy rule engine
    # ------------------------------------------------------------------
    def _run_legacy_pipeline(
        self,
        trading_day: date,
        market_features: Dict,
        sector_features: Dict[str, Dict],
        stock_features: Dict[str, Dict],
        premarket_flags: Optional[Dict[str, Dict]],
        macro_flags: Optional[Dict[str, Dict]],
        news: Optional[Dict],
        output_dir: Optional[Path],
        stage_metrics: Dict[str, Dict[str, object]],
    ) -> PipelineContext:
        risk_start = perf_counter()
        if self.logger:
            log_step(self.logger, "risk_engine", "Evaluating macro risk signals")
        risk_view = self.macro_engine.evaluate(market_features)
        if self.logger:
            target = risk_view.get("target_exposure")
            log_result(
                self.logger,
                "risk_engine",
                f"risk_level={risk_view.get('risk_level')}, target_exposure={target:.2f}" if isinstance(target, (int, float)) else f"risk_level={risk_view.get('risk_level')}",
            )
            log_ok(
                self.logger,
                "risk_engine",
                f"Completed in {perf_counter() - risk_start:.2f}s",
            )
        stage_metrics["risk_engine"] = {
            "risk_level": risk_view.get("risk_level"),
            "target_exposure": risk_view.get("target_exposure"),
        }

        sector_start = perf_counter()
        if self.logger:
            log_step(
                self.logger,
                "sector_scoring",
                f"Scoring {len(sector_features)} sectors",
            )
        sector_scores = self.stock_engine.score_sectors(sector_features)
        if self.logger:
            top_sectors = ", ".join(item["symbol"] for item in sector_scores[:3]) or "n/a"
            log_result(
                self.logger,
                "sector_scoring",
                f"Top sectors: {top_sectors}",
            )
            log_ok(
                self.logger,
                "sector_scoring",
                f"Completed in {perf_counter() - sector_start:.2f}s",
            )
        stage_metrics["sector_scoring"] = {"count": len(sector_scores)}

        stock_start = perf_counter()
        if self.logger:
            log_step(
                self.logger,
                "stock_scoring",
                f"Scoring {len(stock_features)} stocks",
            )
        stock_scores = self.stock_engine.score_stocks(
            stock_features, premarket_flags=premarket_flags or {}
        )
        if self.logger:
            counts = {"buy": 0, "hold": 0, "reduce": 0, "avoid": 0}
            for item in stock_scores:
                action = item.get("action", "hold")
                if action in counts:
                    counts[action] += 1
            distribution = ", ".join(f"{k}={v}" for k, v in counts.items())
            log_result(
                self.logger,
                "stock_scoring",
                f"Actions: {distribution}",
            )
            log_ok(
                self.logger,
                "stock_scoring",
                f"Completed in {perf_counter() - stock_start:.2f}s",
            )
        stage_metrics["stock_scoring"] = {"count": len(stock_scores)}

        sizing_start = perf_counter()
        if self.logger:
            log_step(self.logger, "position_sizer", "Generating orders and allocations")
        orders = self.sizer.generate_orders(
            risk_view,
            stock_scores,
            self.portfolio_state,
        )
        if self.logger:
            buy_notional = sum(order.get("notional", 0.0) for order in orders.get("buy", []))
            log_result(
                self.logger,
                "position_sizer",
                f"orders: buy={len(orders.get('buy', []))}, sell={len(orders.get('sell', []))}, buy_notional={buy_notional:.2f}",
            )
            log_ok(
                self.logger,
                "position_sizer",
                f"Completed in {perf_counter() - sizing_start:.2f}s",
            )
        stage_metrics["position_sizer"] = {
            "buy_orders": len(orders.get("buy", [])),
            "sell_orders": len(orders.get("sell", [])),
        }

        report_start = perf_counter()
        if self.logger:
            log_step(self.logger, "report_builder", "Composing JSON and Markdown reports")
        report_json, report_markdown = self.report_builder.build(
            trading_day=trading_day,
            risk=risk_view,
            sectors=sector_scores,
            stock_scores=stock_scores,
            orders=orders,
            portfolio_state=self.portfolio_state,
            news=news,
        )
        if self.logger:
            log_result(
                self.logger,
                "report_builder",
                f"report_sections={len(report_json)}, markdown_chars={len(report_markdown)}",
            )
            log_ok(
                self.logger,
                "report_builder",
                f"Completed in {perf_counter() - report_start:.2f}s",
            )
        stage_metrics["report_builder"] = {
            "sections": len(report_json),
            "markdown_length": len(report_markdown),
        }

        llm_analysis = None
        if self.analyzer:
            llm_start = perf_counter()
            llm_config = self.config.get("llm", {})
            max_stocks = llm_config.get("max_stock_payload")
            llm_stocks = stock_scores
            if max_stocks and len(stock_scores) > max_stocks:
                llm_stocks = sorted(
                    stock_scores, key=lambda item: item.get("score", 0.0), reverse=True
                )[:max_stocks]
            if self.logger:
                log_step(
                    self.logger,
                    "llm",
                    f"Running staged analysis ({len(llm_stocks)} stocks)",
                )
            llm_analysis = self.analyzer.run(
                trading_day=trading_day,
                risk=risk_view,
                sector_scores=sector_scores,
                stock_scores=llm_stocks,
                orders=orders,
                portfolio_state=self.portfolio_state,
                market_features=market_features,
                premarket_flags=premarket_flags or {},
                news=news,
            )
            if self.logger and llm_analysis:
                log_result(
                    self.logger,
                    "llm",
                    f"Outputs: {', '.join(sorted(llm_analysis.keys()))}",
                )

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "report.json").write_text(
                self.report_builder.dumps_json(report_json), encoding="utf-8"
            )
            (output_dir / "report.md").write_text(report_markdown, encoding="utf-8")
            if llm_analysis is not None:
                (output_dir / "llm_analysis.json").write_text(
                    self.report_builder.dumps_json(llm_analysis), encoding="utf-8"
                )
            if news is not None:
                (output_dir / "news_snapshot.json").write_text(
                    self.report_builder.dumps_json(news), encoding="utf-8"
                )

        if self.analyzer and self.logger:
            log_ok(
                self.logger,
                "llm",
                f"Completed in {perf_counter() - llm_start:.2f}s",
            )
            stage_metrics["llm"] = {
                "stocks": len(llm_stocks),
            }
        elif not self.analyzer:
            if self.logger:
                log_result(self.logger, "llm", "Skipped")
            stage_metrics["llm"] = {"status": "skipped"}

        return PipelineContext(
            risk=risk_view,
            sector_scores=sector_scores,
            stock_scores=stock_scores,
            orders=orders,
            report_json=report_json,
            report_markdown=report_markdown,
            llm_analysis=llm_analysis,
            news=news,
            macro_flags=macro_flags,
            stage_metrics=stage_metrics,
        )

    # ------------------------------------------------------------------
    # LLM orchestrated pipeline
    # ------------------------------------------------------------------
    def _run_llm_pipeline(
        self,
        trading_day: date,
        market_features: Dict,
        sector_features: Dict[str, Dict],
        stock_features: Dict[str, Dict],
        trend_features: Dict,
        macro_flags: Dict[str, Dict],
        premarket_flags: Dict[str, Dict],
        news: Dict,
        output_dir: Optional[Path],
        stage_metrics: Dict[str, Dict[str, object]],
    ) -> PipelineContext:
        if not self.llm_orchestrator:  # pragma: no cover
            raise RuntimeError("LLM orchestrator 未初始化")

        payload = self._build_llm_payload(
            trading_day,
            market_features,
            sector_features,
            stock_features,
            trend_features,
            macro_flags,
            premarket_flags,
            news,
        )

        if self.logger:
            log_step(self.logger, "llm.orchestrator", "Invoking staged operators")

        result = self.llm_orchestrator.run(
            trading_day=trading_day,
            payload=payload,
            portfolio_state=self.portfolio_state,
        )

        stages = result.stages
        safe_mode = stages.get("safe_mode")
        market_view = stages.get("market_analyzer", {})
        exposure_view = stages.get("exposure_planner", {})
        report_view = stages.get("report_composer", {})

        target_exposure = exposure_view.get("target_exposure")
        if target_exposure is None:
            target_exposure = self.config.get("limits", {}).get("max_exposure", 0.8)

        risk_view = {
            "risk_level": market_view.get("risk_level"),
            "bias": market_view.get("bias"),
            "target_exposure": target_exposure,
        }

        sector_scores = self._convert_sector_view(stages.get("sector_analyzer", {}))
        stock_scores = self._convert_stock_view(
            stages.get("stock_classifier", {}), stock_features
        )

        stage_metrics["risk_engine"] = {
            "risk_level": risk_view.get("risk_level"),
            "target_exposure": risk_view.get("target_exposure"),
        }

        stage_metrics["sector_scoring"] = {"count": len(sector_scores)}

        stage_metrics["stock_scoring"] = {"count": len(stock_scores)}

        sizing_start = perf_counter()
        orders = self.sizer.generate_orders(
            risk_view,
            stock_scores,
            self.portfolio_state,
        )
        if self.logger:
            buy_notional = sum(order.get("notional", 0.0) for order in orders.get("buy", []))
            log_result(
                self.logger,
                "position_sizer",
                f"orders: buy={len(orders.get('buy', []))}, sell={len(orders.get('sell', []))}, buy_notional={buy_notional:.2f}",
            )
            log_ok(
                self.logger,
                "position_sizer",
                f"Completed in {perf_counter() - sizing_start:.2f}s",
            )
        stage_metrics["position_sizer"] = {
            "buy_orders": len(orders.get("buy", [])),
            "sell_orders": len(orders.get("sell", [])),
        }

        report_json = report_view.get("sections", {}) if isinstance(report_view, Mapping) else {}
        report_markdown = report_view.get("markdown", "") if isinstance(report_view, Mapping) else ""

        llm_analysis = result.to_dict()

        stage_metrics["llm"] = {
            "safe_mode": bool(safe_mode),
            "stages": list(stages.keys()),
            "artifacts_dir": str(result.artifacts_dir),
        }

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "report.json").write_text(
                self.report_builder.dumps_json(report_json), encoding="utf-8"
            )
            (output_dir / "report.md").write_text(report_markdown, encoding="utf-8")
            (output_dir / "llm_analysis.json").write_text(
                self.report_builder.dumps_json(llm_analysis), encoding="utf-8"
            )
            (output_dir / "news_snapshot.json").write_text(
                self.report_builder.dumps_json(news), encoding="utf-8"
            )

        return PipelineContext(
            risk=risk_view,
            sector_scores=sector_scores,
            stock_scores=stock_scores,
            orders=orders,
            report_json=report_json,
            report_markdown=report_markdown,
            llm_analysis=llm_analysis,
            news=news,
            macro_flags=macro_flags,
            stage_metrics=stage_metrics,
        )

    def _build_llm_payload(
        self,
        trading_day: date,
        market_features: Dict,
        sector_features: Dict[str, Dict],
        stock_features: Dict[str, Dict],
        trend_features: Dict,
        macro_flags: Dict[str, Dict],
        premarket_flags: Dict[str, Dict],
        news: Dict,
    ) -> Dict:
        universe = self.config.get("universe", {})
        schedule = self.config.get("schedule", {})

        return {
            "as_of": trading_day.isoformat(),
            "timezone": schedule.get("tz"),
            "universe": universe,
            "features": {
                "market": market_features,
                "sectors": sector_features,
                "stocks": stock_features,
                "trend": trend_features,
                "news": news,
                "premarket": premarket_flags,
                "macro_flags": macro_flags,
            },
            "constraints": {
                "limits": self.config.get("limits", {}),
                "risk": self.config.get("risk", {}),
            },
            "context": {
                "positions_snapshot": self.portfolio_state.snapshot_dict(),
            },
            "macro_flags": macro_flags,
        }

    def _convert_sector_view(self, view: Mapping) -> List[Dict]:
        results: List[Dict] = []
        if not isinstance(view, Mapping):
            return results
        for rank, item in enumerate(view.get("leading", []) or []):
            if not isinstance(item, Mapping):
                continue
            results.append(
                {
                    "symbol": item.get("symbol"),
                    "category": "leading",
                    "rank": rank,
                    "evidence": item.get("evidence"),
                }
            )
        for rank, item in enumerate(view.get("lagging", []) or []):
            if not isinstance(item, Mapping):
                continue
            results.append(
                {
                    "symbol": item.get("symbol"),
                    "category": "lagging",
                    "rank": rank,
                    "evidence": item.get("evidence"),
                }
            )
        return results

    def _convert_stock_view(
        self, view: Mapping, stock_features: Mapping[str, Mapping]
    ) -> List[Dict]:
        if not isinstance(view, Mapping):
            return []
        categories = (
            view.get("categories", {})
            if isinstance(view.get("categories"), Mapping)
            else {}
        )
        results: List[Dict] = []
        for bucket in ("Buy", "Hold", "Reduce", "Avoid"):
            for item in categories.get(bucket, []) or []:
                if not isinstance(item, Mapping):
                    continue
                symbol = item.get("symbol")
                feature = stock_features.get(symbol, {}) if isinstance(stock_features, Mapping) else {}
                price = feature.get("price", 0.0)
                atr_pct = feature.get("atr_pct", 0.02)
                score = float(item.get("premarket_score", 0.0) or 0.0)
                results.append(
                    {
                        "symbol": symbol,
                        "action": bucket.lower(),
                        "score": score,
                        "price": price,
                        "atr_pct": atr_pct,
                        "confidence": score / 100.0 if score else 0.0,
                        "drivers": item.get("drivers", []),
                        "risks": item.get("risks", []),
                        "momentum_strength": item.get("momentum_strength"),
                        "trend_change": item.get("trend_change"),
                    }
                )
        return results
