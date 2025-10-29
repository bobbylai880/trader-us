"""Orchestrates the daily pre-market workflow.

The base agent wires the macro risk engine, sector and stock scoring layers,
position sizing module, and report builder into a single callable pipeline.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional

from ..decision_engine.stock_scoring import StockDecisionEngine
from ..llm.analyzer import DeepSeekAnalyzer
from ..position_sizer.sizer import PositionSizer
from ..portfolio_manager.state import PortfolioState
from ..report_builder.builder import DailyReportBuilder
from ..risk_engine.macro_engine import MacroRiskEngine
from ..utils import log_ok, log_result, log_step


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
    stage_metrics: Dict[str, Dict]


class BaseAgent:
    """High level pipeline similar to the HKUDS/AI-Trader base mode."""

    def __init__(
        self,
        config: Dict,
        macro_engine: MacroRiskEngine,
        stock_engine: StockDecisionEngine,
        sizer: PositionSizer,
        portfolio_state: PortfolioState,
        report_builder: DailyReportBuilder,
        analyzer: Optional[DeepSeekAnalyzer] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config
        self.macro_engine = macro_engine
        self.stock_engine = stock_engine
        self.sizer = sizer
        self.portfolio_state = portfolio_state
        self.report_builder = report_builder
        self.analyzer = analyzer
        self.logger = logger

    def run(
        self,
        trading_day: date,
        market_features: Dict,
        sector_features: Dict[str, Dict],
        stock_features: Dict[str, Dict],
        premarket_flags: Optional[Dict[str, Dict]] = None,
        news: Optional[Dict] = None,
        output_dir: Optional[Path] = None,
    ) -> PipelineContext:
        """Execute the pipeline.

        Parameters
        ----------
        trading_day: date
            The trading date for the report.
        market_features: Dict
            Market wide metrics consumed by the macro risk engine.
        sector_features: Dict[str, Dict]
            Features per sector ETF.
        stock_features: Dict[str, Dict]
            Per stock technical and context signals.
        premarket_flags: Dict[str, Dict], optional
            Premarket anomaly scores used for risk adjustments.
        output_dir: Path, optional
            Directory where artefacts should be written.
        """

        stage_metrics: Dict[str, Dict[str, object]] = {}

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
            stage_metrics=stage_metrics,
        )
