"""Orchestrates the daily pre-market workflow.

The base agent wires the macro risk engine, sector and stock scoring layers,
position sizing module, and report builder into a single callable pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

from ..decision_engine.stock_scoring import StockDecisionEngine
from ..position_sizer.sizer import PositionSizer
from ..portfolio_manager.state import PortfolioState
from ..report_builder.builder import DailyReportBuilder
from ..risk_engine.macro_engine import MacroRiskEngine


@dataclass
class PipelineContext:
    """Container for pipeline outputs."""

    risk: Dict
    sector_scores: List[Dict]
    stock_scores: List[Dict]
    orders: Dict[str, List[Dict]]
    report_json: Dict
    report_markdown: str


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
    ) -> None:
        self.config = config
        self.macro_engine = macro_engine
        self.stock_engine = stock_engine
        self.sizer = sizer
        self.portfolio_state = portfolio_state
        self.report_builder = report_builder

    def run(
        self,
        trading_day: date,
        market_features: Dict,
        sector_features: Dict[str, Dict],
        stock_features: Dict[str, Dict],
        premarket_flags: Optional[Dict[str, Dict]] = None,
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

        risk_view = self.macro_engine.evaluate(market_features)
        sector_scores = self.stock_engine.score_sectors(sector_features)
        stock_scores = self.stock_engine.score_stocks(
            stock_features, premarket_flags=premarket_flags or {}
        )

        orders = self.sizer.generate_orders(
            risk_view,
            stock_scores,
            self.portfolio_state,
        )

        report_json, report_markdown = self.report_builder.build(
            trading_day=trading_day,
            risk=risk_view,
            sectors=sector_scores,
            stock_scores=stock_scores,
            orders=orders,
            portfolio_state=self.portfolio_state,
        )

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "report.json").write_text(
                self.report_builder.dumps_json(report_json), encoding="utf-8"
            )
            (output_dir / "report.md").write_text(report_markdown, encoding="utf-8")

        return PipelineContext(
            risk=risk_view,
            sector_scores=sector_scores,
            stock_scores=stock_scores,
            orders=orders,
            report_json=report_json,
            report_markdown=report_markdown,
        )
