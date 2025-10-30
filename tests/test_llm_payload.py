from datetime import date

import pytest

from ai_trader_assist.agent.base_agent import BaseAgent
from ai_trader_assist.agent.orchestrator import LLMOrchestrator
from ai_trader_assist.decision_engine.stock_scoring import StockDecisionEngine
from ai_trader_assist.position_sizer.sizer import PositionSizer
from ai_trader_assist.portfolio_manager.state import PortfolioState, Position
from ai_trader_assist.report_builder.builder import DailyReportBuilder
from ai_trader_assist.risk_engine.macro_engine import MacroRiskEngine


def build_agent(state: PortfolioState) -> BaseAgent:
    config = {
        "universe": {"sectors": [], "watchlist": []},
        "schedule": {"tz": "America/Los_Angeles"},
        "limits": {},
        "risk": {},
        "sizer": {},
        "risk_constraints": {},
    }
    return BaseAgent(
        config=config,
        macro_engine=MacroRiskEngine(),
        stock_engine=StockDecisionEngine(),
        sizer=PositionSizer(config["limits"], config["sizer"]),
        portfolio_state=state,
        report_builder=DailyReportBuilder(config["sizer"]),
    )


def test_llm_payload_includes_portfolio_summary() -> None:
    state = PortfolioState(
        cash=10000.0,
        positions=[
            Position(symbol="AAPL", shares=20, avg_cost=100.0, last_price=120.0),
            Position(symbol="TSLA", shares=-5, avg_cost=250.0, last_price=200.0),
        ],
    )

    agent = build_agent(state)

    payload = agent._build_llm_payload(
        trading_day=date(2024, 1, 2),
        market_features={},
        sector_features={},
        stock_features={},
        trend_features={},
        macro_flags={},
        premarket_flags={},
        news={},
    )

    context = payload["context"]
    positions = context["current_positions"]

    risk_constraints = payload["risk_constraints"]
    assert risk_constraints["portfolio_context"]["current_exposure"] == pytest.approx(
        state.current_exposure
    )

    assert context["portfolio_value"] == pytest.approx(state.total_equity)
    assert "AAPL" in positions and "TSLA" in positions
    assert positions["AAPL"]["side"] == "long"
    assert positions["TSLA"]["side"] == "short"

    aapl_weight = positions["AAPL"]["weight"]
    tsla_weight = positions["TSLA"]["weight"]

    expected_aapl = (20 * 120.0) / state.total_equity
    expected_tsla = (-5 * 200.0) / state.total_equity

    assert aapl_weight == pytest.approx(expected_aapl)
    assert tsla_weight == pytest.approx(expected_tsla)


def test_collect_data_gaps_merges_sources() -> None:
    payload = {"data_gaps": ["macro"], "features": {"data_gaps": ["premarket"]}}
    stage_results = {
        "market_analyzer": {"data_gaps": ["macro", "breadth"]},
        "sector_analyzer": {"data_gaps": ["sector_news"]},
        "stock_classifier": {},
        "exposure_planner": {"data_gaps": ["exposure"]},
    }

    merged = LLMOrchestrator._collect_data_gaps(payload, stage_results)

    assert merged == ["macro", "premarket", "breadth", "sector_news", "exposure"]
