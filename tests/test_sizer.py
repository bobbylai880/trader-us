from ai_trader_assist.position_sizer.sizer import PositionSizer
from ai_trader_assist.portfolio_manager.state import PortfolioState, Position


def test_buy_budget_allocation():
    state = PortfolioState(
        cash=5000.0,
        positions=[Position(symbol="AAPL", shares=10, avg_cost=150.0, last_price=160.0)],
    )
    sizer = PositionSizer(
        limits={"max_exposure": 0.85, "max_single_weight": 0.5, "min_ticket": 200},
        sizer_config={"atr_eps": 1e-4},
    )

    risk_view = {"target_exposure": 0.75}
    stock_scores = [
        {
            "symbol": "MSFT",
            "action": "buy",
            "price": 300.0,
            "score": 0.8,
            "confidence": 0.8,
            "atr_pct": 0.02,
        },
        {
            "symbol": "NVDA",
            "action": "buy",
            "price": 400.0,
            "score": 0.7,
            "confidence": 0.7,
            "atr_pct": 0.03,
        },
    ]

    orders = sizer.generate_orders(risk_view, stock_scores, state)
    buy_notional = sum(order["notional"] for order in orders["buy"])
    expected_budget = max(0.0, risk_view["target_exposure"] - state.current_exposure) * state.total_equity
    expected_budget = min(expected_budget, state.cash)

    assert orders["buy"]
    assert abs(buy_notional - expected_budget) / expected_budget < 0.05
