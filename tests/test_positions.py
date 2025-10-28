from ai_trader_assist.portfolio_manager.positions import apply_daily_operations
from ai_trader_assist.portfolio_manager.state import PortfolioState, Position


def test_apply_operations_updates_positions_and_cash():
    state = PortfolioState(
        cash=1000.0,
        positions=[
            Position(symbol="AAPL", shares=10, avg_cost=100.0, last_price=105.0),
            Position(symbol="MSFT", shares=5, avg_cost=200.0, last_price=205.0),
        ],
    )
    operations = [
        {
            "date": "2024-05-01",
            "actions": [
                {"symbol": "AAPL", "action": "buy", "price": 110.0, "shares": 2},
                {"symbol": "MSFT", "action": "sell", "price": 210.0, "shares": 2},
            ],
        }
    ]

    apply_daily_operations(state, operations)

    aapl = state.get_position("AAPL")
    msft = state.get_position("MSFT")

    assert aapl.shares == 12
    assert round(aapl.avg_cost, 2) == round((100.0 * 10 + 110.0 * 2) / 12, 2)
    assert msft.shares == 3
    assert state.cash == 1000.0 - 110.0 * 2 + 210.0 * 2
