from datetime import datetime, timezone

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
    timestamp = datetime(2024, 5, 1, 21, 0, tzinfo=timezone.utc).isoformat()
    operations = [
        {
            "date": "2024-05-01",
            "symbol": "AAPL",
            "action": "BUY",
            "quantity": 2,
            "price": 110.0,
            "timestamp": timestamp,
        },
        {
            "date": "2024-05-01",
            "symbol": "MSFT",
            "action": "SELL",
            "quantity": 2,
            "price": 210.0,
            "timestamp": timestamp,
        },
    ]

    apply_daily_operations(state, operations)

    aapl = state.get_position("AAPL")
    msft = state.get_position("MSFT")

    assert aapl.shares == 12
    assert round(aapl.avg_cost, 2) == round((100.0 * 10 + 110.0 * 2) / 12, 2)
    assert msft.shares == 3
    assert state.cash == 1000.0 - 110.0 * 2 + 210.0 * 2


def test_apply_operations_skips_processed_entries():
    ts_old = datetime(2024, 5, 1, tzinfo=timezone.utc)
    ts_new = datetime(2024, 5, 2, tzinfo=timezone.utc)
    state = PortfolioState(
        cash=1000.0,
        positions=[Position(symbol="AAPL", shares=10, avg_cost=100.0)],
        last_updated=ts_old.isoformat(),
    )
    operations = [
        {
            "date": "2024-05-01",
            "symbol": "AAPL",
            "action": "BUY",
            "quantity": 1,
            "price": 110.0,
            "timestamp": ts_old.isoformat(),
        },
        {
            "date": "2024-05-02",
            "symbol": "AAPL",
            "action": "SELL",
            "quantity": 2,
            "price": 120.0,
            "timestamp": ts_new.isoformat(),
        },
    ]

    apply_daily_operations(state, operations)

    position = state.get_position("AAPL")
    assert position.shares == 8
    assert state.cash == 1000.0 + 120.0 * 2
    assert state.last_updated.startswith("2024-05-02")
