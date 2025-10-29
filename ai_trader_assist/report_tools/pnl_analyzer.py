"""Compute current portfolio profit and loss statistics."""
from __future__ import annotations

from datetime import date
from typing import Dict

from ..portfolio_manager.state import PortfolioState, Position


def _format_position(position: Position, last_price: float) -> Dict[str, float]:
    avg_cost = float(position.avg_cost or 0.0)
    shares = float(position.shares)
    price = float(last_price or avg_cost)
    unrealized = (price - avg_cost) * shares
    pnl_pct = (price / avg_cost - 1.0) if avg_cost else 0.0
    return {
        "symbol": position.symbol,
        "shares": shares,
        "avg_cost": avg_cost,
        "last_price": price,
        "unrealized_pnl": unrealized,
        "pnl_pct": pnl_pct,
    }


def calculate_current_pnl(
    state: PortfolioState,
    prices: Dict[str, float],
    *,
    as_of: date,
) -> Dict[str, object]:
    if prices:
        state.update_prices(prices)

    positions_payload = []
    total_unrealized = 0.0
    for position in state.positions:
        price = prices.get(position.symbol.upper(), position.last_price or position.avg_cost)
        formatted = _format_position(position, price)
        positions_payload.append(formatted)
        total_unrealized += formatted["unrealized_pnl"]

    market_value = state.market_value
    total_equity = market_value + state.cash
    exposure = market_value / total_equity if total_equity else 0.0

    return {
        "date": as_of.isoformat(),
        "equity_value": total_equity,
        "positions": positions_payload,
        "cash": state.cash,
        "total_unrealized_pnl": total_unrealized,
        "total_exposure": exposure,
    }

