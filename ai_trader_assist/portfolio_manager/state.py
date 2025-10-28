"""Portfolio state representation used across modules."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


@dataclass
class Position:
    symbol: str
    shares: int
    avg_cost: float
    last_price: float = 0.0

    @property
    def market_value(self) -> float:
        price = self.last_price or self.avg_cost
        return float(self.shares) * float(price)


class PortfolioState:
    def __init__(
        self,
        cash: float = 0.0,
        positions: Optional[Iterable[Position]] = None,
        latest_prices: Optional[Dict[str, float]] = None,
    ) -> None:
        self.cash = float(cash)
        self.positions: List[Position] = list(positions or [])
        self.latest_prices = {k.upper(): float(v) for k, v in (latest_prices or {}).items()}
        self._apply_latest_prices()

    def _apply_latest_prices(self) -> None:
        for position in self.positions:
            if position.symbol.upper() in self.latest_prices:
                position.last_price = self.latest_prices[position.symbol.upper()]

    @property
    def market_value(self) -> float:
        return sum(p.market_value for p in self.positions)

    @property
    def total_equity(self) -> float:
        return self.cash + self.market_value

    @property
    def current_exposure(self) -> float:
        equity = self.total_equity
        if equity == 0:
            return 0.0
        return self.market_value / equity

    def position_value(self, symbol: str) -> float:
        pos = self.get_position(symbol)
        return pos.market_value if pos else 0.0

    def get_position(self, symbol: str) -> Optional[Position]:
        symbol = symbol.upper()
        for position in self.positions:
            if position.symbol.upper() == symbol:
                return position
        return None

    def update_prices(self, prices: Dict[str, float]) -> None:
        self.latest_prices.update({k.upper(): float(v) for k, v in prices.items()})
        self._apply_latest_prices()

    def apply_operations(self, actions: Iterable[Dict]) -> None:
        positions = {p.symbol.upper(): p for p in self.positions}
        for action in actions:
            symbol = action["symbol"].upper()
            shares = int(action["shares"])
            price = float(action["price"])
            side = action["action"]
            position = positions.setdefault(
                symbol, Position(symbol=symbol, shares=0, avg_cost=0.0)
            )
            if side == "buy":
                total_cost = position.avg_cost * position.shares + shares * price
                position.shares += shares
                position.avg_cost = total_cost / position.shares if position.shares else 0.0
                self.cash -= shares * price
            elif side == "sell":
                position.shares = max(0, position.shares - shares)
                self.cash += shares * price
            else:
                raise ValueError(f"Unsupported action {side}")
            position.last_price = price
        self.positions = [p for p in positions.values() if p.shares > 0]
        self._apply_latest_prices()
