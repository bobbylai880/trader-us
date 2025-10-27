"""Helper to read locally cached prices for sizing and reporting."""
from __future__ import annotations

from typing import Dict, Mapping


class LocalPriceStore:
    def __init__(self, latest_prices: Mapping[str, float]) -> None:
        self.latest_prices = {k.upper(): float(v) for k, v in latest_prices.items()}

    def get(self, symbol: str, default: float = 0.0) -> float:
        return self.latest_prices.get(symbol.upper(), default)

    def update(self, symbol: str, price: float) -> None:
        self.latest_prices[symbol.upper()] = float(price)

    def as_dict(self) -> Dict[str, float]:
        return dict(self.latest_prices)
