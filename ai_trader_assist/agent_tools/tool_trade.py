"""Utility to simulate trade execution on a local portfolio state."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass
class TradeAction:
    symbol: str
    action: str  # "buy" or "sell"
    price: float
    shares: int


class TradeSimulator:
    """Apply hypothetical trades to a portfolio snapshot."""

    def __init__(self, state: Dict) -> None:
        self.state = state

    def apply(self, actions: Iterable[TradeAction]) -> Dict:
        positions = {p["symbol"]: p.copy() for p in self.state.get("positions", [])}
        cash = float(self.state.get("cash", 0.0))

        for action in actions:
            if action.shares <= 0:
                continue
            cost = action.price * action.shares
            position = positions.setdefault(
                action.symbol, {"symbol": action.symbol, "shares": 0, "avg_cost": 0.0}
            )

            if action.action == "buy":
                total_cost = position["avg_cost"] * position["shares"] + cost
                total_shares = position["shares"] + action.shares
                position["shares"] = total_shares
                position["avg_cost"] = total_cost / total_shares if total_shares else 0.0
                cash -= cost
            elif action.action == "sell":
                position["shares"] = max(0, position["shares"] - action.shares)
                cash += cost
            else:
                raise ValueError(f"Unsupported action: {action.action}")

        # remove zero share positions
        filtered_positions: List[Dict] = [
            p for p in positions.values() if p.get("shares", 0) > 0
        ]
        return {"cash": cash, "positions": filtered_positions}
