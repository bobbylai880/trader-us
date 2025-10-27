"""Position sizing logic for the pre-market checklist."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

from ..portfolio_manager.state import PortfolioState


@dataclass
class PositionSizer:
    limits: Dict
    sizer_config: Dict

    def _buy_budget(self, target_exposure: float, state: PortfolioState) -> float:
        equity = state.total_equity
        exposure = state.current_exposure
        delta = max(0.0, target_exposure - exposure)
        return delta * equity

    def _max_single_value(self, state: PortfolioState) -> float:
        equity = state.total_equity
        return self.limits.get("max_single_weight", 0.25) * equity

    def _min_ticket(self) -> float:
        return self.limits.get("min_ticket", 0.0)

    def generate_orders(
        self,
        risk_view: Dict,
        stock_scores: List[Dict],
        portfolio_state: PortfolioState,
    ) -> Dict[str, List[Dict]]:
        max_exposure = self.limits.get("max_exposure", 0.85)
        target_exposure = min(float(risk_view.get("target_exposure", max_exposure)), max_exposure)
        budget = self._buy_budget(target_exposure, portfolio_state)
        buy_orders: List[Dict] = []
        sell_orders: List[Dict] = []

        atr_eps = self.sizer_config.get("atr_eps", 1e-4)

        candidates = [c for c in stock_scores if c["action"] == "buy" and c["price"] > 0]
        if budget >= self._min_ticket() and candidates:
            weights = [c["score"] / max(c.get("atr_pct", 0.01), atr_eps) for c in candidates]
            total_weight = sum(weights)
            remaining_cash = min(budget, portfolio_state.cash)

            for candidate, weight in zip(candidates, weights):
                if total_weight == 0 or remaining_cash <= 0:
                    break
                alloc = budget * (weight / total_weight)
                alloc = min(alloc, remaining_cash)

                max_value = self._max_single_value(portfolio_state)
                current_value = portfolio_state.position_value(candidate["symbol"])
                alloc = min(alloc, max(0.0, max_value - current_value))

                shares = int(math.floor(alloc / candidate["price"]))
                if shares <= 0:
                    continue
                notional = shares * candidate["price"]
                buy_orders.append(
                    {
                        "symbol": candidate["symbol"],
                        "shares": shares,
                        "price": candidate["price"],
                        "notional": notional,
                        "confidence": candidate["confidence"],
                    }
                )
                remaining_cash -= notional

        for candidate in stock_scores:
            if candidate["action"] != "reduce":
                continue
            position = portfolio_state.get_position(candidate["symbol"])
            if not position:
                continue
            shares = max(1, int(math.floor(position.shares * 0.25)))
            sell_orders.append(
                {
                    "symbol": candidate["symbol"],
                    "shares": shares,
                    "price": candidate["price"],
                    "notional": shares * candidate["price"],
                    "confidence": candidate["confidence"],
                }
            )

        return {"buy": buy_orders, "sell": sell_orders}
