"""Utilities for loading and updating portfolio snapshots."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Iterable, List, Optional

from .state import PortfolioState, Position


def read_operations_log(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    entries: List[Dict] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        entries.append(json.loads(line))
    return entries


def load_positions_snapshot(path: Path) -> PortfolioState:
    if not path.exists():
        return PortfolioState()
    payload = json.loads(path.read_text())
    positions = [
        Position(symbol=p["symbol"], shares=p["shares"], avg_cost=p["avg_cost"])
        for p in payload.get("positions", [])
    ]
    state = PortfolioState(cash=payload.get("cash", 0.0), positions=positions)
    return state


def save_positions_snapshot(path: Path, state: PortfolioState, snapshot_date: Optional[date] = None) -> None:
    payload = {
        "date": (snapshot_date or date.today()).isoformat(),
        "cash": state.cash,
        "positions": [
            {
                "symbol": p.symbol,
                "shares": p.shares,
                "avg_cost": p.avg_cost,
            }
            for p in state.positions
        ],
        "equity_value": state.market_value + state.cash,
        "exposure": state.current_exposure,
    }
    path.write_text(json.dumps(payload, indent=2))


def apply_daily_operations(state: PortfolioState, operations: Iterable[Dict]) -> PortfolioState:
    for entry in operations:
        actions = entry.get("actions", [])
        state.apply_operations(actions)
    return state
