"""Utilities for loading and updating portfolio snapshots."""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .state import PortfolioState, Position


def _parse_timestamp(raw: Optional[str]) -> Optional[datetime]:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


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
    state = PortfolioState(
        cash=payload.get("cash", 0.0),
        positions=positions,
        last_updated=payload.get("last_updated"),
    )
    return state


def save_positions_snapshot(
    path: Path,
    state: PortfolioState,
    snapshot_date: Optional[date] = None,
    updated_at: Optional[datetime] = None,
) -> None:
    if updated_at is None and state.last_updated:
        updated_at = _parse_timestamp(state.last_updated)
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
    if updated_at:
        payload["last_updated"] = updated_at.isoformat()
    elif state.last_updated:
        payload["last_updated"] = state.last_updated
    path.write_text(json.dumps(payload, indent=2))


def _normalize_action(entry: Dict) -> List[Dict]:
    if "actions" in entry:
        normalized: List[Dict] = []
        for action in entry.get("actions", []):
            payload = {
                "symbol": action.get("symbol"),
                "action": action.get("action"),
                "shares": action.get("shares"),
                "price": action.get("price"),
            }
            normalized.append(payload)
        return normalized
    payload = {
        "symbol": entry.get("symbol"),
        "action": entry.get("action"),
        "shares": entry.get("shares") if "shares" in entry else entry.get("quantity"),
        "price": entry.get("price"),
    }
    return [payload]


def apply_daily_operations(state: PortfolioState, operations: Iterable[Dict]) -> PortfolioState:
    last_processed = _parse_timestamp(state.last_updated) if state.last_updated else None
    newest_timestamp = last_processed

    for entry in operations:
        timestamp = _parse_timestamp(entry.get("timestamp"))
        if last_processed and timestamp and timestamp <= last_processed:
            continue
        actions = _normalize_action(entry)
        actions = [a for a in actions if a.get("symbol") and a.get("action")]
        if not actions:
            continue
        state.apply_operations(actions)
        if timestamp and (newest_timestamp is None or timestamp > newest_timestamp):
            newest_timestamp = timestamp

    if newest_timestamp:
        state.update_last_updated(newest_timestamp)
    return state
