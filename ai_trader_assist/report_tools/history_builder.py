"""Build daily portfolio history snapshots from operation logs."""
from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime
from typing import Callable, Dict, Iterable, List, Sequence

from ..portfolio_manager.positions import read_operations_log
from ..portfolio_manager.state import PortfolioState

Operation = Dict[str, object]
Snapshot = Dict[str, object]
FetchPrices = Callable[[Sequence[str], date], Dict[str, float]]


def _parse_timestamp(raw: object) -> datetime | None:
    if not raw:
        return None
    if isinstance(raw, datetime):
        return raw
    if isinstance(raw, (int, float)):
        try:
            return datetime.fromtimestamp(float(raw))
        except Exception:
            return None
    if isinstance(raw, str):
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            return None
    return None


def _parse_date(raw_date: object, fallback: datetime | None = None) -> date | None:
    if isinstance(raw_date, date) and not isinstance(raw_date, datetime):
        return raw_date
    if isinstance(raw_date, str):
        for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
            try:
                return datetime.strptime(raw_date, fmt).date()
            except ValueError:
                continue
        ts = _parse_timestamp(raw_date)
        if ts:
            return ts.date()
    if isinstance(raw_date, datetime):
        return raw_date.date()
    if fallback:
        return fallback.date()
    return None


def _normalize_actions(entry: Operation) -> List[Dict[str, object]]:
    if "actions" in entry and isinstance(entry["actions"], Iterable):
        normalized: List[Dict[str, object]] = []
        for action in entry.get("actions", []):
            if not isinstance(action, dict):
                continue
            payload = {
                "symbol": action.get("symbol"),
                "action": action.get("action"),
                "shares": action.get("shares") if "shares" in action else action.get("quantity"),
                "price": action.get("price"),
            }
            normalized.append(payload)
        return normalized
    return [
        {
            "symbol": entry.get("symbol"),
            "action": entry.get("action"),
            "shares": entry.get("shares") if "shares" in entry else entry.get("quantity"),
            "price": entry.get("price"),
        }
    ]


def load_operations(path) -> List[Operation]:
    return read_operations_log(path)


def build_history(
    operations: Iterable[Operation],
    fetch_prices: FetchPrices,
    *,
    final_cash: float | None = None,
    initial_cash: float = 0.0,
    as_of: date | None = None,
) -> List[Snapshot]:
    grouped: Dict[date, List[Operation]] = defaultdict(list)
    for entry in operations:
        timestamp = _parse_timestamp(entry.get("timestamp"))
        entry_date = _parse_date(entry.get("date"), timestamp) or (timestamp.date() if timestamp else None)
        if entry_date is None:
            continue
        if as_of and entry_date > as_of:
            continue
        grouped[entry_date].append(entry)

    if not grouped:
        return []

    state = PortfolioState(cash=initial_cash, positions=[])
    snapshots: List[Snapshot] = []

    for day in sorted(grouped):
        actions: List[Dict[str, object]] = []
        for entry in grouped[day]:
            for action in _normalize_actions(entry):
                symbol = action.get("symbol")
                action_type = action.get("action")
                if not symbol or not action_type:
                    continue
                actions.append(
                    {
                        "symbol": str(symbol).upper(),
                        "action": str(action_type).lower(),
                        "shares": action.get("shares"),
                        "price": action.get("price"),
                    }
                )
        if actions:
            state.apply_operations(actions)
        symbols = [pos.symbol for pos in state.positions]
        prices = fetch_prices(symbols, day) if symbols else {}
        if prices:
            state.update_prices(prices)
        market_value = state.market_value
        snapshot = {
            "date": day.isoformat(),
            "cash": state.cash,
            "market_value": market_value,
            "total_value": market_value + state.cash,
            "exposure": state.current_exposure,
            "holdings": {pos.symbol: pos.shares for pos in state.positions},
        }
        snapshots.append(snapshot)

    if final_cash is not None and snapshots:
        delta = float(final_cash) - float(snapshots[-1]["cash"])
        if abs(delta) > 1e-6:
            for snap in snapshots:
                snap["cash"] = float(snap["cash"]) + delta
                snap["total_value"] = float(snap["market_value"]) + float(snap["cash"])
                total = snap["total_value"]
                snap["exposure"] = float(snap["market_value"]) / total if total else 0.0

    for snap in snapshots:
        snap["cash"] = float(snap["cash"])
        snap["market_value"] = float(snap["market_value"])
        snap["total_value"] = float(snap["total_value"])
        snap["exposure"] = float(snap["exposure"])
    return snapshots

