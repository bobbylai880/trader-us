"""High-level coordinator for portfolio history and PnL reports."""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from ..data_collector.yf_client import YahooFinanceClient
from ..portfolio_manager.positions import load_positions_snapshot, read_operations_log
from ..portfolio_manager.state import PortfolioState, Position
from .history_builder import build_history
from .pnl_analyzer import calculate_current_pnl
from .report_formatter import build_markdown_report


class PriceResolver:
    """Resolve symbol prices using local feature caches or Yahoo Finance."""

    def __init__(self, project_root: Path, allow_fetch: bool = True) -> None:
        self.project_root = project_root
        self.allow_fetch = allow_fetch
        self._features_cache: Dict[str, Dict[str, float]] = {}
        cache_dir = project_root / "storage" / "cache" / "yf"
        self._client = YahooFinanceClient(cache_dir=cache_dir)

    def _feature_path(self, target_date: date) -> Path:
        return self.project_root / "storage" / f"daily_{target_date.isoformat()}" / "stock_features.json"

    def _load_feature_prices(self, target_date: date) -> Dict[str, float]:
        cache_key = target_date.isoformat()
        if cache_key in self._features_cache:
            return self._features_cache[cache_key]
        path = self._feature_path(target_date)
        prices: Dict[str, float] = {}
        if path.exists():
            try:
                data = json.loads(path.read_text())
                for symbol, payload in data.items():
                    price = (
                        payload.get("price")
                        or payload.get("last_price")
                        or payload.get("close")
                    )
                    if price:
                        prices[symbol.upper()] = float(price)
            except Exception:
                prices = {}
        self._features_cache[cache_key] = prices
        return prices

    def _extract_close(self, frame, target_date: date) -> Optional[float]:
        if frame is None:
            return None
        if hasattr(frame, "ndim") and getattr(frame, "ndim") > 1:
            if hasattr(frame, "iloc") and len(frame.index) > 0:
                frame = frame.iloc[-1]
        if hasattr(frame, "__getitem__"):
            try:
                close = frame["Close"]
            except Exception:
                close = frame.get("Close") if hasattr(frame, "get") else None
        elif isinstance(frame, dict):
            close = frame.get("Close")
        else:
            close = None
        if close is None:
            return None
        try:
            return float(close)
        except Exception:
            return None

    def _fetch_price_from_yf(self, symbol: str, target_date: date) -> Optional[float]:
        end = datetime.combine(target_date + timedelta(days=1), datetime.min.time())
        start = end - timedelta(days=10)
        history = self._client.fetch_history(symbol, start=start, end=end, interval="1d")
        if history.empty:
            return None
        key = target_date.strftime("%Y-%m-%d")
        row = None
        try:
            row = history.loc[key]
        except Exception:
            history = history.sort_index()
            subset = history.loc[:key]
            if subset.empty:
                return None
            row = subset.iloc[-1]
        return self._extract_close(row, target_date)

    def get_prices(self, symbols: Sequence[str], target_date: date) -> Dict[str, float]:
        if not symbols:
            return {}
        feature_prices = self._load_feature_prices(target_date)
        results: Dict[str, float] = {}
        missing: List[str] = []
        for symbol in symbols:
            sym = symbol.upper()
            value = feature_prices.get(sym)
            if value:
                results[sym] = value
            else:
                missing.append(sym)
        if missing and self.allow_fetch:
            for sym in missing:
                price = self._fetch_price_from_yf(sym, target_date)
                if price is not None:
                    results[sym] = price
        return results


class PortfolioReporter:
    """Generate portfolio PnL and history artefacts."""

    def __init__(
        self,
        *,
        project_root: Path,
        operations_path: Path,
        positions_path: Path,
        allow_fetch: bool = True,
    ) -> None:
        self.project_root = project_root
        self.operations_path = operations_path
        self.positions_path = positions_path
        self.price_resolver = PriceResolver(project_root, allow_fetch=allow_fetch)

    def _clone_state(self, state: PortfolioState) -> PortfolioState:
        return PortfolioState(
            cash=state.cash,
            positions=[
                Position(symbol=p.symbol, shares=p.shares, avg_cost=p.avg_cost)
                for p in state.positions
            ],
            last_updated=state.last_updated,
        )

    def generate(
        self,
        *,
        as_of: date,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, object]:
        operations = read_operations_log(self.operations_path)
        state = load_positions_snapshot(self.positions_path)

        history = build_history(
            operations,
            self.price_resolver.get_prices,
            final_cash=state.cash,
            as_of=as_of,
        )

        pnl_state = self._clone_state(state)
        symbols = [pos.symbol for pos in pnl_state.positions]
        price_map = self.price_resolver.get_prices(symbols, as_of)
        pnl_payload = calculate_current_pnl(pnl_state, price_map, as_of=as_of)

        markdown = build_markdown_report(
            as_of=as_of,
            pnl_payload=pnl_payload,
            history=history,
        )

        result = {
            "current_pnl": pnl_payload,
            "history": history,
            "markdown": markdown,
        }

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "current_pnl.json").write_text(
                json.dumps(pnl_payload, indent=2),
                encoding="utf-8",
            )
            (output_dir / "history_report.json").write_text(
                json.dumps(history, indent=2),
                encoding="utf-8",
            )
            (output_dir / "portfolio_report.md").write_text(
                markdown,
                encoding="utf-8",
            )

        return result

