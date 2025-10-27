"""Thin wrapper around yfinance with basic caching."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

try:  # pragma: no cover - optional dependency during unit tests
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None


class YahooFinanceClient:
    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        self.cache_dir = cache_dir or Path("storage/cache/yf")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, symbol: str, start: datetime, end: datetime, interval: str) -> Path:
        key = f"{symbol}_{start.date()}_{end.date()}_{interval}.parquet"
        return self.cache_dir / key

    def fetch_history(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "1d",
        force: bool = False,
    ) -> pd.DataFrame:
        """Fetch price history, caching results locally."""
        cache_path = self._cache_path(symbol, start, end, interval)
        if cache_path.exists() and not force:
            try:
                return pd.read_parquet(cache_path)
            except Exception:
                cache_path.unlink(missing_ok=True)

        if yf is None:
            raise RuntimeError("yfinance is not available in this environment")

        try:
            data = yf.download(
                symbol,
                start=start,
                end=end,
                interval=interval,
                progress=False,
            )
        except Exception:
            if cache_path.exists():
                try:
                    return pd.read_parquet(cache_path)
                except Exception:
                    return pd.DataFrame()
            return pd.DataFrame()

        if not data.empty:
            try:
                data.to_parquet(cache_path)
            except Exception:
                pass
        return data

    def latest_price(self, symbol: str) -> Optional[float]:
        if yf is None:
            return None
        try:
            data = yf.Ticker(symbol).history(period="1d")
        except Exception:
            return None
        if data.empty:
            return None
        return float(data["Close"].iloc[-1])
