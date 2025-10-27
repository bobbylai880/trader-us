"""Thin wrapper around yfinance with basic caching and offline fallbacks."""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
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

    def _synthetic_history(
        self, symbol: str, start: datetime, end: datetime, interval: str
    ) -> pd.DataFrame:
        """Generate a deterministic synthetic price series as a fallback."""
        if interval != "1d":
            return pd.DataFrame()

        # yfinance treats ``end`` as exclusive; mirror that behaviour.
        end_exclusive = end - timedelta(days=1)
        index = pd.date_range(start=start, end=end_exclusive, freq="B")
        if index.empty:
            index = pd.date_range(end=end_exclusive, periods=30, freq="B")

        if index.empty:
            return pd.DataFrame()

        # Create a stable pseudo-random seed per symbol so results are repeatable.
        seed = int.from_bytes(hashlib.sha256(symbol.encode("utf-8")).digest()[:8], "big")
        rng = np.random.default_rng(seed)

        base_price = rng.uniform(40, 320)
        drift = rng.normal(0.05, 0.01)
        shocks = rng.normal(0, 1.5, size=len(index))
        close = np.maximum(1.0, base_price + np.cumsum(drift + shocks))

        open_price = close * (1 + rng.normal(0, 0.01, len(index)))
        high = np.maximum(open_price, close) * (1 + np.abs(rng.normal(0.01, 0.005, len(index))))
        low = np.minimum(open_price, close) * (1 - np.abs(rng.normal(0.01, 0.005, len(index))))
        volume = rng.integers(1_000_000, 5_000_000, len(index))

        data = pd.DataFrame(
            {
                "Open": open_price,
                "High": high,
                "Low": low,
                "Close": close,
                "Adj Close": close,
                "Volume": volume,
            },
            index=index,
        )
        return data

    def fetch_history(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "1d",
        force: bool = False,
    ) -> pd.DataFrame:
        """Fetch price history, caching results locally or using a fallback."""
        cache_path = self._cache_path(symbol, start, end, interval)
        if cache_path.exists() and not force:
            try:
                return pd.read_parquet(cache_path)
            except Exception:
                cache_path.unlink(missing_ok=True)

        if yf is None:
            data = pd.DataFrame()
        else:
            try:
                data = yf.download(
                    symbol,
                    start=start,
                    end=end,
                    interval=interval,
                    auto_adjust=False,
                    progress=False,
                )
            except Exception:
                data = pd.DataFrame()

        if data.empty and cache_path.exists():
            try:
                return pd.read_parquet(cache_path)
            except Exception:
                cache_path.unlink(missing_ok=True)

        if data.empty:
            data = self._synthetic_history(symbol, start, end, interval)

        if data.empty:
            return data

        try:
            data.to_parquet(cache_path)
        except Exception:
            pass
        return data

    def latest_price(self, symbol: str) -> Optional[float]:
        end = datetime.utcnow()
        start = end - timedelta(days=10)
        history = self.fetch_history(symbol, start=start, end=end, interval="1d")
        if history.empty or "Close" not in history or history["Close"].empty:
            return None
        return float(history["Close"].iloc[-1])
