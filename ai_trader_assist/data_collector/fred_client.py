"""Client for retrieving macro data from FRED."""
from __future__ import annotations

import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd
import requests


class FredClient:
    """Simple wrapper around the FRED observations API with local caching."""

    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(
        self,
        api_key: Optional[str],
        cache_dir: Optional[Path] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.api_key = api_key
        self.cache_dir = cache_dir or Path("storage/cache/fred")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        self._stats = {
            "requests": 0,
            "cache_hits": 0,
            "series": set(),
            "rows": 0,
        }

    def _cache_path(self, series_id: str) -> Path:
        return self.cache_dir / f"{series_id}.json"

    def _fetch_series_with_status(
        self, series_id: str, start: Optional[str] = None
    ) -> Tuple[pd.DataFrame, bool]:
        """Fetch a series returning both the data frame and fetch status."""

        self._stats["requests"] += 1
        self._stats["series"].add(series_id)
        cache_path = self._cache_path(series_id)

        params: Dict[str, str] = {"series_id": series_id, "api_key": self.api_key or ""}
        params["file_type"] = "json"
        if start:
            params["observation_start"] = start

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            payload = response.json()
            observations = payload.get("observations", [])
            cache_path.write_text(json.dumps(observations))
            self._stats["rows"] += len(observations)
            return pd.DataFrame(observations), True
        except requests.RequestException:
            if cache_path.exists():
                try:
                    data = json.loads(cache_path.read_text())
                    self._stats["cache_hits"] += 1
                    self._stats["rows"] += len(data)
                    return pd.DataFrame(data), False
                except json.JSONDecodeError:
                    cache_path.unlink(missing_ok=True)
            return pd.DataFrame(), False

    def fetch_series(self, series_id: str, start: Optional[str] = None) -> pd.DataFrame:
        data, _ = self._fetch_series_with_status(series_id, start=start)
        return data

    def latest_value(self, series_id: str) -> Optional[float]:
        data = self.fetch_series(series_id)
        if data.empty:
            return None
        data = data.copy()
        data["value"] = pd.to_numeric(data["value"], errors="coerce")
        data = data.dropna(subset=["value"])
        if data.empty:
            return None
        latest = data.iloc[-1]
        return float(latest["value"])

    def snapshot_stats(self) -> Dict[str, object]:
        """Return a copy of the collected usage metrics."""

        serialisable = deepcopy(self._stats)
        series = serialisable.get("series", set())
        serialisable["series"] = sorted(series)
        serialisable["series_count"] = len(series)
        return serialisable

    def fetch_macro_indicators(
        self,
        series_ids: Iterable[str],
        start_date: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> Dict[str, Dict[str, object]]:
        """Fetch a bundle of macro indicators with values and period deltas."""

        indicators: Dict[str, Dict[str, object]] = {}
        active_logger = logger or self.logger

        for series_id in series_ids:
            frame, from_network = self._fetch_series_with_status(series_id, start=start_date)
            if frame.empty or "value" not in frame:
                continue

            frame = frame.copy()
            frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
            frame = frame.dropna(subset=["value"])
            if frame.empty:
                continue

            frame = frame.sort_values("date")
            latest = frame.iloc[-1]
            previous_value = float(frame.iloc[-2]["value"]) if len(frame) > 1 else None
            latest_value = float(latest["value"])
            change = (
                latest_value - previous_value if previous_value is not None else 0.0
            )

            indicators[series_id] = {
                "value": latest_value,
                "change": change,
                "as_of": str(latest.get("date", "")),
            }

            if not from_network and active_logger:
                active_logger.warning(
                    "[WARN] FRED fetch failed, using last cache (series=%s)",
                    series_id,
                )

        return indicators
