"""Client for retrieving macro data from FRED."""
from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests


class FredClient:
    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(self, api_key: Optional[str], cache_dir: Optional[Path] = None) -> None:
        self.api_key = api_key
        self.cache_dir = cache_dir or Path("storage/cache/fred")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._stats = {
            "requests": 0,
            "cache_hits": 0,
            "series": set(),
            "rows": 0,
        }

    def _cache_path(self, series_id: str) -> Path:
        return self.cache_dir / f"{series_id}.json"

    def fetch_series(self, series_id: str, start: Optional[str] = None) -> pd.DataFrame:
        self._stats["requests"] += 1
        self._stats["series"].add(series_id)
        cache_path = self._cache_path(series_id)
        if cache_path.exists():
            try:
                data = json.loads(cache_path.read_text())
                self._stats["cache_hits"] += 1
                self._stats["rows"] += len(data)
                return pd.DataFrame(data)
            except json.JSONDecodeError:
                cache_path.unlink(missing_ok=True)

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
            return pd.DataFrame(observations)
        except requests.RequestException:
            if cache_path.exists():
                try:
                    data = json.loads(cache_path.read_text())
                    self._stats["cache_hits"] += 1
                    self._stats["rows"] += len(data)
                    return pd.DataFrame(data)
                except json.JSONDecodeError:
                    return pd.DataFrame()
            return pd.DataFrame()

    def latest_value(self, series_id: str) -> Optional[float]:
        data = self.fetch_series(series_id)
        if data.empty:
            return None
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
