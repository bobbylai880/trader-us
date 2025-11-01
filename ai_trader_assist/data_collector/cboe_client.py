"""Fetcher for Cboe end-of-day Put/Call Ratio tables."""
from __future__ import annotations

import io
import logging
import re
import time
from datetime import datetime
from typing import Dict, Optional, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup


class CboeClient:
    """Small HTTP client that scrapes Cboe's daily put/call ratio tables."""

    URL = "https://www.cboe.com/us/options/market_statistics/market/"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (compatible; AI-Trader-Assist/1.0; +https://example.com)",
    }
    SECTIONS = {
        "total": "Total",
        "index": "Index Options",
        "equity": "Equity Options",
    }

    def __init__(
        self,
        *,
        session: Optional[requests.Session] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._session = session or requests.Session()
        self._logger = logger
        self._stats: Dict[str, object] = {
            "requests": 0,
            "success": 0,
            "failures": 0,
            "last_error": None,
        }

    def fetch_put_call_ratios(
        self,
        *,
        retries: int = 2,
        timeout: int = 45,
        backoff_seconds: int = 30,
    ) -> Tuple[Optional[datetime], Dict[str, Dict[str, object]]]:
        """Return the most recent EOD put/call ratios.

        The result mirrors the validated standalone script: a tuple with the
        trading day (as a ``datetime`` with no time component) and a mapping of
        ``{"total"|"index"|"equity": {...}}`` records.  Each record contains
        the parsed columns from the final row of the respective table,
        including the ``pc_ratio`` float.
        """

        self._stats["requests"] = int(self._stats.get("requests", 0)) + 1
        last_error: Optional[Exception] = None

        for attempt in range(retries + 1):
            try:
                trade_date, sections = self._fetch_once(timeout=timeout)
                self._stats["success"] = int(self._stats.get("success", 0)) + 1
                self._stats["last_error"] = None
                return trade_date, sections
            except Exception as exc:  # pragma: no cover - network failure path
                last_error = exc
                self._stats["failures"] = int(self._stats.get("failures", 0)) + 1
                self._stats["last_error"] = str(exc)
                if attempt < retries:
                    if self._logger:
                        self._logger.warning(
                            "Cboe PCR fetch failed (attempt %d/%d): %s", attempt + 1, retries + 1, exc
                        )
                    time.sleep(backoff_seconds)
                    continue
                if self._logger:
                    self._logger.error("Cboe PCR fetch exhausted retries: %s", exc)
                raise

        if last_error:
            raise last_error
        raise RuntimeError("Unknown failure while fetching Cboe PCR data")

    def snapshot_stats(self) -> Dict[str, object]:
        """Return a shallow copy of the request statistics."""

        return dict(self._stats)

    def _fetch_once(
        self,
        *,
        timeout: int,
    ) -> Tuple[Optional[datetime], Dict[str, Dict[str, object]]]:
        response = self._session.get(self.URL, headers=self.HEADERS, timeout=timeout)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "lxml")

        trade_date = self._parse_trade_date(soup.get_text(" ", strip=True))
        sections: Dict[str, Dict[str, object]] = {}
        for key, heading in self.SECTIONS.items():
            record = self._extract_section_lastrow(soup, heading)
            if record:
                sections[key] = record

        if not sections:
            raise RuntimeError("未能在页面上解析到任何 P/C 表格。页面结构可能已变更。")

        return trade_date, sections

    @staticmethod
    def _parse_trade_date(text: str) -> Optional[datetime]:
        match = re.search(
            r"Market Statistics for\s+([A-Za-z]+,\s+[A-Za-z]+\s+\d{1,2},\s+\d{4})",
            text,
        )
        if not match:
            return None
        date_str = match.group(1)
        for fmt in ("%A, %B %d, %Y",):
            try:
                parsed = datetime.strptime(date_str, fmt)
                return parsed.replace(hour=0, minute=0, second=0, microsecond=0)
            except ValueError:
                continue
        return None

    @staticmethod
    def _extract_section_lastrow(soup: BeautifulSoup, heading: str) -> Optional[Dict[str, object]]:
        header = soup.find(
            lambda tag: tag.name in ("h2", "h3") and heading.lower() in tag.get_text(strip=True).lower()
        )
        if not header:
            return None
        table = header.find_next("table")
        if table is None:
            return None

        dataframes = pd.read_html(io.StringIO(str(table)), flavor="lxml")
        if not dataframes:
            return None
        df = dataframes[0]
        df.columns = [str(column).strip().upper() for column in df.columns]
        df = df.dropna(how="all")

        if "TIME" in df.columns:
            df = df[df["TIME"].astype(str).str.contains(r"\d{1,2}:\d{2}\s*[AP]M", na=False)]
        if df.empty:
            return None

        last_row = df.iloc[-1]
        pcr_columns = [column for column in df.columns if "P/C" in column]
        if not pcr_columns:
            return None
        pcr_column = pcr_columns[0]

        def _clean_int(value: object) -> Optional[int]:
            string = re.sub(r"[\s,]", "", str(value))
            return int(string) if string.isdigit() else None

        def _to_float(value: object) -> Optional[float]:
            try:
                return float(str(value).replace(",", ""))
            except (TypeError, ValueError):
                return None

        return {
            "time_local_central": last_row.get("TIME"),
            "calls": _clean_int(last_row.get("CALLS")),
            "puts": _clean_int(last_row.get("PUTS")),
            "total_contracts": _clean_int(last_row.get("TOTAL")),
            "pc_ratio": _to_float(last_row.get(pcr_column)),
        }
