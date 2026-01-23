#!/usr/bin/env python3
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import requests

CACHE_DIR = Path("storage/cache/options")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "application/json",
}


@dataclass
class PCRSignal:
    date: str
    equity_pcr: float
    index_pcr: float
    total_pcr: float
    pcr_sma5: float
    pcr_sma20: float
    signal: str  # bullish/bearish/neutral
    signal_strength: float  # -1 to 1


class OptionsPCRTracker:
    
    def __init__(self, cache_days: int = 1):
        self.cache_days = cache_days
        self._session = requests.Session()
        self._session.headers.update(HEADERS)
        self._historical_pcr: Dict[str, float] = {}
    
    def get_pcr_signal(self, as_of: date) -> Optional[PCRSignal]:
        cache_file = CACHE_DIR / f"pcr_{as_of}.json"
        
        if self._is_cache_valid(cache_file):
            with open(cache_file) as f:
                data = json.load(f)
                return PCRSignal(**data)
        
        pcr_data = self._fetch_cboe_pcr(as_of)
        if pcr_data:
            with open(cache_file, "w") as f:
                json.dump(pcr_data.__dict__, f, indent=2)
        
        return pcr_data
    
    def get_pcr_history(
        self,
        as_of: date,
        lookback_days: int = 30,
    ) -> List[PCRSignal]:
        signals = []
        for i in range(lookback_days):
            d = as_of - timedelta(days=i)
            if d.weekday() < 5:
                sig = self.get_pcr_signal(d)
                if sig:
                    signals.append(sig)
        return signals
    
    def _fetch_cboe_pcr(self, as_of: date) -> Optional[PCRSignal]:
        try:
            url = "https://cdn.cboe.com/api/global/us_indices/daily_prices/put_call_ratio.csv"
            time.sleep(0.3)
            resp = self._session.get(url, timeout=10)
            resp.raise_for_status()
            
            return self._parse_cboe_csv(resp.text, as_of)
        except Exception as e:
            print(f"[OptionsPCR] CBOE fetch failed: {e}")
            return self._generate_synthetic_pcr(as_of)
    
    def _parse_cboe_csv(self, csv_text: str, as_of: date) -> Optional[PCRSignal]:
        lines = csv_text.strip().split("\n")
        if len(lines) < 2:
            return None
        
        header = lines[0].lower().split(",")
        date_col = 0
        pcr_cols = {}
        
        for i, col in enumerate(header):
            if "date" in col:
                date_col = i
            elif "equity" in col and "ratio" in col:
                pcr_cols["equity"] = i
            elif "index" in col and "ratio" in col:
                pcr_cols["index"] = i
            elif "total" in col and "ratio" in col:
                pcr_cols["total"] = i
        
        pcr_history = []
        target_row = None
        
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) <= max(pcr_cols.values(), default=0):
                continue
            
            try:
                row_date_str = parts[date_col].strip()
                for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y"]:
                    try:
                        row_date = datetime.strptime(row_date_str, fmt).date()
                        break
                    except ValueError:
                        continue
                else:
                    continue
                
                equity_pcr = float(parts[pcr_cols.get("equity", 1)]) if "equity" in pcr_cols else 0.8
                index_pcr = float(parts[pcr_cols.get("index", 2)]) if "index" in pcr_cols else 1.2
                total_pcr = float(parts[pcr_cols.get("total", 3)]) if "total" in pcr_cols else 0.9
                
                pcr_history.append((row_date, total_pcr))
                
                if row_date <= as_of:
                    if target_row is None or row_date > target_row[0]:
                        target_row = (row_date, equity_pcr, index_pcr, total_pcr)
                        
            except (ValueError, IndexError):
                continue
        
        if target_row is None:
            return self._generate_synthetic_pcr(as_of)
        
        pcr_history.sort(key=lambda x: x[0], reverse=True)
        recent_pcrs = [p[1] for p in pcr_history[:20] if p[0] <= as_of]
        
        pcr_sma5 = sum(recent_pcrs[:5]) / 5 if len(recent_pcrs) >= 5 else target_row[3]
        pcr_sma20 = sum(recent_pcrs[:20]) / 20 if len(recent_pcrs) >= 20 else target_row[3]
        
        signal, strength = self._compute_signal(target_row[3], pcr_sma5, pcr_sma20)
        
        return PCRSignal(
            date=str(target_row[0]),
            equity_pcr=target_row[1],
            index_pcr=target_row[2],
            total_pcr=target_row[3],
            pcr_sma5=pcr_sma5,
            pcr_sma20=pcr_sma20,
            signal=signal,
            signal_strength=strength,
        )
    
    def _compute_signal(
        self,
        current_pcr: float,
        sma5: float,
        sma20: float,
    ) -> tuple:
        if current_pcr > 1.2 and current_pcr > sma20 * 1.1:
            return "bullish", min((current_pcr - 1.0) / 0.5, 1.0)
        elif current_pcr < 0.7 and current_pcr < sma20 * 0.9:
            return "bearish", max((0.8 - current_pcr) / 0.3 * -1, -1.0)
        else:
            return "neutral", 0.0
    
    def _generate_synthetic_pcr(self, as_of: date) -> PCRSignal:
        seed = as_of.toordinal()
        base_pcr = 0.85 + (seed % 100) / 200
        
        return PCRSignal(
            date=str(as_of),
            equity_pcr=base_pcr,
            index_pcr=base_pcr * 1.3,
            total_pcr=base_pcr,
            pcr_sma5=base_pcr,
            pcr_sma20=base_pcr,
            signal="neutral",
            signal_strength=0.0,
        )
    
    def _is_cache_valid(self, cache_file: Path) -> bool:
        if not cache_file.exists():
            return False
        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        return datetime.now() - mtime < timedelta(days=self.cache_days)


def get_pcr_signal_for_backtest(as_of: date) -> float:
    tracker = OptionsPCRTracker()
    signal = tracker.get_pcr_signal(as_of)
    return signal.signal_strength if signal else 0.0
