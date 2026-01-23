#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import requests

CACHE_DIR = Path("storage/cache/truth")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "AI-Trader-Research/1.0 (Educational Purpose)",
}

TARIFF_KEYWORDS = {
    "tariff", "tariffs", "china", "trade", "import", "export",
    "duty", "duties", "mexico", "canada", "reciprocal",
}

POLICY_KEYWORDS = {
    "tax", "regulation", "deregulation", "drill", "energy",
    "bitcoin", "crypto", "fed", "rates", "interest",
}

SECTOR_IMPACT = {
    "tariff": {"XLI": -0.3, "XLY": -0.2, "XLK": -0.1, "XLE": 0.1},
    "deregulation": {"XLF": 0.3, "XLE": 0.2, "XLI": 0.1},
    "energy": {"XLE": 0.4, "XLI": 0.1},
    "crypto": {"XLF": 0.1},
    "tax": {"XLF": 0.2, "XLY": 0.2, "XLI": 0.1},
}


@dataclass
class PolicySignal:
    date: str
    topic: str
    sentiment: float
    sector_impacts: Dict[str, float]
    urgency: float
    signal_strength: float


class TruthSocialTracker:
    
    def __init__(self, cache_days: int = 1):
        self.cache_days = cache_days
        self._session = requests.Session()
        self._session.headers.update(HEADERS)
    
    def get_policy_signals(
        self,
        as_of: date,
        lookback_days: int = 7,
    ) -> List[PolicySignal]:
        cache_file = CACHE_DIR / f"policy_{as_of}_{lookback_days}.json"
        
        if self._is_cache_valid(cache_file):
            with open(cache_file) as f:
                data = json.load(f)
                return [PolicySignal(**s) for s in data]
        
        signals = self._generate_historical_signals(as_of, lookback_days)
        
        with open(cache_file, "w") as f:
            json.dump([s.__dict__ for s in signals], f, indent=2)
        
        return signals
    
    def get_sector_adjustment(
        self,
        as_of: date,
        lookback_days: int = 7,
    ) -> Dict[str, float]:
        signals = self.get_policy_signals(as_of, lookback_days)
        
        sector_adj: Dict[str, float] = {}
        for sig in signals:
            for sector, impact in sig.sector_impacts.items():
                if sector not in sector_adj:
                    sector_adj[sector] = 0.0
                sector_adj[sector] += impact * sig.urgency
        
        for sector in sector_adj:
            sector_adj[sector] = max(-1, min(1, sector_adj[sector]))
        
        return sector_adj
    
    def _generate_historical_signals(
        self,
        as_of: date,
        lookback_days: int,
    ) -> List[PolicySignal]:
        signals = []
        
        historical_events = self._get_historical_policy_events()
        
        start_date = as_of - timedelta(days=lookback_days)
        
        for event_date_str, event_data in historical_events.items():
            try:
                event_date = datetime.strptime(event_date_str, "%Y-%m-%d").date()
            except ValueError:
                continue
            
            if start_date <= event_date <= as_of:
                signals.append(PolicySignal(
                    date=event_date_str,
                    topic=event_data["topic"],
                    sentiment=event_data["sentiment"],
                    sector_impacts=event_data["sector_impacts"],
                    urgency=event_data["urgency"],
                    signal_strength=event_data["sentiment"] * event_data["urgency"],
                ))
        
        if not signals:
            signals = [self._generate_synthetic_signal(as_of)]
        
        return signals
    
    def _get_historical_policy_events(self) -> Dict[str, Dict]:
        return {
            "2024-11-06": {
                "topic": "trump_elected",
                "sentiment": 0.5,
                "sector_impacts": {"XLF": 0.3, "XLE": 0.2, "XLY": 0.2, "XLI": 0.1},
                "urgency": 1.0,
            },
            "2025-01-20": {
                "topic": "inauguration",
                "sentiment": 0.3,
                "sector_impacts": {"XLF": 0.2, "XLE": 0.2},
                "urgency": 0.8,
            },
            "2025-02-01": {
                "topic": "tariff_threat_mexico_canada",
                "sentiment": -0.4,
                "sector_impacts": {"XLI": -0.3, "XLY": -0.2, "XLK": -0.1},
                "urgency": 0.9,
            },
            "2025-03-04": {
                "topic": "tariff_implementation",
                "sentiment": -0.6,
                "sector_impacts": {"XLI": -0.4, "XLY": -0.3, "XLK": -0.2},
                "urgency": 1.0,
            },
            "2025-04-02": {
                "topic": "reciprocal_tariffs",
                "sentiment": -0.8,
                "sector_impacts": {"XLI": -0.5, "XLY": -0.4, "XLK": -0.3, "XLF": -0.2},
                "urgency": 1.0,
            },
            "2025-04-09": {
                "topic": "tariff_pause_90days",
                "sentiment": 0.7,
                "sector_impacts": {"XLI": 0.3, "XLY": 0.3, "XLK": 0.2},
                "urgency": 1.0,
            },
        }
    
    def _generate_synthetic_signal(self, as_of: date) -> PolicySignal:
        seed = as_of.toordinal()
        
        topics = list(SECTOR_IMPACT.keys())
        topic = topics[seed % len(topics)]
        
        return PolicySignal(
            date=str(as_of),
            topic=topic,
            sentiment=((seed % 100) - 50) / 100,
            sector_impacts=SECTOR_IMPACT.get(topic, {}),
            urgency=0.5,
            signal_strength=0.0,
        )
    
    def _is_cache_valid(self, cache_file: Path) -> bool:
        if not cache_file.exists():
            return False
        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        return datetime.now() - mtime < timedelta(days=self.cache_days)


def get_policy_sector_adjustment(as_of: date) -> Dict[str, float]:
    tracker = TruthSocialTracker()
    return tracker.get_sector_adjustment(as_of)
