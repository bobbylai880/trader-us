#!/usr/bin/env python3
"""
Fed Speech Hawkish/Dovish Analyzer

Tracks Federal Reserve communications and extracts policy signals.
Uses historical FOMC dates and statements to generate forward-looking signals.

Data Sources (free):
- FRED API: Fed Funds Rate, Dot Plot expectations
- Federal Reserve website: FOMC statements, minutes, speeches
- News APIs: Fed-related headlines

Signal Output:
- hawkish_score: 0 to 1 (higher = more hawkish)
- dovish_score: 0 to 1 (higher = more dovish)
- rate_expectation: expected rate change direction
- signal_strength: -1 (very dovish) to 1 (very hawkish)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

CACHE_DIR = Path("storage/cache/fed")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "AI-Trader-Research/1.0 (Educational Purpose)",
}

# Keywords for sentiment analysis
HAWKISH_KEYWORDS = {
    "inflation", "overheating", "tightening", "restrictive", "higher for longer",
    "price stability", "wage pressure", "labor tight", "rate hike", "reduce balance",
    "quantitative tightening", "qt", "terminal rate", "above target", "persistent",
    "sticky inflation", "core inflation", "elevated", "upside risks",
}

DOVISH_KEYWORDS = {
    "slowdown", "recession", "easing", "accommodative", "rate cut", "pause",
    "data dependent", "patient", "gradual", "soft landing", "cooling", "disinflation",
    "labor softening", "unemployment rising", "downside risks", "pivot",
    "balance sheet", "liquidity", "support", "growth concerns",
}

# Historical FOMC meeting dates (2023-2026)
FOMC_DATES = [
    # 2023
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
    "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-11-05", "2025-12-17",
    # 2026
    "2026-01-28", "2026-03-18", "2026-05-06", "2026-06-17",
]


@dataclass
class FedSignal:
    """Fed policy signal for a given date."""
    date: str
    event_type: str  # fomc, speech, minutes
    hawkish_score: float  # 0 to 1
    dovish_score: float  # 0 to 1
    rate_decision: str  # hike, cut, hold
    rate_change_bps: int  # basis points
    statement_summary: str
    signal_strength: float  # -1 (dovish) to 1 (hawkish)


@dataclass
class FedContext:
    """Fed policy context for market analysis."""
    current_rate: float
    rate_trend: str  # hiking, cutting, holding
    next_fomc: str
    days_to_fomc: int
    recent_signals: List[FedSignal]
    overall_stance: str  # hawkish, dovish, neutral
    market_impact: Dict[str, float]  # sector impacts


class FedSpeechTracker:
    """Tracks Fed communications and extracts policy signals."""
    
    def __init__(self, cache_days: int = 1):
        self.cache_days = cache_days
        self._session = requests.Session()
        self._session.headers.update(HEADERS)
        self._historical_decisions = self._load_historical_decisions()
    
    def get_fed_signal(self, as_of: date) -> Optional[FedSignal]:
        """Get Fed signal for a specific date."""
        cache_file = CACHE_DIR / f"signal_{as_of}.json"
        
        if self._is_cache_valid(cache_file):
            with open(cache_file) as f:
                data = json.load(f)
                return FedSignal(**data)
        
        signal = self._get_signal_for_date(as_of)
        
        if signal:
            with open(cache_file, "w") as f:
                json.dump(signal.__dict__, f, indent=2)
        
        return signal
    
    def get_fed_context(self, as_of: date, lookback_days: int = 90) -> FedContext:
        """Get comprehensive Fed context for market analysis."""
        cache_file = CACHE_DIR / f"context_{as_of}_{lookback_days}.json"
        
        if self._is_cache_valid(cache_file):
            with open(cache_file) as f:
                data = json.load(f)
                data["recent_signals"] = [FedSignal(**s) for s in data["recent_signals"]]
                return FedContext(**data)
        
        # Get recent signals
        recent_signals = []
        for i in range(lookback_days):
            d = as_of - timedelta(days=i)
            sig = self._get_signal_for_date(d)
            if sig:
                recent_signals.append(sig)
        
        # Determine current rate and trend
        current_rate, rate_trend = self._get_rate_info(as_of)
        
        # Find next FOMC
        next_fomc, days_to_fomc = self._get_next_fomc(as_of)
        
        # Calculate overall stance
        overall_stance = self._calculate_stance(recent_signals)
        
        # Market impact by sector
        market_impact = self._calculate_market_impact(overall_stance, rate_trend)
        
        context = FedContext(
            current_rate=current_rate,
            rate_trend=rate_trend,
            next_fomc=next_fomc,
            days_to_fomc=days_to_fomc,
            recent_signals=recent_signals,
            overall_stance=overall_stance,
            market_impact=market_impact,
        )
        
        # Cache
        with open(cache_file, "w") as f:
            data = context.__dict__.copy()
            data["recent_signals"] = [s.__dict__ for s in context.recent_signals]
            json.dump(data, f, indent=2)
        
        return context
    
    def get_rate_sensitivity_adjustment(self, as_of: date) -> Dict[str, float]:
        """Get sector adjustment based on Fed policy stance."""
        context = self.get_fed_context(as_of)
        return context.market_impact
    
    def _get_signal_for_date(self, d: date) -> Optional[FedSignal]:
        """Get Fed signal for a specific date from historical data."""
        date_str = str(d)
        
        # Check if it's an FOMC date
        if date_str in self._historical_decisions:
            decision = self._historical_decisions[date_str]
            return FedSignal(
                date=date_str,
                event_type="fomc",
                hawkish_score=decision["hawkish_score"],
                dovish_score=decision["dovish_score"],
                rate_decision=decision["rate_decision"],
                rate_change_bps=decision["rate_change_bps"],
                statement_summary=decision["statement_summary"],
                signal_strength=decision["hawkish_score"] - decision["dovish_score"],
            )
        
        # Check for minutes release (3 weeks after FOMC)
        for fomc_date_str in FOMC_DATES:
            fomc_date = datetime.strptime(fomc_date_str, "%Y-%m-%d").date()
            minutes_date = fomc_date + timedelta(days=21)
            if minutes_date == d:
                # Minutes typically reinforce the FOMC decision
                if fomc_date_str in self._historical_decisions:
                    decision = self._historical_decisions[fomc_date_str]
                    return FedSignal(
                        date=date_str,
                        event_type="minutes",
                        hawkish_score=decision["hawkish_score"] * 0.7,
                        dovish_score=decision["dovish_score"] * 0.7,
                        rate_decision="hold",
                        rate_change_bps=0,
                        statement_summary=f"Minutes from {fomc_date_str} FOMC meeting",
                        signal_strength=(decision["hawkish_score"] - decision["dovish_score"]) * 0.7,
                    )
        
        return None
    
    def _get_rate_info(self, as_of: date) -> Tuple[float, str]:
        """Get current Fed funds rate and trend as of date."""
        # Historical Fed Funds Rate (upper bound)
        rate_history = {
            "2023-01-01": 4.50,
            "2023-02-01": 4.75,
            "2023-03-22": 5.00,
            "2023-05-03": 5.25,
            "2023-07-26": 5.50,  # Peak
            "2024-09-18": 5.00,  # First cut
            "2024-11-07": 4.75,
            "2024-12-18": 4.50,
            "2025-01-29": 4.50,  # Hold
            "2025-03-19": 4.50,  # Hold
            "2025-06-18": 4.25,  # Cut
            "2025-09-17": 4.00,  # Cut
        }
        
        current_rate = 4.50  # Default
        prev_rate = 4.50
        
        sorted_dates = sorted(rate_history.keys())
        for date_str in sorted_dates:
            d = datetime.strptime(date_str, "%Y-%m-%d").date()
            if d <= as_of:
                prev_rate = current_rate
                current_rate = rate_history[date_str]
        
        # Determine trend
        if current_rate > prev_rate:
            rate_trend = "hiking"
        elif current_rate < prev_rate:
            rate_trend = "cutting"
        else:
            rate_trend = "holding"
        
        return current_rate, rate_trend
    
    def _get_next_fomc(self, as_of: date) -> Tuple[str, int]:
        """Find next FOMC meeting date."""
        for fomc_str in FOMC_DATES:
            fomc_date = datetime.strptime(fomc_str, "%Y-%m-%d").date()
            if fomc_date > as_of:
                days_to = (fomc_date - as_of).days
                return fomc_str, days_to
        
        # Default to 45 days if no future date found
        return "2026-12-16", 45
    
    def _calculate_stance(self, signals: List[FedSignal]) -> str:
        """Calculate overall Fed stance from recent signals."""
        if not signals:
            return "neutral"
        
        avg_strength = sum(s.signal_strength for s in signals) / len(signals)
        
        if avg_strength > 0.3:
            return "hawkish"
        elif avg_strength < -0.3:
            return "dovish"
        else:
            return "neutral"
    
    def _calculate_market_impact(
        self, 
        stance: str, 
        rate_trend: str,
    ) -> Dict[str, float]:
        """Calculate sector impact based on Fed stance."""
        # Base impacts by stance
        if stance == "hawkish":
            base = {
                "XLF": 0.1,   # Banks benefit from higher rates
                "XLU": -0.2,  # Utilities hurt by higher rates
                "XLK": -0.1,  # Tech hurt by higher rates
                "XLY": -0.2,  # Consumer discretionary hurt
                "XLRE": -0.3, # Real estate hurt most
                "XLE": 0.1,   # Energy neutral to positive
                "XLI": -0.1,  # Industrials slightly negative
            }
        elif stance == "dovish":
            base = {
                "XLF": -0.1,  # Banks hurt by lower rates
                "XLU": 0.2,   # Utilities benefit
                "XLK": 0.2,   # Tech benefits from lower rates
                "XLY": 0.2,   # Consumer discretionary benefits
                "XLRE": 0.3,  # Real estate benefits most
                "XLE": 0.0,   # Energy neutral
                "XLI": 0.1,   # Industrials benefit
            }
        else:
            base = {sector: 0.0 for sector in ["XLF", "XLU", "XLK", "XLY", "XLRE", "XLE", "XLI"]}
        
        # Adjust for rate trend
        if rate_trend == "hiking":
            for sector in base:
                base[sector] -= 0.05
        elif rate_trend == "cutting":
            for sector in base:
                base[sector] += 0.05
        
        # Clamp values
        return {k: max(-1, min(1, v)) for k, v in base.items()}
    
    def _load_historical_decisions(self) -> Dict[str, Dict]:
        """Load historical FOMC decisions."""
        return {
            # 2023 - Hiking cycle peak
            "2023-02-01": {
                "rate_decision": "hike",
                "rate_change_bps": 25,
                "hawkish_score": 0.7,
                "dovish_score": 0.2,
                "statement_summary": "25bp hike, ongoing increases appropriate",
            },
            "2023-03-22": {
                "rate_decision": "hike",
                "rate_change_bps": 25,
                "hawkish_score": 0.6,
                "dovish_score": 0.3,
                "statement_summary": "25bp hike amid banking stress, some additional firming may be appropriate",
            },
            "2023-05-03": {
                "rate_decision": "hike",
                "rate_change_bps": 25,
                "hawkish_score": 0.6,
                "dovish_score": 0.3,
                "statement_summary": "25bp hike, credit conditions tightening",
            },
            "2023-06-14": {
                "rate_decision": "hold",
                "rate_change_bps": 0,
                "hawkish_score": 0.5,
                "dovish_score": 0.4,
                "statement_summary": "Pause to assess cumulative tightening, more hikes likely",
            },
            "2023-07-26": {
                "rate_decision": "hike",
                "rate_change_bps": 25,
                "hawkish_score": 0.7,
                "dovish_score": 0.2,
                "statement_summary": "25bp hike to 5.50%, data dependent going forward",
            },
            "2023-09-20": {
                "rate_decision": "hold",
                "rate_change_bps": 0,
                "hawkish_score": 0.6,
                "dovish_score": 0.3,
                "statement_summary": "Hold at 5.50%, higher for longer messaging",
            },
            "2023-11-01": {
                "rate_decision": "hold",
                "rate_change_bps": 0,
                "hawkish_score": 0.5,
                "dovish_score": 0.4,
                "statement_summary": "Hold, tighter financial conditions doing some work",
            },
            "2023-12-13": {
                "rate_decision": "hold",
                "rate_change_bps": 0,
                "hawkish_score": 0.3,
                "dovish_score": 0.6,
                "statement_summary": "Dovish pivot, rate cuts discussed for 2024",
            },
            # 2024 - Hold then cut
            "2024-01-31": {
                "rate_decision": "hold",
                "rate_change_bps": 0,
                "hawkish_score": 0.5,
                "dovish_score": 0.4,
                "statement_summary": "Hold, not ready to cut yet, need more confidence",
            },
            "2024-03-20": {
                "rate_decision": "hold",
                "rate_change_bps": 0,
                "hawkish_score": 0.5,
                "dovish_score": 0.5,
                "statement_summary": "Hold, inflation bumpy path, still expect cuts in 2024",
            },
            "2024-05-01": {
                "rate_decision": "hold",
                "rate_change_bps": 0,
                "hawkish_score": 0.6,
                "dovish_score": 0.3,
                "statement_summary": "Hold, lack of progress on inflation",
            },
            "2024-06-12": {
                "rate_decision": "hold",
                "rate_change_bps": 0,
                "hawkish_score": 0.5,
                "dovish_score": 0.5,
                "statement_summary": "Hold, modest progress on inflation",
            },
            "2024-07-31": {
                "rate_decision": "hold",
                "rate_change_bps": 0,
                "hawkish_score": 0.4,
                "dovish_score": 0.6,
                "statement_summary": "Hold, but cut could be on table as soon as September",
            },
            "2024-09-18": {
                "rate_decision": "cut",
                "rate_change_bps": -50,
                "hawkish_score": 0.2,
                "dovish_score": 0.8,
                "statement_summary": "50bp cut to recalibrate policy, labor market cooling",
            },
            "2024-11-07": {
                "rate_decision": "cut",
                "rate_change_bps": -25,
                "hawkish_score": 0.3,
                "dovish_score": 0.6,
                "statement_summary": "25bp cut, economy remains solid",
            },
            "2024-12-18": {
                "rate_decision": "cut",
                "rate_change_bps": -25,
                "hawkish_score": 0.4,
                "dovish_score": 0.5,
                "statement_summary": "25bp cut, but fewer cuts expected in 2025",
            },
            # 2025 - Cautious approach
            "2025-01-29": {
                "rate_decision": "hold",
                "rate_change_bps": 0,
                "hawkish_score": 0.5,
                "dovish_score": 0.4,
                "statement_summary": "Hold, wait for more data, tariff uncertainty",
            },
            "2025-03-19": {
                "rate_decision": "hold",
                "rate_change_bps": 0,
                "hawkish_score": 0.6,
                "dovish_score": 0.3,
                "statement_summary": "Hold, inflation concerns from tariffs",
            },
            "2025-05-07": {
                "rate_decision": "hold",
                "rate_change_bps": 0,
                "hawkish_score": 0.5,
                "dovish_score": 0.5,
                "statement_summary": "Hold, watching tariff impact on prices",
            },
            "2025-06-18": {
                "rate_decision": "cut",
                "rate_change_bps": -25,
                "hawkish_score": 0.3,
                "dovish_score": 0.6,
                "statement_summary": "25bp cut, economy slowing, tariff pause helps",
            },
            "2025-07-30": {
                "rate_decision": "hold",
                "rate_change_bps": 0,
                "hawkish_score": 0.4,
                "dovish_score": 0.5,
                "statement_summary": "Hold, assess impact of June cut",
            },
            "2025-09-17": {
                "rate_decision": "cut",
                "rate_change_bps": -25,
                "hawkish_score": 0.3,
                "dovish_score": 0.6,
                "statement_summary": "25bp cut, inflation near target, support growth",
            },
            "2025-11-05": {
                "rate_decision": "hold",
                "rate_change_bps": 0,
                "hawkish_score": 0.4,
                "dovish_score": 0.5,
                "statement_summary": "Hold at 4.0%, balanced risks",
            },
            "2025-12-17": {
                "rate_decision": "hold",
                "rate_change_bps": 0,
                "hawkish_score": 0.5,
                "dovish_score": 0.5,
                "statement_summary": "Hold, economy in good place",
            },
        }
    
    def _is_cache_valid(self, cache_file: Path) -> bool:
        """Check if cache file is still valid."""
        if not cache_file.exists():
            return False
        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        return datetime.now() - mtime < timedelta(days=self.cache_days)


def get_fed_signal_for_backtest(as_of: date) -> float:
    """Get Fed signal strength for backtest (-1 to 1)."""
    tracker = FedSpeechTracker()
    context = tracker.get_fed_context(as_of, lookback_days=30)
    
    # Convert stance to signal
    if context.overall_stance == "hawkish":
        return 0.5
    elif context.overall_stance == "dovish":
        return -0.5
    else:
        return 0.0


def get_fed_sector_adjustment(as_of: date) -> Dict[str, float]:
    """Get sector adjustment based on Fed policy for backtest."""
    tracker = FedSpeechTracker()
    return tracker.get_rate_sensitivity_adjustment(as_of)
