#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set

from .insider.sec_form4 import InsiderTracker, get_insider_signals_for_backtest
from .options.pcr_tracker import OptionsPCRTracker, get_pcr_signal_for_backtest
from .social.reddit_tracker import RedditSentimentTracker, get_reddit_signals_for_backtest
from .social.twitter_tracker import TwitterSentimentTracker, get_twitter_signals_for_backtest
from .social.truth_tracker import TruthSocialTracker, get_policy_sector_adjustment
from .fed.hawkish_dovish import FedSpeechTracker, get_fed_sector_adjustment
from .analyst.ratings_tracker import AnalystRatingsTracker, get_analyst_signal_for_backtest

CACHE_DIR = Path("storage/cache/theme")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"]

SECTOR_STOCKS = {
    "XLK": ["NVDA", "AAPL", "MSFT", "GOOGL", "META", "AMD", "AVGO", "ORCL", "CRM", "ADBE"],
    "XLF": ["JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "USB"],
    "XLE": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "HAL"],
    "XLV": ["UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY"],
    "XLI": ["CAT", "HON", "UNP", "BA", "RTX", "DE", "LMT", "GE", "MMM", "UPS"],
    "XLY": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "TJX", "BKNG", "CMG"],
    "XLP": ["PG", "KO", "PEP", "COST", "WMT", "PM", "MDLZ", "CL", "MO", "GIS"],
    "XLU": ["NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC", "XEL", "ED", "WEC"],
}


@dataclass
class ThemeConfig:
    focus_sectors: List[str] = field(default_factory=list)
    focus_stocks: List[str] = field(default_factory=list)
    avoid_sectors: List[str] = field(default_factory=list)
    avoid_stocks: List[str] = field(default_factory=list)
    sector_bonus: Dict[str, float] = field(default_factory=dict)
    stock_bonus: Dict[str, float] = field(default_factory=dict)
    risk_level: str = "normal"
    theme_drivers: List[str] = field(default_factory=list)


@dataclass
class DataSourceSignals:
    insider_signals: Dict[str, float] = field(default_factory=dict)
    pcr_signal: float = 0.0
    reddit_signals: Dict[str, float] = field(default_factory=dict)
    twitter_signals: Dict[str, float] = field(default_factory=dict)
    policy_sector_adj: Dict[str, float] = field(default_factory=dict)
    fed_sector_adj: Dict[str, float] = field(default_factory=dict)
    analyst_signals: Dict[str, float] = field(default_factory=dict)


class ForwardThemeGenerator:
    
    WEIGHTS = {
        "momentum": 0.30,
        "insider": 0.20,
        "analyst": 0.15,
        "options": 0.10,
        "social": 0.10,
        "policy": 0.10,
        "fed": 0.05,
    }
    
    def __init__(
        self,
        universe: Optional[List[str]] = None,
        cache_days: int = 1,
    ):
        self.universe = universe or [
            "NVDA", "META", "GOOGL", "AMZN", "MSFT", "AAPL", 
            "AMD", "AVGO", "NFLX", "TSLA",
        ]
        self.cache_days = cache_days
        
        self._insider_tracker = InsiderTracker()
        self._pcr_tracker = OptionsPCRTracker()
        self._reddit_tracker = RedditSentimentTracker()
        self._twitter_tracker = TwitterSentimentTracker()
        self._truth_tracker = TruthSocialTracker()
        self._fed_tracker = FedSpeechTracker()
        self._analyst_tracker = AnalystRatingsTracker()
    
    def generate_theme(
        self,
        as_of: date,
        momentum_scores: Optional[Dict[str, float]] = None,
    ) -> ThemeConfig:
        cache_file = CACHE_DIR / f"theme_{as_of}.json"
        
        if self._is_cache_valid(cache_file):
            with open(cache_file) as f:
                data = json.load(f)
                return ThemeConfig(**data)
        
        signals = self._collect_all_signals(as_of)
        
        if momentum_scores is None:
            momentum_scores = self._generate_synthetic_momentum(as_of)
        
        stock_scores = self._compute_stock_scores(signals, momentum_scores)
        sector_scores = self._compute_sector_scores(signals, stock_scores)
        
        theme = self._build_theme_config(stock_scores, sector_scores, signals)
        
        with open(cache_file, "w") as f:
            json.dump(theme.__dict__, f, indent=2)
        
        return theme
    
    def _collect_all_signals(self, as_of: date) -> DataSourceSignals:
        insider_signals = get_insider_signals_for_backtest(self.universe, as_of)
        
        pcr_signal = get_pcr_signal_for_backtest(as_of)
        reddit_signals = get_reddit_signals_for_backtest(self.universe, as_of)
        twitter_signals = get_twitter_signals_for_backtest(self.universe, as_of)
        policy_sector_adj = get_policy_sector_adjustment(as_of)
        fed_sector_adj = get_fed_sector_adjustment(as_of)
        analyst_signals = get_analyst_signal_for_backtest(self.universe, as_of)
        
        return DataSourceSignals(
            insider_signals=insider_signals,
            pcr_signal=pcr_signal,
            reddit_signals=reddit_signals,
            twitter_signals=twitter_signals,
            policy_sector_adj=policy_sector_adj,
            fed_sector_adj=fed_sector_adj,
            analyst_signals=analyst_signals,
        )
    
    def _compute_stock_scores(
        self,
        signals: DataSourceSignals,
        momentum_scores: Dict[str, float],
    ) -> Dict[str, float]:
        scores = {}
        
        for symbol in self.universe:
            momentum = momentum_scores.get(symbol, 0.0)
            insider = signals.insider_signals.get(symbol, 0.0)
            analyst = signals.analyst_signals.get(symbol, 0.0)
            reddit = signals.reddit_signals.get(symbol, 0.0)
            twitter = signals.twitter_signals.get(symbol, 0.0)
            
            social = (reddit + twitter) / 2
            
            score = (
                momentum * self.WEIGHTS["momentum"] +
                insider * self.WEIGHTS["insider"] +
                analyst * self.WEIGHTS["analyst"] +
                signals.pcr_signal * self.WEIGHTS["options"] +
                social * self.WEIGHTS["social"]
            )
            
            scores[symbol] = max(-1, min(1, score))
        
        return scores
    
    def _compute_sector_scores(
        self,
        signals: DataSourceSignals,
        stock_scores: Dict[str, float],
    ) -> Dict[str, float]:
        sector_scores = {}
        
        for sector in SECTOR_ETFS:
            policy_adj = signals.policy_sector_adj.get(sector, 0.0)
            fed_adj = signals.fed_sector_adj.get(sector, 0.0)
            
            sector_stocks = SECTOR_STOCKS.get(sector, [])
            stock_score_sum = sum(
                stock_scores.get(s, 0.0) for s in sector_stocks if s in stock_scores
            )
            stock_count = sum(1 for s in sector_stocks if s in stock_scores)
            avg_stock_score = stock_score_sum / stock_count if stock_count > 0 else 0.0
            
            score = (
                avg_stock_score * 0.5 +
                policy_adj * self.WEIGHTS["policy"] +
                fed_adj * self.WEIGHTS["fed"]
            )
            
            sector_scores[sector] = max(-1, min(1, score))
        
        return sector_scores
    
    def _build_theme_config(
        self,
        stock_scores: Dict[str, float],
        sector_scores: Dict[str, float],
        signals: DataSourceSignals,
    ) -> ThemeConfig:
        sorted_stocks = sorted(stock_scores.items(), key=lambda x: -x[1])
        sorted_sectors = sorted(sector_scores.items(), key=lambda x: -x[1])
        
        focus_stocks = [s for s, score in sorted_stocks if score > 0.2][:5]
        avoid_stocks = [s for s, score in sorted_stocks if score < -0.2]
        
        focus_sectors = [s for s, score in sorted_sectors if score > 0.15][:3]
        avoid_sectors = [s for s, score in sorted_sectors if score < -0.15]
        
        stock_bonus = {s: round(score * 0.1, 3) for s, score in stock_scores.items() if abs(score) > 0.1}
        sector_bonus = {s: round(score * 0.1, 3) for s, score in sector_scores.items() if abs(score) > 0.1}
        
        if signals.pcr_signal > 0.5:
            risk_level = "low"
        elif signals.pcr_signal < -0.5:
            risk_level = "high"
        else:
            risk_level = "normal"
        
        theme_drivers = self._identify_theme_drivers(signals, stock_scores, sector_scores)
        
        return ThemeConfig(
            focus_sectors=focus_sectors,
            focus_stocks=focus_stocks,
            avoid_sectors=avoid_sectors,
            avoid_stocks=avoid_stocks,
            sector_bonus=sector_bonus,
            stock_bonus=stock_bonus,
            risk_level=risk_level,
            theme_drivers=theme_drivers,
        )
    
    def _identify_theme_drivers(
        self,
        signals: DataSourceSignals,
        stock_scores: Dict[str, float],
        sector_scores: Dict[str, float],
    ) -> List[str]:
        drivers = []
        
        strong_insider = [s for s, v in signals.insider_signals.items() if v > 0.5]
        if strong_insider:
            drivers.append(f"Insider buying: {', '.join(strong_insider[:3])}")
        
        if signals.pcr_signal > 0.5:
            drivers.append("High PCR - contrarian bullish signal")
        elif signals.pcr_signal < -0.5:
            drivers.append("Low PCR - excessive optimism warning")
        
        policy_bullish = [s for s, v in signals.policy_sector_adj.items() if v > 0.2]
        policy_bearish = [s for s, v in signals.policy_sector_adj.items() if v < -0.2]
        if policy_bullish:
            drivers.append(f"Policy tailwind: {', '.join(policy_bullish)}")
        if policy_bearish:
            drivers.append(f"Policy headwind: {', '.join(policy_bearish)}")
        
        fed_impact = [s for s, v in signals.fed_sector_adj.items() if abs(v) > 0.15]
        if fed_impact:
            drivers.append(f"Fed sensitivity: {', '.join(fed_impact[:3])}")
        
        strong_analyst = [s for s, v in signals.analyst_signals.items() if v > 0.3]
        if strong_analyst:
            drivers.append(f"Analyst upgrades: {', '.join(strong_analyst[:3])}")
        
        return drivers[:5]
    
    def _generate_synthetic_momentum(self, as_of: date) -> Dict[str, float]:
        seed = as_of.toordinal()
        momentum = {}
        
        for i, symbol in enumerate(self.universe):
            score = ((seed + hash(symbol)) % 100 - 50) / 100
            momentum[symbol] = score
        
        return momentum
    
    def _is_cache_valid(self, cache_file: Path) -> bool:
        if not cache_file.exists():
            return False
        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        return datetime.now() - mtime < timedelta(days=self.cache_days)


def generate_forward_theme(
    as_of: date,
    universe: Optional[List[str]] = None,
    momentum_scores: Optional[Dict[str, float]] = None,
) -> ThemeConfig:
    generator = ForwardThemeGenerator(universe=universe)
    return generator.generate_theme(as_of, momentum_scores)
