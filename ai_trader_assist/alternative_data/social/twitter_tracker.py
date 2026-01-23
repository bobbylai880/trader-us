#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import requests

CACHE_DIR = Path("storage/cache/twitter")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "AI-Trader-Research/1.0 (Educational Purpose)",
}

FINTWIT_ACCOUNTS = [
    "zaborsky", "unusual_whales", "DeItaone", "FirstSquawk",
    "Fxhedgers", "LiveSquawk", "SquawkCNBC", "markets",
]

BULLISH_KEYWORDS = {
    "bullish", "buy", "long", "calls", "breakout", "rally",
    "upgrade", "beat", "strong", "surge", "moon", "higher",
}

BEARISH_KEYWORDS = {
    "bearish", "sell", "short", "puts", "breakdown", "crash",
    "downgrade", "miss", "weak", "plunge", "tank", "lower",
}


@dataclass
class TweetSignal:
    ticker: str
    tweet_count: int
    avg_sentiment: float
    influencer_mentions: int
    breaking_news_count: int
    signal_strength: float


class TwitterSentimentTracker:
    
    def __init__(self, cache_days: int = 1):
        self.cache_days = cache_days
        self._session = requests.Session()
        self._session.headers.update(HEADERS)
    
    def get_stock_signals(
        self,
        symbols: List[str],
        as_of: date,
        lookback_hours: int = 24,
    ) -> Dict[str, TweetSignal]:
        cache_file = CACHE_DIR / f"signals_{as_of}_{lookback_hours}.json"
        
        if self._is_cache_valid(cache_file):
            with open(cache_file) as f:
                data = json.load(f)
                return {k: TweetSignal(**v) for k, v in data.items()}
        
        signals = {}
        for symbol in symbols:
            signals[symbol] = self._generate_synthetic_signal(symbol, as_of)
        
        with open(cache_file, "w") as f:
            json.dump({k: v.__dict__ for k, v in signals.items()}, f, indent=2)
        
        return signals
    
    def get_breaking_news(
        self,
        as_of: date,
        lookback_hours: int = 6,
    ) -> List[Dict]:
        return self._generate_synthetic_news(as_of)
    
    def _generate_synthetic_signal(
        self,
        symbol: str,
        as_of: date,
    ) -> TweetSignal:
        seed = as_of.toordinal() + hash(symbol)
        
        tweet_count = (seed % 50) + 5
        avg_sentiment = ((seed % 100) - 50) / 100
        influencer_mentions = seed % 5
        breaking_news = seed % 3
        
        signal_strength = avg_sentiment * 0.5 + (influencer_mentions / 5) * 0.3 + (breaking_news / 3) * 0.2
        signal_strength = max(-1, min(1, signal_strength))
        
        return TweetSignal(
            ticker=symbol,
            tweet_count=tweet_count,
            avg_sentiment=avg_sentiment,
            influencer_mentions=influencer_mentions,
            breaking_news_count=breaking_news,
            signal_strength=signal_strength,
        )
    
    def _generate_synthetic_news(self, as_of: date) -> List[Dict]:
        seed = as_of.toordinal()
        news_items = []
        
        tickers = ["NVDA", "TSLA", "META", "AAPL", "MSFT"]
        for i, ticker in enumerate(tickers[:seed % 5]):
            news_items.append({
                "ticker": ticker,
                "headline": f"Breaking: {ticker} news on {as_of}",
                "sentiment": ((seed + i) % 100 - 50) / 100,
                "source": FINTWIT_ACCOUNTS[i % len(FINTWIT_ACCOUNTS)],
            })
        
        return news_items
    
    def _is_cache_valid(self, cache_file: Path) -> bool:
        if not cache_file.exists():
            return False
        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        return datetime.now() - mtime < timedelta(days=self.cache_days)


def get_twitter_signals_for_backtest(
    symbols: List[str],
    as_of: date,
) -> Dict[str, float]:
    tracker = TwitterSentimentTracker()
    signals = tracker.get_stock_signals(symbols, as_of)
    return {sym: sig.signal_strength for sym, sig in signals.items()}
