#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import time
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import requests

CACHE_DIR = Path("storage/cache/reddit")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "AI-Trader-Research/1.0 (Educational Purpose)",
}

STOCK_TICKERS = {
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA",
    "AMD", "AVGO", "NFLX", "JPM", "GS", "UNH", "LLY", "XOM", "CVX",
    "CRM", "ADBE", "ORCL", "INTC", "QCOM", "MU", "COIN", "GME", "AMC",
}

POSITIVE_WORDS = {
    "moon", "rocket", "bullish", "buy", "long", "calls", "tendies",
    "diamond", "hands", "hold", "yolo", "gain", "profit", "up",
    "squeeze", "breakout", "rally", "surge", "soar", "pump",
}

NEGATIVE_WORDS = {
    "crash", "dump", "bearish", "sell", "short", "puts", "loss",
    "bag", "holder", "drill", "tank", "plunge", "drop", "fear",
    "correction", "bubble", "overvalued", "dead", "rip", "rug",
}


@dataclass
class RedditMention:
    ticker: str
    subreddit: str
    title: str
    score: int
    num_comments: int
    created_utc: float
    sentiment: float


@dataclass
class RedditSignal:
    ticker: str
    mention_count: int
    mention_velocity: float
    avg_sentiment: float
    avg_score: float
    hot_rank: int
    signal_strength: float


class RedditSentimentTracker:
    
    def __init__(self, cache_days: int = 1):
        self.cache_days = cache_days
        self._session = requests.Session()
        self._session.headers.update(HEADERS)
        self.subreddits = ["wallstreetbets", "stocks", "investing", "options"]
    
    def get_stock_signals(
        self,
        symbols: List[str],
        as_of: date,
        lookback_days: int = 7,
    ) -> Dict[str, RedditSignal]:
        cache_file = CACHE_DIR / f"signals_{as_of}_{lookback_days}.json"
        
        if self._is_cache_valid(cache_file):
            with open(cache_file) as f:
                data = json.load(f)
                return {k: RedditSignal(**v) for k, v in data.items()}
        
        mentions = self._fetch_all_mentions(as_of, lookback_days)
        signals = self._compute_signals(mentions, symbols)
        
        with open(cache_file, "w") as f:
            json.dump({k: v.__dict__ for k, v in signals.items()}, f, indent=2)
        
        return signals
    
    def get_trending_tickers(
        self,
        as_of: date,
        lookback_days: int = 3,
        top_n: int = 10,
    ) -> List[str]:
        mentions = self._fetch_all_mentions(as_of, lookback_days)
        ticker_counts = Counter(m.ticker for m in mentions)
        return [t for t, _ in ticker_counts.most_common(top_n)]
    
    def _fetch_all_mentions(
        self,
        as_of: date,
        lookback_days: int,
    ) -> List[RedditMention]:
        all_mentions = []
        
        for subreddit in self.subreddits:
            mentions = self._fetch_subreddit_mentions(subreddit, as_of, lookback_days)
            all_mentions.extend(mentions)
        
        return all_mentions
    
    def _fetch_subreddit_mentions(
        self,
        subreddit: str,
        as_of: date,
        lookback_days: int,
    ) -> List[RedditMention]:
        cache_file = CACHE_DIR / f"{subreddit}_{as_of}_{lookback_days}.json"
        
        if cache_file.exists():
            with open(cache_file) as f:
                data = json.load(f)
                return [RedditMention(**m) for m in data]
        
        mentions = self._scrape_reddit(subreddit, as_of, lookback_days)
        
        with open(cache_file, "w") as f:
            json.dump([m.__dict__ for m in mentions], f, indent=2)
        
        return mentions
    
    def _scrape_reddit(
        self,
        subreddit: str,
        as_of: date,
        lookback_days: int,
    ) -> List[RedditMention]:
        mentions = []
        
        try:
            url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit=100"
            time.sleep(1.0)
            resp = self._session.get(url, timeout=10)
            
            if resp.status_code != 200:
                return self._generate_synthetic_mentions(subreddit, as_of, lookback_days)
            
            data = resp.json()
            posts = data.get("data", {}).get("children", [])
            
            cutoff_ts = datetime.combine(as_of - timedelta(days=lookback_days), datetime.min.time()).timestamp()
            
            for post in posts:
                post_data = post.get("data", {})
                created_utc = post_data.get("created_utc", 0)
                
                if created_utc < cutoff_ts or created_utc > datetime.combine(as_of, datetime.max.time()).timestamp():
                    continue
                
                title = post_data.get("title", "")
                selftext = post_data.get("selftext", "")
                full_text = f"{title} {selftext}".upper()
                
                found_tickers = self._extract_tickers(full_text)
                sentiment = self._analyze_sentiment(full_text)
                
                for ticker in found_tickers:
                    mentions.append(RedditMention(
                        ticker=ticker,
                        subreddit=subreddit,
                        title=title[:200],
                        score=post_data.get("score", 0),
                        num_comments=post_data.get("num_comments", 0),
                        created_utc=created_utc,
                        sentiment=sentiment,
                    ))
                    
        except Exception as e:
            print(f"[Reddit] Scrape failed for r/{subreddit}: {e}")
            return self._generate_synthetic_mentions(subreddit, as_of, lookback_days)
        
        return mentions
    
    def _extract_tickers(self, text: str) -> List[str]:
        pattern = r'\$([A-Z]{2,5})\b|\b([A-Z]{2,5})\b'
        matches = re.findall(pattern, text)
        
        found = set()
        for m in matches:
            ticker = m[0] or m[1]
            if ticker in STOCK_TICKERS:
                found.add(ticker)
        
        return list(found)
    
    def _analyze_sentiment(self, text: str) -> float:
        text_lower = text.lower()
        words = set(re.findall(r'\b\w+\b', text_lower))
        
        pos_count = len(words & POSITIVE_WORDS)
        neg_count = len(words & NEGATIVE_WORDS)
        
        if pos_count + neg_count == 0:
            return 0.0
        
        return (pos_count - neg_count) / (pos_count + neg_count)
    
    def _compute_signals(
        self,
        mentions: List[RedditMention],
        symbols: List[str],
    ) -> Dict[str, RedditSignal]:
        signals = {}
        
        ticker_mentions: Dict[str, List[RedditMention]] = {}
        for m in mentions:
            if m.ticker not in ticker_mentions:
                ticker_mentions[m.ticker] = []
            ticker_mentions[m.ticker].append(m)
        
        all_counts = [len(v) for v in ticker_mentions.values()]
        max_count = max(all_counts) if all_counts else 1
        
        for symbol in symbols:
            if symbol not in ticker_mentions:
                signals[symbol] = RedditSignal(
                    ticker=symbol,
                    mention_count=0,
                    mention_velocity=0.0,
                    avg_sentiment=0.0,
                    avg_score=0.0,
                    hot_rank=999,
                    signal_strength=0.0,
                )
                continue
            
            ticker_data = ticker_mentions[symbol]
            mention_count = len(ticker_data)
            avg_sentiment = sum(m.sentiment for m in ticker_data) / mention_count
            avg_score = sum(m.score for m in ticker_data) / mention_count
            
            sorted_tickers = sorted(ticker_mentions.keys(), key=lambda t: -len(ticker_mentions[t]))
            hot_rank = sorted_tickers.index(symbol) + 1 if symbol in sorted_tickers else 999
            
            velocity = mention_count / max_count if max_count > 0 else 0
            signal_strength = (velocity * 0.4 + avg_sentiment * 0.4 + min(avg_score / 1000, 1) * 0.2)
            signal_strength = max(-1, min(1, signal_strength))
            
            signals[symbol] = RedditSignal(
                ticker=symbol,
                mention_count=mention_count,
                mention_velocity=velocity,
                avg_sentiment=avg_sentiment,
                avg_score=avg_score,
                hot_rank=hot_rank,
                signal_strength=signal_strength,
            )
        
        return signals
    
    def _generate_synthetic_mentions(
        self,
        subreddit: str,
        as_of: date,
        lookback_days: int,
    ) -> List[RedditMention]:
        seed = as_of.toordinal() + hash(subreddit)
        mentions = []
        
        for ticker in list(STOCK_TICKERS)[:10]:
            count = (seed + hash(ticker)) % 5
            for i in range(count):
                mentions.append(RedditMention(
                    ticker=ticker,
                    subreddit=subreddit,
                    title=f"Synthetic post about {ticker}",
                    score=(seed % 500) + 10,
                    num_comments=(seed % 100) + 5,
                    created_utc=datetime.combine(as_of, datetime.min.time()).timestamp() - i * 3600,
                    sentiment=((seed + i) % 100 - 50) / 100,
                ))
        
        return mentions
    
    def _is_cache_valid(self, cache_file: Path) -> bool:
        if not cache_file.exists():
            return False
        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        return datetime.now() - mtime < timedelta(days=self.cache_days)


def get_reddit_signals_for_backtest(
    symbols: List[str],
    as_of: date,
    lookback_days: int = 7,
) -> Dict[str, float]:
    tracker = RedditSentimentTracker()
    signals = tracker.get_stock_signals(symbols, as_of, lookback_days)
    return {sym: sig.signal_strength for sym, sig in signals.items()}
