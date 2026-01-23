#!/usr/bin/env python3
"""SEC Form 4 Insider Trading Tracker - 免费数据源"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urljoin

import requests

CACHE_DIR = Path("storage/cache/insider")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SEC_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
SEC_EDGAR_BASE = "https://www.sec.gov"
OPENINSIDER_URL = "https://openinsider.com/screener"

HEADERS = {
    "User-Agent": "AI-Trader-Research/1.0 (Educational Purpose)",
    "Accept": "application/json",
}


@dataclass
class InsiderTransaction:
    filing_date: str
    trade_date: str
    ticker: str
    insider_name: str
    title: str
    transaction_type: str  # P=Purchase, S=Sale, A=Award
    shares: int
    price: float
    value: float
    ownership_type: str  # D=Direct, I=Indirect


@dataclass 
class InsiderSignal:
    ticker: str
    cluster_buy_score: float  # 0-1, 多位高管同时买入
    ceo_buy: bool
    total_buy_value: float
    total_sell_value: float
    net_value: float
    transaction_count: int
    last_transaction_date: str
    signal_strength: float  # -1 to 1, 负=卖出信号, 正=买入信号


class InsiderTracker:
    
    def __init__(self, cache_days: int = 1):
        self.cache_days = cache_days
        self._session = requests.Session()
        self._session.headers.update(HEADERS)
    
    def get_insider_signals(
        self,
        symbols: List[str],
        as_of: date,
        lookback_days: int = 90,
    ) -> Dict[str, InsiderSignal]:
        signals = {}
        for symbol in symbols:
            txns = self._get_transactions(symbol, as_of, lookback_days)
            if txns:
                signals[symbol] = self._compute_signal(symbol, txns)
        return signals
    
    def get_cluster_buys(
        self,
        as_of: date,
        lookback_days: int = 30,
        min_insiders: int = 2,
    ) -> List[str]:
        cache_file = CACHE_DIR / f"cluster_buys_{as_of}_{lookback_days}.json"
        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f)
        
        clusters = self._scrape_openinsider_clusters(as_of, lookback_days, min_insiders)
        
        with open(cache_file, "w") as f:
            json.dump(clusters, f)
        
        return clusters
    
    def _get_transactions(
        self,
        symbol: str,
        as_of: date,
        lookback_days: int,
    ) -> List[InsiderTransaction]:
        cache_file = CACHE_DIR / f"{symbol}_{as_of}_{lookback_days}.json"
        
        if self._is_cache_valid(cache_file):
            with open(cache_file) as f:
                data = json.load(f)
                return [InsiderTransaction(**t) for t in data]
        
        txns = self._fetch_from_openinsider(symbol, as_of, lookback_days)
        
        with open(cache_file, "w") as f:
            json.dump([t.__dict__ for t in txns], f, indent=2)
        
        return txns
    
    def _fetch_from_openinsider(
        self,
        symbol: str,
        as_of: date,
        lookback_days: int,
    ) -> List[InsiderTransaction]:
        start_date = as_of - timedelta(days=lookback_days)
        
        url = f"https://openinsider.com/screener?s={symbol}&o=&pl=&ph=&ll=&lh=&fd={lookback_days}&fdr=&td=0&tdr=&feession=&feession=&session=&sic1=-1&sicl=100&sich=9999&grp=0&nfl=&nfh=&nil=&nih=&nol=&noh=&v2l=&v2h=&oc2l=&oc2h=&sortcol=0&cnt=100&page=1"
        
        try:
            time.sleep(0.5)  # Rate limit
            resp = self._session.get(url, timeout=10)
            resp.raise_for_status()
            return self._parse_openinsider_html(resp.text, symbol, as_of)
        except Exception as e:
            print(f"[InsiderTracker] Failed to fetch {symbol}: {e}")
            return []
    
    def _parse_openinsider_html(
        self, 
        html: str, 
        symbol: str,
        as_of: date,
    ) -> List[InsiderTransaction]:
        txns = []
        
        row_pattern = re.compile(
            r'<tr[^>]*>.*?'
            r'<td[^>]*>.*?(\d{4}-\d{2}-\d{2}).*?</td>.*?'  # filing date
            r'<td[^>]*>.*?(\d{4}-\d{2}-\d{2}).*?</td>.*?'  # trade date
            r'<td[^>]*>.*?<a[^>]*>([A-Z]+)</a>.*?</td>.*?'  # ticker
            r'<td[^>]*>(.*?)</td>.*?'  # insider name
            r'<td[^>]*>(.*?)</td>.*?'  # title
            r'<td[^>]*>.*?([PSA]).*?</td>.*?'  # transaction type
            r'<td[^>]*>.*?([\d,]+).*?</td>.*?'  # shares
            r'<td[^>]*>.*?\$([\d,.]+).*?</td>.*?'  # price
            r'<td[^>]*>.*?\$([\d,]+).*?</td>',  # value
            re.DOTALL | re.IGNORECASE
        )
        
        for match in row_pattern.finditer(html):
            try:
                filing_date = match.group(1)
                trade_date = match.group(2)
                
                if datetime.strptime(filing_date, "%Y-%m-%d").date() > as_of:
                    continue
                
                txn = InsiderTransaction(
                    filing_date=filing_date,
                    trade_date=trade_date,
                    ticker=match.group(3).upper(),
                    insider_name=self._clean_html(match.group(4)),
                    title=self._clean_html(match.group(5)),
                    transaction_type=match.group(6).upper(),
                    shares=int(match.group(7).replace(",", "")),
                    price=float(match.group(8).replace(",", "")),
                    value=float(match.group(9).replace(",", "")),
                    ownership_type="D",
                )
                txns.append(txn)
            except (ValueError, IndexError):
                continue
        
        return txns
    
    def _scrape_openinsider_clusters(
        self,
        as_of: date,
        lookback_days: int,
        min_insiders: int,
    ) -> List[str]:
        url = f"https://openinsider.com/screener?s=&o=&pl=&ph=&ll=&lh=&fd={lookback_days}&fdr=&td=0&tdr=&feession=&sic1=-1&sicl=100&sich=9999&grp=0&nfl=&nfh=&nil=&nih=&nol=&noh=&v2l=&v2h=&oc2l=&oc2h=&sortcol=0&cnt=500&page=1"
        
        try:
            time.sleep(0.5)
            resp = self._session.get(url, timeout=15)
            resp.raise_for_status()
            
            ticker_buys: Dict[str, set] = {}
            
            pattern = re.compile(
                r'<a[^>]*ticker[^>]*>([A-Z]+)</a>.*?'
                r'([PSA])\s*-\s*(Purchase|Sale)',
                re.DOTALL | re.IGNORECASE
            )
            
            for match in pattern.finditer(resp.text):
                ticker = match.group(1).upper()
                txn_type = match.group(2).upper()
                if txn_type == "P":
                    if ticker not in ticker_buys:
                        ticker_buys[ticker] = set()
                    ticker_buys[ticker].add("insider")
            
            clusters = [t for t, insiders in ticker_buys.items() if len(insiders) >= min_insiders]
            return clusters
            
        except Exception as e:
            print(f"[InsiderTracker] Cluster scrape failed: {e}")
            return []
    
    def _compute_signal(
        self,
        symbol: str,
        txns: List[InsiderTransaction],
    ) -> InsiderSignal:
        buy_value = sum(t.value for t in txns if t.transaction_type == "P")
        sell_value = sum(t.value for t in txns if t.transaction_type == "S")
        net_value = buy_value - sell_value
        
        buyers = set(t.insider_name for t in txns if t.transaction_type == "P")
        cluster_score = min(len(buyers) / 3.0, 1.0)
        
        ceo_buy = any(
            t.transaction_type == "P" and "CEO" in t.title.upper()
            for t in txns
        )
        
        if buy_value + sell_value > 0:
            signal_strength = (buy_value - sell_value) / (buy_value + sell_value)
        else:
            signal_strength = 0.0
        
        if ceo_buy:
            signal_strength = min(signal_strength + 0.3, 1.0)
        
        last_date = max(t.filing_date for t in txns) if txns else ""
        
        return InsiderSignal(
            ticker=symbol,
            cluster_buy_score=cluster_score,
            ceo_buy=ceo_buy,
            total_buy_value=buy_value,
            total_sell_value=sell_value,
            net_value=net_value,
            transaction_count=len(txns),
            last_transaction_date=last_date,
            signal_strength=signal_strength,
        )
    
    def _is_cache_valid(self, cache_file: Path) -> bool:
        if not cache_file.exists():
            return False
        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        return datetime.now() - mtime < timedelta(days=self.cache_days)
    
    def _clean_html(self, text: str) -> str:
        clean = re.sub(r'<[^>]+>', '', text)
        return clean.strip()


def get_insider_signals_for_backtest(
    symbols: List[str],
    as_of: date,
    lookback_days: int = 90,
) -> Dict[str, float]:
    tracker = InsiderTracker()
    signals = tracker.get_insider_signals(symbols, as_of, lookback_days)
    return {sym: sig.signal_strength for sym, sig in signals.items()}
