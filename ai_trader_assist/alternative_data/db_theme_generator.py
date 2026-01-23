#!/usr/bin/env python3
"""
V8.1 回测专用的数据库数据源

从 PostgreSQL 读取 Alternative Data，避免网络请求，加速回测。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional

import psycopg2


def get_connection():
    return psycopg2.connect(
        host=os.getenv("PG_HOST", "192.168.10.11"),
        port=os.getenv("PG_PORT", "5432"),
        database=os.getenv("PG_DATABASE", "trader"),
        user=os.getenv("PG_USER", "trader"),
        password=os.getenv("PG_PASSWORD", "")
    )


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


class DatabaseThemeGenerator:
    
    WEIGHTS = {
        "momentum": 0.30,
        "insider": 0.20,
        "analyst": 0.15,
        "options": 0.10,
        "social": 0.10,
        "policy": 0.10,
        "fed": 0.05,
    }
    
    SECTOR_STOCKS = {
        "XLK": ["NVDA", "AAPL", "MSFT", "GOOGL", "META", "AMD", "AVGO"],
        "XLF": ["JPM", "GS", "BAC", "WFC", "MS"],
        "XLE": ["XOM", "CVX", "COP"],
        "XLV": ["UNH", "LLY", "JNJ", "PFE"],
        "XLI": ["CAT", "HON", "BA", "GE"],
        "XLY": ["AMZN", "TSLA", "HD", "MCD", "NKE"],
        "XLC": ["META", "GOOGL", "NFLX"],
    }
    
    STOCK_SECTOR = {
        "NVDA": "XLK", "AAPL": "XLK", "MSFT": "XLK", "AMD": "XLK", "AVGO": "XLK",
        "META": "XLC", "GOOGL": "XLC", "NFLX": "XLC",
        "AMZN": "XLY", "TSLA": "XLY",
        "JPM": "XLF", "GS": "XLF",
        "UNH": "XLV", "LLY": "XLV",
        "XOM": "XLE", "CVX": "XLE",
    }
    
    def __init__(self, universe: Optional[List[str]] = None):
        self.universe = universe or [
            "NVDA", "META", "GOOGL", "AMZN", "MSFT", "AAPL", 
            "AMD", "AVGO", "NFLX", "TSLA",
        ]
        self._conn = None
        self._cache = {}
    
    def _get_conn(self):
        if self._conn is None or self._conn.closed:
            self._conn = get_connection()
        return self._conn
    
    def generate_theme(
        self,
        as_of: date,
        momentum_scores: Optional[Dict[str, float]] = None,
    ) -> ThemeConfig:
        cache_key = str(as_of)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        if momentum_scores is None:
            momentum_scores = {}
        
        insider_signals = self._get_insider_signals(as_of)
        analyst_signals = self._get_analyst_signals(as_of)
        pcr_signal = self._get_pcr_signal(as_of)
        social_signals = self._get_social_signals(as_of)
        policy_adj = self._get_policy_adjustment(as_of)
        fed_adj = self._get_fed_adjustment(as_of)
        
        stock_scores = self._compute_stock_scores(
            momentum_scores, insider_signals, analyst_signals, 
            pcr_signal, social_signals
        )
        
        sector_scores = self._compute_sector_scores(stock_scores, policy_adj, fed_adj)
        
        theme = self._build_theme(stock_scores, sector_scores, pcr_signal)
        
        self._cache[cache_key] = theme
        return theme
    
    def _get_insider_signals(self, as_of: date) -> Dict[str, float]:
        conn = self._get_conn()
        cur = conn.cursor()
        
        start = as_of - timedelta(days=90)
        
        cur.execute("""
            SELECT symbol,
                   SUM(CASE WHEN transaction_type = 'P' THEN value ELSE 0 END) as buy_value,
                   SUM(CASE WHEN transaction_type = 'S' THEN value ELSE 0 END) as sell_value
            FROM insider_transactions
            WHERE filing_date BETWEEN %s AND %s
              AND symbol IN %s
            GROUP BY symbol
        """, (start, as_of, tuple(self.universe)))
        
        signals = {}
        for row in cur.fetchall():
            symbol, buy_val, sell_val = row
            total = float(buy_val or 0) + float(sell_val or 0)
            if total > 0:
                signals[symbol] = (float(buy_val or 0) - float(sell_val or 0)) / total
            else:
                signals[symbol] = 0.0
        
        for sym in self.universe:
            if sym not in signals:
                signals[sym] = 0.0
        
        return signals
    
    def _get_analyst_signals(self, as_of: date) -> Dict[str, float]:
        conn = self._get_conn()
        cur = conn.cursor()
        
        start = as_of - timedelta(days=60)
        
        cur.execute("""
            SELECT symbol,
                   AVG((new_target - old_target) / NULLIF(old_target, 0)) as avg_target_change,
                   COUNT(*) as rating_count
            FROM analyst_ratings
            WHERE rating_date BETWEEN %s AND %s
              AND symbol IN %s
            GROUP BY symbol
        """, (start, as_of, tuple(self.universe)))
        
        signals = {}
        for row in cur.fetchall():
            symbol, avg_change, count = row
            if avg_change is not None:
                signals[symbol] = float(avg_change) * 2
            else:
                signals[symbol] = 0.0
        
        for sym in self.universe:
            if sym not in signals:
                signals[sym] = 0.0
        
        return signals
    
    def _get_pcr_signal(self, as_of: date) -> float:
        conn = self._get_conn()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT total_pcr FROM options_pcr
            WHERE trade_date <= %s
            ORDER BY trade_date DESC LIMIT 1
        """, (as_of,))
        
        row = cur.fetchone()
        if row and row[0]:
            pcr = float(row[0])
            if pcr > 1.2:
                return 0.5
            elif pcr > 1.0:
                return 0.2
            elif pcr < 0.7:
                return -0.3
            else:
                return 0.0
        return 0.0
    
    def _get_social_signals(self, as_of: date) -> Dict[str, float]:
        conn = self._get_conn()
        cur = conn.cursor()
        
        start = as_of - timedelta(days=7)
        
        cur.execute("""
            SELECT symbol, AVG(signal_strength) as avg_signal
            FROM social_sentiment
            WHERE trade_date BETWEEN %s AND %s
              AND symbol IN %s
            GROUP BY symbol
        """, (start, as_of, tuple(self.universe)))
        
        signals = {}
        for row in cur.fetchall():
            symbol, avg_signal = row
            signals[symbol] = float(avg_signal) if avg_signal else 0.0
        
        for sym in self.universe:
            if sym not in signals:
                signals[sym] = 0.0
        
        return signals
    
    def _get_policy_adjustment(self, as_of: date) -> Dict[str, float]:
        conn = self._get_conn()
        cur = conn.cursor()
        
        start = as_of - timedelta(days=30)
        
        cur.execute("""
            SELECT sector_impacts FROM policy_events
            WHERE event_date BETWEEN %s AND %s
            ORDER BY event_date DESC
        """, (start, as_of))
        
        sector_adj = {}
        for row in cur.fetchall():
            if row[0]:
                impacts = row[0]
                for sector, impact in impacts.items():
                    if sector not in sector_adj:
                        sector_adj[sector] = 0.0
                    sector_adj[sector] += impact * 0.5
        
        return sector_adj
    
    def _get_fed_adjustment(self, as_of: date) -> Dict[str, float]:
        conn = self._get_conn()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT hawkish_score, dovish_score, rate_decision
            FROM fed_decisions
            WHERE meeting_date <= %s
            ORDER BY meeting_date DESC LIMIT 1
        """, (as_of,))
        
        row = cur.fetchone()
        if not row:
            return {}
        
        hawkish, dovish, decision = row
        stance = (hawkish or 0) - (dovish or 0)
        
        if stance > 0.2:
            return {"XLF": 0.1, "XLU": -0.2, "XLK": -0.1, "XLRE": -0.3}
        elif stance < -0.2:
            return {"XLF": -0.1, "XLU": 0.2, "XLK": 0.2, "XLRE": 0.3}
        else:
            return {}
    
    def _compute_stock_scores(
        self,
        momentum: Dict[str, float],
        insider: Dict[str, float],
        analyst: Dict[str, float],
        pcr: float,
        social: Dict[str, float],
    ) -> Dict[str, float]:
        scores = {}
        
        for symbol in self.universe:
            score = (
                momentum.get(symbol, 0) * self.WEIGHTS["momentum"] +
                insider.get(symbol, 0) * self.WEIGHTS["insider"] +
                analyst.get(symbol, 0) * self.WEIGHTS["analyst"] +
                pcr * self.WEIGHTS["options"] +
                social.get(symbol, 0) * self.WEIGHTS["social"]
            )
            scores[symbol] = max(-1, min(1, score))
        
        return scores
    
    def _compute_sector_scores(
        self,
        stock_scores: Dict[str, float],
        policy_adj: Dict[str, float],
        fed_adj: Dict[str, float],
    ) -> Dict[str, float]:
        sector_scores = {}
        
        for sector, stocks in self.SECTOR_STOCKS.items():
            stock_vals = [stock_scores.get(s, 0) for s in stocks if s in stock_scores]
            avg_stock = sum(stock_vals) / len(stock_vals) if stock_vals else 0
            
            policy = policy_adj.get(sector, 0)
            fed = fed_adj.get(sector, 0)
            
            score = avg_stock * 0.6 + policy * 0.25 + fed * 0.15
            sector_scores[sector] = max(-1, min(1, score))
        
        return sector_scores
    
    def _build_theme(
        self,
        stock_scores: Dict[str, float],
        sector_scores: Dict[str, float],
        pcr_signal: float,
    ) -> ThemeConfig:
        sorted_stocks = sorted(stock_scores.items(), key=lambda x: -x[1])
        sorted_sectors = sorted(sector_scores.items(), key=lambda x: -x[1])
        
        focus_stocks = [s for s, v in sorted_stocks if v > 0.1][:5]
        avoid_stocks = [s for s, v in sorted_stocks if v < -0.2]
        
        focus_sectors = [s for s, v in sorted_sectors if v > 0.1][:3]
        avoid_sectors = [s for s, v in sorted_sectors if v < -0.1]
        
        stock_bonus = {s: round(v * 0.1, 3) for s, v in stock_scores.items() if abs(v) > 0.05}
        sector_bonus = {s: round(v * 0.1, 3) for s, v in sector_scores.items() if abs(v) > 0.05}
        
        if pcr_signal > 0.3:
            risk_level = "low"
        elif pcr_signal < -0.2:
            risk_level = "high"
        else:
            risk_level = "normal"
        
        drivers = []
        if focus_sectors:
            drivers.append(f"Focus: {', '.join(focus_sectors[:2])}")
        if avoid_sectors:
            drivers.append(f"Avoid: {', '.join(avoid_sectors[:2])}")
        
        return ThemeConfig(
            focus_sectors=focus_sectors,
            focus_stocks=focus_stocks,
            avoid_sectors=avoid_sectors,
            avoid_stocks=avoid_stocks,
            sector_bonus=sector_bonus,
            stock_bonus=stock_bonus,
            risk_level=risk_level,
            theme_drivers=drivers,
        )
    
    def close(self):
        if self._conn and not self._conn.closed:
            self._conn.close()
