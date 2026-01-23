#!/usr/bin/env python3
"""
V6.3 "Neuro-Adaptive Pro" 交易系统回测 - 优化版

基于V6.2的优化:
1. 波动率仓位放宽: target_risk 1%→2%, max_position 20%→25%
2. RRG阈值收紧: RS>105才加分，避免频繁切换
3. 熔断冷却期缩短: danger 10→5天, caution 5→3天
4. 科技龙头保护: 核心科技股额外加分，避免被轮出

目标: 年化收益 > 15%, 最大回撤 < 15%, Alpha > 15%
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
import psycopg2

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# V6.3 核心配置 (优化版)
# ============================================================

# 核心科技龙头 (保护机制)
CORE_TECH_LEADERS = ["NVDA", "META", "GOOGL", "AMZN", "MSFT", "AAPL", "AMD", "AVGO"]

# 板块ETF
SECTOR_ETFS = {
    "XLK": "科技", "XLC": "通讯", "XLY": "可选消费",
    "XLF": "金融", "XLV": "医疗", "XLE": "能源",
    "XLI": "工业", "XLP": "必需消费", "XLU": "公用事业",
}

SYMBOL_TO_SECTOR = {
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "AVGO": "XLK", "AMD": "XLK",
    "ADBE": "XLK", "CRM": "XLK", "ORCL": "XLK", "CSCO": "XLK", "INTC": "XLK",
    "META": "XLC", "GOOGL": "XLC", "GOOG": "XLC", "NFLX": "XLC", "DIS": "XLC",
    "CMCSA": "XLC", "T": "XLC", "VZ": "XLC", "TMUS": "XLC",
    "AMZN": "XLY", "TSLA": "XLY", "HD": "XLY", "MCD": "XLY",
    "NKE": "XLY", "SBUX": "XLY", "LOW": "XLY", "TJX": "XLY",
    "JPM": "XLF", "BAC": "XLF", "WFC": "XLF", "GS": "XLF", "MS": "XLF", "BLK": "XLF",
    "UNH": "XLV", "JNJ": "XLV", "LLY": "XLV", "PFE": "XLV", "MRK": "XLV", "ABBV": "XLV",
    "XOM": "XLE", "CVX": "XLE", "COP": "XLE",
}

# 熔断规则 (优化: 缩短冷却期)
CIRCUIT_BREAKER = {
    "vix_danger": 28,
    "vix_caution": 22,
    "vix_watch": 20,
    "vix_rising_fast": 0.25,  # 从0.20提高到0.25
    "market_crash_pct": 0.025,  # 从0.02提高到0.025
    "cooldown_danger": 5,   # 从10天缩短到5天
    "cooldown_caution": 3,  # 从5天缩短到3天
    "cooldown_watch": 2,    # 从3天缩短到2天
    "recovery_vix": 20,     # 从18提高到20
}

ATR_MULTIPLIER = {
    "offensive": 5.0,
    "neutral": 4.0,
    "defensive": 2.5,
}

# 波动率目标仓位 (优化: 放宽参数)
VOLATILITY_TARGET = {
    "target_risk_per_trade": 0.02,  # 从1%提高到2%
    "max_position_pct": 0.25,       # 从20%提高到25%
    "min_position_pct": 0.08,       # 从5%提高到8%
}

PROFIT_LOCK_TIERS = [
    {"threshold": 0.30, "lock_pct": 0.90},
    {"threshold": 0.15, "lock_pct": 1.02},
]

MIN_STOP_DISTANCE = 0.12

INITIAL_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AVGO", "AMD", "ADBE", "CRM", "ORCL", "CSCO", "INTC",
    "META", "GOOGL", "GOOG", "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS",
    "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "TJX",
    "JPM", "BAC", "WFC", "GS", "MS", "BLK",
    "UNH", "JNJ", "LLY", "PFE", "MRK", "ABBV",
    "XOM", "CVX", "COP",
]


@dataclass
class RRGScore:
    etf: str
    sector_name: str
    rs: float
    rs_momentum: float
    quadrant: str
    score: int


@dataclass
class CircuitBreakerState:
    level: str = "normal"
    trigger_date: Optional[str] = None
    trigger_reason: str = ""
    cooldown_until: Optional[str] = None


@dataclass
class MacroView:
    date: str
    market_regime: str
    target_exposure: float
    vix_level: float
    vix_5d_change: float
    spy_change_1d: float
    spy_vs_sma200: float
    score: int
    reasoning: str
    circuit_breaker: Optional[CircuitBreakerState] = None


@dataclass
class Position:
    symbol: str
    shares: int
    avg_cost: float
    entry_date: str
    highest_price: float
    atr_at_entry: float
    is_index_etf: bool = False


@dataclass
class Trade:
    date: str
    symbol: str
    action: str
    price: float
    shares: int
    pnl: float = 0.0
    pnl_pct: float = 0.0
    reason: str = ""


@dataclass 
class DynamicLeader:
    symbol: str
    quant_score: float
    rrg_score: float
    momentum_score: float
    core_bonus: float  # 科技龙头加分
    total_score: float
    sector_etf: str
    reason: str


class V63BacktestEngine:
    """V6.3 优化版回测引擎"""
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.conn = psycopg2.connect(
            host=os.getenv("PG_HOST", "192.168.10.11"),
            port=os.getenv("PG_PORT", "5432"),
            database=os.getenv("PG_DATABASE", "trader"),
            user=os.getenv("PG_USER", "trader"),
            password=os.getenv("PG_PASSWORD", "")
        )
        
        self._prices: Dict[str, pd.DataFrame] = {}
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[str, float, float]] = []
        self.macro_history: List[MacroView] = []
        self.current_leaders: List[str] = []
        self.leader_history: List[Dict] = []
        self.rrg_history: List[Dict] = []
        self.circuit_breaker = CircuitBreakerState()
        self._current_macro: Optional[MacroView] = None
        self._cold_start_days = 0
        self._in_cold_start = False
    
    def _load_data(self, start: date, end: date):
        print("  加载价格数据...")
        all_symbols = set(INITIAL_UNIVERSE)
        all_symbols.update(['SPY', 'QQQ', 'VIX'])
        all_symbols.update(SECTOR_ETFS.keys())
        
        query = """
            SELECT symbol, trade_date, open, high, low, close, volume
            FROM daily_prices
            WHERE trade_date BETWEEN %s AND %s
              AND symbol IN %s
            ORDER BY symbol, trade_date
        """
        df = pd.read_sql(query, self.conn, params=(start - timedelta(days=400), end, tuple(all_symbols)))
        
        for sym in df['symbol'].unique():
            sdf = df[df['symbol'] == sym].copy()
            sdf.set_index('trade_date', inplace=True)
            sdf['sma20'] = sdf['close'].rolling(20).mean()
            sdf['sma50'] = sdf['close'].rolling(50).mean()
            sdf['sma200'] = sdf['close'].rolling(200).mean()
            sdf['rsi'] = self._calc_rsi(sdf['close'], 14)
            sdf['atr'] = self._calc_atr(sdf, 14)
            sdf['mom5'] = sdf['close'].pct_change(5)
            sdf['mom20'] = sdf['close'].pct_change(20)
            sdf['mom60'] = sdf['close'].pct_change(60)
            sdf['change_1d'] = sdf['close'].pct_change(1)
            self._prices[sym] = sdf
        
        print(f"    已加载 {len(self._prices)} 只标的")
    
    def _calc_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calc_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        high, low, close = df['high'], df['low'], df['close']
        tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    
    def _get(self, sym: str, dt: date, col: str) -> Optional[float]:
        if sym not in self._prices:
            return None
        df = self._prices[sym]
        valid = df[df.index <= dt]
        if len(valid) == 0:
            return None
        val = valid[col].iloc[-1]
        return float(val) if pd.notna(val) else None
    
    def _get_prev(self, sym: str, dt: date, col: str, days: int = 1) -> Optional[float]:
        if sym not in self._prices:
            return None
        df = self._prices[sym]
        valid = df[df.index <= dt]
        if len(valid) < days + 1:
            return None
        val = valid[col].iloc[-(days + 1)]
        return float(val) if pd.notna(val) else None
    
    # ================================================================
    # 优化2: RRG评分 (收紧阈值)
    # ================================================================
    
    def _calc_rrg_scores(self, dt: date) -> Dict[str, RRGScore]:
        """RRG评分 - 优化版: 只有RS>105才给满分"""
        rrg_scores = {}
        spy_mom60 = self._get('SPY', dt, 'mom60') or 0.001
        
        for etf, sector_name in SECTOR_ETFS.items():
            if etf not in self._prices:
                continue
            
            etf_mom60 = self._get(etf, dt, 'mom60')
            etf_mom60_prev = self._get_prev(etf, dt, 'mom60', 10)
            
            if etf_mom60 is None:
                continue
            
            rs = ((1 + etf_mom60) / (1 + spy_mom60)) * 100
            rs_momentum = (rs - ((1 + (etf_mom60_prev or etf_mom60)) / (1 + spy_mom60)) * 100) if etf_mom60_prev else 0
            
            # 优化: 收紧阈值，RS>105才给满分
            if rs > 105 and rs_momentum > 0:
                quadrant = "Leading"
                score = 3
            elif rs > 102 and rs_momentum > 0:
                quadrant = "Leading"
                score = 2  # 弱领先只给2分
            elif rs > 100 and rs_momentum <= 0:
                quadrant = "Weakening"
                score = 0  # 减弱不加分
            elif rs <= 100 and rs_momentum > 0:
                quadrant = "Improving"
                score = 1  # 改善给1分
            else:
                quadrant = "Lagging"
                score = -1  # 落后只扣1分
            
            rrg_scores[etf] = RRGScore(etf, sector_name, rs, rs_momentum, quadrant, score)
        
        return rrg_scores
    
    # ================================================================
    # 熔断检查器 (优化: 缩短冷却期)
    # ================================================================
    
    def _check_circuit_breaker(self, dt: date) -> CircuitBreakerState:
        vix = self._get('VIX', dt, 'close') or 20
        vix_5d_ago = self._get_prev('VIX', dt, 'close', 5) or vix
        vix_5d_change = (vix - vix_5d_ago) / vix_5d_ago if vix_5d_ago > 0 else 0
        
        spy_change = self._get('SPY', dt, 'change_1d') or 0
        spy_close = self._get('SPY', dt, 'close') or 0
        spy_sma50 = self._get('SPY', dt, 'sma50') or spy_close
        spy_sma200 = self._get('SPY', dt, 'sma200') or spy_close
        
        if self.circuit_breaker.cooldown_until:
            cooldown_date = date.fromisoformat(self.circuit_breaker.cooldown_until)
            if dt <= cooldown_date:
                return self.circuit_breaker
        
        if self.circuit_breaker.level != "normal":
            if vix < CIRCUIT_BREAKER["recovery_vix"] and spy_close > spy_sma50:
                self.circuit_breaker = CircuitBreakerState(level="normal")
                return self.circuit_breaker
        
        new_level = "normal"
        trigger_reason = ""
        cooldown_days = 0
        
        if vix > CIRCUIT_BREAKER["vix_danger"]:
            new_level = "danger"
            trigger_reason = f"VIX恐慌({vix:.1f})"
            cooldown_days = CIRCUIT_BREAKER["cooldown_danger"]
        elif vix_5d_change > CIRCUIT_BREAKER["vix_rising_fast"] and vix > 22:
            new_level = "danger"
            trigger_reason = f"VIX急升({vix_5d_change*100:.0f}%)"
            cooldown_days = CIRCUIT_BREAKER["cooldown_danger"]
        elif spy_change < -CIRCUIT_BREAKER["market_crash_pct"]:
            new_level = "danger"
            trigger_reason = f"SPY暴跌({spy_change*100:.1f}%)"
            cooldown_days = CIRCUIT_BREAKER["cooldown_danger"]
        elif vix > CIRCUIT_BREAKER["vix_caution"] and vix_5d_change > 0.15:
            new_level = "caution"
            trigger_reason = f"VIX警戒({vix:.1f})"
            cooldown_days = CIRCUIT_BREAKER["cooldown_caution"]
        elif spy_close < spy_sma200 * 0.97:  # 从0.98收紧到0.97
            new_level = "caution"
            trigger_reason = f"SPY跌破SMA200"
            cooldown_days = CIRCUIT_BREAKER["cooldown_caution"]
        elif vix > CIRCUIT_BREAKER["vix_watch"] and vix_5d_change > 0.10:
            new_level = "watch"
            trigger_reason = f"VIX观察({vix:.1f})"
            cooldown_days = CIRCUIT_BREAKER["cooldown_watch"]
        
        level_order = {"normal": 0, "watch": 1, "caution": 2, "danger": 3}
        if level_order.get(new_level, 0) > level_order.get(self.circuit_breaker.level, 0):
            cooldown = dt + timedelta(days=cooldown_days)
            self.circuit_breaker = CircuitBreakerState(new_level, str(dt), trigger_reason, str(cooldown))
        
        return self.circuit_breaker
    
    def _analyze_macro(self, dt: date) -> MacroView:
        breaker = self._check_circuit_breaker(dt)
        
        vix = self._get('VIX', dt, 'close') or 20
        vix_20d_ago = self._get_prev('VIX', dt, 'close', 20) or vix
        vix_5d_ago = self._get_prev('VIX', dt, 'close', 5) or vix
        vix_5d_change = (vix - vix_5d_ago) / vix_5d_ago if vix_5d_ago > 0 else 0
        
        spy_close = self._get('SPY', dt, 'close') or 0
        spy_sma50 = self._get('SPY', dt, 'sma50') or spy_close
        spy_sma200 = self._get('SPY', dt, 'sma200') or spy_close
        spy_mom = self._get('SPY', dt, 'mom20') or 0
        spy_change = self._get('SPY', dt, 'change_1d') or 0
        
        # 优化: 提高defensive的目标仓位
        if breaker.level == "danger":
            return MacroView(str(dt), "defensive", 0.3, vix, vix_5d_change, spy_change,
                           (spy_close / spy_sma200 - 1) if spy_sma200 else 0, -99,
                           f"🚨DANGER: {breaker.trigger_reason}", breaker)
        elif breaker.level == "caution":
            return MacroView(str(dt), "defensive", 0.5, vix, vix_5d_change, spy_change,
                           (spy_close / spy_sma200 - 1) if spy_sma200 else 0, -50,
                           f"⚠️CAUTION: {breaker.trigger_reason}", breaker)
        elif breaker.level == "watch":
            return MacroView(str(dt), "neutral", 0.7, vix, vix_5d_change, spy_change,
                           (spy_close / spy_sma200 - 1) if spy_sma200 else 0, -20,
                           f"👀WATCH: {breaker.trigger_reason}", breaker)
        
        score = 0
        reasoning_parts = []
        
        if vix < 15:
            score += 2
            reasoning_parts.append("VIX极低")
        elif vix < 18:
            score += 1
            reasoning_parts.append("VIX正常")
        elif vix < 22:
            score -= 1
            reasoning_parts.append("VIX偏高")
        else:
            score -= 2
            reasoning_parts.append("VIX警告")
        
        if spy_close > spy_sma50 and spy_close > spy_sma200 and spy_mom > 0.03:
            score += 2
            reasoning_parts.append("SPY强势")
        elif spy_close > spy_sma50 and spy_close > spy_sma200:
            score += 1
            reasoning_parts.append("SPY趋势向上")
        elif spy_close < spy_sma200:
            score -= 2
            reasoning_parts.append("SPY跌破年线")
        elif spy_close < spy_sma50:
            score -= 1
            reasoning_parts.append("SPY跌破50日线")
        
        if vix < vix_20d_ago * 0.8:
            score += 1
            reasoning_parts.append("VIX下降")
        elif vix > vix_20d_ago * 1.3:
            score -= 1
            reasoning_parts.append("VIX上升")
        
        if score >= 2:
            regime, exposure = "offensive", 1.0
        elif score >= 0:
            regime, exposure = "neutral", 0.8  # 从0.7提高到0.8
        else:
            regime, exposure = "defensive", 0.5  # 从0.4提高到0.5
        
        return MacroView(str(dt), regime, exposure, vix, vix_5d_change, spy_change,
                        (spy_close / spy_sma200 - 1) if spy_sma200 else 0, score,
                        " | ".join(reasoning_parts), breaker)
    
    def _check_cold_start(self, dt: date) -> bool:
        key_symbols = ["NVDA", "AAPL", "MSFT", "META", "GOOGL"]
        ready_count = sum(1 for sym in key_symbols if self._get(sym, dt, 'sma200') is not None)
        return ready_count < 3
    
    # ================================================================
    # 优化4: 科技龙头保护机制
    # ================================================================
    
    def _build_dynamic_universe(self, dt: date, rrg_scores: Dict[str, RRGScore]) -> List[DynamicLeader]:
        if self._check_cold_start(dt):
            self._in_cold_start = True
            return []
        
        self._in_cold_start = False
        candidates = []
        seen_symbols = set()
        spy_mom60 = self._get('SPY', dt, 'mom60') or 0
        
        for sym in INITIAL_UNIVERSE:
            if sym in seen_symbols or sym not in self._prices:
                continue
            seen_symbols.add(sym)
            
            close = self._get(sym, dt, 'close')
            sma50 = self._get(sym, dt, 'sma50')
            sma200 = self._get(sym, dt, 'sma200')
            rsi = self._get(sym, dt, 'rsi')
            mom20 = self._get(sym, dt, 'mom20')
            mom60 = self._get(sym, dt, 'mom60')
            
            if close is None or sma200 is None:
                continue
            
            sector_etf = SYMBOL_TO_SECTOR.get(sym, "XLK")
            
            # Quant评分
            quant_score = 0
            if close > sma200:
                quant_score += 2
            elif close > sma50:
                quant_score += 1
            else:
                continue
            
            if rsi and rsi > 40:
                quant_score += 1
            if mom20 and mom20 > 0.08:
                quant_score += 2
            elif mom20 and mom20 > 0.02:
                quant_score += 1
            elif mom20 and mom20 < -0.05:
                quant_score -= 1
            
            # 动量评分
            momentum_score = 0
            if mom60 and mom60 > 0.20:
                momentum_score += 3
            elif mom60 and mom60 > 0.10:
                momentum_score += 2
            elif mom60 and mom60 > 0:
                momentum_score += 1
            elif mom60 and mom60 < -0.10:
                momentum_score -= 2
            
            rs = (mom60 or 0) - spy_mom60
            if rs > 0.15:
                momentum_score += 2
            elif rs > 0.05:
                momentum_score += 1
            elif rs < -0.10:
                momentum_score -= 1
            
            # RRG板块评分
            rrg_score = rrg_scores.get(sector_etf, RRGScore(sector_etf, "", 100, 0, "Neutral", 0)).score
            
            # 优化4: 科技龙头保护加分
            core_bonus = 0
            if sym in CORE_TECH_LEADERS:
                core_bonus = 2  # 核心科技龙头额外+2分
            
            total_score = quant_score + momentum_score + rrg_score + core_bonus
            
            candidates.append(DynamicLeader(
                sym, quant_score, rrg_score, momentum_score, core_bonus,
                total_score, sector_etf, f"RS:{rs:.2f}, RRG:{rrg_score}, Core:{core_bonus}"
            ))
        
        candidates.sort(key=lambda x: -x.total_score)
        
        # 板块分散
        sector_count: Dict[str, int] = {}
        final_leaders = []
        for c in candidates[:15]:
            if len(final_leaders) >= 10:
                break
            count = sector_count.get(c.sector_etf, 0)
            if count >= 3:
                continue
            final_leaders.append(c)
            sector_count[c.sector_etf] = count + 1
        
        return final_leaders
    
    # ================================================================
    # 优化1: 波动率仓位 (放宽参数)
    # ================================================================
    
    def _calc_position_size(self, sym: str, dt: date, available_capital: float) -> float:
        atr = self._get(sym, dt, 'atr')
        price = self._get(sym, dt, 'close')
        
        if not atr or not price or atr <= 0:
            return available_capital * 0.12
        
        stop_distance_pct = (atr * 5) / price
        target_risk = self.initial_capital * VOLATILITY_TARGET["target_risk_per_trade"]
        position_value = target_risk / stop_distance_pct if stop_distance_pct > 0 else 0
        
        max_pos = self.initial_capital * VOLATILITY_TARGET["max_position_pct"]
        min_pos = self.initial_capital * VOLATILITY_TARGET["min_position_pct"]
        
        return min(max(min_pos, position_value), max_pos, available_capital)
    
    def _calc_stop_price(self, pos: Position, dt: date, regime: str) -> float:
        if pos.is_index_etf:
            return 0
        
        current_price = self._get(pos.symbol, dt, 'close') or pos.avg_cost
        current_atr = self._get(pos.symbol, dt, 'atr') or pos.atr_at_entry
        pos.highest_price = max(pos.highest_price, current_price)
        
        multiplier = ATR_MULTIPLIER.get(regime, 4.0)
        atr_stop = pos.highest_price - (multiplier * current_atr)
        min_stop = pos.highest_price * (1 - MIN_STOP_DISTANCE)
        atr_stop = min(atr_stop, min_stop)
        
        pnl_pct = (current_price - pos.avg_cost) / pos.avg_cost
        profit_stop = 0
        for tier in PROFIT_LOCK_TIERS:
            if pnl_pct >= tier["threshold"]:
                profit_stop = pos.avg_cost * tier["lock_pct"] if tier["lock_pct"] > 1 else pos.highest_price * tier["lock_pct"]
                break
        
        return max(atr_stop, profit_stop)
    
    def _portfolio_value(self, dt: date) -> float:
        return self.cash + sum(p.shares * (self._get(s, dt, 'close') or p.avg_cost) for s, p in self.positions.items())
    
    def _buy(self, sym: str, dt: date, budget: float, reason: str, is_index_etf: bool = False) -> bool:
        price = self._get(sym, dt, 'close')
        if not price or budget < 500:
            return False
        shares = int(budget / price)
        if shares <= 0 or shares * price > self.cash:
            return False
        
        self.cash -= shares * price
        atr = self._get(sym, dt, 'atr') or (price * 0.02)
        
        if sym in self.positions:
            p = self.positions[sym]
            total = p.shares + shares
            p.avg_cost = (p.avg_cost * p.shares + price * shares) / total
            p.shares = total
            p.highest_price = max(p.highest_price, price)
        else:
            self.positions[sym] = Position(sym, shares, price, str(dt), price, atr, is_index_etf)
        
        self.trades.append(Trade(str(dt), sym, "BUY", price, shares, reason=reason))
        return True
    
    def _sell(self, sym: str, dt: date, reason: str) -> float:
        if sym not in self.positions:
            return 0
        p = self.positions[sym]
        price = self._get(sym, dt, 'close') or p.avg_cost
        proceeds = p.shares * price
        pnl = proceeds - p.shares * p.avg_cost
        pnl_pct = pnl / (p.shares * p.avg_cost)
        self.cash += proceeds
        self.trades.append(Trade(str(dt), sym, "SELL", price, p.shares, pnl, pnl_pct, reason))
        del self.positions[sym]
        return pnl
    
    def _check_stops(self, dt: date, regime: str):
        to_sell = [(sym, f"止损触发(${self._calc_stop_price(pos, dt, regime):.2f}, {((self._get(sym, dt, 'close') or pos.avg_cost) - pos.avg_cost) / pos.avg_cost:+.1%})")
                   for sym, pos in self.positions.items()
                   if not pos.is_index_etf and self._get(sym, dt, 'close') and self._get(sym, dt, 'close') < self._calc_stop_price(pos, dt, regime)]
        for sym, reason in to_sell:
            self._sell(sym, dt, reason)
    
    def _liquidate_weak_positions(self, dt: date, keep_pct: float = 0.10):
        to_sell = [(sym, (self._get(sym, dt, 'close') or pos.avg_cost) / pos.avg_cost - 1)
                   for sym, pos in self.positions.items() if not pos.is_index_etf]
        to_sell = [(s, p, f"风控清仓({p:+.1%})") for s, p in to_sell if p < keep_pct]
        to_sell.sort(key=lambda x: x[1])
        for sym, _, reason in to_sell[:max(1, len(to_sell) // 2)]:
            self._sell(sym, dt, reason)
    
    def _rebalance(self, dt: date, macro: MacroView, leaders: List[str]):
        pv = self._portfolio_value(dt)
        regime = macro.market_regime
        
        if self._in_cold_start:
            if "QQQ" not in self.positions:
                for sym in list(self.positions.keys()):
                    self._sell(sym, dt, "冷启动切换QQQ")
                self._buy("QQQ", dt, self.cash * 0.95, "冷启动持有QQQ", is_index_etf=True)
            return
        
        if "QQQ" in self.positions:
            self._sell("QQQ", dt, "退出冷启动")
        
        if regime == "defensive":
            self._liquidate_weak_positions(dt, keep_pct=0.10)
            return
        
        for sym in list(self.positions.keys()):
            if sym not in leaders and not self.positions[sym].is_index_etf:
                self._sell(sym, dt, "轮出龙头池")
        
        stock_budget = pv * macro.target_exposure
        current_val = sum(p.shares * (self._get(s, dt, 'close') or p.avg_cost) for s, p in self.positions.items() if not p.is_index_etf)
        
        if current_val < stock_budget * 0.85:
            available = min(stock_budget - current_val, self.cash * 0.95)
            max_pos = 6 if regime == "offensive" else 5
            current_count = len([p for p in self.positions.values() if not p.is_index_etf])
            
            for sym in leaders:
                if sym in self.positions or current_count >= max_pos:
                    continue
                close, sma50, mom20 = self._get(sym, dt, 'close'), self._get(sym, dt, 'sma50'), self._get(sym, dt, 'mom20')
                if not close or not sma50 or close < sma50 * 0.92 or (mom20 and mom20 < -0.12):
                    continue
                
                budget = self._calc_position_size(sym, dt, available)
                if self._buy(sym, dt, budget, f"龙头买入({regime})"):
                    current_count += 1
                    available -= budget
    
    def run(self, start: date, end: date) -> dict:
        print("\n" + "=" * 70)
        print("V6.3 Neuro-Adaptive Pro 策略回测 (优化版)")
        print("=" * 70)
        print("  优化内容:")
        print("    1. 波动率仓位放宽: risk 2%, max 25%")
        print("    2. RRG阈值收紧: RS>105才满分")
        print("    3. 熔断冷却期缩短: danger 5天")
        print("    4. 科技龙头保护: +2分加成")
        
        self._load_data(start, end)
        
        trading_days = sorted([d for d in self._prices['SPY'].index.tolist() if start <= d <= end])
        actual_start = trading_days[0]
        
        print(f"\n  回测区间: {actual_start} ~ {end}")
        print(f"  交易日数: {len(trading_days)}")
        print(f"  初始资金: ${self.initial_capital:,.0f}")
        
        last_macro_month = last_universe_month = None
        self._spy_start_price = self._get('SPY', actual_start, 'close') or 1
        
        for i, dt in enumerate(trading_days):
            current_month = dt.strftime("%Y-%m")
            self._current_macro = self._analyze_macro(dt)
            
            if current_month != last_macro_month:
                self.macro_history.append(self._current_macro)
                last_macro_month = current_month
                if i % 50 == 0 or len(self.macro_history) <= 3:
                    breaker_info = f" [{self._current_macro.circuit_breaker.level.upper()}]" if self._current_macro.circuit_breaker and self._current_macro.circuit_breaker.level != "normal" else ""
                    print(f"\n  📊 [{dt}] {self._current_macro.market_regime}{breaker_info} (仓位:{self._current_macro.target_exposure:.0%}) - {self._current_macro.reasoning}")
            
            if current_month != last_universe_month:
                rrg_scores = self._calc_rrg_scores(dt)
                self.rrg_history.append({"date": str(dt), "scores": {e: {"quadrant": s.quadrant, "score": s.score} for e, s in rrg_scores.items()}})
                leaders = self._build_dynamic_universe(dt, rrg_scores)
                self.current_leaders = [l.symbol for l in leaders]
                self.leader_history.append({"date": str(dt), "cold_start": self._in_cold_start, "leaders": [{"symbol": l.symbol, "score": l.total_score} for l in leaders]})
                last_universe_month = current_month
                
                if self._in_cold_start:
                    self._cold_start_days += 1
                    print(f"  ❄️ [{dt}] 冷启动: 持有QQQ")
                elif leaders:
                    leading = [s.sector_name for s in rrg_scores.values() if s.quadrant == "Leading"]
                    print(f"  🔄 [{dt}] 龙头池: {', '.join(self.current_leaders[:6])}" + (f" | RRG领先: {', '.join(leading)}" if leading else ""))
            
            self._check_stops(dt, self._current_macro.market_regime)
            if i % 5 == 0:
                self._rebalance(dt, self._current_macro, self.current_leaders)
            
            pv = self._portfolio_value(dt)
            spy_val = self.initial_capital * self._get('SPY', dt, 'close') / self._spy_start_price
            self.equity_curve.append((str(dt), pv, spy_val))
            
            if i % 150 == 0:
                print(f"  [{i+1}/{len(trading_days)}] {dt}: ${pv:,.0f} (SPY: ${spy_val:,.0f})")
        
        return self._calc_results(start, end)
    
    def _calc_results(self, start: date, end: date) -> dict:
        final, spy_final = self.equity_curve[-1][1], self.equity_curve[-1][2]
        total_ret = final / self.initial_capital - 1
        spy_ret = spy_final / self.initial_capital - 1
        years = (end - start).days / 365
        ann_ret = (1 + total_ret) ** (1/years) - 1 if years > 0 else 0
        
        values = [e[1] for e in self.equity_curve]
        peak, max_dd = self.initial_capital, 0
        for v in values:
            peak = max(peak, v)
            max_dd = max(max_dd, (peak - v) / peak)
        
        rets = pd.Series(values).pct_change().dropna()
        sharpe = np.sqrt(252) * rets.mean() / rets.std() if rets.std() > 0 else 0
        
        sells = [t for t in self.trades if t.action == "SELL" and t.symbol not in ["QQQ", "SPY"]]
        wins = [t for t in sells if t.pnl > 0]
        win_rate = len(wins) / len(sells) if sells else 0
        profit_factor = sum(t.pnl for t in wins) / abs(sum(t.pnl for t in sells if t.pnl < 0)) if any(t.pnl < 0 for t in sells) else float('inf')
        
        breaker_triggers = {"watch": 0, "caution": 0, "danger": 0}
        for m in self.macro_history:
            if m.circuit_breaker and m.circuit_breaker.level != "normal":
                breaker_triggers[m.circuit_breaker.level] += 1
        
        regime_dist = {}
        for m in self.macro_history:
            regime_dist[m.market_regime] = regime_dist.get(m.market_regime, 0) + 1
        
        return {
            "final_value": final, "total_return": total_ret, "annualized_return": ann_ret,
            "spy_return": spy_ret, "alpha": total_ret - spy_ret, "max_drawdown": max_dd,
            "sharpe": sharpe, "win_rate": win_rate, "profit_factor": profit_factor,
            "total_trades": len(self.trades), "stock_trades": len(sells),
            "avg_win": np.mean([t.pnl for t in wins]) if wins else 0,
            "avg_loss": np.mean([t.pnl for t in sells if t.pnl < 0]) if any(t.pnl < 0 for t in sells) else 0,
            "regime_distribution": regime_dist, "circuit_breaker_triggers": breaker_triggers,
            "cold_start_months": self._cold_start_days,
        }


def main():
    bt = V63BacktestEngine(100000.0)
    result = bt.run(date(2022, 1, 3), date(2026, 1, 16))
    
    print("\n" + "=" * 70)
    print("V6.3 回测结果")
    print("=" * 70)
    print(f"\n  最终价值: ${result['final_value']:,.0f}")
    print(f"  总收益率: {result['total_return']:+.2%}")
    print(f"  年化收益: {result['annualized_return']:+.2%}")
    print(f"  SPY收益:  {result['spy_return']:+.2%}")
    print(f"  超额收益: {result['alpha']:+.2%}")
    print(f"\n  最大回撤: {result['max_drawdown']:.2%}")
    print(f"  夏普比率: {result['sharpe']:.2f}")
    print(f"  胜率: {result['win_rate']:.1%}")
    print(f"  盈亏比: {result['profit_factor']:.2f}")
    print(f"  交易次数: {result['stock_trades']} 笔")
    print(f"  平均盈利: ${result['avg_win']:,.0f}")
    print(f"  平均亏损: ${result['avg_loss']:,.0f}")
    print(f"\n  冷启动月数: {result['cold_start_months']}")
    print(f"  熔断触发: {result['circuit_breaker_triggers']}")
    print(f"  状态分布: {result['regime_distribution']}")
    
    # 保存
    output = Path("storage/backtest_v6_3")
    output.mkdir(parents=True, exist_ok=True)
    with open(output / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    
    trades_data = [{"date": t.date, "symbol": t.symbol, "action": t.action, "price": t.price, "shares": t.shares, "pnl": t.pnl, "pnl_pct": t.pnl_pct, "reason": t.reason} for t in bt.trades]
    with open(output / "trades.json", "w") as f:
        json.dump(trades_data, f, indent=2)
    
    pd.DataFrame(bt.equity_curve, columns=['date', 'portfolio', 'spy']).to_csv(output / "equity_curve.csv", index=False)
    
    print(f"\n📁 保存到: {output}")
    
    # Top交易
    sells = [t for t in bt.trades if t.action == "SELL" and t.symbol not in ["QQQ", "SPY"]]
    print("\n【最大盈利】")
    for t in sorted(sells, key=lambda x: -x.pnl)[:5]:
        print(f"  {t.date} {t.symbol}: ${t.pnl:+,.0f} ({t.pnl_pct:+.1%})")
    print("\n【最大亏损】")
    for t in sorted(sells, key=lambda x: x.pnl)[:5]:
        print(f"  {t.date} {t.symbol}: ${t.pnl:+,.0f} ({t.pnl_pct:+.1%})")
    
    # 年度
    print("\n【年度收益】")
    eq = pd.DataFrame(bt.equity_curve, columns=['date', 'portfolio', 'spy'])
    eq['date'] = pd.to_datetime(eq['date'])
    eq['year'] = eq['date'].dt.year
    for year in sorted(eq['year'].unique()):
        yd = eq[eq['year'] == year]
        if len(yd) < 2:
            continue
        pr = (yd['portfolio'].iloc[-1] / yd['portfolio'].iloc[0] - 1) * 100
        sr = (yd['spy'].iloc[-1] / yd['spy'].iloc[0] - 1) * 100
        print(f"  {year}: 策略 {pr:+.1f}% | SPY {sr:+.1f}% | Alpha {pr-sr:+.1f}%")
    
    # 对比
    print("\n" + "=" * 70)
    print("【V6.1 vs V6.2 vs V6.3 对比】")
    print("=" * 70)
    v61 = {"total_return": 0.7898, "max_drawdown": 0.2036, "sharpe": 0.96, "alpha": 0.3419}
    v62 = {"total_return": 0.2820, "max_drawdown": 0.0968, "sharpe": 0.90, "alpha": -0.1659}
    print(f"  {'指标':<12} {'V6.1':<12} {'V6.2':<12} {'V6.3':<12}")
    print(f"  {'-'*48}")
    print(f"  {'总收益':<12} {v61['total_return']:+.1%}{'':>4} {v62['total_return']:+.1%}{'':>4} {result['total_return']:+.1%}")
    print(f"  {'Alpha':<12} {v61['alpha']:+.1%}{'':>4} {v62['alpha']:+.1%}{'':>4} {result['alpha']:+.1%}")
    print(f"  {'最大回撤':<12} {v61['max_drawdown']:.1%}{'':>5} {v62['max_drawdown']:.1%}{'':>5} {result['max_drawdown']:.1%}")
    print(f"  {'夏普':<12} {v61['sharpe']:.2f}{'':>8} {v62['sharpe']:.2f}{'':>8} {result['sharpe']:.2f}")


if __name__ == "__main__":
    main()
