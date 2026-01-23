#!/usr/bin/env python3
"""
V7.1 "Rule-Based Theme" 交易系统回测 - 规则生成主题版

对比 V7.0 (人工主题) vs V7.1 (规则主题):
- V7.0: 人工基于历史事件判断每季度主题
- V7.1: 系统基于历史事件自动生成主题 (规则模式)

验证规则分析器的有效性，为后续 LLM 分析打基础。
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psycopg2

sys.path.insert(0, str(Path(__file__).parent.parent))

# 从主题配置文件加载 (支持规则/LLM两种来源)
THEMES_FILE = Path("storage/intelligence_cache/backtest_themes.json")
LLM_THEMES_FILE = Path("storage/intelligence_cache/llm_themes.json")

def load_quarterly_themes() -> Dict:
    """加载规则生成的季度主题"""
    if THEMES_FILE.exists():
        with open(THEMES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    # 回退到默认配置
    return {
        "default": {
            "theme": "均衡配置",
            "focus_sectors": ["XLK", "XLV"],
            "focus_stocks": ["NVDA", "AAPL", "MSFT", "UNH"],
            "avoid_sectors": [],
            "sector_bonus": {"XLK": 2, "XLV": 1},
        }
    }

QUARTERLY_THEMES = load_quarterly_themes()

SECTOR_ETFS = {
    "XLK": "科技", "XLC": "通讯", "XLY": "可选消费",
    "XLF": "金融", "XLV": "医疗", "XLE": "能源",
    "XLI": "工业", "XLP": "必需消费", "XLU": "公用事业",
    "XLRE": "房地产", "XLB": "材料",
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
    "CAT": "XLI", "DE": "XLI", "UNP": "XLI", "HON": "XLI", "GE": "XLI", "RTX": "XLI",
    "PG": "XLP", "KO": "XLP", "PEP": "XLP", "COST": "XLP", "WMT": "XLP",
    "AMT": "XLRE", "PLD": "XLRE", "CCI": "XLRE",
}

CIRCUIT_BREAKER = {
    "vix_danger": 28, "vix_caution": 22, "vix_watch": 20,
    "vix_rising_fast": 0.25, "market_crash_pct": 0.025,
    "cooldown_danger": 5, "cooldown_caution": 3, "cooldown_watch": 2,
    "recovery_vix": 20,
}

ATR_MULTIPLIER = {"offensive": 5.0, "neutral": 4.0, "defensive": 2.5}

VOLATILITY_TARGET = {
    "target_risk_per_trade": 0.02,
    "max_position_pct": 0.25,
    "min_position_pct": 0.08,
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
    "CAT", "DE", "UNP", "HON", "GE", "RTX",
    "PG", "KO", "PEP", "COST", "WMT",
]


def get_quarter_key(dt: date) -> str:
    q = (dt.month - 1) // 3 + 1
    return f"{dt.year}-Q{q}"


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
    score: int
    reasoning: str
    circuit_breaker: Optional[CircuitBreakerState] = None
    theme: str = ""


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
    theme_bonus: float
    focus_bonus: float
    momentum_score: float
    total_score: float
    sector_etf: str


class V71BacktestEngine:
    """V7.1 回测引擎 - 规则生成主题版"""
    
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
        self.circuit_breaker = CircuitBreakerState()
        self._current_macro: Optional[MacroView] = None
        self._cold_start_days = 0
        self._in_cold_start = False
        self._current_theme: Optional[Dict] = None
        self._last_universe_update: Optional[date] = None
    
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
    
    def _get_theme(self, dt: date) -> Dict:
        qkey = get_quarter_key(dt)
        if qkey in QUARTERLY_THEMES:
            return QUARTERLY_THEMES[qkey]
        keys = sorted(QUARTERLY_THEMES.keys())
        for k in reversed(keys):
            if k < qkey:
                return QUARTERLY_THEMES[k]
        return QUARTERLY_THEMES.get("default", QUARTERLY_THEMES[keys[0]])
    
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
        elif spy_close < spy_sma200 * 0.97:
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
        theme = self._get_theme(dt)
        self._current_theme = theme
        
        vix = self._get('VIX', dt, 'close') or 20
        vix_20d_ago = self._get_prev('VIX', dt, 'close', 20) or vix
        
        spy_close = self._get('SPY', dt, 'close') or 0
        spy_sma50 = self._get('SPY', dt, 'sma50') or spy_close
        spy_sma200 = self._get('SPY', dt, 'sma200') or spy_close
        spy_mom = self._get('SPY', dt, 'mom20') or 0
        
        if breaker.level == "danger":
            return MacroView(str(dt), "defensive", 0.3, vix, -99,
                           f"🚨DANGER: {breaker.trigger_reason}", breaker, theme["theme"])
        elif breaker.level == "caution":
            return MacroView(str(dt), "defensive", 0.5, vix, -50,
                           f"⚠️CAUTION: {breaker.trigger_reason}", breaker, theme["theme"])
        elif breaker.level == "watch":
            return MacroView(str(dt), "neutral", 0.7, vix, -20,
                           f"👀WATCH: {breaker.trigger_reason}", breaker, theme["theme"])
        
        score = 0
        reasoning_parts = [f"主题:{theme['theme'][:15]}"]
        
        if vix < 15:
            score += 2
        elif vix < 18:
            score += 1
        elif vix < 22:
            score -= 1
        else:
            score -= 2
        
        if spy_close > spy_sma50 and spy_close > spy_sma200 and spy_mom > 0.03:
            score += 2
            reasoning_parts.append("SPY强势")
        elif spy_close > spy_sma50 and spy_close > spy_sma200:
            score += 1
        elif spy_close < spy_sma200:
            score -= 2
            reasoning_parts.append("SPY破年线")
        
        if vix < vix_20d_ago * 0.8:
            score += 1
        elif vix > vix_20d_ago * 1.3:
            score -= 1
        
        if score >= 2:
            regime, exposure = "offensive", 1.0
        elif score >= 0:
            regime, exposure = "neutral", 0.8
        else:
            regime, exposure = "defensive", 0.5
        
        return MacroView(str(dt), regime, exposure, vix, score,
                        " | ".join(reasoning_parts), breaker, theme["theme"])
    
    def _check_cold_start(self, dt: date) -> bool:
        key_symbols = ["NVDA", "AAPL", "MSFT", "META", "GOOGL"]
        ready_count = sum(1 for sym in key_symbols if self._get(sym, dt, 'sma200') is not None)
        return ready_count < 3
    
    def _build_dynamic_universe(self, dt: date) -> List[DynamicLeader]:
        if self._check_cold_start(dt):
            self._in_cold_start = True
            return []
        
        self._in_cold_start = False
        theme = self._current_theme or self._get_theme(dt)
        candidates = []
        spy_mom60 = self._get('SPY', dt, 'mom60') or 0
        
        focus_sectors = set(theme.get("focus_sectors", []))
        focus_stocks = set(theme.get("focus_stocks", []))
        avoid_sectors = set(theme.get("avoid_sectors", []))
        sector_bonus_map = theme.get("sector_bonus", {})
        
        for sym in INITIAL_UNIVERSE:
            if sym not in self._prices:
                continue
            
            close = self._get(sym, dt, 'close')
            sma50 = self._get(sym, dt, 'sma50')
            sma200 = self._get(sym, dt, 'sma200')
            rsi = self._get(sym, dt, 'rsi')
            mom20 = self._get(sym, dt, 'mom20')
            mom60 = self._get(sym, dt, 'mom60')
            
            if close is None or sma200 is None:
                continue
            
            sector_etf = SYMBOL_TO_SECTOR.get(sym, "XLK")
            
            if sector_etf in avoid_sectors:
                continue
            
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
            
            theme_bonus = sector_bonus_map.get(sector_etf, 0)
            
            focus_bonus = 0
            if sym in focus_stocks:
                focus_bonus = 3
            elif sector_etf in focus_sectors:
                focus_bonus = 1
            
            total_score = quant_score + momentum_score + theme_bonus + focus_bonus
            
            candidates.append(DynamicLeader(
                sym, quant_score, theme_bonus, focus_bonus,
                momentum_score, total_score, sector_etf
            ))
        
        candidates.sort(key=lambda x: -x.total_score)
        
        sector_count: Dict[str, int] = {}
        final_leaders = []
        for c in candidates[:15]:
            if len(final_leaders) >= 10:
                break
            count = sector_count.get(c.sector_etf, 0)
            if count >= 4:
                continue
            final_leaders.append(c)
            sector_count[c.sector_etf] = count + 1
        
        return final_leaders
    
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
        to_sell = []
        for sym, pos in self.positions.items():
            if pos.is_index_etf:
                continue
            price = self._get(sym, dt, 'close')
            stop = self._calc_stop_price(pos, dt, regime)
            if price and price < stop:
                pnl_pct = (price - pos.avg_cost) / pos.avg_cost
                to_sell.append((sym, f"止损({pnl_pct:+.1%})"))
        for sym, reason in to_sell:
            self._sell(sym, dt, reason)
    
    def _liquidate_weak_positions(self, dt: date):
        to_sell = []
        for sym, pos in self.positions.items():
            if pos.is_index_etf:
                continue
            price = self._get(sym, dt, 'close') or pos.avg_cost
            pnl_pct = (price - pos.avg_cost) / pos.avg_cost
            if pnl_pct < 0.10:
                to_sell.append((sym, pnl_pct))
        to_sell.sort(key=lambda x: x[1])
        for sym, pnl in to_sell[:max(1, len(to_sell) // 2)]:
            self._sell(sym, dt, f"风控清仓({pnl:+.1%})")
    
    def _should_update_universe(self, dt: date) -> bool:
        if self._last_universe_update is None:
            return True
        return (dt - self._last_universe_update).days >= 14
    
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
            self._liquidate_weak_positions(dt)
            return
        
        for sym in list(self.positions.keys()):
            if sym not in leaders and not self.positions[sym].is_index_etf:
                self._sell(sym, dt, "轮出龙头池")
        
        stock_budget = pv * macro.target_exposure
        current_val = sum(p.shares * (self._get(s, dt, 'close') or p.avg_cost) 
                         for s, p in self.positions.items() if not p.is_index_etf)
        
        if current_val < stock_budget * 0.85:
            available = min(stock_budget - current_val, self.cash * 0.95)
            max_pos = 6 if regime == "offensive" else 5
            current_count = len([p for p in self.positions.values() if not p.is_index_etf])
            
            for sym in leaders:
                if sym in self.positions or current_count >= max_pos:
                    continue
                close = self._get(sym, dt, 'close')
                sma50 = self._get(sym, dt, 'sma50')
                mom20 = self._get(sym, dt, 'mom20')
                if not close or not sma50:
                    continue
                if close < sma50 * 0.92 or (mom20 and mom20 < -0.12):
                    continue
                
                budget = self._calc_position_size(sym, dt, available)
                if self._buy(sym, dt, budget, f"龙头买入({regime})"):
                    current_count += 1
                    available -= budget
    
    def run(self, start: date, end: date) -> dict:
        print("\n" + "=" * 70)
        print("V7.1 Rule-Based Theme 策略回测 (规则生成主题版)")
        print("=" * 70)
        print("  核心改进:")
        print("    1. 规则自动生成季度主题 (基于历史事件)")
        print("    2. 对比 V7.0 人工主题的有效性")
        print("    3. 为 LLM 分析打基础")
        
        self._load_data(start, end)
        
        trading_days = sorted([d for d in self._prices['SPY'].index.tolist() if start <= d <= end])
        actual_start = trading_days[0]
        
        print(f"\n  回测区间: {actual_start} ~ {end}")
        print(f"  交易日数: {len(trading_days)}")
        print(f"  初始资金: ${self.initial_capital:,.0f}")
        
        last_quarter = None
        self._spy_start_price = self._get('SPY', actual_start, 'close') or 1
        
        for i, dt in enumerate(trading_days):
            current_quarter = get_quarter_key(dt)
            self._current_macro = self._analyze_macro(dt)
            
            if current_quarter != last_quarter:
                theme = self._get_theme(dt)
                breaker_info = f" [{self._current_macro.circuit_breaker.level.upper()}]" if self._current_macro.circuit_breaker and self._current_macro.circuit_breaker.level != "normal" else ""
                print(f"\n  📊 [{current_quarter}] 主题: {theme['theme']}{breaker_info}")
                print(f"      焦点板块: {theme['focus_sectors']} | 焦点股票: {theme['focus_stocks'][:5]}")
                self.macro_history.append(self._current_macro)
                last_quarter = current_quarter
            
            if self._should_update_universe(dt):
                leaders = self._build_dynamic_universe(dt)
                self.current_leaders = [l.symbol for l in leaders]
                self.leader_history.append({
                    "date": str(dt), 
                    "cold_start": self._in_cold_start,
                    "theme": self._current_theme["theme"] if self._current_theme else "",
                    "leaders": [{"symbol": l.symbol, "score": l.total_score} for l in leaders]
                })
                self._last_universe_update = dt
                
                if self._in_cold_start:
                    self._cold_start_days += 1
                elif leaders:
                    print(f"  🔄 [{dt}] 龙头池: {', '.join(self.current_leaders[:6])}")
            
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
        losses = [t for t in sells if t.pnl < 0]
        profit_factor = sum(t.pnl for t in wins) / abs(sum(t.pnl for t in losses)) if losses else float('inf')
        
        return {
            "final_value": final, "total_return": total_ret, "annualized_return": ann_ret,
            "spy_return": spy_ret, "alpha": total_ret - spy_ret, "max_drawdown": max_dd,
            "sharpe": sharpe, "win_rate": win_rate, "profit_factor": profit_factor,
            "total_trades": len(self.trades), "stock_trades": len(sells),
            "avg_win": np.mean([t.pnl for t in wins]) if wins else 0,
            "avg_loss": np.mean([t.pnl for t in losses]) if losses else 0,
            "cold_start_months": self._cold_start_days,
            "universe_updates": len(self.leader_history),
        }


def main():
    bt = V71BacktestEngine(100000.0)
    result = bt.run(date(2023, 1, 3), date(2026, 1, 16))
    
    print("\n" + "=" * 70)
    print("V7.1 回测结果 (3年)")
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
    print(f"\n  龙头池更新: {result['universe_updates']} 次")
    
    output = Path("storage/backtest_v7_1")
    output.mkdir(parents=True, exist_ok=True)
    with open(output / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    
    trades_data = [{"date": t.date, "symbol": t.symbol, "action": t.action, 
                    "price": t.price, "shares": t.shares, "pnl": t.pnl, 
                    "pnl_pct": t.pnl_pct, "reason": t.reason} for t in bt.trades]
    with open(output / "trades.json", "w") as f:
        json.dump(trades_data, f, indent=2)
    
    pd.DataFrame(bt.equity_curve, columns=['date', 'portfolio', 'spy']).to_csv(output / "equity_curve.csv", index=False)
    
    with open(output / "quarterly_themes.json", "w") as f:
        json.dump(QUARTERLY_THEMES, f, indent=2, ensure_ascii=False)
    
    print(f"\n📁 保存到: {output}")
    
    sells = [t for t in bt.trades if t.action == "SELL" and t.symbol not in ["QQQ", "SPY"]]
    print("\n【最大盈利】")
    for t in sorted(sells, key=lambda x: -x.pnl)[:5]:
        print(f"  {t.date} {t.symbol}: ${t.pnl:+,.0f} ({t.pnl_pct:+.1%})")
    print("\n【最大亏损】")
    for t in sorted(sells, key=lambda x: x.pnl)[:5]:
        print(f"  {t.date} {t.symbol}: ${t.pnl:+,.0f} ({t.pnl_pct:+.1%})")
    
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
    
    print("\n" + "=" * 70)
    print("【V7.0 vs V7.1 策略对比】")
    print("=" * 70)
    
    # 加载 V7.0 结果进行对比
    v70_result_file = Path("storage/backtest_v7_0/result.json")
    if v70_result_file.exists():
        with open(v70_result_file) as f:
            v70 = json.load(f)
        
        print(f"\n  {'指标':<12} {'V7.0(人工)':<15} {'V7.1(规则)':<15} {'差异':<10}")
        print(f"  {'-'*52}")
        print(f"  {'总收益':<12} {v70['total_return']:+.1%}{'':<7} {result['total_return']:+.1%}{'':<7} {(result['total_return']-v70['total_return'])*100:+.1f}pp")
        print(f"  {'Alpha':<12} {v70['alpha']:+.1%}{'':<7} {result['alpha']:+.1%}{'':<7} {(result['alpha']-v70['alpha'])*100:+.1f}pp")
        print(f"  {'回撤':<12} {v70['max_drawdown']:.1%}{'':<8} {result['max_drawdown']:.1%}{'':<8} {(result['max_drawdown']-v70['max_drawdown'])*100:+.1f}pp")
        print(f"  {'夏普':<12} {v70['sharpe']:.2f}{'':<10} {result['sharpe']:.2f}{'':<10} {result['sharpe']-v70['sharpe']:+.2f}")
        print(f"  {'胜率':<12} {v70['win_rate']:.1%}{'':<8} {result['win_rate']:.1%}{'':<8} {(result['win_rate']-v70['win_rate'])*100:+.1f}pp")
        
        # 判断结果
        print("\n  📊 结论:")
        alpha_diff = (result['alpha'] - v70['alpha']) * 100
        if alpha_diff > 1:
            print(f"     ✅ 规则主题优于人工主题 (Alpha +{alpha_diff:.1f}pp)")
        elif alpha_diff < -1:
            print(f"     ❌ 规则主题不如人工主题 (Alpha {alpha_diff:.1f}pp)")
        else:
            print(f"     ➖ 规则主题与人工主题相当 (Alpha {alpha_diff:+.1f}pp)")
    else:
        print("\n  ⚠️ V7.0 结果不存在，请先运行 run_backtest_v7_0.py")


if __name__ == "__main__":
    main()
