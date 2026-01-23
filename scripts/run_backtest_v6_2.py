#!/usr/bin/env python3
"""
V6.2 "Neuro-Adaptive Pro" 交易系统回测

四大改进:
1. RRG动态板块轮动: 废除硬编码SECTOR_WEIGHT，跟随资金流向
2. 诚实冷启动: 删除静态池回退，冷启动期持有QQQ/SPY
3. 波动率目标仓位: ATR动态调整仓位大小，控制回撤
4. LLM否决权: 新闻风险过滤(模拟版，真实版需API)

目标: 年化收益 > 20%, 最大回撤 < 15%, Alpha > 10%
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
from psycopg2.extras import RealDictCursor

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# V6.2 核心配置
# ============================================================

# 板块ETF列表 (用于RRG计算)
SECTOR_ETFS = {
    "XLK": "科技", "XLC": "通讯", "XLY": "可选消费",
    "XLF": "金融", "XLV": "医疗", "XLE": "能源",
    "XLI": "工业", "XLP": "必需消费", "XLU": "公用事业",
}

# 股票到板块的映射
SYMBOL_TO_SECTOR = {
    # 科技
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "AVGO": "XLK", "AMD": "XLK",
    "ADBE": "XLK", "CRM": "XLK", "ORCL": "XLK", "CSCO": "XLK", "INTC": "XLK",
    # 通讯
    "META": "XLC", "GOOGL": "XLC", "GOOG": "XLC", "NFLX": "XLC", "DIS": "XLC",
    "CMCSA": "XLC", "T": "XLC", "VZ": "XLC", "TMUS": "XLC",
    # 可选消费
    "AMZN": "XLY", "TSLA": "XLY", "HD": "XLY", "MCD": "XLY",
    "NKE": "XLY", "SBUX": "XLY", "LOW": "XLY", "TJX": "XLY",
    # 金融
    "JPM": "XLF", "BAC": "XLF", "WFC": "XLF", "GS": "XLF", "MS": "XLF", "BLK": "XLF",
    # 医疗
    "UNH": "XLV", "JNJ": "XLV", "LLY": "XLV", "PFE": "XLV", "MRK": "XLV", "ABBV": "XLV",
    # 能源
    "XOM": "XLE", "CVX": "XLE", "COP": "XLE",
}

# 熔断规则
CIRCUIT_BREAKER = {
    "vix_danger": 28,
    "vix_caution": 22,
    "vix_watch": 20,
    "vix_rising_fast": 0.20,
    "market_crash_pct": 0.02,
    "cooldown_danger": 10,
    "cooldown_caution": 5,
    "cooldown_watch": 3,
    "recovery_vix": 18,
}

# ATR 止损乘数
ATR_MULTIPLIER = {
    "offensive": 5.0,
    "neutral": 4.0,
    "defensive": 2.5,
}

# 波动率目标仓位参数 (改进3)
VOLATILITY_TARGET = {
    "target_risk_per_trade": 0.01,  # 每笔交易风险 1%
    "max_position_pct": 0.20,       # 单只股票最大仓位 20%
    "min_position_pct": 0.05,       # 单只股票最小仓位 5%
}

# 利润锁定层级
PROFIT_LOCK_TIERS = [
    {"threshold": 0.30, "lock_pct": 0.90},
    {"threshold": 0.15, "lock_pct": 1.02},
]

# 最小止损距离
MIN_STOP_DISTANCE = 0.12

# 初始股票池
INITIAL_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AVGO", "AMD", "ADBE", "CRM", "ORCL", "CSCO", "INTC",
    "META", "GOOGL", "GOOG", "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS",
    "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "TJX",
    "JPM", "BAC", "WFC", "GS", "MS", "BLK",
    "UNH", "JNJ", "LLY", "PFE", "MRK", "ABBV",
    "XOM", "CVX", "COP",
]

# LLM黑名单模拟 (改进4: 真实版需调用API)
# 这里模拟一些历史上的暴雷事件
LLM_BLACKLIST_EVENTS = {
    # 格式: "YYYY-MM": ["SYMBOL", ...]
    # 模拟: 这些日期前后，LLM检测到负面新闻
}


@dataclass
class RRGScore:
    """RRG (相对强弱图谱) 评分"""
    etf: str
    sector_name: str
    rs: float           # 相对强度 (vs SPY)
    rs_momentum: float  # 相对动量
    quadrant: str       # Leading/Weakening/Lagging/Improving
    score: int          # 动态评分


@dataclass
class CircuitBreakerState:
    """熔断状态"""
    level: str = "normal"
    trigger_date: Optional[str] = None
    trigger_reason: str = ""
    cooldown_until: Optional[str] = None


@dataclass
class MacroView:
    """宏观视图"""
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
    """持仓"""
    symbol: str
    shares: int
    avg_cost: float
    entry_date: str
    highest_price: float
    atr_at_entry: float
    is_index_etf: bool = False  # 是否为指数ETF (QQQ/SPY)


@dataclass
class Trade:
    """交易记录"""
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
    """动态龙头"""
    symbol: str
    quant_score: float
    rrg_score: float      # RRG板块评分 (改进1)
    momentum_score: float
    total_score: float
    sector_etf: str
    reason: str


class V62BacktestEngine:
    """V6.2 Neuro-Adaptive Pro 回测引擎"""
    
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
        self.rrg_history: List[Dict] = []  # RRG历史
        
        self.circuit_breaker = CircuitBreakerState()
        self._current_macro: Optional[MacroView] = None
        
        # 冷启动统计
        self._cold_start_days = 0
        self._in_cold_start = False
    
    def _load_data(self, start: date, end: date):
        """加载数据"""
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
            
            # RRG需要的相对强度计算
            sdf['rs_ratio'] = None  # 将在运行时计算
            sdf['rs_momentum'] = None
            
            self._prices[sym] = sdf
        
        print(f"    已加载 {len(self._prices)} 只标的")
    
    def _calc_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calc_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        high = df['high']
        low = df['low']
        close = df['close']
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
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
    # 改进1: RRG动态板块轮动
    # ================================================================
    
    def _calc_rrg_scores(self, dt: date) -> Dict[str, RRGScore]:
        """
        计算RRG (相对强弱图谱) 评分
        
        RS = ETF 60日收益 / SPY 60日收益
        RS_Momentum = RS 的 10日变化率
        
        象限:
        - Leading (领先): RS > 100 且 Momentum > 0 -> +3分
        - Weakening (减弱): RS > 100 且 Momentum < 0 -> +1分
        - Lagging (落后): RS < 100 且 Momentum < 0 -> -2分
        - Improving (改善): RS < 100 且 Momentum > 0 -> 0分
        """
        rrg_scores = {}
        
        spy_mom60 = self._get('SPY', dt, 'mom60') or 0.001  # 避免除零
        
        for etf, sector_name in SECTOR_ETFS.items():
            if etf not in self._prices:
                continue
            
            etf_mom60 = self._get(etf, dt, 'mom60')
            etf_mom60_prev = self._get_prev(etf, dt, 'mom60', 10)
            
            if etf_mom60 is None:
                continue
            
            # RS = 相对强度 (以100为基准)
            rs = ((1 + etf_mom60) / (1 + spy_mom60)) * 100
            
            # RS Momentum = RS的变化
            if etf_mom60_prev is not None:
                rs_prev = ((1 + etf_mom60_prev) / (1 + spy_mom60)) * 100
                rs_momentum = rs - rs_prev
            else:
                rs_momentum = 0
            
            # 确定象限和评分
            if rs > 100 and rs_momentum > 0:
                quadrant = "Leading"
                score = 3
            elif rs > 100 and rs_momentum <= 0:
                quadrant = "Weakening"
                score = 1
            elif rs <= 100 and rs_momentum > 0:
                quadrant = "Improving"
                score = 0
            else:  # rs <= 100 and rs_momentum <= 0
                quadrant = "Lagging"
                score = -2
            
            rrg_scores[etf] = RRGScore(
                etf=etf,
                sector_name=sector_name,
                rs=rs,
                rs_momentum=rs_momentum,
                quadrant=quadrant,
                score=score
            )
        
        return rrg_scores
    
    # ================================================================
    # 改进4: LLM否决权 (模拟版)
    # ================================================================
    
    def _llm_veto_check(self, symbol: str, dt: date) -> Tuple[bool, str]:
        """
        LLM否决权检查 (模拟版)
        
        真实版本应该:
        1. 调用新闻API获取最近3天新闻
        2. 调用LLM分析是否存在致命风险
        3. 返回YES/NO
        
        这里使用模拟黑名单
        """
        month_key = dt.strftime("%Y-%m")
        
        if month_key in LLM_BLACKLIST_EVENTS:
            if symbol in LLM_BLACKLIST_EVENTS[month_key]:
                return True, "LLM检测到负面新闻风险"
        
        # 模拟: 随机概率检测 (实际应该调用API)
        # 这里不做随机，保持回测可重复性
        
        return False, ""
    
    # ================================================================
    # 熔断检查器
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
        elif vix > CIRCUIT_BREAKER["vix_caution"] and vix_5d_change > 0.10:
            new_level = "caution"
            trigger_reason = f"VIX警戒({vix:.1f})"
            cooldown_days = CIRCUIT_BREAKER["cooldown_caution"]
        elif spy_close < spy_sma200 * 0.98:
            new_level = "caution"
            trigger_reason = f"SPY跌破SMA200"
            cooldown_days = CIRCUIT_BREAKER["cooldown_caution"]
        elif vix > CIRCUIT_BREAKER["vix_watch"] and vix_5d_change > 0.05:
            new_level = "watch"
            trigger_reason = f"VIX观察({vix:.1f})"
            cooldown_days = CIRCUIT_BREAKER["cooldown_watch"]
        
        level_order = {"normal": 0, "watch": 1, "caution": 2, "danger": 3}
        if level_order.get(new_level, 0) > level_order.get(self.circuit_breaker.level, 0):
            cooldown = dt + timedelta(days=cooldown_days)
            self.circuit_breaker = CircuitBreakerState(
                level=new_level,
                trigger_date=str(dt),
                trigger_reason=trigger_reason,
                cooldown_until=str(cooldown)
            )
        
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
        
        if breaker.level == "danger":
            return MacroView(
                date=str(dt), market_regime="defensive", target_exposure=0.2,
                vix_level=vix, vix_5d_change=vix_5d_change, spy_change_1d=spy_change,
                spy_vs_sma200=(spy_close / spy_sma200 - 1) if spy_sma200 else 0,
                score=-99, reasoning=f"🚨DANGER: {breaker.trigger_reason}",
                circuit_breaker=breaker
            )
        elif breaker.level == "caution":
            return MacroView(
                date=str(dt), market_regime="defensive", target_exposure=0.4,
                vix_level=vix, vix_5d_change=vix_5d_change, spy_change_1d=spy_change,
                spy_vs_sma200=(spy_close / spy_sma200 - 1) if spy_sma200 else 0,
                score=-50, reasoning=f"⚠️CAUTION: {breaker.trigger_reason}",
                circuit_breaker=breaker
            )
        elif breaker.level == "watch":
            return MacroView(
                date=str(dt), market_regime="neutral", target_exposure=0.6,
                vix_level=vix, vix_5d_change=vix_5d_change, spy_change_1d=spy_change,
                spy_vs_sma200=(spy_close / spy_sma200 - 1) if spy_sma200 else 0,
                score=-20, reasoning=f"👀WATCH: {breaker.trigger_reason}",
                circuit_breaker=breaker
            )
        
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
            regime = "offensive"
            target_exposure = 1.0
        elif score >= 0:
            regime = "neutral"
            target_exposure = 0.7
        else:
            regime = "defensive"
            target_exposure = 0.4
        
        return MacroView(
            date=str(dt), market_regime=regime, target_exposure=target_exposure,
            vix_level=vix, vix_5d_change=vix_5d_change, spy_change_1d=spy_change,
            spy_vs_sma200=(spy_close / spy_sma200 - 1) if spy_sma200 else 0,
            score=score, reasoning=" | ".join(reasoning_parts),
            circuit_breaker=breaker
        )
    
    # ================================================================
    # 改进2: 诚实冷启动
    # ================================================================
    
    def _check_cold_start(self, dt: date) -> bool:
        """
        检查是否处于冷启动期
        条件: SMA200 数据不足
        """
        # 检查关键股票的SMA200是否可用
        key_symbols = ["NVDA", "AAPL", "MSFT", "META", "GOOGL"]
        ready_count = 0
        
        for sym in key_symbols:
            sma200 = self._get(sym, dt, 'sma200')
            if sma200 is not None:
                ready_count += 1
        
        return ready_count < 3  # 至少3只关键股票有SMA200
    
    def _build_dynamic_universe(self, dt: date, rrg_scores: Dict[str, RRGScore]) -> List[DynamicLeader]:
        """
        构建动态龙头池 (改进版)
        
        改进1: 使用RRG动态板块评分
        改进2: 冷启动时返回空列表 (由调用方处理)
        改进4: LLM否决权检查
        """
        # 检查冷启动
        if self._check_cold_start(dt):
            self._in_cold_start = True
            return []  # 返回空，让调用方持有QQQ
        
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
            
            # 获取板块ETF
            sector_etf = SYMBOL_TO_SECTOR.get(sym, "XLK")
            
            # ========== Quant 评分 ==========
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
            
            # ========== 动量评分 ==========
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
            
            # ========== RRG板块评分 (改进1) ==========
            rrg_score = 0
            if sector_etf in rrg_scores:
                rrg_score = rrg_scores[sector_etf].score
            
            # ========== 总分 ==========
            total_score = quant_score + momentum_score + rrg_score
            
            candidates.append(DynamicLeader(
                symbol=sym,
                quant_score=quant_score,
                rrg_score=rrg_score,
                momentum_score=momentum_score,
                total_score=total_score,
                sector_etf=sector_etf,
                reason=f"RS:{rs:.2f}, RRG:{rrg_score}"
            ))
        
        # 排序取Top 15 (给LLM筛选留余量)
        candidates.sort(key=lambda x: -x.total_score)
        top_candidates = candidates[:15]
        
        # ========== LLM否决权检查 (改进4) ==========
        final_leaders = []
        for c in top_candidates:
            vetoed, reason = self._llm_veto_check(c.symbol, dt)
            if not vetoed:
                final_leaders.append(c)
            # else:
            #     print(f"    ⛔ LLM否决: {c.symbol} - {reason}")
        
        # 板块分散: 每个板块最多3只
        sector_count: Dict[str, int] = {}
        dispersed_leaders = []
        
        for c in final_leaders:
            if len(dispersed_leaders) >= 10:
                break
            count = sector_count.get(c.sector_etf, 0)
            if count >= 3:
                continue
            dispersed_leaders.append(c)
            sector_count[c.sector_etf] = count + 1
        
        return dispersed_leaders
    
    # ================================================================
    # 改进3: 波动率目标仓位
    # ================================================================
    
    def _calc_position_size(self, sym: str, dt: date, available_capital: float) -> float:
        """
        波动率目标仓位计算
        
        公式: PositionSize = (TotalCapital × TargetRisk%) / (ATR × 5)
        
        效果: 高波动股票仓位小，低波动股票仓位大
        """
        atr = self._get(sym, dt, 'atr')
        price = self._get(sym, dt, 'close')
        
        if not atr or not price or atr <= 0:
            # 无法计算ATR，使用默认仓位
            return available_capital * 0.10  # 10%
        
        # ATR止损距离 (5倍ATR)
        stop_distance = atr * 5
        stop_distance_pct = stop_distance / price
        
        # 目标风险 = 每笔交易风险1%
        target_risk = self.initial_capital * VOLATILITY_TARGET["target_risk_per_trade"]
        
        # 仓位大小 = 目标风险 / 止损距离百分比
        position_value = target_risk / stop_distance_pct if stop_distance_pct > 0 else 0
        
        # 限制仓位范围
        max_position = self.initial_capital * VOLATILITY_TARGET["max_position_pct"]
        min_position = self.initial_capital * VOLATILITY_TARGET["min_position_pct"]
        
        position_value = max(min_position, min(position_value, max_position))
        position_value = min(position_value, available_capital)  # 不超过可用资金
        
        return position_value
    
    def _calc_stop_price(self, pos: Position, dt: date, regime: str) -> float:
        if pos.is_index_etf:
            return 0  # 指数ETF不设止损
        
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
                if tier["lock_pct"] > 1:
                    profit_stop = pos.avg_cost * tier["lock_pct"]
                else:
                    profit_stop = pos.highest_price * tier["lock_pct"]
                break
        
        return max(atr_stop, profit_stop)
    
    # ================================================================
    # 交易执行
    # ================================================================
    
    def _portfolio_value(self, dt: date) -> float:
        pos_val = sum(
            p.shares * (self._get(s, dt, 'close') or p.avg_cost)
            for s, p in self.positions.items()
        )
        return self.cash + pos_val
    
    def _buy(self, sym: str, dt: date, budget: float, reason: str, is_index_etf: bool = False) -> bool:
        price = self._get(sym, dt, 'close')
        if not price or budget < 500:
            return False
        
        shares = int(budget / price)
        if shares <= 0:
            return False
        
        cost = shares * price
        if cost > self.cash:
            return False
        
        self.cash -= cost
        atr = self._get(sym, dt, 'atr') or (price * 0.02)
        
        if sym in self.positions:
            p = self.positions[sym]
            total = p.shares + shares
            p.avg_cost = (p.avg_cost * p.shares + price * shares) / total
            p.shares = total
            p.highest_price = max(p.highest_price, price)
        else:
            self.positions[sym] = Position(
                symbol=sym, shares=shares, avg_cost=price,
                entry_date=str(dt), highest_price=price,
                atr_at_entry=atr, is_index_etf=is_index_etf
            )
        
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
            if not price:
                continue
            
            stop_price = self._calc_stop_price(pos, dt, regime)
            
            if price < stop_price:
                pnl_pct = (price - pos.avg_cost) / pos.avg_cost
                to_sell.append((sym, f"止损触发(${stop_price:.2f}, {pnl_pct:+.1%})"))
        
        for sym, reason in to_sell:
            self._sell(sym, dt, reason)
    
    def _liquidate_weak_positions(self, dt: date, keep_pct: float = 0.15):
        to_sell = []
        
        for sym, pos in self.positions.items():
            if pos.is_index_etf:
                continue
            
            price = self._get(sym, dt, 'close') or pos.avg_cost
            pnl_pct = (price - pos.avg_cost) / pos.avg_cost
            
            if pnl_pct < keep_pct:
                to_sell.append((sym, pnl_pct, f"风控清仓({pnl_pct:+.1%})"))
        
        to_sell.sort(key=lambda x: x[1])
        max_sell = max(1, len(to_sell) // 2)
        for sym, _, reason in to_sell[:max_sell]:
            self._sell(sym, dt, reason)
    
    def _rebalance(self, dt: date, macro: MacroView, leaders: List[str]):
        pv = self._portfolio_value(dt)
        regime = macro.market_regime
        
        # 检查是否在冷启动期
        if self._in_cold_start:
            # 冷启动期: 持有QQQ (改进2)
            if "QQQ" not in self.positions:
                # 卖出其他持仓
                for sym in list(self.positions.keys()):
                    self._sell(sym, dt, "冷启动切换QQQ")
                
                # 买入QQQ
                budget = self.cash * 0.95
                self._buy("QQQ", dt, budget, "冷启动持有QQQ", is_index_etf=True)
            return
        
        # 正常模式: 如果持有QQQ，卖出
        if "QQQ" in self.positions:
            self._sell("QQQ", dt, "退出冷启动")
        
        # 风控减仓
        if regime == "defensive":
            self._liquidate_weak_positions(dt, keep_pct=0.15)
            return
        
        # 卖出不在龙头池的持仓
        for sym in list(self.positions.keys()):
            if sym not in leaders and not self.positions[sym].is_index_etf:
                self._sell(sym, dt, "轮出龙头池")
        
        stock_exposure = macro.target_exposure
        stock_budget = pv * stock_exposure
        
        current_stock_value = sum(
            p.shares * (self._get(s, dt, 'close') or p.avg_cost)
            for s, p in self.positions.items()
            if not p.is_index_etf
        )
        
        if current_stock_value < stock_budget * 0.85:
            available = min(stock_budget - current_stock_value, self.cash * 0.95)
            
            max_positions = 6 if regime == "offensive" else 4
            current_positions = len([p for p in self.positions.values() if not p.is_index_etf])
            
            for sym in leaders:
                if sym in self.positions:
                    continue
                if current_positions >= max_positions:
                    break
                
                close = self._get(sym, dt, 'close')
                sma50 = self._get(sym, dt, 'sma50')
                mom20 = self._get(sym, dt, 'mom20')
                
                if not close or not sma50:
                    continue
                if close < sma50 * 0.92:
                    continue
                if mom20 and mom20 < -0.12:
                    continue
                
                # 使用波动率目标仓位 (改进3)
                position_budget = self._calc_position_size(sym, dt, available)
                
                if self._buy(sym, dt, position_budget, f"龙头买入({regime})"):
                    current_positions += 1
                    available -= position_budget
    
    # ================================================================
    # 主运行循环
    # ================================================================
    
    def run(self, start: date, end: date) -> dict:
        print("\n" + "=" * 70)
        print("V6.2 Neuro-Adaptive Pro 策略回测")
        print("=" * 70)
        print("  四大改进:")
        print("    1. RRG动态板块轮动: 废除硬编码，跟随资金流向")
        print("    2. 诚实冷启动: 数据不足时持有QQQ")
        print("    3. 波动率目标仓位: ATR动态调整仓位大小")
        print("    4. LLM否决权: 新闻风险过滤")
        
        self._load_data(start, end)
        
        if 'SPY' not in self._prices:
            raise ValueError("SPY 数据缺失")
        
        trading_days = sorted(self._prices['SPY'].index.tolist())
        trading_days = [d for d in trading_days if start <= d <= end]
        
        if len(trading_days) == 0:
            raise ValueError(f"没有交易日在 {start} ~ {end} 范围内")
        
        actual_start = trading_days[0]
        
        print(f"\n  回测区间: {actual_start} ~ {end}")
        print(f"  交易日数: {len(trading_days)}")
        print(f"  初始资金: ${self.initial_capital:,.0f}")
        
        last_macro_month = None
        last_universe_month = None
        
        self._spy_start_price = self._get('SPY', actual_start, 'close') or 1
        
        for i, dt in enumerate(trading_days):
            current_month = dt.strftime("%Y-%m")
            
            # 每日宏观分析
            self._current_macro = self._analyze_macro(dt)
            
            # 月初记录宏观状态
            if current_month != last_macro_month:
                self.macro_history.append(self._current_macro)
                last_macro_month = current_month
                
                if i % 50 == 0 or len(self.macro_history) <= 3:
                    breaker_info = ""
                    if self._current_macro.circuit_breaker and self._current_macro.circuit_breaker.level != "normal":
                        breaker_info = f" [{self._current_macro.circuit_breaker.level.upper()}]"
                    print(f"\n  📊 [{dt}] {self._current_macro.market_regime}{breaker_info} "
                          f"(分数:{self._current_macro.score}, 仓位:{self._current_macro.target_exposure:.0%}) "
                          f"- {self._current_macro.reasoning}")
            
            # 月度更新龙头池
            if current_month != last_universe_month:
                # 计算RRG评分 (改进1)
                rrg_scores = self._calc_rrg_scores(dt)
                
                # 记录RRG状态
                self.rrg_history.append({
                    "date": str(dt),
                    "scores": {etf: {"quadrant": s.quadrant, "score": s.score, "rs": s.rs} 
                              for etf, s in rrg_scores.items()}
                })
                
                leaders = self._build_dynamic_universe(dt, rrg_scores)
                self.current_leaders = [l.symbol for l in leaders]
                self.leader_history.append({
                    "date": str(dt),
                    "cold_start": self._in_cold_start,
                    "leaders": [{"symbol": l.symbol, "score": l.total_score, 
                                "rrg": l.rrg_score, "sector": l.sector_etf} for l in leaders]
                })
                last_universe_month = current_month
                
                if self._in_cold_start:
                    self._cold_start_days += 1
                    print(f"  ❄️ [{dt}] 冷启动模式: 持有QQQ")
                elif len(leaders) > 0:
                    # 显示RRG领先板块
                    leading_sectors = [s.sector_name for s in rrg_scores.values() if s.quadrant == "Leading"]
                    print(f"  🔄 [{dt}] 龙头池: {', '.join(self.current_leaders[:6])}")
                    if leading_sectors:
                        print(f"      RRG领先板块: {', '.join(leading_sectors)}")
            
            # 每日止损检查
            self._check_stops(dt, self._current_macro.market_regime)
            
            # 每5天再平衡
            if i % 5 == 0:
                self._rebalance(dt, self._current_macro, self.current_leaders)
            
            pv = self._portfolio_value(dt)
            spy_price = self._get('SPY', dt, 'close') or 0
            spy_val = self.initial_capital * spy_price / self._spy_start_price
            self.equity_curve.append((str(dt), pv, spy_val))
            
            if i % 150 == 0:
                print(f"  [{i+1}/{len(trading_days)}] {dt}: ${pv:,.0f} (SPY: ${spy_val:,.0f})")
        
        return self._calc_results(start, end)
    
    def _calc_results(self, start: date, end: date) -> dict:
        final = self.equity_curve[-1][1]
        spy_final = self.equity_curve[-1][2]
        
        total_ret = final / self.initial_capital - 1
        spy_ret = spy_final / self.initial_capital - 1
        
        years = (end - start).days / 365
        ann_ret = (1 + total_ret) ** (1/years) - 1 if years > 0 else 0
        
        values = [e[1] for e in self.equity_curve]
        peak = self.initial_capital
        max_dd = 0
        for v in values:
            peak = max(peak, v)
            dd = (peak - v) / peak
            max_dd = max(max_dd, dd)
        
        rets = pd.Series(values).pct_change().dropna()
        sharpe = np.sqrt(252) * rets.mean() / rets.std() if rets.std() > 0 else 0
        
        sells = [t for t in self.trades if t.action == "SELL" and t.symbol not in ["QQQ", "SPY"]]
        wins = [t for t in sells if t.pnl > 0]
        win_rate = len(wins) / len(sells) if sells else 0
        
        total_win = sum(t.pnl for t in wins)
        total_loss = abs(sum(t.pnl for t in sells if t.pnl < 0))
        profit_factor = total_win / total_loss if total_loss > 0 else float('inf')
        
        breaker_triggers = {"watch": 0, "caution": 0, "danger": 0}
        for m in self.macro_history:
            if m.circuit_breaker and m.circuit_breaker.level != "normal":
                breaker_triggers[m.circuit_breaker.level] = breaker_triggers.get(m.circuit_breaker.level, 0) + 1
        
        regime_dist = {}
        for m in self.macro_history:
            regime_dist[m.market_regime] = regime_dist.get(m.market_regime, 0) + 1
        
        # RRG板块统计
        rrg_leading_count = {}
        for h in self.rrg_history:
            for etf, data in h["scores"].items():
                if data["quadrant"] == "Leading":
                    rrg_leading_count[etf] = rrg_leading_count.get(etf, 0) + 1
        
        return {
            "final_value": final,
            "total_return": total_ret,
            "annualized_return": ann_ret,
            "spy_return": spy_ret,
            "alpha": total_ret - spy_ret,
            "max_drawdown": max_dd,
            "sharpe": sharpe,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_trades": len(self.trades),
            "stock_trades": len(sells),
            "avg_win": np.mean([t.pnl for t in wins]) if wins else 0,
            "avg_loss": np.mean([t.pnl for t in sells if t.pnl < 0]) if any(t.pnl < 0 for t in sells) else 0,
            "regime_distribution": regime_dist,
            "circuit_breaker_triggers": breaker_triggers,
            "cold_start_months": self._cold_start_days,
            "rrg_leading_sectors": rrg_leading_count,
        }


def main():
    bt = V62BacktestEngine(100000.0)
    result = bt.run(date(2022, 1, 3), date(2026, 1, 16))
    
    print("\n" + "=" * 70)
    print("V6.2 Neuro-Adaptive Pro 回测结果")
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
    print(f"  股票交易: {result['stock_trades']} 笔")
    print(f"  平均盈利: ${result['avg_win']:,.0f}")
    print(f"  平均亏损: ${result['avg_loss']:,.0f}")
    
    print(f"\n  冷启动月数: {result['cold_start_months']}")
    print(f"  熔断触发: {result['circuit_breaker_triggers']}")
    
    print(f"\n  宏观状态分布:")
    for regime, count in result['regime_distribution'].items():
        print(f"    {regime}: {count} 月")
    
    print(f"\n  RRG领先板块统计 (月数):")
    sorted_rrg = sorted(result['rrg_leading_sectors'].items(), key=lambda x: -x[1])
    for etf, count in sorted_rrg[:5]:
        sector_name = SECTOR_ETFS.get(etf, etf)
        print(f"    {sector_name} ({etf}): {count} 月")
    
    # 保存结果
    output = Path("storage/backtest_v6_2")
    output.mkdir(parents=True, exist_ok=True)
    
    with open(output / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    
    trades_data = [
        {"date": t.date, "symbol": t.symbol, "action": t.action,
         "price": t.price, "shares": t.shares, "pnl": t.pnl, 
         "pnl_pct": t.pnl_pct, "reason": t.reason}
        for t in bt.trades
    ]
    with open(output / "trades.json", "w") as f:
        json.dump(trades_data, f, indent=2)
    
    macro_data = [
        {"date": m.date, "regime": m.market_regime, "exposure": m.target_exposure,
         "score": m.score, "vix": m.vix_level, "reasoning": m.reasoning,
         "circuit_breaker_level": m.circuit_breaker.level if m.circuit_breaker else "normal"}
        for m in bt.macro_history
    ]
    with open(output / "macro_history.json", "w") as f:
        json.dump(macro_data, f, indent=2)
    
    with open(output / "leader_history.json", "w") as f:
        json.dump(bt.leader_history, f, indent=2)
    
    with open(output / "rrg_history.json", "w") as f:
        json.dump(bt.rrg_history, f, indent=2)
    
    equity_df = pd.DataFrame(bt.equity_curve, columns=['date', 'portfolio', 'spy'])
    equity_df.to_csv(output / "equity_curve.csv", index=False)
    
    print(f"\n📁 保存到: {output}")
    
    # 最大盈利交易
    print("\n【最大盈利交易】")
    stock_sells = [t for t in bt.trades if t.action == "SELL" and t.symbol not in ["QQQ", "SPY"]]
    top = sorted(stock_sells, key=lambda x: -x.pnl)[:5]
    for t in top:
        print(f"  {t.date} {t.symbol}: ${t.pnl:+,.0f} ({t.pnl_pct:+.1%}) - {t.reason}")
    
    # 最大亏损交易
    print("\n【最大亏损交易】")
    bottom = sorted(stock_sells, key=lambda x: x.pnl)[:5]
    for t in bottom:
        print(f"  {t.date} {t.symbol}: ${t.pnl:+,.0f} ({t.pnl_pct:+.1%}) - {t.reason}")
    
    # 年度收益分解
    print("\n【年度收益分解】")
    equity_df['date'] = pd.to_datetime(equity_df['date'])
    equity_df['year'] = equity_df['date'].dt.year
    
    for year in sorted(equity_df['year'].unique()):
        year_data = equity_df[equity_df['year'] == year]
        if len(year_data) < 2:
            continue
        port_start = year_data['portfolio'].iloc[0]
        port_end = year_data['portfolio'].iloc[-1]
        spy_start = year_data['spy'].iloc[0]
        spy_end = year_data['spy'].iloc[-1]
        
        port_ret = (port_end / port_start - 1) * 100
        spy_ret = (spy_end / spy_start - 1) * 100
        alpha = port_ret - spy_ret
        
        print(f"  {year}: 策略 {port_ret:+.1f}% | SPY {spy_ret:+.1f}% | Alpha {alpha:+.1f}%")
    
    # V6.1 vs V6.2 对比
    print("\n" + "=" * 70)
    print("【V6.1 vs V6.2 对比】")
    print("=" * 70)
    
    v61_results = {
        "total_return": 0.7898,
        "max_drawdown": 0.2036,
        "sharpe": 0.96,
        "win_rate": 0.436,
        "alpha": 0.3419,
    }
    
    print(f"  {'指标':<15} {'V6.1':<15} {'V6.2':<15} {'改进':<15}")
    print(f"  {'-'*60}")
    print(f"  {'总收益率':<15} {v61_results['total_return']:+.2%}{'':>5} {result['total_return']:+.2%}{'':>5} {result['total_return'] - v61_results['total_return']:+.2%}")
    print(f"  {'Alpha':<15} {v61_results['alpha']:+.2%}{'':>5} {result['alpha']:+.2%}{'':>5} {result['alpha'] - v61_results['alpha']:+.2%}")
    print(f"  {'最大回撤':<15} {v61_results['max_drawdown']:.2%}{'':>6} {result['max_drawdown']:.2%}{'':>6} {v61_results['max_drawdown'] - result['max_drawdown']:+.2%}")
    print(f"  {'夏普比率':<15} {v61_results['sharpe']:.2f}{'':>10} {result['sharpe']:.2f}{'':>10} {result['sharpe'] - v61_results['sharpe']:+.2f}")
    print(f"  {'胜率':<15} {v61_results['win_rate']:.1%}{'':>8} {result['win_rate']:.1%}{'':>8} {result['win_rate'] - v61_results['win_rate']:+.1%}")


if __name__ == "__main__":
    main()
