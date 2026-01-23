#!/usr/bin/env python3
"""
V6.1 "Neuro-Adaptive" 交易系统回测 - 修复版

修复的核心漏洞:
1. P0 冷启动问题: 数据不足时回退到静态龙头池
2. P0 价值陷阱: Quant评分增加成长股权重 + 板块加分
3. P1 止损过敏: ATR乘数从3x放宽到5x + 最小12%止损距离
4. P1 熔断失效: 阈值从VIX>30降到25 + 预警机制
5. P2 更新频率: 从季度更新改为月度更新

目标: 年化收益 > 25%, 最大回撤 < 20%
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
# V6.1 核心配置 (修复版)
# ============================================================

# 静态龙头池 - 冷启动回退用
STATIC_TECH_LEADERS = ["NVDA", "META", "GOOGL", "AMZN", "MSFT", "AAPL", "AMD", "AVGO", "NFLX", "TSLA"]

# 板块权重 - 解决价值陷阱
SECTOR_WEIGHT = {
    "科技": 3,       # 成长型加分
    "通讯": 2,
    "可选消费": 2,
    "金融": 0,       # 中性
    "医疗": 1,
    "能源": -1,      # 价值型减分
    "公用事业": -2,
    "必需消费": -1,
    "其他": 0,
}

# 股票到板块的映射
SYMBOL_TO_SECTOR = {
    # 科技
    "AAPL": "科技", "MSFT": "科技", "NVDA": "科技", "AVGO": "科技", "AMD": "科技",
    "ADBE": "科技", "CRM": "科技", "ORCL": "科技", "CSCO": "科技", "INTC": "科技",
    # 通讯
    "META": "通讯", "GOOGL": "通讯", "GOOG": "通讯", "NFLX": "通讯", "DIS": "通讯",
    "CMCSA": "通讯", "T": "通讯", "VZ": "通讯", "TMUS": "通讯",
    # 可选消费
    "AMZN": "可选消费", "TSLA": "可选消费", "HD": "可选消费", "MCD": "可选消费",
    "NKE": "可选消费", "SBUX": "可选消费", "LOW": "可选消费", "TJX": "可选消费",
    # 金融
    "JPM": "金融", "BAC": "金融", "WFC": "金融", "GS": "金融", "MS": "金融", "BLK": "金融",
    # 医疗
    "UNH": "医疗", "JNJ": "医疗", "LLY": "医疗", "PFE": "医疗", "MRK": "医疗", "ABBV": "医疗",
    # 能源
    "XOM": "能源", "CVX": "能源", "COP": "能源",
}

# 熔断规则 (修复: 降低阈值 + 预警)
CIRCUIT_BREAKER = {
    "vix_danger": 28,         # 危险模式 (从30降到28)
    "vix_caution": 22,        # 警戒模式 (新增)
    "vix_watch": 20,          # 观察模式 (新增)
    "vix_rising_fast": 0.20,  # VIX 5日涨幅 > 20% 触发
    "market_crash_pct": 0.02, # 单日跌幅 > 2% (从2.5%降到2%)
    "cooldown_danger": 10,    # 危险模式冷却天数
    "cooldown_caution": 5,    # 警戒模式冷却天数
    "cooldown_watch": 3,      # 观察模式冷却天数
    "recovery_vix": 18,       # VIX < 18 可恢复 (从25降到18)
}

# ATR 止损乘数 (修复: 放宽到5x)
ATR_MULTIPLIER = {
    "offensive": 5.0,   # 从 3.0 放宽到 5.0
    "neutral": 4.0,     # 从 2.0 放宽到 4.0
    "defensive": 2.5,   # 从 1.5 放宽到 2.5
}

# 最小止损距离 (新增)
MIN_STOP_DISTANCE = 0.12  # 至少 12% 止损距离

# 利润锁定层级
PROFIT_LOCK_TIERS = [
    {"threshold": 0.30, "lock_pct": 0.90},  # 30%盈利 → 锁定90%最高价
    {"threshold": 0.15, "lock_pct": 1.02},  # 15%盈利 → 保本+2%
]

# 板块ETF映射
SECTOR_ETFS = {
    "XLK": "科技", "XLC": "通讯", "XLY": "可选消费",
    "XLF": "金融", "XLV": "医疗", "XLE": "能源",
    "XLI": "工业", "XLP": "必需消费", "XLU": "公用事业",
}

INITIAL_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AVGO", "AMD", "ADBE", "CRM", "ORCL", "CSCO", "INTC",
    "META", "GOOGL", "GOOG", "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS",
    "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "TJX",
    "JPM", "BAC", "WFC", "GS", "MS", "BLK",
    "UNH", "JNJ", "LLY", "PFE", "MRK", "ABBV",
    "XOM", "CVX", "COP",
]


@dataclass
class CircuitBreakerState:
    """熔断状态 (增强版)"""
    level: str = "normal"  # normal / watch / caution / danger
    trigger_date: Optional[str] = None
    trigger_reason: str = ""
    cooldown_until: Optional[str] = None


@dataclass
class MacroView:
    """宏观视图"""
    date: str
    market_regime: str  # offensive / neutral / defensive
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
    is_safe_haven: bool = False


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
    sector_score: float
    momentum_score: float
    total_score: float
    sector: str
    reason: str


class V61BacktestEngine:
    """V6.1 Neuro-Adaptive 回测引擎 (修复版)"""
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.conn = psycopg2.connect(
            host=os.getenv("PG_HOST", "192.168.10.11"),
            port=os.getenv("PG_PORT", "5432"),
            database=os.getenv("PG_DATABASE", "trader"),
            user=os.getenv("PG_USER", "trader"),
            password=os.getenv("PG_PASSWORD", "")
        )
        
        # 数据缓存
        self._prices: Dict[str, pd.DataFrame] = {}
        
        # 状态
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[str, float, float]] = []
        self.macro_history: List[MacroView] = []
        
        # 动态龙头池 (月度更新)
        self.current_leaders: List[str] = []
        self.leader_history: List[Dict] = []
        
        # 熔断状态 (增强版)
        self.circuit_breaker = CircuitBreakerState()
        
        # 当前宏观状态
        self._current_macro: Optional[MacroView] = None
        
        # 冷启动计数
        self._cold_start_months = 0
    
    def _load_data(self, start: date, end: date):
        """加载数据"""
        print("  加载价格数据...")
        
        # 构建完整股票池
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
            sdf['vix_5d_change'] = sdf['close'].pct_change(5) if sym == 'VIX' else None
            self._prices[sym] = sdf
        
        print(f"    已加载 {len(self._prices)} 只标的")
    
    def _calc_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """计算 RSI"""
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calc_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """计算 ATR"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        return tr.rolling(period).mean()
    
    def _get(self, sym: str, dt: date, col: str) -> Optional[float]:
        """获取指定日期的数据"""
        if sym not in self._prices:
            return None
        df = self._prices[sym]
        valid = df[df.index <= dt]
        if len(valid) == 0:
            return None
        val = valid[col].iloc[-1]
        return float(val) if pd.notna(val) else None
    
    def _get_prev(self, sym: str, dt: date, col: str, days: int = 1) -> Optional[float]:
        """获取N天前的数据"""
        if sym not in self._prices:
            return None
        df = self._prices[sym]
        valid = df[df.index <= dt]
        if len(valid) < days + 1:
            return None
        val = valid[col].iloc[-(days + 1)]
        return float(val) if pd.notna(val) else None
    
    # ================================================================
    # Phase 1: 增强熔断检查器 (预警机制)
    # ================================================================
    
    def _check_circuit_breaker(self, dt: date) -> CircuitBreakerState:
        """
        增强版熔断检查 - 分级预警
        返回: 熔断状态
        """
        vix = self._get('VIX', dt, 'close') or 20
        vix_5d_ago = self._get_prev('VIX', dt, 'close', 5) or vix
        vix_5d_change = (vix - vix_5d_ago) / vix_5d_ago if vix_5d_ago > 0 else 0
        
        spy_change = self._get('SPY', dt, 'change_1d') or 0
        spy_close = self._get('SPY', dt, 'close') or 0
        spy_sma50 = self._get('SPY', dt, 'sma50') or spy_close
        spy_sma200 = self._get('SPY', dt, 'sma200') or spy_close
        
        # 检查冷却期
        if self.circuit_breaker.cooldown_until:
            cooldown_date = date.fromisoformat(self.circuit_breaker.cooldown_until)
            if dt <= cooldown_date:
                return self.circuit_breaker  # 仍在冷却期
        
        # 检查恢复条件
        if self.circuit_breaker.level != "normal":
            if vix < CIRCUIT_BREAKER["recovery_vix"] and spy_close > spy_sma50:
                # 恢复正常
                self.circuit_breaker = CircuitBreakerState(level="normal")
                return self.circuit_breaker
        
        # 分级检查
        new_level = "normal"
        trigger_reason = ""
        cooldown_days = 0
        
        # 条件1: DANGER - VIX恐慌 或 急升
        if vix > CIRCUIT_BREAKER["vix_danger"]:
            new_level = "danger"
            trigger_reason = f"VIX恐慌({vix:.1f})"
            cooldown_days = CIRCUIT_BREAKER["cooldown_danger"]
        elif vix_5d_change > CIRCUIT_BREAKER["vix_rising_fast"] and vix > 22:
            new_level = "danger"
            trigger_reason = f"VIX急升({vix_5d_change*100:.0f}%)"
            cooldown_days = CIRCUIT_BREAKER["cooldown_danger"]
        # 条件2: DANGER - 单日暴跌
        elif spy_change < -CIRCUIT_BREAKER["market_crash_pct"]:
            new_level = "danger"
            trigger_reason = f"SPY暴跌({spy_change*100:.1f}%)"
            cooldown_days = CIRCUIT_BREAKER["cooldown_danger"]
        # 条件3: CAUTION - VIX 偏高 + 趋势上升
        elif vix > CIRCUIT_BREAKER["vix_caution"] and vix_5d_change > 0.10:
            new_level = "caution"
            trigger_reason = f"VIX警戒({vix:.1f}, +{vix_5d_change*100:.0f}%)"
            cooldown_days = CIRCUIT_BREAKER["cooldown_caution"]
        # 条件4: CAUTION - SPY 跌破年线
        elif spy_close < spy_sma200 * 0.98:
            new_level = "caution"
            trigger_reason = f"SPY跌破SMA200"
            cooldown_days = CIRCUIT_BREAKER["cooldown_caution"]
        # 条件5: WATCH - VIX 上升趋势
        elif vix > CIRCUIT_BREAKER["vix_watch"] and vix_5d_change > 0.05:
            new_level = "watch"
            trigger_reason = f"VIX观察({vix:.1f})"
            cooldown_days = CIRCUIT_BREAKER["cooldown_watch"]
        
        # 只升级不降级 (在冷却期外)
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
    
    # ================================================================
    # Phase 1: 宏观分析 (融合分级熔断)
    # ================================================================
    
    def _analyze_macro(self, dt: date) -> MacroView:
        """宏观分析 (含分级熔断)"""
        
        # 先检查熔断
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
        
        # 根据熔断级别强制状态
        if breaker.level == "danger":
            return MacroView(
                date=str(dt),
                market_regime="defensive",
                target_exposure=0.2,  # 只保留20%
                vix_level=vix,
                vix_5d_change=vix_5d_change,
                spy_change_1d=spy_change,
                spy_vs_sma200=(spy_close / spy_sma200 - 1) if spy_sma200 else 0,
                score=-99,
                reasoning=f"🚨DANGER: {breaker.trigger_reason}",
                circuit_breaker=breaker
            )
        elif breaker.level == "caution":
            return MacroView(
                date=str(dt),
                market_regime="defensive",
                target_exposure=0.4,  # 保留40%
                vix_level=vix,
                vix_5d_change=vix_5d_change,
                spy_change_1d=spy_change,
                spy_vs_sma200=(spy_close / spy_sma200 - 1) if spy_sma200 else 0,
                score=-50,
                reasoning=f"⚠️CAUTION: {breaker.trigger_reason}",
                circuit_breaker=breaker
            )
        elif breaker.level == "watch":
            return MacroView(
                date=str(dt),
                market_regime="neutral",
                target_exposure=0.6,  # 保留60%
                vix_level=vix,
                vix_5d_change=vix_5d_change,
                spy_change_1d=spy_change,
                spy_vs_sma200=(spy_close / spy_sma200 - 1) if spy_sma200 else 0,
                score=-20,
                reasoning=f"👀WATCH: {breaker.trigger_reason}",
                circuit_breaker=breaker
            )
        
        # 正常评分逻辑
        score = 0
        reasoning_parts = []
        
        # VIX 评分
        if vix < 15:
            score += 2
            reasoning_parts.append("VIX极低(贪婪)")
        elif vix < 18:
            score += 1
            reasoning_parts.append("VIX正常")
        elif vix < 22:
            score -= 1
            reasoning_parts.append("VIX偏高")
        else:
            score -= 2
            reasoning_parts.append("VIX警告")
        
        # SPY 趋势评分
        if spy_close > spy_sma50 and spy_close > spy_sma200 and spy_mom > 0.03:
            score += 2
            reasoning_parts.append("SPY强势上涨")
        elif spy_close > spy_sma50 and spy_close > spy_sma200:
            score += 1
            reasoning_parts.append("SPY趋势向上")
        elif spy_close < spy_sma200:
            score -= 2
            reasoning_parts.append("SPY跌破年线")
        elif spy_close < spy_sma50:
            score -= 1
            reasoning_parts.append("SPY跌破50日线")
        
        # VIX 趋势
        if vix < vix_20d_ago * 0.8:
            score += 1
            reasoning_parts.append("VIX下降趋势")
        elif vix > vix_20d_ago * 1.3:
            score -= 1
            reasoning_parts.append("VIX上升趋势")
        
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
            date=str(dt),
            market_regime=regime,
            target_exposure=target_exposure,
            vix_level=vix,
            vix_5d_change=vix_5d_change,
            spy_change_1d=spy_change,
            spy_vs_sma200=(spy_close / spy_sma200 - 1) if spy_sma200 else 0,
            score=score,
            reasoning=" | ".join(reasoning_parts),
            circuit_breaker=breaker
        )
    
    # ================================================================
    # Phase 2: 动态龙头池构建 (修复版)
    # ================================================================
    
    def _build_dynamic_universe(self, dt: date) -> List[DynamicLeader]:
        """
        构建动态龙头池 (修复版)
        
        修复:
        1. 冷启动回退: 数据不足时使用静态龙头池
        2. 成长股权重: 增加板块加分
        3. 长期动量: 增加60日动量权重
        4. 去重: 避免重复股票
        """
        candidates = []
        seen_symbols = set()
        
        spy_mom60 = self._get('SPY', dt, 'mom60') or 0
        
        # 合并科技龙头和初始宇宙 (去重)
        all_candidates = []
        for sym in STATIC_TECH_LEADERS:
            if sym not in seen_symbols:
                all_candidates.append(sym)
                seen_symbols.add(sym)
        for sym in INITIAL_UNIVERSE:
            if sym not in seen_symbols:
                all_candidates.append(sym)
                seen_symbols.add(sym)
        
        data_ready_count = 0
        
        for sym in all_candidates:
            if sym not in self._prices:
                continue
            
            close = self._get(sym, dt, 'close')
            sma50 = self._get(sym, dt, 'sma50')
            sma200 = self._get(sym, dt, 'sma200')
            rsi = self._get(sym, dt, 'rsi')
            mom20 = self._get(sym, dt, 'mom20')
            mom60 = self._get(sym, dt, 'mom60')
            atr = self._get(sym, dt, 'atr')
            
            if close is None or sma50 is None:
                continue
            
            # 检查数据是否充足
            if sma200 is not None:
                data_ready_count += 1
            
            # 获取板块
            sector = SYMBOL_TO_SECTOR.get(sym, "其他")
            
            # ========== Quant 评分 (修复版) ==========
            quant_score = 0
            
            # 1. 趋势条件 (放宽: SMA50 或 SMA200)
            if sma200 and close > sma200:
                quant_score += 2  # 高于年线加2分
            elif close > sma50:
                quant_score += 1  # 至少高于50日线
            else:
                continue  # 跌破50日线的不考虑
            
            # 2. RSI 条件 (放宽)
            if rsi and rsi > 40:  # 从45放宽到40
                quant_score += 1
            
            # 3. 短期动量
            if mom20 and mom20 > 0.08:
                quant_score += 2
            elif mom20 and mom20 > 0.02:
                quant_score += 1
            elif mom20 and mom20 < -0.05:
                quant_score -= 1  # 短期下跌扣分
            
            # ========== 长期动量评分 (新增) ==========
            momentum_score = 0
            
            if mom60 and mom60 > 0.20:
                momentum_score += 3  # 60日涨幅>20%
            elif mom60 and mom60 > 0.10:
                momentum_score += 2
            elif mom60 and mom60 > 0:
                momentum_score += 1
            elif mom60 and mom60 < -0.10:
                momentum_score -= 2  # 长期下跌扣分
            
            # 相对强度 (vs SPY)
            rs = (mom60 or 0) - spy_mom60
            if rs > 0.15:
                momentum_score += 2
            elif rs > 0.05:
                momentum_score += 1
            elif rs < -0.10:
                momentum_score -= 1
            
            # ========== 板块评分 (新增) ==========
            sector_score = SECTOR_WEIGHT.get(sector, 0)
            
            # ========== 总分 ==========
            total_score = quant_score + momentum_score + sector_score
            
            candidates.append(DynamicLeader(
                symbol=sym,
                quant_score=quant_score,
                sector_score=sector_score,
                momentum_score=momentum_score,
                total_score=total_score,
                sector=sector,
                reason=f"RS:{rs:.2f}, Mom60:{mom60 or 0:.1%}, Sector:{sector}"
            ))
        
        # ========== 冷启动处理 ==========
        if data_ready_count < 5:
            # 数据不足，使用静态龙头池
            self._cold_start_months += 1
            print(f"    ⚠️ 冷启动模式 (数据充足: {data_ready_count}/10), 使用静态龙头池")
            
            static_leaders = []
            for sym in STATIC_TECH_LEADERS[:6]:
                if sym in self._prices:
                    close = self._get(sym, dt, 'close')
                    sma50 = self._get(sym, dt, 'sma50')
                    # 只要价格在50日线以上就可以
                    if close and sma50 and close > sma50 * 0.95:
                        sector = SYMBOL_TO_SECTOR.get(sym, "科技")
                        static_leaders.append(DynamicLeader(
                            symbol=sym,
                            quant_score=5,
                            sector_score=SECTOR_WEIGHT.get(sector, 0),
                            momentum_score=0,
                            total_score=5,
                            sector=sector,
                            reason="静态龙头池"
                        ))
            return static_leaders
        
        # 排序并取 Top 10
        candidates.sort(key=lambda x: -x.total_score)
        
        # 板块分散: 每个板块最多3只
        sector_count: Dict[str, int] = {}
        final_leaders = []
        
        for c in candidates:
            if len(final_leaders) >= 10:
                break
            count = sector_count.get(c.sector, 0)
            if count >= 3:
                continue  # 该板块已满
            final_leaders.append(c)
            sector_count[c.sector] = count + 1
        
        return final_leaders
    
    # ================================================================
    # Phase 1: ATR 自适应止损 + 利润锁定 (修复版)
    # ================================================================
    
    def _calc_stop_price(self, pos: Position, dt: date, regime: str) -> float:
        """
        计算动态止损价 (修复版)
        
        修复:
        1. ATR 乘数放宽到 5x
        2. 增加最小止损距离 12%
        """
        if pos.is_safe_haven:
            return 0  # 避险资产不设止损
        
        current_price = self._get(pos.symbol, dt, 'close') or pos.avg_cost
        current_atr = self._get(pos.symbol, dt, 'atr') or pos.atr_at_entry
        
        # 更新最高价
        pos.highest_price = max(pos.highest_price, current_price)
        
        # ATR 止损基准 (修复: 使用更宽的乘数)
        multiplier = ATR_MULTIPLIER.get(regime, 4.0)
        atr_stop = pos.highest_price - (multiplier * current_atr)
        
        # 最小止损距离 (新增)
        min_stop = pos.highest_price * (1 - MIN_STOP_DISTANCE)
        atr_stop = min(atr_stop, min_stop)  # 取更宽松的止损
        
        # 利润锁定
        pnl_pct = (current_price - pos.avg_cost) / pos.avg_cost
        
        profit_stop = 0
        for tier in PROFIT_LOCK_TIERS:
            if pnl_pct >= tier["threshold"]:
                if tier["lock_pct"] > 1:
                    # 保本微利模式
                    profit_stop = pos.avg_cost * tier["lock_pct"]
                else:
                    # 锁定最高价百分比
                    profit_stop = pos.highest_price * tier["lock_pct"]
                break
        
        # 取较高的止损价 (更保守)
        final_stop = max(atr_stop, profit_stop)
        
        return final_stop
    
    # ================================================================
    # 交易执行
    # ================================================================
    
    def _portfolio_value(self, dt: date) -> float:
        """计算组合价值"""
        pos_val = sum(
            p.shares * (self._get(s, dt, 'close') or p.avg_cost)
            for s, p in self.positions.items()
        )
        return self.cash + pos_val
    
    def _buy(self, sym: str, dt: date, budget: float, reason: str, is_safe_haven: bool = False) -> bool:
        """买入"""
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
                symbol=sym,
                shares=shares,
                avg_cost=price,
                entry_date=str(dt),
                highest_price=price,
                atr_at_entry=atr,
                is_safe_haven=is_safe_haven
            )
        
        self.trades.append(Trade(str(dt), sym, "BUY", price, shares, reason=reason))
        return True
    
    def _sell(self, sym: str, dt: date, reason: str) -> float:
        """卖出"""
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
        """检查止损"""
        to_sell = []
        
        for sym, pos in self.positions.items():
            if pos.is_safe_haven:
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
    
    def _liquidate_weak_positions(self, dt: date, keep_pct: float = 0.20):
        """
        清仓弱势持仓 (渐进式)
        keep_pct: 保留盈利超过此比例的持仓
        """
        to_sell = []
        
        for sym, pos in self.positions.items():
            if pos.is_safe_haven:
                continue
            
            price = self._get(sym, dt, 'close') or pos.avg_cost
            pnl_pct = (price - pos.avg_cost) / pos.avg_cost
            
            # 只保留盈利超过阈值的仓位
            if pnl_pct < keep_pct:
                to_sell.append((sym, pnl_pct, f"风控清仓({pnl_pct:+.1%})"))
        
        # 按亏损程度排序，优先卖出亏损最大的
        to_sell.sort(key=lambda x: x[1])
        
        # 每次最多卖出一半 (渐进式)
        max_sell = max(1, len(to_sell) // 2)
        for sym, _, reason in to_sell[:max_sell]:
            self._sell(sym, dt, reason)
    
    def _rebalance(self, dt: date, macro: MacroView, leaders: List[str]):
        """再平衡"""
        pv = self._portfolio_value(dt)
        regime = macro.market_regime
        
        # 风控减仓
        if regime == "defensive":
            self._liquidate_weak_positions(dt, keep_pct=0.15)
            return  # 不新开仓
        
        # 卖出不在龙头池的持仓
        for sym in list(self.positions.keys()):
            if sym not in leaders:
                self._sell(sym, dt, "轮出龙头池")
        
        # 计算股票目标仓位
        stock_exposure = macro.target_exposure
        stock_budget = pv * stock_exposure
        
        # 当前股票持仓价值
        current_stock_value = sum(
            p.shares * (self._get(s, dt, 'close') or p.avg_cost)
            for s, p in self.positions.items()
            if not p.is_safe_haven
        )
        
        # 需要加仓
        if current_stock_value < stock_budget * 0.85:
            available = min(stock_budget - current_stock_value, self.cash * 0.95)
            
            max_positions = 6 if regime == "offensive" else 4
            current_positions = len([p for p in self.positions.values() if not p.is_safe_haven])
            slots = max(1, max_positions - current_positions)
            position_budget = available / slots
            
            for sym in leaders:
                if sym in self.positions:
                    continue
                if len([p for p in self.positions.values() if not p.is_safe_haven]) >= max_positions:
                    break
                
                # 检查技术条件
                close = self._get(sym, dt, 'close')
                sma50 = self._get(sym, dt, 'sma50')
                mom20 = self._get(sym, dt, 'mom20')
                
                if not close or not sma50:
                    continue
                if close < sma50 * 0.92:  # 跌破50日线8%不买
                    continue
                if mom20 and mom20 < -0.12:  # 过滤明显下跌
                    continue
                
                self._buy(sym, dt, position_budget, f"龙头买入({regime})")
    
    # ================================================================
    # 主运行循环
    # ================================================================
    
    def run(self, start: date, end: date) -> dict:
        """运行回测"""
        print("\n" + "=" * 70)
        print("V6.1 Neuro-Adaptive 策略回测 (修复版)")
        print("=" * 70)
        print("  核心修复:")
        print("    1. 冷启动回退: 数据不足时使用静态龙头池")
        print("    2. 成长股权重: 板块加分 + 长期动量")
        print("    3. 止损放宽: ATR 5x + 最小12%距离")
        print("    4. 分级预警: watch/caution/danger 三级")
        print("    5. 月度更新: 龙头池每月更新")
        
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
        last_universe_month = None  # 改为月度更新
        
        self._spy_start_price = self._get('SPY', actual_start, 'close') or 1
        
        for i, dt in enumerate(trading_days):
            current_month = dt.strftime("%Y-%m")
            
            # 每日宏观分析 (含熔断检查)
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
            
            # 月度更新龙头池 (修复: 从季度改为月度)
            if current_month != last_universe_month:
                leaders = self._build_dynamic_universe(dt)
                self.current_leaders = [l.symbol for l in leaders]
                self.leader_history.append({
                    "date": str(dt),
                    "leaders": [{"symbol": l.symbol, "score": l.total_score, "sector": l.sector} for l in leaders]
                })
                last_universe_month = current_month
                
                if len(leaders) > 0:
                    print(f"  🔄 [{dt}] 更新龙头池: {', '.join(self.current_leaders[:6])}")
            
            # 每日止损检查
            self._check_stops(dt, self._current_macro.market_regime)
            
            # 每5天再平衡
            if i % 5 == 0 and self.current_leaders:
                self._rebalance(dt, self._current_macro, self.current_leaders)
            
            pv = self._portfolio_value(dt)
            spy_price = self._get('SPY', dt, 'close') or 0
            spy_val = self.initial_capital * spy_price / self._spy_start_price
            self.equity_curve.append((str(dt), pv, spy_val))
            
            if i % 150 == 0:
                print(f"  [{i+1}/{len(trading_days)}] {dt}: ${pv:,.0f} (SPY: ${spy_val:,.0f})")
        
        return self._calc_results(start, end)
    
    def _calc_results(self, start: date, end: date) -> dict:
        """计算结果"""
        final = self.equity_curve[-1][1]
        spy_final = self.equity_curve[-1][2]
        
        total_ret = final / self.initial_capital - 1
        spy_ret = spy_final / self.initial_capital - 1
        
        years = (end - start).days / 365
        ann_ret = (1 + total_ret) ** (1/years) - 1 if years > 0 else 0
        
        # 最大回撤
        values = [e[1] for e in self.equity_curve]
        peak = self.initial_capital
        max_dd = 0
        for v in values:
            peak = max(peak, v)
            dd = (peak - v) / peak
            max_dd = max(max_dd, dd)
        
        # 夏普比率
        rets = pd.Series(values).pct_change().dropna()
        sharpe = np.sqrt(252) * rets.mean() / rets.std() if rets.std() > 0 else 0
        
        # 交易统计
        sells = [t for t in self.trades if t.action == "SELL"]
        wins = [t for t in sells if t.pnl > 0]
        win_rate = len(wins) / len(sells) if sells else 0
        
        total_win = sum(t.pnl for t in wins)
        total_loss = abs(sum(t.pnl for t in sells if t.pnl < 0))
        profit_factor = total_win / total_loss if total_loss > 0 else float('inf')
        
        # 熔断统计
        breaker_triggers = {
            "watch": 0,
            "caution": 0,
            "danger": 0
        }
        for m in self.macro_history:
            if m.circuit_breaker and m.circuit_breaker.level != "normal":
                breaker_triggers[m.circuit_breaker.level] = breaker_triggers.get(m.circuit_breaker.level, 0) + 1
        
        # 状态分布
        regime_dist = {}
        for m in self.macro_history:
            regime_dist[m.market_regime] = regime_dist.get(m.market_regime, 0) + 1
        
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
            "cold_start_months": self._cold_start_months,
        }


def main():
    """主函数"""
    # 4年回测 (包含2022熊市)
    bt = V61BacktestEngine(100000.0)
    result = bt.run(date(2022, 1, 3), date(2026, 1, 16))
    
    print("\n" + "=" * 70)
    print("V6.1 Neuro-Adaptive 回测结果 (修复版)")
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
    
    # 保存结果
    output = Path("storage/backtest_v6_1")
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
         "score": m.score, "vix": m.vix_level, "vix_5d_change": m.vix_5d_change,
         "reasoning": m.reasoning,
         "circuit_breaker_level": m.circuit_breaker.level if m.circuit_breaker else "normal",
         "circuit_breaker_reason": m.circuit_breaker.trigger_reason if m.circuit_breaker else ""}
        for m in bt.macro_history
    ]
    with open(output / "macro_history.json", "w") as f:
        json.dump(macro_data, f, indent=2)
    
    leader_data = bt.leader_history
    with open(output / "leader_history.json", "w") as f:
        json.dump(leader_data, f, indent=2)
    
    equity_df = pd.DataFrame(bt.equity_curve, columns=['date', 'portfolio', 'spy'])
    equity_df.to_csv(output / "equity_curve.csv", index=False)
    
    print(f"\n📁 保存到: {output}")
    
    # 最大盈利交易
    print("\n【最大盈利交易】")
    stock_sells = [t for t in bt.trades if t.action == "SELL"]
    top = sorted(stock_sells, key=lambda x: -x.pnl)[:5]
    for t in top:
        print(f"  {t.date} {t.symbol}: ${t.pnl:+,.0f} ({t.pnl_pct:+.1%}) - {t.reason}")
    
    # 最大亏损交易
    print("\n【最大亏损交易】")
    bottom = sorted(stock_sells, key=lambda x: x.pnl)[:5]
    for t in bottom:
        print(f"  {t.date} {t.symbol}: ${t.pnl:+,.0f} ({t.pnl_pct:+.1%}) - {t.reason}")
    
    # 熔断事件
    print("\n【熔断事件】")
    for m in bt.macro_history:
        if m.circuit_breaker and m.circuit_breaker.level != "normal":
            print(f"  {m.date}: [{m.circuit_breaker.level.upper()}] {m.circuit_breaker.trigger_reason}")
    
    # 龙头池更新 (最近5次)
    print("\n【龙头池更新 (最近5次)】")
    for h in bt.leader_history[-5:]:
        leaders_str = ", ".join([f"{l['symbol']}({l['score']:.0f})" for l in h['leaders'][:5]])
        print(f"  {h['date']}: {leaders_str}")
    
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
    
    # 与 V6 原版对比
    print("\n" + "=" * 70)
    print("【V6 vs V6.1 对比】")
    print("=" * 70)
    print(f"  {'指标':<15} {'V6 原版':<15} {'V6.1 修复版':<15} {'改进':<15}")
    print(f"  {'-'*60}")
    
    v6_results = {
        "total_return": -0.1060,
        "max_drawdown": 0.3268,
        "sharpe": -0.20,
        "win_rate": 0.389,
        "stock_trades": 126,
    }
    
    print(f"  {'总收益率':<15} {v6_results['total_return']:+.2%}{'':>5} {result['total_return']:+.2%}{'':>5} {result['total_return'] - v6_results['total_return']:+.2%}")
    print(f"  {'最大回撤':<15} {v6_results['max_drawdown']:.2%}{'':>6} {result['max_drawdown']:.2%}{'':>6} {v6_results['max_drawdown'] - result['max_drawdown']:+.2%}")
    print(f"  {'夏普比率':<15} {v6_results['sharpe']:.2f}{'':>10} {result['sharpe']:.2f}{'':>10} {result['sharpe'] - v6_results['sharpe']:+.2f}")
    print(f"  {'胜率':<15} {v6_results['win_rate']:.1%}{'':>8} {result['win_rate']:.1%}{'':>8} {result['win_rate'] - v6_results['win_rate']:+.1%}")
    print(f"  {'交易次数':<15} {v6_results['stock_trades']}{'':>12} {result['stock_trades']}{'':>12}")


if __name__ == "__main__":
    main()
