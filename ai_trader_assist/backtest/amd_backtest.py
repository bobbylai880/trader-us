"""AMD Backtest System - 使用 AMD 历史数据回测新旧系统对比

功能：
1. 使用 AMD 历史数据回测验证新系统表现
2. 基于回测结果调优各状态参数边界值
3. 添加状态转换平滑逻辑，避免频繁切换
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ai_trader_assist.data_collector.yf_client import YahooFinanceClient
from ai_trader_assist.feature_engineering.indicators import (
    rsi as calc_rsi,
    macd as calc_macd,
    atr as calc_atr,
    zscore as calc_zscore,
    bollinger_position_score as calc_bb_score,
)
from ai_trader_assist.risk_engine.market_regime import (
    MarketRegime,
    MarketRegimeDetector,
    RegimeSignals,
)
from ai_trader_assist.risk_engine.adaptive_params import (
    AdaptiveParameterManager,
    RegimeParameters,
    DEFAULT_REGIME_PARAMETERS,
)
from ai_trader_assist.risk_engine.macro_engine import MacroRiskEngine


@dataclass
class TradeRecord:
    date: str
    action: str  # BUY, SELL, HOLD
    price: float
    shares: int
    reason: str
    regime: str
    score: float
    pnl: float = 0.0


@dataclass
class BacktestResult:
    symbol: str
    start_date: str
    end_date: str
    initial_capital: float
    final_value: float
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    total_trades: int
    trades: List[TradeRecord] = field(default_factory=list)
    regime_distribution: Dict[str, int] = field(default_factory=dict)
    daily_values: List[Dict] = field(default_factory=list)
    drawdown_events: List[Dict] = field(default_factory=list)  # 回撤事件记录


@dataclass
class DrawdownController:
    """动态回撤控制器 - 根据组合回撤调整目标仓位"""
    severe_threshold: float = 0.08  # 严重回撤阈值 (8%)
    moderate_threshold: float = 0.05  # 中度回撤阈值 (5%)
    severe_multiplier: float = 0.6  # 严重回撤时仓位乘数
    moderate_multiplier: float = 0.8  # 中度回撤时仓位乘数
    recovery_threshold: float = 0.03  # 回撤恢复阈值 (3%)
    recovery_steps: int = 3  # 恢复步数
    
    _current_multiplier: float = 1.0
    _in_drawdown: bool = False
    _recovery_step: int = 0
    _peak_value: float = 0.0
    
    def update(self, portfolio_value: float, peak_value: float) -> Tuple[float, str]:
        """
        更新回撤状态并返回仓位乘数
        
        Returns:
            Tuple[float, str]: (仓位乘数, 状态描述)
        """
        self._peak_value = peak_value
        current_drawdown = (peak_value - portfolio_value) / peak_value if peak_value > 0 else 0
        
        # 检测严重回撤
        if current_drawdown >= self.severe_threshold:
            self._in_drawdown = True
            self._recovery_step = 0
            self._current_multiplier = self.severe_multiplier
            return self._current_multiplier, f"severe_drawdown_{current_drawdown:.1%}"
        
        # 检测中度回撤
        if current_drawdown >= self.moderate_threshold:
            self._in_drawdown = True
            self._recovery_step = 0
            self._current_multiplier = self.moderate_multiplier
            return self._current_multiplier, f"moderate_drawdown_{current_drawdown:.1%}"
        
        # 检测回撤恢复
        if self._in_drawdown:
            if current_drawdown <= self.recovery_threshold:
                # 逐步恢复仓位
                self._recovery_step += 1
                if self._recovery_step >= self.recovery_steps:
                    self._in_drawdown = False
                    self._current_multiplier = 1.0
                    self._recovery_step = 0
                    return 1.0, "fully_recovered"
                else:
                    # 渐进恢复: 0.6 -> 0.73 -> 0.87 -> 1.0
                    recovery_progress = self._recovery_step / self.recovery_steps
                    self._current_multiplier = self.severe_multiplier + (1.0 - self.severe_multiplier) * recovery_progress
                    return self._current_multiplier, f"recovering_step_{self._recovery_step}"
            else:
                # 仍在回撤中但未达阈值，保持当前乘数
                return self._current_multiplier, f"in_drawdown_{current_drawdown:.1%}"
        
        return 1.0, "normal"
    
    def reset(self):
        """重置控制器状态"""
        self._current_multiplier = 1.0
        self._in_drawdown = False
        self._recovery_step = 0
        self._peak_value = 0.0


@dataclass
class ConcentrationController:
    """集中度控制器 - 限制单只股票对组合的贡献度"""
    max_contribution_pct: float = 0.30  # 单只最大贡献 30%
    partial_take_profit_levels: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.50, 0.25),  # 盈利 50% 时卖出 25%
        (1.00, 0.25),  # 盈利 100% 时再卖 25%
        (2.00, 0.25),  # 盈利 200% 时再卖 25%
    ])
    
    def check_concentration(
        self,
        position_value: float,
        portfolio_value: float,
    ) -> Tuple[bool, float]:
        """
        检查集中度是否超标
        
        Returns:
            Tuple[bool, float]: (是否需要减仓, 建议卖出比例)
        """
        if portfolio_value <= 0:
            return False, 0.0
        
        contribution = position_value / portfolio_value
        if contribution > self.max_contribution_pct:
            # 计算需要卖出多少才能降到上限
            target_value = portfolio_value * self.max_contribution_pct
            excess_value = position_value - target_value
            sell_ratio = excess_value / position_value
            return True, min(0.5, sell_ratio)  # 最多一次卖 50%
        
        return False, 0.0
    
    def check_partial_profit(
        self,
        current_price: float,
        entry_price: float,
        taken_profit_levels: List[float],
    ) -> Tuple[bool, float, float]:
        """
        检查是否触发渐进止盈
        
        Args:
            current_price: 当前价格
            entry_price: 入场价格
            taken_profit_levels: 已触发的止盈水平列表
            
        Returns:
            Tuple[bool, float, float]: (是否触发止盈, 卖出比例, 触发的盈利水平)
        """
        if entry_price <= 0:
            return False, 0.0, 0.0
        
        profit_pct = (current_price - entry_price) / entry_price
        
        for level, sell_ratio in self.partial_take_profit_levels:
            if profit_pct >= level and level not in taken_profit_levels:
                return True, sell_ratio, level
        
        return False, 0.0, 0.0


@dataclass
class RegimeTransitionSmoother:
    """状态转换平滑器 - 避免频繁切换"""
    confirmation_periods: int = 3
    _history: List[MarketRegime] = field(default_factory=list)
    _confirmed_regime: MarketRegime = MarketRegime.UNKNOWN
    
    def update(self, new_regime: MarketRegime) -> MarketRegime:
        self._history.append(new_regime)
        if len(self._history) > self.confirmation_periods:
            self._history.pop(0)
        
        if len(self._history) < self.confirmation_periods:
            return self._confirmed_regime
        
        if all(r == new_regime for r in self._history[-self.confirmation_periods:]):
            self._confirmed_regime = new_regime
        
        return self._confirmed_regime
    
    def reset(self):
        self._history = []
        self._confirmed_regime = MarketRegime.UNKNOWN


class AMDBacktester:
    """AMD 回测引擎"""
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        use_smoothing: bool = True,
        smoothing_periods: int = 3,
        use_drawdown_control: bool = True,
        use_concentration_control: bool = True,
    ):
        self.initial_capital = initial_capital
        self.use_smoothing = use_smoothing
        self.use_drawdown_control = use_drawdown_control
        self.use_concentration_control = use_concentration_control
        self.yf_client = YahooFinanceClient()
        self.regime_detector = MarketRegimeDetector()
        self.param_manager = AdaptiveParameterManager()
        self.smoother = RegimeTransitionSmoother(confirmation_periods=smoothing_periods)
        self.drawdown_controller = DrawdownController()
        self.concentration_controller = ConcentrationController()
        
    def _fetch_data(
        self,
        symbol: str,
        market_symbol: str,
        start: datetime,
        end: datetime,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """获取股票和市场数据"""
        stock_data = self.yf_client.fetch_history(symbol, start, end, interval="1d")
        market_data = self.yf_client.fetch_history(market_symbol, start, end, interval="1d")
        return stock_data, market_data
    
    def _calculate_features(
        self,
        stock_df: pd.DataFrame,
        market_df: pd.DataFrame,
        idx: int,
    ) -> Dict[str, float]:
        """计算当日特征"""
        lookback = min(idx + 1, 60)
        stock_slice = stock_df.iloc[max(0, idx - lookback + 1):idx + 1]
        market_slice = market_df.iloc[max(0, idx - lookback + 1):idx + 1]
        
        if len(stock_slice) < 20 or len(market_slice) < 20:
            return {}
        
        close = stock_slice["Close"]
        market_close = market_slice["Close"]
        
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        if isinstance(market_close, pd.DataFrame):
            market_close = market_close.iloc[:, 0]
        
        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean() if len(close) >= 50 else close.rolling(20).mean()
        sma200 = close.rolling(200).mean() if len(close) >= 200 else sma50
        
        market_sma50 = market_close.rolling(50).mean() if len(market_close) >= 50 else market_close.rolling(20).mean()
        market_sma200 = market_close.rolling(200).mean() if len(market_close) >= 200 else market_sma50
        
        current_price = float(close.iloc[-1])
        current_sma50 = float(sma50.iloc[-1]) if not pd.isna(sma50.iloc[-1]) else current_price
        current_sma200 = float(sma200.iloc[-1]) if not pd.isna(sma200.iloc[-1]) else current_price
        
        market_price = float(market_close.iloc[-1])
        market_sma50_val = float(market_sma50.iloc[-1]) if not pd.isna(market_sma50.iloc[-1]) else market_price
        market_sma200_val = float(market_sma200.iloc[-1]) if not pd.isna(market_sma200.iloc[-1]) else market_price
        
        rsi = calc_rsi(close, 14)
        rsi_val = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
        
        atr = calc_atr(stock_slice["High"], stock_slice["Low"], close, 14)
        atr_val = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else current_price * 0.02
        
        stock_return_20d = (current_price / float(close.iloc[-20]) - 1) if len(close) >= 20 else 0
        market_return_20d = (market_price / float(market_close.iloc[-20]) - 1) if len(market_close) >= 20 else 0
        
        slope_5d = (float(sma50.iloc[-1]) - float(sma50.iloc[-5])) / 5 / current_sma50 if len(sma50) >= 5 else 0
        
        # 布林带均值回归评分
        bb_score = calc_bb_score(close, window=20)
        bb_score_val = float(bb_score.iloc[-1]) if not pd.isna(bb_score.iloc[-1]) else 0.5
        
        features = {
            "price": current_price,
            "rsi": rsi_val,
            "rsi_norm": rsi_val / 100.0,
            "atr": atr_val,
            "atr_pct": atr_val / current_price,
            "sma20": float(sma20.iloc[-1]) if not pd.isna(sma20.iloc[-1]) else current_price,
            "sma50": current_sma50,
            "sma200": current_sma200,
            "price_vs_sma50": (current_price - current_sma50) / current_sma50 * 100,
            "price_vs_sma200": (current_price - current_sma200) / current_sma200 * 100,
            "momentum_20d": stock_return_20d,
            "relative_strength": stock_return_20d - market_return_20d,
            "spy_vs_sma200": (market_price - market_sma200_val) / market_sma200_val * 100,
            "spy_vs_sma50": (market_price - market_sma50_val) / market_sma50_val * 100,
            "sma50_slope": slope_5d,
            "spy_momentum_20d": market_return_20d,
            "qqq_momentum_20d": market_return_20d * 1.1,
            "bb_score": bb_score_val,  # 布林带均值回归评分 (0-1, 高=超卖)
        }
        
        return features
    
    def _detect_regime(self, features: Dict[str, float]) -> Tuple[MarketRegime, float]:
        """识别市场状态"""
        signals = RegimeSignals(
            spy_vs_sma200=features.get("spy_vs_sma200", 0),
            spy_vs_sma50=features.get("spy_vs_sma50", 0),
            sma50_slope=features.get("sma50_slope", 0),
            breadth=0.5 + features.get("spy_momentum_20d", 0) * 2,
            nh_nl_ratio=1.0 + features.get("spy_momentum_20d", 0) * 5,
            vix_value=20.0 - features.get("spy_momentum_20d", 0) * 50,
            vix_term_contango=features.get("spy_momentum_20d", 0) > -0.05,
            spy_momentum_20d=features.get("spy_momentum_20d", 0),
            qqq_momentum_20d=features.get("qqq_momentum_20d", 0),
        )
        
        result = self.regime_detector.detect(signals)
        
        if self.use_smoothing:
            smoothed_regime = self.smoother.update(result.regime)
            return smoothed_regime, result.confidence
        
        return result.regime, result.confidence
    
    def _calculate_score(
        self,
        features: Dict[str, float],
        regime: MarketRegime,
    ) -> float:
        """计算综合评分"""
        self.param_manager.set_regime(regime)
        weights = self.param_manager.scoring_weights
        
        trend_score = 0.0
        if features.get("price_vs_sma50", 0) > 0:
            trend_score += 0.5
        if features.get("price_vs_sma200", 0) > 0:
            trend_score += 0.5
        
        rsi = features.get("rsi", 50)
        if regime in [MarketRegime.BEAR_TREND, MarketRegime.RANGE_BOUND]:
            if rsi < 30:
                momentum_score = 0.8
            elif rsi < 40:
                momentum_score = 0.6
            elif rsi > 70:
                momentum_score = 0.2
            else:
                momentum_score = 0.5
        else:
            if 40 < rsi < 70:
                momentum_score = 0.7
            elif rsi > 70:
                momentum_score = 0.5
            else:
                momentum_score = 0.4
        
        rs = features.get("relative_strength", 0)
        rs_score = 0.5 + min(1, max(-1, rs * 10)) * 0.5
        
        mean_rev_score = 0.5
        if regime in [MarketRegime.BEAR_TREND, MarketRegime.RANGE_BOUND]:
            # 使用布林带评分增强均值回归信号
            bb_score = features.get("bb_score", 0.5)
            price_deviation = features.get("price_vs_sma50", 0)
            
            # 组合布林带评分和价格偏离度
            if bb_score > 0.8:  # 严重超卖 (接近下轨)
                mean_rev_score = 0.95
            elif bb_score > 0.65:  # 中度超卖
                mean_rev_score = 0.8
            elif bb_score < 0.2:  # 严重超买 (接近上轨)
                mean_rev_score = 0.1
            elif bb_score < 0.35:  # 中度超买
                mean_rev_score = 0.3
            elif price_deviation < -10:
                mean_rev_score = 0.85
            elif price_deviation < -5:
                mean_rev_score = 0.7
            elif price_deviation > 10:
                mean_rev_score = 0.2
            else:
                # 正常区间，使用布林带位置
                mean_rev_score = bb_score * 0.6 + 0.2
        
        score = (
            weights.trend * trend_score +
            weights.momentum * momentum_score +
            weights.relative_strength * rs_score +
            weights.mean_reversion * mean_rev_score +
            weights.volume * 0.5 +
            weights.structure * 0.5 +
            weights.news * 0.5
        )
        
        return min(1.0, max(0.0, score))
    
    def _should_buy(
        self,
        score: float,
        regime: MarketRegime,
        current_position: int,
    ) -> bool:
        """判断是否应该买入"""
        if current_position > 0:
            return False
        
        params = self.param_manager.get_params(regime)
        return score >= params.buy_threshold
    
    def _should_sell(
        self,
        score: float,
        regime: MarketRegime,
        entry_price: float,
        current_price: float,
        atr: float,
        features: Optional[Dict[str, float]] = None,
        peak_price: float = 0.0,
    ) -> Tuple[bool, str]:
        """判断是否应该卖出 - 牛市只用移动止损"""
        params = self.param_manager.get_params(regime)
        
        stop_loss = entry_price - atr * params.stop_atr_mult
        if current_price <= stop_loss:
            return True, "stop_loss"
        
        if regime == MarketRegime.BULL_TREND:
            if peak_price > 0:
                trailing_pct = 0.25
                trailing_stop = peak_price * (1 - trailing_pct)
                if current_price <= trailing_stop and current_price > entry_price:
                    return True, f"trailing_stop_{trailing_pct:.0%}"
            
            return False, ""
        
        target = entry_price + atr * params.target_atr_mult
        if current_price >= target:
            return True, "take_profit"
        
        if score < params.reduce_threshold:
            return True, "score_below_threshold"
        
        return False, ""
    
    def run_backtest(
        self,
        symbol: str = "AMD",
        market_symbol: str = "SPY",
        start_date: str = "2023-01-01",
        end_date: str = "2024-12-31",
    ) -> BacktestResult:
        """运行回测"""
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        lookback_start = start - timedelta(days=250)
        
        stock_df, market_df = self._fetch_data(symbol, market_symbol, lookback_start, end)
        
        if stock_df.empty or market_df.empty:
            raise ValueError(f"Failed to fetch data for {symbol} or {market_symbol}")
        
        stock_df = stock_df.loc[stock_df.index >= pd.Timestamp(lookback_start)]
        market_df = market_df.loc[market_df.index >= pd.Timestamp(lookback_start)]
        
        common_dates = stock_df.index.intersection(market_df.index)
        stock_df = stock_df.loc[common_dates]
        market_df = market_df.loc[common_dates]
        
        backtest_start_idx = len(stock_df[stock_df.index < pd.Timestamp(start)])
        
        cash = self.initial_capital
        shares = 0
        entry_price = 0.0
        entry_atr = 0.0
        position_peak_price = 0.0  # 持仓期间最高价（用于移动止损）
        
        trades: List[TradeRecord] = []
        daily_values: List[Dict] = []
        regime_counts: Dict[str, int] = {r.value: 0 for r in MarketRegime}
        
        peak_value = self.initial_capital
        max_drawdown = 0.0
        daily_returns: List[float] = []
        prev_value = self.initial_capital
        drawdown_events: List[Dict] = []  # 回撤事件记录
        taken_profit_levels: List[float] = []  # 已触发的止盈水平
        
        self.smoother.reset()
        self.drawdown_controller.reset()
        
        for idx in range(backtest_start_idx, len(stock_df)):
            date = stock_df.index[idx]
            
            if isinstance(stock_df["Close"].iloc[idx], pd.Series):
                current_price = float(stock_df["Close"].iloc[idx].iloc[0])
            else:
                current_price = float(stock_df["Close"].iloc[idx])
            
            features = self._calculate_features(stock_df, market_df, idx)
            if not features:
                continue
            
            regime, confidence = self._detect_regime(features)
            regime_counts[regime.value] += 1
            
            score = self._calculate_score(features, regime)
            
            action = "HOLD"
            pnl = 0.0
            reason = ""
            
            if shares > 0:
                # 更新持仓期间最高价（用于移动止损）
                if current_price > position_peak_price:
                    position_peak_price = current_price
                
                # 牛市中只使用移动止损，禁用渐进止盈和集中度控制
                if regime == MarketRegime.BULL_TREND:
                    pass  # 牛市中跳过渐进止盈，只依赖移动止损
                elif self.use_concentration_control:
                    position_value = shares * current_price
                    portfolio_value_now = cash + position_value
                    
                    # 检查集中度是否超标
                    conc_triggered, conc_sell_ratio = self.concentration_controller.check_concentration(
                        position_value, portfolio_value_now
                    )
                    
                    # 检查渐进止盈
                    profit_triggered, profit_sell_ratio, profit_level = self.concentration_controller.check_partial_profit(
                        current_price, entry_price, taken_profit_levels
                    )
                    
                    # 优先处理集中度超标（强制减仓）
                    if conc_triggered:
                        shares_to_sell = max(1, int(shares * conc_sell_ratio))
                        sell_value = shares_to_sell * current_price
                        pnl = (current_price - entry_price) * shares_to_sell
                        cash += sell_value
                        shares -= shares_to_sell
                        
                        trades.append(TradeRecord(
                            date=str(date.date()),
                            action="PARTIAL_SELL",
                            price=current_price,
                            shares=shares_to_sell,
                            reason=f"concentration_limit_{conc_sell_ratio:.0%}",
                            regime=regime.value,
                            score=score,
                            pnl=pnl,
                        ))
                    
                    # 处理渐进止盈
                    elif profit_triggered and shares > 0:
                        shares_to_sell = max(1, int(shares * profit_sell_ratio))
                        sell_value = shares_to_sell * current_price
                        pnl = (current_price - entry_price) * shares_to_sell
                        cash += sell_value
                        shares -= shares_to_sell
                        taken_profit_levels.append(profit_level)
                        
                        trades.append(TradeRecord(
                            date=str(date.date()),
                            action="PARTIAL_SELL",
                            price=current_price,
                            shares=shares_to_sell,
                            reason=f"partial_profit_{profit_level:.0%}",
                            regime=regime.value,
                            score=score,
                            pnl=pnl,
                        ))
                
                # 再检查常规卖出条件（止损/止盈/评分）
                if shares > 0:
                    should_sell, sell_reason = self._should_sell(
                        score, regime, entry_price, current_price, entry_atr, features, position_peak_price
                    )
                    if should_sell:
                        pnl = (current_price - entry_price) * shares
                        cash += shares * current_price
                        action = "SELL"
                        reason = sell_reason
                        
                        trades.append(TradeRecord(
                            date=str(date.date()),
                            action=action,
                            price=current_price,
                            shares=shares,
                            reason=reason,
                            regime=regime.value,
                            score=score,
                            pnl=pnl,
                        ))
                        
                        shares = 0
                        entry_price = 0.0
                        entry_atr = 0.0
                        position_peak_price = 0.0  # 重置最高价
                        taken_profit_levels = []  # 重置止盈水平
            
            elif shares == 0 and self._should_buy(score, regime, shares):
                params = self.param_manager.get_params(regime)
                
                # 动态回撤控制：根据回撤状态调整可用资金
                if self.use_drawdown_control:
                    portfolio_value_now = cash + shares * current_price
                    dd_multiplier, dd_status = self.drawdown_controller.update(portfolio_value_now, peak_value)
                    
                    # 记录回撤事件
                    if dd_status != "normal":
                        drawdown_events.append({
                            "date": str(date.date()),
                            "status": dd_status,
                            "multiplier": dd_multiplier,
                            "portfolio_value": portfolio_value_now,
                            "peak_value": peak_value,
                        })
                else:
                    dd_multiplier = 1.0
                
                # 应用回撤乘数到仓位计算
                effective_cash = cash * dd_multiplier
                max_position_value = effective_cash * params.max_single_weight
                
                atr = features.get("atr", current_price * 0.02)
                risk_per_share = atr * params.stop_atr_mult
                risk_budget = cash * 0.02
                shares_by_risk = int(risk_budget / risk_per_share) if risk_per_share > 0 else 0
                shares_by_capital = int(max_position_value / current_price)
                
                shares_to_buy = min(shares_by_risk, shares_by_capital)
                
                if shares_to_buy > 0 and shares_to_buy * current_price <= cash:
                    cost = shares_to_buy * current_price
                    cash -= cost
                    shares = shares_to_buy
                    entry_price = current_price
                    entry_atr = atr
                    action = "BUY"
                    reason = f"score={score:.2f}, regime={regime.value}"
                    
                    trades.append(TradeRecord(
                        date=str(date.date()),
                        action=action,
                        price=current_price,
                        shares=shares,
                        reason=reason,
                        regime=regime.value,
                        score=score,
                        pnl=0.0,
                    ))
            
            portfolio_value = cash + shares * current_price
            
            if portfolio_value > peak_value:
                peak_value = portfolio_value
            drawdown = (peak_value - portfolio_value) / peak_value
            if drawdown > max_drawdown:
                max_drawdown = drawdown
            
            daily_return = (portfolio_value - prev_value) / prev_value if prev_value > 0 else 0
            daily_returns.append(daily_return)
            prev_value = portfolio_value
            
            daily_values.append({
                "date": str(date.date()),
                "price": current_price,
                "portfolio_value": portfolio_value,
                "cash": cash,
                "shares": shares,
                "regime": regime.value,
                "score": score,
                "drawdown": drawdown,
            })
        
        final_value = cash + shares * current_price if shares > 0 else cash
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        if len(daily_returns) > 0 and np.std(daily_returns) > 0:
            sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        winning_trades = sum(1 for t in trades if t.action == "SELL" and t.pnl > 0)
        total_sells = sum(1 for t in trades if t.action == "SELL")
        win_rate = winning_trades / total_sells if total_sells > 0 else 0.0
        
        return BacktestResult(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_value=final_value,
            total_return=total_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            win_rate=win_rate,
            total_trades=len(trades),
            trades=trades,
            regime_distribution=regime_counts,
            daily_values=daily_values,
            drawdown_events=drawdown_events,
        )


class OldSystemBacktester(AMDBacktester):
    """旧系统回测器（固定参数）"""
    
    def __init__(self, initial_capital: float = 100000.0):
        super().__init__(initial_capital, use_smoothing=False, use_drawdown_control=False, use_concentration_control=False)
        
        fixed_params = RegimeParameters(
            max_exposure=0.85,
            max_single_weight=0.25,
            buy_threshold=0.60,
            hold_threshold=0.45,
            reduce_threshold=0.40,
            stop_atr_mult=1.5,
            target_atr_mult=2.5,
        )
        
        for regime in MarketRegime:
            self.param_manager.regime_params[regime] = fixed_params


def run_comparison_backtest(
    symbol: str = "AMD",
    start_date: str = "2023-01-01",
    end_date: str = "2024-12-31",
    initial_capital: float = 100000.0,
) -> Dict[str, Any]:
    """运行新旧系统对比回测"""
    
    print(f"Running backtest for {symbol} from {start_date} to {end_date}")
    print("=" * 60)
    
    print("\n[1/3] Running NEW system (adaptive + smoothing)...")
    new_system = AMDBacktester(initial_capital, use_smoothing=True, smoothing_periods=3)
    new_result = new_system.run_backtest(symbol, "SPY", start_date, end_date)
    
    print("[2/3] Running NEW system (adaptive, no smoothing)...")
    new_no_smooth = AMDBacktester(initial_capital, use_smoothing=False)
    new_no_smooth_result = new_no_smooth.run_backtest(symbol, "SPY", start_date, end_date)
    
    print("[3/3] Running OLD system (fixed params)...")
    old_system = OldSystemBacktester(initial_capital)
    old_result = old_system.run_backtest(symbol, "SPY", start_date, end_date)
    
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS COMPARISON")
    print("=" * 60)
    
    print(f"\n{'Metric':<25} {'Old System':<15} {'New (no smooth)':<15} {'New (smooth)':<15}")
    print("-" * 70)
    print(f"{'Total Return':<25} {old_result.total_return:>14.2%} {new_no_smooth_result.total_return:>14.2%} {new_result.total_return:>14.2%}")
    print(f"{'Max Drawdown':<25} {old_result.max_drawdown:>14.2%} {new_no_smooth_result.max_drawdown:>14.2%} {new_result.max_drawdown:>14.2%}")
    print(f"{'Sharpe Ratio':<25} {old_result.sharpe_ratio:>14.2f} {new_no_smooth_result.sharpe_ratio:>14.2f} {new_result.sharpe_ratio:>14.2f}")
    print(f"{'Win Rate':<25} {old_result.win_rate:>14.2%} {new_no_smooth_result.win_rate:>14.2%} {new_result.win_rate:>14.2%}")
    print(f"{'Total Trades':<25} {old_result.total_trades:>14} {new_no_smooth_result.total_trades:>14} {new_result.total_trades:>14}")
    print(f"{'Final Value':<25} ${old_result.final_value:>13,.2f} ${new_no_smooth_result.final_value:>13,.2f} ${new_result.final_value:>13,.2f}")
    
    print("\n" + "-" * 60)
    print("REGIME DISTRIBUTION (New System with Smoothing)")
    print("-" * 60)
    for regime, count in new_result.regime_distribution.items():
        if count > 0:
            pct = count / sum(new_result.regime_distribution.values()) * 100
            print(f"  {regime:<20} {count:>5} days ({pct:>5.1f}%)")
    
    return {
        "old_system": {
            "total_return": old_result.total_return,
            "max_drawdown": old_result.max_drawdown,
            "sharpe_ratio": old_result.sharpe_ratio,
            "win_rate": old_result.win_rate,
            "total_trades": old_result.total_trades,
            "final_value": old_result.final_value,
        },
        "new_no_smoothing": {
            "total_return": new_no_smooth_result.total_return,
            "max_drawdown": new_no_smooth_result.max_drawdown,
            "sharpe_ratio": new_no_smooth_result.sharpe_ratio,
            "win_rate": new_no_smooth_result.win_rate,
            "total_trades": new_no_smooth_result.total_trades,
            "final_value": new_no_smooth_result.final_value,
        },
        "new_with_smoothing": {
            "total_return": new_result.total_return,
            "max_drawdown": new_result.max_drawdown,
            "sharpe_ratio": new_result.sharpe_ratio,
            "win_rate": new_result.win_rate,
            "total_trades": new_result.total_trades,
            "final_value": new_result.final_value,
            "regime_distribution": new_result.regime_distribution,
        },
        "improvement": {
            "return_improvement": new_result.total_return - old_result.total_return,
            "drawdown_reduction": old_result.max_drawdown - new_result.max_drawdown,
            "sharpe_improvement": new_result.sharpe_ratio - old_result.sharpe_ratio,
            "trade_reduction": old_result.total_trades - new_result.total_trades,
        },
        "trades": {
            "new_system": [
                {
                    "date": t.date,
                    "action": t.action,
                    "price": t.price,
                    "shares": t.shares,
                    "regime": t.regime,
                    "score": t.score,
                    "pnl": t.pnl,
                }
                for t in new_result.trades
            ]
        }
    }


def test_backtest_runs():
    """Test that backtest completes without errors"""
    backtester = AMDBacktester(initial_capital=100000.0, use_smoothing=True)
    result = backtester.run_backtest("AMD", "SPY", "2024-01-01", "2024-06-30")
    
    assert result.symbol == "AMD"
    assert result.initial_capital == 100000.0
    assert result.final_value > 0
    assert 0 <= result.max_drawdown <= 1
    assert len(result.daily_values) > 0


def test_regime_smoother():
    """Test regime transition smoother"""
    smoother = RegimeTransitionSmoother(confirmation_periods=3)
    
    result1 = smoother.update(MarketRegime.BULL_TREND)
    assert result1 == MarketRegime.UNKNOWN
    
    result2 = smoother.update(MarketRegime.BULL_TREND)
    assert result2 == MarketRegime.UNKNOWN
    
    result3 = smoother.update(MarketRegime.BULL_TREND)
    assert result3 == MarketRegime.BULL_TREND
    
    result4 = smoother.update(MarketRegime.BEAR_TREND)
    assert result4 == MarketRegime.BULL_TREND
    
    result5 = smoother.update(MarketRegime.BEAR_TREND)
    result6 = smoother.update(MarketRegime.BEAR_TREND)
    assert result6 == MarketRegime.BEAR_TREND


if __name__ == "__main__":
    results = run_comparison_backtest(
        symbol="AMD",
        start_date="2023-01-01",
        end_date="2024-12-31",
        initial_capital=100000.0,
    )
    
    output_path = Path("storage/amd_backtest_results.json")
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to: {output_path}")
