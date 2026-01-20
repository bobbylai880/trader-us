"""
自适应策略回测引擎 - 根据市场状态自动切换三套策略
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
    atr as calc_atr,
    bollinger_position_score as calc_bb_score,
)
from ai_trader_assist.risk_engine.market_regime import (
    MarketRegime,
    MarketRegimeDetector,
    RegimeSignals,
)
from ai_trader_assist.backtest.regime_strategies import (
    AdaptiveStrategyEngine,
    StrategyMode,
    BULL_STRATEGY,
    RANGE_STRATEGY,
    BEAR_STRATEGY,
)


@dataclass
class TradeRecord:
    date: str
    action: str
    price: float
    shares: int
    reason: str
    regime: str
    strategy: str
    pnl: float = 0.0


@dataclass
class AdaptiveBacktestResult:
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
    buy_hold_return: float
    capture_rate: float
    trades: List[TradeRecord] = field(default_factory=list)
    regime_distribution: Dict[str, int] = field(default_factory=dict)
    strategy_distribution: Dict[str, int] = field(default_factory=dict)
    daily_values: List[Dict] = field(default_factory=list)


class AdaptiveBacktester:
    """自适应策略回测引擎"""
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.yf_client = YahooFinanceClient()
        self.regime_detector = MarketRegimeDetector()
        self.strategy_engine = AdaptiveStrategyEngine()
        
    def _fetch_data(
        self,
        symbol: str,
        market_symbol: str,
        start: datetime,
        end: datetime,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        stock_data = self.yf_client.fetch_history(symbol, start, end, interval="1d")
        market_data = self.yf_client.fetch_history(market_symbol, start, end, interval="1d")
        return stock_data, market_data
    
    def _calculate_features(
        self,
        stock_df: pd.DataFrame,
        market_df: pd.DataFrame,
        idx: int,
    ) -> Dict[str, float]:
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
        current_sma20 = float(sma20.iloc[-1]) if not pd.isna(sma20.iloc[-1]) else current_price
        current_sma50 = float(sma50.iloc[-1]) if not pd.isna(sma50.iloc[-1]) else current_price
        current_sma200 = float(sma200.iloc[-1]) if not pd.isna(sma200.iloc[-1]) else current_price
        
        market_price = float(market_close.iloc[-1])
        market_sma50_val = float(market_sma50.iloc[-1]) if not pd.isna(market_sma50.iloc[-1]) else market_price
        market_sma200_val = float(market_sma200.iloc[-1]) if not pd.isna(market_sma200.iloc[-1]) else market_price
        
        rsi = calc_rsi(close, 14)
        rsi_val = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
        
        atr = calc_atr(stock_slice["High"], stock_slice["Low"], close, 14)
        atr_val = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else current_price * 0.02
        
        bb_score = calc_bb_score(close, window=20)
        bb_score_val = float(bb_score.iloc[-1]) if not pd.isna(bb_score.iloc[-1]) else 0.5
        
        stock_return_20d = (current_price / float(close.iloc[-20]) - 1) if len(close) >= 20 else 0
        market_return_20d = (market_price / float(market_close.iloc[-20]) - 1) if len(market_close) >= 20 else 0
        
        slope_5d = (float(sma50.iloc[-1]) - float(sma50.iloc[-5])) / 5 / current_sma50 if len(sma50) >= 5 and current_sma50 > 0 else 0
        
        return {
            "price": current_price,
            "rsi": rsi_val,
            "atr": atr_val,
            "sma20": current_sma20,
            "sma50": current_sma50,
            "sma200": current_sma200,
            "bb_score": bb_score_val,
            "price_vs_sma50": (current_price - current_sma50) / current_sma50 * 100 if current_sma50 > 0 else 0,
            "price_vs_sma200": (current_price - current_sma200) / current_sma200 * 100 if current_sma200 > 0 else 0,
            "spy_vs_sma200": (market_price - market_sma200_val) / market_sma200_val * 100 if market_sma200_val > 0 else 0,
            "spy_vs_sma50": (market_price - market_sma50_val) / market_sma50_val * 100 if market_sma50_val > 0 else 0,
            "sma50_slope": slope_5d,
            "spy_momentum_20d": market_return_20d,
            "qqq_momentum_20d": market_return_20d * 1.1,
            "relative_strength": stock_return_20d - market_return_20d,
        }
    
    def _detect_regime(self, features: Dict[str, float]) -> MarketRegime:
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
        return result.regime
    
    def _calculate_score(self, features: Dict[str, float], regime: MarketRegime) -> float:
        trend_score = 0.0
        if features.get("price_vs_sma50", 0) > 0:
            trend_score += 0.5
        if features.get("price_vs_sma200", 0) > 0:
            trend_score += 0.5
        
        rsi = features.get("rsi", 50)
        if 40 < rsi < 70:
            momentum_score = 0.7
        elif rsi > 70:
            momentum_score = 0.4
        elif rsi < 30:
            momentum_score = 0.8
        else:
            momentum_score = 0.5
        
        rs = features.get("relative_strength", 0)
        rs_score = 0.5 + min(1, max(-1, rs * 10)) * 0.5
        
        bb_score = features.get("bb_score", 0.5)
        
        if regime in [MarketRegime.BULL_TREND, MarketRegime.BULL_PULLBACK]:
            score = trend_score * 0.4 + momentum_score * 0.3 + rs_score * 0.3
        elif regime == MarketRegime.RANGE_BOUND:
            score = bb_score * 0.4 + momentum_score * 0.3 + rs_score * 0.3
        else:
            score = momentum_score * 0.4 + rs_score * 0.3 + (1 - trend_score) * 0.3
        
        return min(1.0, max(0.0, score))
    
    def run_backtest(
        self,
        symbol: str = "NVDA",
        market_symbol: str = "SPY",
        start_date: str = "2024-01-01",
        end_date: str = "2024-12-31",
    ) -> AdaptiveBacktestResult:
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
        peak_price = 0.0
        
        trades: List[TradeRecord] = []
        daily_values: List[Dict] = []
        regime_counts: Dict[str, int] = {r.value: 0 for r in MarketRegime}
        strategy_counts: Dict[str, int] = {m.value: 0 for m in StrategyMode}
        
        portfolio_peak = self.initial_capital
        max_drawdown = 0.0
        daily_returns: List[float] = []
        prev_value = self.initial_capital
        
        self.strategy_engine = AdaptiveStrategyEngine()
        
        first_price = None
        last_price = None
        
        for day_idx, idx in enumerate(range(backtest_start_idx, len(stock_df))):
            date = stock_df.index[idx]
            
            if isinstance(stock_df["Close"].iloc[idx], pd.Series):
                current_price = float(stock_df["Close"].iloc[idx].iloc[0])
            else:
                current_price = float(stock_df["Close"].iloc[idx])
            
            if first_price is None:
                first_price = current_price
            last_price = current_price
            
            features = self._calculate_features(stock_df, market_df, idx)
            if not features:
                continue
            
            regime = self._detect_regime(features)
            regime_counts[regime.value] += 1
            
            strategy = self.strategy_engine.update_strategy(regime, day_idx)
            strategy_counts[strategy.mode.value] += 1
            
            score = self._calculate_score(features, regime)
            
            current_exposure = (shares * current_price) / (cash + shares * current_price) if (cash + shares * current_price) > 0 else 0
            
            if shares > 0:
                if current_price > peak_price:
                    peak_price = current_price
                
                should_sell, sell_reason = self.strategy_engine.should_sell(
                    score=score,
                    entry_price=entry_price,
                    current_price=current_price,
                    peak_price=peak_price,
                    current_day=day_idx,
                    bb_score=features.get("bb_score", 0.5),
                )
                
                if should_sell:
                    pnl = (current_price - entry_price) * shares
                    cash += shares * current_price
                    
                    trades.append(TradeRecord(
                        date=str(date.date()),
                        action="SELL",
                        price=current_price,
                        shares=shares,
                        reason=sell_reason,
                        regime=regime.value,
                        strategy=strategy.mode.value,
                        pnl=pnl,
                    ))
                    
                    self.strategy_engine.record_exit(day_idx)
                    shares = 0
                    entry_price = 0.0
                    peak_price = 0.0
            
            else:
                should_buy, buy_reason = self.strategy_engine.should_buy(
                    score=score,
                    current_day=day_idx,
                    current_exposure=current_exposure,
                )
                
                if should_buy:
                    shares_to_buy = self.strategy_engine.get_position_size(cash, current_price)
                    
                    if shares_to_buy > 0 and shares_to_buy * current_price <= cash:
                        cost = shares_to_buy * current_price
                        cash -= cost
                        shares = shares_to_buy
                        entry_price = current_price
                        peak_price = current_price
                        
                        self.strategy_engine.record_entry(day_idx)
                        
                        trades.append(TradeRecord(
                            date=str(date.date()),
                            action="BUY",
                            price=current_price,
                            shares=shares,
                            reason=buy_reason,
                            regime=regime.value,
                            strategy=strategy.mode.value,
                            pnl=0.0,
                        ))
            
            portfolio_value = cash + shares * current_price
            
            if portfolio_value > portfolio_peak:
                portfolio_peak = portfolio_value
            drawdown = (portfolio_peak - portfolio_value) / portfolio_peak
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
                "strategy": strategy.mode.value,
                "score": score,
            })
        
        final_value = cash + shares * current_price if shares > 0 else cash
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        buy_hold_return = (last_price / first_price - 1) if first_price and last_price else 0
        capture_rate = (total_return / buy_hold_return * 100) if buy_hold_return != 0 else 0
        
        if len(daily_returns) > 0 and np.std(daily_returns) > 0:
            sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        winning_trades = sum(1 for t in trades if t.action == "SELL" and t.pnl > 0)
        total_sells = sum(1 for t in trades if t.action == "SELL")
        win_rate = winning_trades / total_sells if total_sells > 0 else 0.0
        
        return AdaptiveBacktestResult(
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
            buy_hold_return=buy_hold_return,
            capture_rate=capture_rate,
            trades=trades,
            regime_distribution=regime_counts,
            strategy_distribution=strategy_counts,
            daily_values=daily_values,
        )
