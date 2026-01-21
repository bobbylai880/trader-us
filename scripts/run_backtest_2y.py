"""2年回测脚本 - 使用PostgreSQL数据源运行自适应策略回测."""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_trader_assist.data_collector.pg_client import PostgresMarketDB, get_db
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
)


@dataclass
class TradeRecord:
    date: str
    symbol: str
    action: str
    price: float
    shares: int
    reason: str
    regime: str
    strategy: str
    pnl: float = 0.0
    pnl_pct: float = 0.0


@dataclass
class PortfolioBacktestResult:
    start_date: str
    end_date: str
    initial_capital: float
    final_value: float
    total_return: float
    annualized_return: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    total_trades: int
    spy_return: float
    alpha: float
    trades: List[TradeRecord] = field(default_factory=list)
    regime_distribution: Dict[str, int] = field(default_factory=dict)
    monthly_returns: Dict[str, float] = field(default_factory=dict)
    daily_values: List[Dict] = field(default_factory=list)


class PostgresBacktester:
    
    def __init__(
        self,
        db: PostgresMarketDB,
        initial_capital: float = 100000.0,
        max_positions: int = 5,
    ):
        self.db = db
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.regime_detector = MarketRegimeDetector()
        self.strategy_engines: Dict[str, AdaptiveStrategyEngine] = {}
    
    def _get_strategy_engine(self, symbol: str) -> AdaptiveStrategyEngine:
        if symbol not in self.strategy_engines:
            self.strategy_engines[symbol] = AdaptiveStrategyEngine()
        return self.strategy_engines[symbol]
    
    def _load_data(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        lookback_start = start_date - timedelta(days=250)
        
        prices_df = self.db.get_daily_prices(symbols + ["SPY"], lookback_start, end_date)
        indicators_df = self.db.get_indicators(symbols, lookback_start, end_date)
        spy_df = self.db.get_daily_prices_single("SPY", lookback_start, end_date)
        
        return prices_df, indicators_df, spy_df
    
    def _detect_regime(self, spy_data: pd.DataFrame, current_idx: int) -> MarketRegime:
        if len(spy_data) < 50:
            return MarketRegime.RANGE_BOUND
        
        spy_slice = spy_data.iloc[:current_idx + 1].tail(250)
        close = spy_slice["close"]
        
        sma50 = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()
        
        current_price = float(close.iloc[-1])
        sma50_val = float(sma50.iloc[-1]) if not pd.isna(sma50.iloc[-1]) else current_price
        sma200_val = float(sma200.iloc[-1]) if not pd.isna(sma200.iloc[-1]) else current_price
        
        momentum_20d = (current_price / float(close.iloc[-20]) - 1) if len(close) >= 20 else 0
        slope_5d = (float(sma50.iloc[-1]) - float(sma50.iloc[-5])) / 5 / sma50_val if len(sma50) >= 5 and sma50_val > 0 else 0
        
        signals = RegimeSignals(
            spy_vs_sma200=(current_price - sma200_val) / sma200_val * 100 if sma200_val > 0 else 0,
            spy_vs_sma50=(current_price - sma50_val) / sma50_val * 100 if sma50_val > 0 else 0,
            sma50_slope=slope_5d,
            breadth=0.5 + momentum_20d * 2,
            nh_nl_ratio=1.0 + momentum_20d * 5,
            vix_value=20.0 - momentum_20d * 50,
            vix_term_contango=momentum_20d > -0.05,
            spy_momentum_20d=momentum_20d,
            qqq_momentum_20d=momentum_20d * 1.1,
        )
        
        result = self.regime_detector.detect(signals)
        return result.regime
    
    def _calculate_score(
        self,
        symbol: str,
        prices_df: pd.DataFrame,
        indicators_df: pd.DataFrame,
        current_date: date,
        regime: MarketRegime,
    ) -> float:
        symbol_prices = prices_df[prices_df["symbol"] == symbol]
        current_ts = pd.Timestamp(current_date)
        symbol_indicators = indicators_df[
            (indicators_df["symbol"] == symbol) &
            (pd.to_datetime(indicators_df["trade_date"]) <= current_ts)
        ]
        
        if symbol_indicators.empty or symbol_prices.empty:
            return 0.0
        
        latest_ind = symbol_indicators.iloc[-1]
        latest_price = symbol_prices[symbol_prices["trade_date"] <= pd.Timestamp(current_date)].iloc[-1]
        
        trend_score = 0.0
        current_price = float(latest_price["close"])
        sma50 = latest_ind.get("sma_50")
        sma200 = latest_ind.get("sma_200")
        
        if pd.notna(sma50) and current_price > sma50:
            trend_score += 0.5
        if pd.notna(sma200) and current_price > sma200:
            trend_score += 0.5
        
        rsi = latest_ind.get("rsi_14", 50)
        if pd.isna(rsi):
            rsi = 50
        if 40 < rsi < 70:
            momentum_score = 0.7
        elif rsi > 70:
            momentum_score = 0.4
        elif rsi < 30:
            momentum_score = 0.8
        else:
            momentum_score = 0.5
        
        momentum_10d = latest_ind.get("momentum_10d", 0)
        if pd.isna(momentum_10d):
            momentum_10d = 0
        rs_score = 0.5 + min(1, max(-1, momentum_10d * 10)) * 0.5
        
        bb_lower = latest_ind.get("bb_lower")
        bb_upper = latest_ind.get("bb_upper")
        if pd.notna(bb_lower) and pd.notna(bb_upper) and bb_upper > bb_lower:
            bb_score = 1 - (current_price - bb_lower) / (bb_upper - bb_lower)
            bb_score = max(0, min(1, bb_score))
        else:
            bb_score = 0.5
        
        if regime in [MarketRegime.BULL_TREND, MarketRegime.BULL_PULLBACK]:
            score = trend_score * 0.4 + momentum_score * 0.3 + rs_score * 0.3
        elif regime == MarketRegime.RANGE_BOUND:
            score = bb_score * 0.4 + momentum_score * 0.3 + rs_score * 0.3
        else:
            score = momentum_score * 0.4 + rs_score * 0.3 + (1 - trend_score) * 0.3
        
        return min(1.0, max(0.0, score))
    
    def run_backtest(
        self,
        symbols: List[str],
        start_date: str = "2024-01-20",
        end_date: str = "2026-01-20",
    ) -> PortfolioBacktestResult:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
        
        print(f"加载数据: {len(symbols)} 只股票, {start} ~ {end}")
        prices_df, indicators_df, spy_df = self._load_data(symbols, start, end)
        
        if prices_df.empty:
            raise ValueError("No price data available")
        
        spy_backtest = spy_df[spy_df["trade_date"] >= pd.Timestamp(start)].copy()
        spy_backtest = spy_backtest.sort_values("trade_date").reset_index(drop=True)
        
        if spy_backtest.empty:
            raise ValueError("No SPY data for backtest period")
        
        trading_dates = spy_backtest["trade_date"].tolist()
        
        cash = self.initial_capital
        positions: Dict[str, Dict[str, Any]] = {}
        
        trades: List[TradeRecord] = []
        daily_values: List[Dict] = []
        regime_counts: Dict[str, int] = {r.value: 0 for r in MarketRegime}
        
        portfolio_peak = self.initial_capital
        max_drawdown = 0.0
        daily_returns: List[float] = []
        prev_value = self.initial_capital
        
        monthly_returns: Dict[str, float] = {}
        month_start_value = self.initial_capital
        current_month = None
        
        print(f"开始回测: {len(trading_dates)} 个交易日")
        
        for day_idx, trade_date in enumerate(trading_dates):
            current_date = trade_date.date() if hasattr(trade_date, 'date') else trade_date
            
            spy_idx = spy_backtest[spy_backtest["trade_date"] <= trade_date].index[-1]
            regime = self._detect_regime(spy_backtest, spy_idx)
            regime_counts[regime.value] += 1
            
            day_prices: Dict[str, float] = {}
            for symbol in symbols:
                symbol_prices = prices_df[
                    (prices_df["symbol"] == symbol) &
                    (prices_df["trade_date"] <= trade_date)
                ]
                if not symbol_prices.empty:
                    day_prices[symbol] = float(symbol_prices.iloc[-1]["close"])
            
            for symbol in list(positions.keys()):
                if symbol not in day_prices:
                    continue
                
                current_price = day_prices[symbol]
                pos = positions[symbol]
                
                if current_price > pos["peak_price"]:
                    pos["peak_price"] = current_price
                
                engine = self._get_strategy_engine(symbol)
                engine.update_strategy(regime, day_idx)
                
                score = self._calculate_score(symbol, prices_df, indicators_df, current_date, regime)
                
                should_sell, sell_reason = engine.should_sell(
                    score=score,
                    entry_price=pos["entry_price"],
                    current_price=current_price,
                    peak_price=pos["peak_price"],
                    current_day=day_idx,
                    bb_score=0.5,
                )
                
                if should_sell:
                    pnl = (current_price - pos["entry_price"]) * pos["shares"]
                    pnl_pct = (current_price - pos["entry_price"]) / pos["entry_price"]
                    cash += pos["shares"] * current_price
                    
                    trades.append(TradeRecord(
                        date=str(current_date),
                        symbol=symbol,
                        action="SELL",
                        price=current_price,
                        shares=pos["shares"],
                        reason=sell_reason,
                        regime=regime.value,
                        strategy=engine.current_strategy.mode.value if engine.current_strategy else "",
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                    ))
                    
                    engine.record_exit(day_idx)
                    del positions[symbol]
            
            if len(positions) < self.max_positions:
                candidates: List[Tuple[str, float]] = []
                
                for symbol in symbols:
                    if symbol in positions or symbol not in day_prices:
                        continue
                    
                    engine = self._get_strategy_engine(symbol)
                    engine.update_strategy(regime, day_idx)
                    
                    score = self._calculate_score(symbol, prices_df, indicators_df, current_date, regime)
                    
                    current_exposure = sum(
                        pos["shares"] * day_prices.get(s, 0) 
                        for s, pos in positions.items()
                    ) / (cash + sum(pos["shares"] * day_prices.get(s, 0) for s, pos in positions.items()))
                    
                    should_buy, buy_reason = engine.should_buy(
                        score=score,
                        current_day=day_idx,
                        current_exposure=current_exposure,
                    )
                    
                    if should_buy:
                        candidates.append((symbol, score))
                
                candidates.sort(key=lambda x: x[1], reverse=True)
                
                for symbol, score in candidates[:self.max_positions - len(positions)]:
                    if symbol not in day_prices:
                        continue
                    
                    current_price = day_prices[symbol]
                    engine = self._get_strategy_engine(symbol)
                    
                    position_budget = cash * 0.2
                    shares_to_buy = int(position_budget / current_price)
                    
                    if shares_to_buy > 0 and shares_to_buy * current_price <= cash:
                        cost = shares_to_buy * current_price
                        cash -= cost
                        
                        positions[symbol] = {
                            "shares": shares_to_buy,
                            "entry_price": current_price,
                            "peak_price": current_price,
                            "entry_day": day_idx,
                        }
                        
                        engine.record_entry(day_idx)
                        
                        trades.append(TradeRecord(
                            date=str(current_date),
                            symbol=symbol,
                            action="BUY",
                            price=current_price,
                            shares=shares_to_buy,
                            reason=f"score_{score:.2f}",
                            regime=regime.value,
                            strategy=engine.current_strategy.mode.value if engine.current_strategy else "",
                        ))
            
            position_value = sum(
                pos["shares"] * day_prices.get(s, pos["entry_price"])
                for s, pos in positions.items()
            )
            portfolio_value = cash + position_value
            
            if portfolio_value > portfolio_peak:
                portfolio_peak = portfolio_value
            drawdown = (portfolio_peak - portfolio_value) / portfolio_peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
            
            daily_return = (portfolio_value - prev_value) / prev_value if prev_value > 0 else 0
            daily_returns.append(daily_return)
            prev_value = portfolio_value
            
            month_key = current_date.strftime("%Y-%m")
            if current_month != month_key:
                if current_month is not None:
                    monthly_returns[current_month] = (prev_value / month_start_value - 1) * 100
                current_month = month_key
                month_start_value = portfolio_value
            
            daily_values.append({
                "date": str(current_date),
                "portfolio_value": portfolio_value,
                "cash": cash,
                "position_count": len(positions),
                "regime": regime.value,
            })
            
            if day_idx % 50 == 0:
                print(f"  Day {day_idx}/{len(trading_dates)}: ${portfolio_value:,.0f} ({regime.value})")
        
        if current_month:
            monthly_returns[current_month] = (portfolio_value / month_start_value - 1) * 100
        
        for symbol, pos in positions.items():
            if symbol in day_prices:
                pnl = (day_prices[symbol] - pos["entry_price"]) * pos["shares"]
                cash += pos["shares"] * day_prices[symbol]
        
        final_value = cash
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        years = (end - start).days / 365.25
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else total_return
        
        spy_first = float(spy_backtest.iloc[0]["close"])
        spy_last = float(spy_backtest.iloc[-1]["close"])
        spy_return = (spy_last / spy_first - 1)
        
        alpha = total_return - spy_return
        
        if len(daily_returns) > 0 and np.std(daily_returns) > 0:
            sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        winning_trades = sum(1 for t in trades if t.action == "SELL" and t.pnl > 0)
        total_sells = sum(1 for t in trades if t.action == "SELL")
        win_rate = winning_trades / total_sells if total_sells > 0 else 0.0
        
        return PortfolioBacktestResult(
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_value=final_value,
            total_return=total_return,
            annualized_return=annualized_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            win_rate=win_rate,
            total_trades=len(trades),
            spy_return=spy_return,
            alpha=alpha,
            trades=trades,
            regime_distribution=regime_counts,
            monthly_returns=monthly_returns,
            daily_values=daily_values,
        )


def print_report(result: PortfolioBacktestResult) -> None:
    print("\n" + "=" * 70)
    print("2年回测报告 - 自适应策略")
    print("=" * 70)
    
    print(f"\n【回测区间】")
    print(f"  起始日期: {result.start_date}")
    print(f"  结束日期: {result.end_date}")
    
    print(f"\n【收益概览】")
    print(f"  初始资金: ${result.initial_capital:,.0f}")
    print(f"  最终价值: ${result.final_value:,.0f}")
    print(f"  总收益率: {result.total_return:+.2%}")
    print(f"  年化收益: {result.annualized_return:+.2%}")
    print(f"  SPY收益:  {result.spy_return:+.2%}")
    print(f"  超额收益: {result.alpha:+.2%}")
    
    print(f"\n【风险指标】")
    print(f"  最大回撤: {result.max_drawdown:.2%}")
    print(f"  夏普比率: {result.sharpe_ratio:.2f}")
    print(f"  胜率:     {result.win_rate:.1%}")
    print(f"  交易次数: {result.total_trades}")
    
    print(f"\n【市场状态分布】")
    total_days = sum(result.regime_distribution.values())
    for regime, count in sorted(result.regime_distribution.items(), key=lambda x: -x[1]):
        pct = count / total_days * 100 if total_days > 0 else 0
        bar = "█" * int(pct / 5)
        print(f"  {regime:20s}: {count:4d}天 ({pct:5.1f}%) {bar}")
    
    print(f"\n【月度收益】")
    for month, ret in sorted(result.monthly_returns.items()):
        bar_char = "█" if ret >= 0 else "▓"
        bar_len = int(abs(ret) / 2)
        bar = bar_char * bar_len
        sign = "+" if ret >= 0 else ""
        print(f"  {month}: {sign}{ret:6.2f}% {bar}")
    
    if result.trades:
        print(f"\n【最大盈利交易 Top 5】")
        sells = [t for t in result.trades if t.action == "SELL"]
        top_winners = sorted(sells, key=lambda x: x.pnl, reverse=True)[:5]
        for t in top_winners:
            print(f"  {t.date} {t.symbol:5s} ${t.pnl:+,.0f} ({t.pnl_pct:+.1%}) - {t.reason}")
        
        print(f"\n【最大亏损交易 Top 5】")
        top_losers = sorted(sells, key=lambda x: x.pnl)[:5]
        for t in top_losers:
            print(f"  {t.date} {t.symbol:5s} ${t.pnl:+,.0f} ({t.pnl_pct:+.1%}) - {t.reason}")
    
    print("\n" + "=" * 70)


def main():
    db = get_db()
    
    symbols = [
        "NVDA", "AAPL", "MSFT", "AMD", "GOOGL", "META", "AMZN", "TSLA",
        "AVGO", "CRM", "ADBE", "NFLX", "QCOM", "MU",
        "JPM", "GS", "V", "MA",
        "UNH", "LLY", "JNJ",
        "XOM", "CVX",
        "HD", "COST", "WMT",
        "CAT", "BA", "GE",
    ]
    
    backtester = PostgresBacktester(
        db=db,
        initial_capital=100000.0,
        max_positions=5,
    )
    
    result = backtester.run_backtest(
        symbols=symbols,
        start_date="2024-01-20",
        end_date="2026-01-17",
    )
    
    print_report(result)
    
    run_id = db.save_backtest_run(
        name="2Y_Adaptive_Strategy",
        strategy="adaptive_regime",
        symbols=symbols,
        start_date=datetime.strptime(result.start_date, "%Y-%m-%d").date(),
        end_date=datetime.strptime(result.end_date, "%Y-%m-%d").date(),
        initial_capital=result.initial_capital,
        final_capital=result.final_value,
        total_return=result.total_return,
        max_drawdown=result.max_drawdown,
        sharpe_ratio=result.sharpe_ratio,
        win_rate=result.win_rate,
        total_trades=result.total_trades,
        config={"max_positions": 5, "symbols_count": len(symbols)},
        monthly_returns=result.monthly_returns,
    )
    
    if run_id and result.trades:
        trade_records = [
            {
                "symbol": t.symbol,
                "trade_date": t.date,
                "action": t.action,
                "shares": t.shares,
                "price": t.price,
                "reason": t.reason,
                "pnl": t.pnl,
                "pnl_pct": t.pnl_pct,
            }
            for t in result.trades
        ]
        db.save_backtest_trades(run_id, trade_records)
        print(f"\n回测结果已保存到数据库 (run_id: {run_id})")
    
    return result


if __name__ == "__main__":
    main()
