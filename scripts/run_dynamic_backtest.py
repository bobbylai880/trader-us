"""动态选股回测脚本 - 使用 UniverseBuilder 进行自适应选股."""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_trader_assist.data_collector.pg_client import PostgresMarketDB, get_db
from ai_trader_assist.universe.builder import UniverseBuilder, UniverseConfig, StockCandidate, PoolType
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
    pool: str = ""
    pnl: float = 0.0
    pnl_pct: float = 0.0


@dataclass
class DynamicBacktestResult:
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
    pool_distribution: Dict[str, int] = field(default_factory=dict)
    monthly_returns: Dict[str, float] = field(default_factory=dict)
    universe_changes: int = 0


class DynamicUniverseBacktester:
    
    def __init__(
        self,
        db: Optional[PostgresMarketDB] = None,
        initial_capital: float = 100000.0,
        max_positions: int = 5,
        rotation_frequency_days: int = 5,
    ):
        self.db = db or get_db()
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.rotation_frequency_days = rotation_frequency_days
        self.regime_detector = MarketRegimeDetector()
        self.universe_builder = UniverseBuilder(db=self.db)
        self.strategy_engines: Dict[str, AdaptiveStrategyEngine] = {}
        
        self._prices_cache: Dict[str, pd.DataFrame] = {}
        self._indicators_cache: Dict[str, pd.DataFrame] = {}
        
        self._core_symbols = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "UNH", "JNJ"]
        self._sector_etfs = ["XLK", "XLC", "XLY", "XLF", "XLV", "XLE", "XLI", "XLP", "XLU", "XLB", "XLRE"]
        self._all_symbols: List[str] = []
    
    def _build_fast_universe(self, trade_date: date, regime: MarketRegime) -> List[StockCandidate]:
        candidates: List[StockCandidate] = []
        
        for symbol in self._core_symbols:
            if symbol in self._prices_cache:
                candidates.append(StockCandidate(
                    symbol=symbol,
                    pool=PoolType.CORE,
                    sector_etf=None,
                    score=0.8,
                ))
        
        sector_scores: List[Tuple[str, float]] = []
        for etf in self._sector_etfs:
            ret_20d = self._calc_return(etf, trade_date, 20)
            ret_5d = self._calc_return(etf, trade_date, 5)
            spy_ret_20d = self._calc_return("SPY", trade_date, 20)
            rs_vs_spy = ret_20d - spy_ret_20d
            composite = 0.5 * ret_20d + 0.3 * rs_vs_spy + 0.2 * ret_5d
            sector_scores.append((etf, composite))
        
        sector_scores.sort(key=lambda x: x[1], reverse=True)
        
        k = 4 if regime in [MarketRegime.BULL_TREND, MarketRegime.BULL_PULLBACK] else 2
        top_sectors = [s[0] for s in sector_scores[:k] if s[1] > 0]
        
        if regime in [MarketRegime.BEAR_TREND, MarketRegime.BEAR_RALLY]:
            for d in ["XLP", "XLV", "XLU"]:
                if d not in top_sectors:
                    top_sectors.append(d)
        
        stock_scores: List[Tuple[str, float, PoolType]] = []
        rotation_symbols = set(self._core_symbols)
        
        for symbol in self._all_symbols:
            if symbol in rotation_symbols or symbol in self._sector_etfs:
                continue
            
            momentum = self._get_indicator(symbol, trade_date, "momentum_10d") or 0
            volume_ratio = self._get_indicator(symbol, trade_date, "volume_ratio") or 1
            rsi = self._get_indicator(symbol, trade_date, "rsi_14") or 50
            sma50 = self._get_indicator(symbol, trade_date, "sma_50")
            price = self._get_price(symbol, trade_date)
            
            trend_above_sma50 = price and sma50 and price > sma50
            
            if regime in [MarketRegime.BULL_TREND, MarketRegime.BULL_PULLBACK]:
                if momentum > 0.02 and volume_ratio >= 1.0 and trend_above_sma50:
                    score = momentum * 0.5 + (volume_ratio - 1) * 0.3 + (rsi - 50) / 100 * 0.2
                    stock_scores.append((symbol, score, PoolType.ROTATION))
                elif momentum > 0 and volume_ratio >= 0.8:
                    score = momentum * 0.4 + (volume_ratio - 1) * 0.3
                    stock_scores.append((symbol, score, PoolType.CANDIDATE))
            elif regime == MarketRegime.RANGE_BOUND:
                if 30 <= rsi <= 50 and volume_ratio >= 0.8 and trend_above_sma50:
                    score = (50 - rsi) / 50 * 0.4 + volume_ratio * 0.3 + momentum * 0.3
                    stock_scores.append((symbol, score, PoolType.ROTATION))
            else:
                if momentum > -0.02 and rsi < 60 and volume_ratio >= 0.6:
                    score = (60 - rsi) / 60 * 0.4 + (1 - abs(momentum)) * 0.3 + volume_ratio * 0.3
                    stock_scores.append((symbol, score, PoolType.CANDIDATE))
        
        stock_scores.sort(key=lambda x: x[1], reverse=True)
        
        rotation_count = 0
        candidate_count = 0
        max_rotation = 18
        max_candidate = 12
        
        for symbol, score, pool in stock_scores:
            if pool == PoolType.ROTATION and rotation_count < max_rotation:
                candidates.append(StockCandidate(symbol=symbol, pool=pool, score=score))
                rotation_count += 1
            elif pool == PoolType.CANDIDATE and candidate_count < max_candidate:
                candidates.append(StockCandidate(symbol=symbol, pool=pool, score=score))
                candidate_count += 1
            
            if rotation_count >= max_rotation and candidate_count >= max_candidate:
                break
        
        return candidates
    
    def _calc_return(self, symbol: str, trade_date: date, days: int) -> float:
        if symbol not in self._prices_cache:
            return 0.0
        df = self._prices_cache[symbol]
        if df.empty:
            return 0.0
        
        mask = pd.to_datetime(df["trade_date"]).dt.date <= trade_date
        filtered = df[mask].tail(days + 1)
        if len(filtered) < 2:
            return 0.0
        
        start_price = float(filtered.iloc[0]["close"])
        end_price = float(filtered.iloc[-1]["close"])
        return (end_price / start_price - 1) if start_price > 0 else 0.0
    
    def _get_strategy_engine(self, symbol: str) -> AdaptiveStrategyEngine:
        if symbol not in self.strategy_engines:
            self.strategy_engines[symbol] = AdaptiveStrategyEngine()
        return self.strategy_engines[symbol]
    
    def _preload_all_data(self, start_date: date, end_date: date) -> None:
        lookback_start = start_date - timedelta(days=250)
        
        all_symbols_query = "SELECT DISTINCT symbol FROM daily_prices WHERE symbol NOT LIKE '^%'"
        with self.db.get_cursor() as cur:
            cur.execute(all_symbols_query)
            all_symbols = [row[0] for row in cur.fetchall()]
        
        all_symbols.append("SPY")
        
        print(f"预加载数据: {len(all_symbols)} 只股票...")
        prices_df = self.db.get_daily_prices(all_symbols, lookback_start, end_date)
        indicators_df = self.db.get_indicators(all_symbols, lookback_start, end_date)
        
        self._all_symbols = [s for s in all_symbols if s not in self._sector_etfs and s != "SPY"]
        
        for symbol in all_symbols:
            self._prices_cache[symbol] = prices_df[prices_df["symbol"] == symbol].copy()
            self._indicators_cache[symbol] = indicators_df[indicators_df["symbol"] == symbol].copy()
        
        print(f"数据加载完成: {len(prices_df)} 行价格, {len(indicators_df)} 行指标")
    
    def _get_price(self, symbol: str, trade_date: date) -> Optional[float]:
        if symbol not in self._prices_cache:
            return None
        df = self._prices_cache[symbol]
        if df.empty:
            return None
        
        mask = pd.to_datetime(df["trade_date"]).dt.date <= trade_date
        filtered = df[mask]
        if filtered.empty:
            return None
        return float(filtered.iloc[-1]["close"])
    
    def _get_indicator(self, symbol: str, trade_date: date, field: str) -> Optional[float]:
        if symbol not in self._indicators_cache:
            return None
        df = self._indicators_cache[symbol]
        if df.empty:
            return None
        
        mask = pd.to_datetime(df["trade_date"]).dt.date <= trade_date
        filtered = df[mask]
        if filtered.empty:
            return None
        val = filtered.iloc[-1].get(field)
        return float(val) if pd.notna(val) else None
    
    def _detect_regime(self, trade_date: date) -> MarketRegime:
        if "SPY" not in self._prices_cache:
            return MarketRegime.RANGE_BOUND
        
        spy_df = self._prices_cache["SPY"]
        mask = pd.to_datetime(spy_df["trade_date"]).dt.date <= trade_date
        spy_slice = spy_df[mask].tail(250)
        
        if len(spy_slice) < 50:
            return MarketRegime.RANGE_BOUND
        
        close = spy_slice["close"].astype(float)
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
    
    def _calculate_score(self, symbol: str, trade_date: date, regime: MarketRegime) -> float:
        momentum = self._get_indicator(symbol, trade_date, "momentum_10d") or 0
        rsi = self._get_indicator(symbol, trade_date, "rsi_14") or 50
        volume_ratio = self._get_indicator(symbol, trade_date, "volume_ratio") or 1
        sma50 = self._get_indicator(symbol, trade_date, "sma_50")
        sma200 = self._get_indicator(symbol, trade_date, "sma_200")
        price = self._get_price(symbol, trade_date) or 0
        
        trend_score = 0.0
        if sma50 and price > sma50:
            trend_score += 0.5
        if sma200 and price > sma200:
            trend_score += 0.5
        
        if 40 < rsi < 70:
            momentum_score = 0.7
        elif rsi > 70:
            momentum_score = 0.4
        elif rsi < 30:
            momentum_score = 0.8
        else:
            momentum_score = 0.5
        
        rs_score = 0.5 + min(1, max(-1, momentum * 10)) * 0.5
        
        bb_lower = self._get_indicator(symbol, trade_date, "bb_lower")
        bb_upper = self._get_indicator(symbol, trade_date, "bb_upper")
        if bb_lower and bb_upper and bb_upper > bb_lower and price:
            bb_score = 1 - (price - bb_lower) / (bb_upper - bb_lower)
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
        start_date: str = "2024-01-20",
        end_date: str = "2026-01-17",
    ) -> DynamicBacktestResult:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
        
        self._preload_all_data(start, end)
        
        spy_df = self._prices_cache.get("SPY", pd.DataFrame())
        if spy_df.empty:
            raise ValueError("No SPY data available")
        
        spy_backtest = spy_df[pd.to_datetime(spy_df["trade_date"]).dt.date >= start].copy()
        spy_backtest = spy_backtest.sort_values("trade_date").reset_index(drop=True)
        trading_dates = [d.date() if hasattr(d, 'date') else d for d in spy_backtest["trade_date"].tolist()]
        
        cash = self.initial_capital
        positions: Dict[str, Dict[str, Any]] = {}
        
        trades: List[TradeRecord] = []
        regime_counts: Dict[str, int] = {r.value: 0 for r in MarketRegime}
        pool_counts: Dict[str, int] = {p.value: 0 for p in PoolType}
        
        portfolio_peak = self.initial_capital
        max_drawdown = 0.0
        daily_returns: List[float] = []
        prev_value = self.initial_capital
        
        monthly_returns: Dict[str, float] = {}
        month_start_value = self.initial_capital
        current_month: Optional[str] = None
        
        current_universe: List[StockCandidate] = []
        last_rotation_date: Optional[date] = None
        universe_changes = 0
        cached_universe_symbols: Set[str] = set()
        
        initial_universe = self._build_fast_universe(trading_dates[0], MarketRegime.RANGE_BOUND)
        current_universe = initial_universe
        cached_universe_symbols = set(c.symbol for c in current_universe)
        
        print(f"开始动态选股回测: {len(trading_dates)} 个交易日")
        
        for day_idx, trade_date in enumerate(trading_dates):
            regime = self._detect_regime(trade_date)
            regime_counts[regime.value] += 1
            
            should_rotate = (
                last_rotation_date is None or 
                (trade_date - last_rotation_date).days >= self.rotation_frequency_days
            )
            
            if should_rotate:
                new_universe = self._build_fast_universe(trade_date, regime)
                new_symbols = set(c.symbol for c in new_universe)
                if new_symbols != cached_universe_symbols:
                    universe_changes += 1
                    cached_universe_symbols = new_symbols
                current_universe = new_universe
                last_rotation_date = trade_date
            
            universe_symbols = {c.symbol: c for c in current_universe}
            
            for symbol in list(positions.keys()):
                price = self._get_price(symbol, trade_date)
                if not price:
                    continue
                
                pos = positions[symbol]
                if price > pos["peak_price"]:
                    pos["peak_price"] = price
                
                engine = self._get_strategy_engine(symbol)
                engine.update_strategy(regime, day_idx)
                
                score = self._calculate_score(symbol, trade_date, regime)
                
                force_sell = symbol not in universe_symbols and pos.get("hold_days", 0) > 10
                
                should_sell, sell_reason = engine.should_sell(
                    score=score,
                    entry_price=pos["entry_price"],
                    current_price=price,
                    peak_price=pos["peak_price"],
                    current_day=day_idx,
                    bb_score=0.5,
                )
                
                if should_sell or force_sell:
                    pnl = (price - pos["entry_price"]) * pos["shares"]
                    pnl_pct = (price - pos["entry_price"]) / pos["entry_price"]
                    cash += pos["shares"] * price
                    
                    reason = "universe_exit" if force_sell else sell_reason
                    
                    trades.append(TradeRecord(
                        date=str(trade_date),
                        symbol=symbol,
                        action="SELL",
                        price=price,
                        shares=pos["shares"],
                        reason=reason,
                        regime=regime.value,
                        strategy=engine.current_strategy.mode.value if engine.current_strategy else "",
                        pool=pos.get("pool", ""),
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                    ))
                    
                    engine.record_exit(day_idx)
                    del positions[symbol]
                else:
                    pos["hold_days"] = pos.get("hold_days", 0) + 1
            
            if len(positions) < self.max_positions:
                candidates: List[Tuple[str, float, StockCandidate]] = []
                
                for symbol, candidate in universe_symbols.items():
                    if symbol in positions:
                        continue
                    
                    price = self._get_price(symbol, trade_date)
                    if not price:
                        continue
                    
                    engine = self._get_strategy_engine(symbol)
                    engine.update_strategy(regime, day_idx)
                    
                    score = self._calculate_score(symbol, trade_date, regime)
                    
                    pool_boost = 0.1 if candidate.pool == PoolType.CORE else 0
                    adjusted_score = score + pool_boost
                    
                    total_value = cash + sum(
                        pos["shares"] * (self._get_price(s, trade_date) or pos["entry_price"])
                        for s, pos in positions.items()
                    )
                    current_exposure = (total_value - cash) / total_value if total_value > 0 else 0
                    
                    should_buy, buy_reason = engine.should_buy(
                        score=score,
                        current_day=day_idx,
                        current_exposure=current_exposure,
                    )
                    
                    if should_buy:
                        candidates.append((symbol, adjusted_score, candidate))
                
                candidates.sort(key=lambda x: x[1], reverse=True)
                
                for symbol, adj_score, candidate in candidates[:self.max_positions - len(positions)]:
                    price = self._get_price(symbol, trade_date)
                    if not price:
                        continue
                    
                    engine = self._get_strategy_engine(symbol)
                    
                    position_budget = cash * 0.2
                    shares_to_buy = int(position_budget / price)
                    
                    if shares_to_buy > 0 and shares_to_buy * price <= cash:
                        cost = shares_to_buy * price
                        cash -= cost
                        
                        positions[symbol] = {
                            "shares": shares_to_buy,
                            "entry_price": price,
                            "peak_price": price,
                            "entry_day": day_idx,
                            "hold_days": 0,
                            "pool": candidate.pool.value,
                        }
                        
                        pool_counts[candidate.pool.value] = pool_counts.get(candidate.pool.value, 0) + 1
                        
                        engine.record_entry(day_idx)
                        
                        trades.append(TradeRecord(
                            date=str(trade_date),
                            symbol=symbol,
                            action="BUY",
                            price=price,
                            shares=shares_to_buy,
                            reason=f"score_{adj_score:.2f}",
                            regime=regime.value,
                            strategy=engine.current_strategy.mode.value if engine.current_strategy else "",
                            pool=candidate.pool.value,
                        ))
            
            position_value = sum(
                pos["shares"] * (self._get_price(s, trade_date) or pos["entry_price"])
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
            
            month_key = trade_date.strftime("%Y-%m")
            if current_month != month_key:
                if current_month is not None:
                    monthly_returns[current_month] = (prev_value / month_start_value - 1) * 100
                current_month = month_key
                month_start_value = portfolio_value
            
            if day_idx % 50 == 0:
                print(f"  Day {day_idx}/{len(trading_dates)}: ${portfolio_value:,.0f} "
                      f"({regime.value}) Universe={len(current_universe)}")
        
        if current_month:
            monthly_returns[current_month] = (portfolio_value / month_start_value - 1) * 100
        
        for symbol, pos in positions.items():
            price = self._get_price(symbol, trading_dates[-1])
            if price:
                cash += pos["shares"] * price
        
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
        
        return DynamicBacktestResult(
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
            pool_distribution=pool_counts,
            monthly_returns=monthly_returns,
            universe_changes=universe_changes,
        )


def print_comparison_report(
    dynamic_result: DynamicBacktestResult,
    static_result: Optional[Dict] = None,
) -> None:
    print("\n" + "=" * 70)
    print("动态选股 vs 固定池 回测对比报告")
    print("=" * 70)
    
    print(f"\n【回测区间】{dynamic_result.start_date} ~ {dynamic_result.end_date}")
    
    print(f"\n{'指标':<20} {'动态选股':>15} {'固定池(参考)':>15} {'差异':>10}")
    print("-" * 60)
    
    static_return = 0.3086 if static_result is None else static_result.get("total_return", 0)
    static_sharpe = 1.34 if static_result is None else static_result.get("sharpe_ratio", 0)
    static_drawdown = 0.1003 if static_result is None else static_result.get("max_drawdown", 0)
    static_win_rate = 0.475 if static_result is None else static_result.get("win_rate", 0)
    
    print(f"{'总收益率':<20} {dynamic_result.total_return:>14.2%} {static_return:>14.2%} "
          f"{(dynamic_result.total_return - static_return):>+9.2%}")
    print(f"{'年化收益':<20} {dynamic_result.annualized_return:>14.2%}")
    print(f"{'SPY收益':<20} {dynamic_result.spy_return:>14.2%}")
    print(f"{'超额收益 (Alpha)':<20} {dynamic_result.alpha:>14.2%}")
    print(f"{'最大回撤':<20} {dynamic_result.max_drawdown:>14.2%} {static_drawdown:>14.2%}")
    print(f"{'夏普比率':<20} {dynamic_result.sharpe_ratio:>14.2f} {static_sharpe:>14.2f}")
    print(f"{'胜率':<20} {dynamic_result.win_rate:>14.1%} {static_win_rate:>14.1%}")
    print(f"{'交易次数':<20} {dynamic_result.total_trades:>14}")
    print(f"{'Universe变更次数':<20} {dynamic_result.universe_changes:>14}")
    
    print(f"\n【交易来源分布】")
    total_buys = sum(1 for t in dynamic_result.trades if t.action == "BUY")
    for pool, count in sorted(dynamic_result.pool_distribution.items(), key=lambda x: -x[1]):
        pct = count / total_buys * 100 if total_buys > 0 else 0
        bar = "█" * int(pct / 5)
        print(f"  {pool:12s}: {count:4d} ({pct:5.1f}%) {bar}")
    
    print(f"\n【市场状态分布】")
    total_days = sum(dynamic_result.regime_distribution.values())
    for regime, count in sorted(dynamic_result.regime_distribution.items(), key=lambda x: -x[1]):
        pct = count / total_days * 100 if total_days > 0 else 0
        bar = "█" * int(pct / 5)
        print(f"  {regime:20s}: {count:4d}天 ({pct:5.1f}%) {bar}")
    
    print(f"\n【月度收益】")
    for month, ret in sorted(dynamic_result.monthly_returns.items()):
        bar_char = "█" if ret >= 0 else "▓"
        bar_len = int(abs(ret) / 2)
        bar = bar_char * bar_len
        sign = "+" if ret >= 0 else ""
        print(f"  {month}: {sign}{ret:6.2f}% {bar}")
    
    if dynamic_result.trades:
        print(f"\n【最大盈利交易 Top 5】")
        sells = [t for t in dynamic_result.trades if t.action == "SELL"]
        top_winners = sorted(sells, key=lambda x: x.pnl, reverse=True)[:5]
        for t in top_winners:
            print(f"  {t.date} {t.symbol:5s} [{t.pool:8s}] ${t.pnl:+,.0f} ({t.pnl_pct:+.1%})")
        
        print(f"\n【最大亏损交易 Top 5】")
        top_losers = sorted(sells, key=lambda x: x.pnl)[:5]
        for t in top_losers:
            print(f"  {t.date} {t.symbol:5s} [{t.pool:8s}] ${t.pnl:+,.0f} ({t.pnl_pct:+.1%})")
    
    print("\n" + "=" * 70)
    
    improvement = dynamic_result.total_return - static_return
    if improvement > 0:
        print(f"✅ 动态选股策略超越固定池 {improvement:+.2%}")
    else:
        print(f"⚠️ 动态选股策略落后固定池 {improvement:.2%}")
    
    print("=" * 70)


def main():
    backtester = DynamicUniverseBacktester(
        initial_capital=100000.0,
        max_positions=5,
        rotation_frequency_days=5,
    )
    
    result = backtester.run_backtest(
        start_date="2024-01-20",
        end_date="2026-01-17",
    )
    
    print_comparison_report(result)
    
    return result


if __name__ == "__main__":
    main()
