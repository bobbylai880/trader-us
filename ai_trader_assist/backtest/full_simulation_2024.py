"""
2024年完整闭环回测 - 模拟每一个交易日的完整决策流程

每日流程:
1. 获取市场数据 (SPY/QQQ/VIX)
2. 获取板块数据 (11个SPDR ETF)
3. 获取个股数据 + 技术指标
4. 模拟新闻情绪
5. 市场状态识别
6. 策略选择与交易决策
7. 记录每日报告
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
    bollinger_bands,
    bollinger_position_score,
)
from ai_trader_assist.risk_engine.market_regime import (
    MarketRegime,
    MarketRegimeDetector,
    RegimeSignals,
)
from ai_trader_assist.backtest.regime_strategies import (
    AdaptiveStrategyEngine,
    StrategyMode,
    get_strategy_for_regime,
)


MARKET_INDICES = ["SPY", "QQQ", "DIA", "IWM"]
SECTOR_ETFS = ["XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"]
WATCHLIST = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "AMD", "GOOGL", "TSLA"]

SECTOR_NAMES = {
    "XLK": "科技", "XLF": "金融", "XLV": "医疗", "XLE": "能源",
    "XLI": "工业", "XLY": "非必需消费", "XLP": "必需消费", "XLU": "公用事业",
    "XLB": "材料", "XLRE": "房地产", "XLC": "通信"
}


@dataclass
class DailyMarketSnapshot:
    date: str
    spy_price: float
    spy_change: float
    qqq_price: float
    qqq_change: float
    vix_value: float
    market_regime: str
    strategy_mode: str
    leading_sectors: List[str]
    lagging_sectors: List[str]


@dataclass
class DailyStockSnapshot:
    symbol: str
    price: float
    change_1d: float
    change_5d: float
    rsi: float
    bb_position: float
    relative_strength: float
    news_sentiment: float
    score: float
    signal: str


@dataclass
class DailyTradeAction:
    symbol: str
    action: str
    shares: int
    price: float
    reason: str
    pnl: float = 0.0


@dataclass
class DailyReport:
    date: str
    market: DailyMarketSnapshot
    stocks: List[DailyStockSnapshot]
    trades: List[DailyTradeAction]
    portfolio_value: float
    cash: float
    positions: Dict[str, Dict]
    daily_return: float
    cumulative_return: float


@dataclass 
class FullYearSimulationResult:
    start_date: str
    end_date: str
    initial_capital: float
    final_value: float
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    total_trades: int
    win_rate: float
    daily_reports: List[DailyReport] = field(default_factory=list)
    monthly_returns: Dict[str, float] = field(default_factory=dict)
    regime_distribution: Dict[str, int] = field(default_factory=dict)
    strategy_distribution: Dict[str, int] = field(default_factory=dict)


class FullYearSimulator:
    """2024年完整闭环回测模拟器"""
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.yf_client = YahooFinanceClient()
        self.regime_detector = MarketRegimeDetector()
        self.strategy_engine = AdaptiveStrategyEngine()
        
        self.cash = initial_capital
        self.positions: Dict[str, Dict] = {}
        self.trade_history: List[DailyTradeAction] = []
        
    def _load_all_data(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """预加载所有需要的数据"""
        start = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=60)
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        data = {}
        all_symbols = MARKET_INDICES + SECTOR_ETFS + WATCHLIST
        
        for symbol in all_symbols:
            try:
                df = self.yf_client.fetch_history(symbol, start, end, interval="1d")
                if not df.empty:
                    data[symbol] = df
            except:
                pass
        
        return data
    
    def _get_price_on_date(self, data: Dict[str, pd.DataFrame], symbol: str, date: pd.Timestamp) -> Optional[float]:
        """获取指定日期的收盘价"""
        if symbol not in data:
            return None
        df = data[symbol]
        if date in df.index:
            price = df.loc[date, "Close"]
            if isinstance(price, pd.Series):
                return float(price.iloc[0])
            return float(price)
        return None
    
    def _calculate_change(self, data: Dict[str, pd.DataFrame], symbol: str, date: pd.Timestamp, days: int = 1) -> float:
        """计算N日涨跌幅"""
        if symbol not in data:
            return 0.0
        df = data[symbol]
        idx = df.index.get_loc(date) if date in df.index else -1
        if idx < days:
            return 0.0
        current = df.iloc[idx]["Close"]
        previous = df.iloc[idx - days]["Close"]
        if isinstance(current, pd.Series):
            current = current.iloc[0]
        if isinstance(previous, pd.Series):
            previous = previous.iloc[0]
        return (current / previous - 1) if previous > 0 else 0.0
    
    def _calculate_indicators(self, data: Dict[str, pd.DataFrame], symbol: str, date: pd.Timestamp) -> Dict[str, float]:
        """计算技术指标"""
        if symbol not in data:
            return {}
        
        df = data[symbol]
        idx = df.index.get_loc(date) if date in df.index else -1
        if idx < 20:
            return {}
        
        slice_df = df.iloc[:idx+1]
        close = slice_df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        
        rsi = calc_rsi(close, 14)
        rsi_val = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
        
        bb_score = bollinger_position_score(close, 20)
        bb_val = float(bb_score.iloc[-1]) if not pd.isna(bb_score.iloc[-1]) else 0.5
        
        atr = calc_atr(slice_df["High"], slice_df["Low"], close, 14)
        atr_val = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0.0
        
        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean() if len(close) >= 50 else sma20
        
        current_price = float(close.iloc[-1])
        
        return {
            "price": current_price,
            "rsi": rsi_val,
            "bb_position": bb_val,
            "atr": atr_val,
            "sma20": float(sma20.iloc[-1]) if not pd.isna(sma20.iloc[-1]) else current_price,
            "sma50": float(sma50.iloc[-1]) if not pd.isna(sma50.iloc[-1]) else current_price,
        }
    
    def _simulate_news_sentiment(self, symbol: str, price_change: float) -> float:
        """模拟新闻情绪 (基于价格动量)"""
        base_sentiment = price_change * 5
        noise = np.random.uniform(-0.1, 0.1)
        return max(-1.0, min(1.0, base_sentiment + noise))
    
    def _detect_regime(self, data: Dict[str, pd.DataFrame], date: pd.Timestamp) -> MarketRegime:
        """识别市场状态"""
        spy_df = data.get("SPY")
        if spy_df is None or date not in spy_df.index:
            return MarketRegime.UNKNOWN
        
        idx = spy_df.index.get_loc(date)
        if idx < 50:
            return MarketRegime.UNKNOWN
        
        slice_df = spy_df.iloc[:idx+1]
        close = slice_df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        
        sma50 = close.rolling(50).mean()
        sma200 = close.rolling(200).mean() if len(close) >= 200 else sma50
        
        current = float(close.iloc[-1])
        sma50_val = float(sma50.iloc[-1]) if not pd.isna(sma50.iloc[-1]) else current
        sma200_val = float(sma200.iloc[-1]) if not pd.isna(sma200.iloc[-1]) else current
        
        momentum_20d = (current / float(close.iloc[-20]) - 1) if len(close) >= 20 else 0
        
        signals = RegimeSignals(
            spy_vs_sma200=(current - sma200_val) / sma200_val * 100 if sma200_val > 0 else 0,
            spy_vs_sma50=(current - sma50_val) / sma50_val * 100 if sma50_val > 0 else 0,
            sma50_slope=(float(sma50.iloc[-1]) - float(sma50.iloc[-5])) / 5 / sma50_val if len(sma50) >= 5 and sma50_val > 0 else 0,
            breadth=0.5 + momentum_20d * 2,
            nh_nl_ratio=1.0 + momentum_20d * 5,
            vix_value=20.0,
            vix_term_contango=True,
            spy_momentum_20d=momentum_20d,
            qqq_momentum_20d=momentum_20d * 1.1,
        )
        
        result = self.regime_detector.detect(signals)
        return result.regime
    
    def _get_sector_rankings(self, data: Dict[str, pd.DataFrame], date: pd.Timestamp) -> Tuple[List[str], List[str]]:
        """获取板块排名"""
        sector_returns = {}
        for etf in SECTOR_ETFS:
            change = self._calculate_change(data, etf, date, 20)
            sector_returns[etf] = change
        
        sorted_sectors = sorted(sector_returns.items(), key=lambda x: x[1], reverse=True)
        leading = [s[0] for s in sorted_sectors[:3]]
        lagging = [s[0] for s in sorted_sectors[-3:]]
        
        return leading, lagging
    
    def _calculate_stock_score(self, indicators: Dict[str, float], regime: MarketRegime, rel_strength: float, sentiment: float) -> float:
        """计算个股评分"""
        rsi = indicators.get("rsi", 50)
        bb = indicators.get("bb_position", 0.5)
        price = indicators.get("price", 0)
        sma20 = indicators.get("sma20", price)
        sma50 = indicators.get("sma50", price)
        
        trend_score = 0.5
        if price > sma20:
            trend_score += 0.25
        if price > sma50:
            trend_score += 0.25
        
        if 40 < rsi < 70:
            momentum_score = 0.7
        elif rsi < 30:
            momentum_score = 0.8
        elif rsi > 70:
            momentum_score = 0.4
        else:
            momentum_score = 0.5
        
        rs_score = 0.5 + min(1, max(-1, rel_strength * 10)) * 0.5
        
        sentiment_score = 0.5 + sentiment * 0.3
        
        if regime in [MarketRegime.BULL_TREND, MarketRegime.BULL_PULLBACK]:
            score = trend_score * 0.35 + momentum_score * 0.25 + rs_score * 0.25 + sentiment_score * 0.15
        elif regime == MarketRegime.RANGE_BOUND:
            score = bb * 0.35 + momentum_score * 0.25 + rs_score * 0.25 + sentiment_score * 0.15
        else:
            score = momentum_score * 0.30 + rs_score * 0.30 + (1 - trend_score) * 0.25 + sentiment_score * 0.15
        
        return min(1.0, max(0.0, score))
    
    def _get_signal(self, score: float, regime: MarketRegime) -> str:
        """根据评分获取信号"""
        strategy = get_strategy_for_regime(regime)
        if score >= strategy.buy_threshold:
            return "BUY"
        elif score < 0.35:
            return "AVOID"
        else:
            return "HOLD"
    
    def _execute_trades(
        self,
        date: str,
        stocks: List[DailyStockSnapshot],
        regime: MarketRegime,
        day_idx: int,
    ) -> List[DailyTradeAction]:
        """执行交易决策"""
        trades = []
        strategy = self.strategy_engine.update_strategy(regime, day_idx)
        
        for symbol, pos in list(self.positions.items()):
            stock = next((s for s in stocks if s.symbol == symbol), None)
            if stock is None:
                continue
            
            current_price = stock.price
            entry_price = pos["entry_price"]
            peak_price = pos.get("peak_price", entry_price)
            
            if current_price > peak_price:
                pos["peak_price"] = current_price
                peak_price = current_price
            
            should_sell, reason = self.strategy_engine.should_sell(
                score=stock.score,
                entry_price=entry_price,
                current_price=current_price,
                peak_price=peak_price,
                current_day=day_idx,
                bb_score=stock.bb_position,
            )
            
            if should_sell:
                shares = pos["shares"]
                pnl = (current_price - entry_price) * shares
                self.cash += shares * current_price
                
                trades.append(DailyTradeAction(
                    symbol=symbol,
                    action="SELL",
                    shares=shares,
                    price=current_price,
                    reason=reason,
                    pnl=pnl,
                ))
                
                del self.positions[symbol]
                self.strategy_engine.record_exit(day_idx)
        
        portfolio_value = self.cash + sum(
            pos["shares"] * next((s.price for s in stocks if s.symbol == sym), pos["entry_price"])
            for sym, pos in self.positions.items()
        )
        current_exposure = (portfolio_value - self.cash) / portfolio_value if portfolio_value > 0 else 0
        
        buy_candidates = [s for s in stocks if s.signal == "BUY" and s.symbol not in self.positions]
        buy_candidates.sort(key=lambda x: x.score, reverse=True)
        
        for stock in buy_candidates[:2]:
            should_buy, reason = self.strategy_engine.should_buy(
                score=stock.score,
                current_day=day_idx,
                current_exposure=current_exposure,
            )
            
            if should_buy:
                shares = self.strategy_engine.get_position_size(self.cash, stock.price)
                if shares > 0 and shares * stock.price <= self.cash:
                    cost = shares * stock.price
                    self.cash -= cost
                    
                    self.positions[stock.symbol] = {
                        "shares": shares,
                        "entry_price": stock.price,
                        "entry_date": date,
                        "peak_price": stock.price,
                    }
                    
                    trades.append(DailyTradeAction(
                        symbol=stock.symbol,
                        action="BUY",
                        shares=shares,
                        price=stock.price,
                        reason=reason,
                    ))
                    
                    self.strategy_engine.record_entry(day_idx)
                    
                    portfolio_value = self.cash + sum(
                        pos["shares"] * next((s.price for s in stocks if s.symbol == sym), pos["entry_price"])
                        for sym, pos in self.positions.items()
                    )
                    current_exposure = (portfolio_value - self.cash) / portfolio_value if portfolio_value > 0 else 0
        
        return trades
    
    def run_simulation(
        self,
        start_date: str = "2024-01-01",
        end_date: str = "2024-12-31",
    ) -> FullYearSimulationResult:
        """运行完整年度模拟"""
        print("加载数据...")
        data = self._load_all_data(start_date, end_date)
        
        spy_df = data.get("SPY")
        if spy_df is None:
            raise ValueError("无法获取SPY数据")
        
        trading_dates = spy_df.loc[spy_df.index >= pd.Timestamp(start_date)].index
        
        self.cash = self.initial_capital
        self.positions = {}
        self.strategy_engine = AdaptiveStrategyEngine()
        
        daily_reports: List[DailyReport] = []
        regime_counts: Dict[str, int] = {r.value: 0 for r in MarketRegime}
        strategy_counts: Dict[str, int] = {m.value: 0 for m in StrategyMode}
        monthly_returns: Dict[str, float] = {}
        
        portfolio_peak = self.initial_capital
        max_drawdown = 0.0
        daily_returns: List[float] = []
        prev_value = self.initial_capital
        
        print(f"开始模拟 {len(trading_dates)} 个交易日...")
        
        for day_idx, date in enumerate(trading_dates):
            date_str = str(date.date())
            
            spy_price = self._get_price_on_date(data, "SPY", date) or 0
            spy_change = self._calculate_change(data, "SPY", date, 1)
            qqq_price = self._get_price_on_date(data, "QQQ", date) or 0
            qqq_change = self._calculate_change(data, "QQQ", date, 1)
            
            regime = self._detect_regime(data, date)
            regime_counts[regime.value] += 1
            
            strategy = self.strategy_engine.update_strategy(regime, day_idx)
            strategy_counts[strategy.mode.value] += 1
            
            leading, lagging = self._get_sector_rankings(data, date)
            
            market = DailyMarketSnapshot(
                date=date_str,
                spy_price=spy_price,
                spy_change=spy_change,
                qqq_price=qqq_price,
                qqq_change=qqq_change,
                vix_value=20.0,
                market_regime=regime.value,
                strategy_mode=strategy.mode.value,
                leading_sectors=leading,
                lagging_sectors=lagging,
            )
            
            stocks: List[DailyStockSnapshot] = []
            spy_return_20d = self._calculate_change(data, "SPY", date, 20)
            
            for symbol in WATCHLIST:
                price = self._get_price_on_date(data, symbol, date)
                if price is None:
                    continue
                
                change_1d = self._calculate_change(data, symbol, date, 1)
                change_5d = self._calculate_change(data, symbol, date, 5)
                change_20d = self._calculate_change(data, symbol, date, 20)
                
                indicators = self._calculate_indicators(data, symbol, date)
                if not indicators:
                    continue
                
                rel_strength = change_20d - spy_return_20d
                sentiment = self._simulate_news_sentiment(symbol, change_5d)
                
                score = self._calculate_stock_score(indicators, regime, rel_strength, sentiment)
                signal = self._get_signal(score, regime)
                
                stocks.append(DailyStockSnapshot(
                    symbol=symbol,
                    price=price,
                    change_1d=change_1d,
                    change_5d=change_5d,
                    rsi=indicators.get("rsi", 50),
                    bb_position=indicators.get("bb_position", 0.5),
                    relative_strength=rel_strength,
                    news_sentiment=sentiment,
                    score=score,
                    signal=signal,
                ))
            
            trades = self._execute_trades(date_str, stocks, regime, day_idx)
            
            portfolio_value = self.cash
            positions_snapshot = {}
            for sym, pos in self.positions.items():
                stock = next((s for s in stocks if s.symbol == sym), None)
                if stock:
                    pos_value = pos["shares"] * stock.price
                    portfolio_value += pos_value
                    positions_snapshot[sym] = {
                        "shares": pos["shares"],
                        "entry_price": pos["entry_price"],
                        "current_price": stock.price,
                        "pnl": (stock.price - pos["entry_price"]) * pos["shares"],
                        "pnl_pct": (stock.price / pos["entry_price"] - 1),
                    }
            
            if portfolio_value > portfolio_peak:
                portfolio_peak = portfolio_value
            drawdown = (portfolio_peak - portfolio_value) / portfolio_peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
            
            daily_return = (portfolio_value - prev_value) / prev_value if prev_value > 0 else 0
            daily_returns.append(daily_return)
            prev_value = portfolio_value
            
            cumulative_return = (portfolio_value - self.initial_capital) / self.initial_capital
            
            month_key = date_str[:7]
            if month_key not in monthly_returns:
                monthly_returns[month_key] = 0
            monthly_returns[month_key] = cumulative_return
            
            daily_reports.append(DailyReport(
                date=date_str,
                market=market,
                stocks=stocks,
                trades=trades,
                portfolio_value=portfolio_value,
                cash=self.cash,
                positions=positions_snapshot,
                daily_return=daily_return,
                cumulative_return=cumulative_return,
            ))
            
            if (day_idx + 1) % 50 == 0:
                print(f"  已处理 {day_idx + 1}/{len(trading_dates)} 天, 收益: {cumulative_return:+.1%}")
        
        final_value = portfolio_value
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        if len(daily_returns) > 0 and np.std(daily_returns) > 0:
            sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        all_trades = [t for r in daily_reports for t in r.trades]
        winning = sum(1 for t in all_trades if t.action == "SELL" and t.pnl > 0)
        total_sells = sum(1 for t in all_trades if t.action == "SELL")
        win_rate = winning / total_sells if total_sells > 0 else 0
        
        return FullYearSimulationResult(
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_value=final_value,
            total_return=total_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            total_trades=len(all_trades),
            win_rate=win_rate,
            daily_reports=daily_reports,
            monthly_returns=monthly_returns,
            regime_distribution=regime_counts,
            strategy_distribution=strategy_counts,
        )


def generate_report(result: FullYearSimulationResult) -> str:
    """生成完整年度报告"""
    lines = []
    lines.append("=" * 80)
    lines.append("2024年完整闭环回测报告")
    lines.append("=" * 80)
    lines.append("")
    
    lines.append("【一、年度总览】")
    lines.append("-" * 40)
    lines.append(f"回测期间: {result.start_date} ~ {result.end_date}")
    lines.append(f"初始资金: ${result.initial_capital:,.0f}")
    lines.append(f"最终资金: ${result.final_value:,.0f}")
    lines.append(f"总收益率: {result.total_return:+.2%}")
    lines.append(f"最大回撤: {result.max_drawdown:.2%}")
    lines.append(f"夏普比率: {result.sharpe_ratio:.2f}")
    lines.append(f"总交易次数: {result.total_trades}")
    lines.append(f"胜率: {result.win_rate:.0%}")
    lines.append("")
    
    lines.append("【二、月度收益】")
    lines.append("-" * 40)
    prev_ret = 0
    for month, cum_ret in sorted(result.monthly_returns.items()):
        month_ret = cum_ret - prev_ret
        prev_ret = cum_ret
        bar = "█" * int(abs(month_ret) * 100)
        sign = "+" if month_ret >= 0 else ""
        lines.append(f"  {month}: {sign}{month_ret:.2%} {bar}")
    lines.append("")
    
    lines.append("【三、市场状态分布】")
    lines.append("-" * 40)
    total_days = sum(result.regime_distribution.values())
    for regime, count in result.regime_distribution.items():
        if count > 0:
            pct = count / total_days * 100
            lines.append(f"  {regime:<20}: {count:>4} 天 ({pct:>5.1f}%)")
    lines.append("")
    
    lines.append("【四、策略使用分布】")
    lines.append("-" * 40)
    for strategy, count in result.strategy_distribution.items():
        if count > 0:
            pct = count / total_days * 100
            lines.append(f"  {strategy:<20}: {count:>4} 天 ({pct:>5.1f}%)")
    lines.append("")
    
    lines.append("【五、重要交易记录】")
    lines.append("-" * 40)
    all_trades = [(r.date, t) for r in result.daily_reports for t in r.trades]
    sells = [(d, t) for d, t in all_trades if t.action == "SELL"]
    sells.sort(key=lambda x: abs(x[1].pnl), reverse=True)
    
    lines.append("最大盈利交易:")
    for date, trade in sells[:5]:
        if trade.pnl > 0:
            lines.append(f"  {date} {trade.symbol} +${trade.pnl:,.0f} ({trade.reason})")
    
    lines.append("")
    lines.append("最大亏损交易:")
    for date, trade in reversed(sells[-5:]):
        if trade.pnl < 0:
            lines.append(f"  {date} {trade.symbol} -${abs(trade.pnl):,.0f} ({trade.reason})")
    lines.append("")
    
    lines.append("【六、每日样本 (每月第一个交易日)】")
    lines.append("-" * 40)
    shown_months = set()
    for report in result.daily_reports:
        month = report.date[:7]
        if month not in shown_months:
            shown_months.add(month)
            lines.append(f"\n{report.date}:")
            lines.append(f"  市场: SPY ${report.market.spy_price:.2f} ({report.market.spy_change:+.2%})")
            lines.append(f"  状态: {report.market.market_regime} → {report.market.strategy_mode}")
            lines.append(f"  领先板块: {', '.join(report.market.leading_sectors)}")
            lines.append(f"  组合价值: ${report.portfolio_value:,.0f} ({report.cumulative_return:+.2%})")
            if report.trades:
                for t in report.trades:
                    lines.append(f"  交易: {t.action} {t.symbol} {t.shares}股 @ ${t.price:.2f}")
    
    lines.append("")
    lines.append("=" * 80)
    lines.append("报告生成完成")
    lines.append("=" * 80)
    
    return "\n".join(lines)


if __name__ == "__main__":
    simulator = FullYearSimulator(initial_capital=100000.0)
    result = simulator.run_simulation("2024-01-01", "2024-12-31")
    
    report = generate_report(result)
    print(report)
    
    output_path = Path("storage/full_simulation_2024_report.txt")
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(report)
    print(f"\n报告已保存到: {output_path}")
