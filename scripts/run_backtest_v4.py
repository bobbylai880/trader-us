#!/usr/bin/env python3
"""
3年回测 V4 - 分层决策策略

核心架构 (从大到小):
┌─────────────────────────────────────────────────────────────┐
│ 第一层: 宏观趋势 (月度回顾)                                    │
│   - 新闻情绪聚合 (整体市场情绪)                                │
│   - VIX 水平与趋势                                           │
│   - SPY 月度动量                                             │
│   → 输出: 市场倾向 (进攻/中性/防御) + 目标仓位                  │
├─────────────────────────────────────────────────────────────┤
│ 第二层: 板块轮动 (周度调整)                                    │
│   - 11个板块ETF相对强度排名                                   │
│   - 板块动量 + 相对SPY强度                                    │
│   → 输出: Top 3 强势板块                                      │
├─────────────────────────────────────────────────────────────┤
│ 第三层: 个股精选 (在板块内选股)                                │
│   - 板块内个股动量排名                                        │
│   - 技术指标过滤 (趋势/成交量)                                 │
│   → 输出: 每板块 Top 2 个股                                   │
└─────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# 板块与个股映射
# ============================================================

SECTOR_STOCKS = {
    "XLK": ["AAPL", "MSFT", "NVDA", "AVGO", "AMD", "ADBE", "CRM", "ORCL", "CSCO", "INTC"],
    "XLC": ["META", "GOOGL", "GOOG", "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS"],
    "XLY": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "TJX", "TGT"],
    "XLF": ["JPM", "BAC", "WFC", "GS", "MS", "BLK", "C", "AXP", "SCHW", "PNC"],
    "XLV": ["UNH", "JNJ", "LLY", "PFE", "MRK", "ABBV", "TMO", "ABT", "DHR", "BMY"],
    "XLE": ["XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO", "OXY"],
    "XLI": ["CAT", "UNP", "HON", "UPS", "BA", "RTX", "DE", "LMT", "GE", "FDX"],
    "XLP": ["PG", "KO", "PEP", "COST", "WMT", "PM", "MO"],
    "XLU": ["NEE", "DUK", "SO", "D", "AEP"],
    "XLB": ["LIN", "APD", "SHW", "ECL", "DD"],
    "XLRE": ["AMT", "PLD", "CCI", "EQIX", "SPG"],
}

DEFENSIVE_SECTORS = ["XLP", "XLV", "XLU"]
GROWTH_SECTORS = ["XLK", "XLC", "XLY"]


@dataclass
class MacroView:
    """宏观视图 - 月度更新"""
    date: str
    market_regime: str  # "offensive", "neutral", "defensive"
    target_exposure: float
    vix_level: float
    vix_trend: str  # "rising", "falling", "stable"
    news_sentiment: float  # -1 to 1
    spy_momentum: float
    reasoning: str


@dataclass
class SectorAllocation:
    """板块配置 - 周度更新"""
    date: str
    top_sectors: List[str]
    sector_scores: Dict[str, float]
    rotation_signal: str  # "rotate_in", "hold", "rotate_out"


@dataclass
class Position:
    symbol: str
    shares: int
    avg_cost: float
    entry_date: str
    sector: str
    highest_price: float


@dataclass
class Trade:
    date: str
    symbol: str
    action: str
    price: float
    shares: int
    sector: str
    pnl: float = 0.0
    pnl_pct: float = 0.0
    reason: str = ""


class HierarchicalBacktest:
    """分层决策回测引擎"""
    
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
        self._news_sentiment: Dict[str, Dict[str, float]] = {}
        
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[str, float, float]] = []
        self.macro_history: List[MacroView] = []
        self.sector_history: List[SectorAllocation] = []
        
        self._current_macro: Optional[MacroView] = None
        self._current_sectors: List[str] = []
    
    def _load_data(self, start: date, end: date):
        print("  加载价格数据...")
        
        all_symbols = set(['SPY', 'VIX'])
        for etf in SECTOR_STOCKS:
            all_symbols.add(etf)
            all_symbols.update(SECTOR_STOCKS[etf])
        
        query = """
            SELECT symbol, trade_date, open, high, low, close, volume
            FROM daily_prices
            WHERE trade_date BETWEEN %s AND %s
              AND symbol IN %s
            ORDER BY symbol, trade_date
        """
        df = pd.read_sql(query, self.conn, params=(start - timedelta(days=100), end, tuple(all_symbols)))
        
        for sym in df['symbol'].unique():
            sdf = df[df['symbol'] == sym].copy()
            sdf.set_index('trade_date', inplace=True)
            sdf['sma20'] = sdf['close'].rolling(20).mean()
            sdf['sma50'] = sdf['close'].rolling(50).mean()
            sdf['mom5'] = sdf['close'].pct_change(5)
            sdf['mom20'] = sdf['close'].pct_change(20)
            sdf['vol_ratio'] = sdf['volume'] / sdf['volume'].rolling(20).mean()
            self._prices[sym] = sdf
        
        print(f"    已加载 {len(self._prices)} 只标的")
        
        print("  加载新闻情绪...")
        query2 = """
            SELECT symbol, DATE(published_at) as news_date, 
                   AVG(sentiment_score) as sentiment
            FROM news
            GROUP BY symbol, DATE(published_at)
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query2)
            for row in cur.fetchall():
                sym = row['symbol']
                dt = str(row['news_date'])
                if sym not in self._news_sentiment:
                    self._news_sentiment[sym] = {}
                self._news_sentiment[sym][dt] = float(row['sentiment'] or 0)
        
        print(f"    已加载 {len(self._news_sentiment)} 只标的新闻")
    
    def _get(self, sym: str, dt: date, col: str) -> Optional[float]:
        if sym not in self._prices:
            return None
        df = self._prices[sym]
        valid = df[df.index <= dt]
        if len(valid) == 0:
            return None
        val = valid[col].iloc[-1]
        return float(val) if pd.notna(val) else None
    
    def _get_news_sentiment(self, symbols: List[str], dt: date, lookback: int = 30) -> float:
        sentiments = []
        for sym in symbols:
            if sym not in self._news_sentiment:
                continue
            for i in range(lookback):
                check_dt = str(dt - timedelta(days=i))
                if check_dt in self._news_sentiment[sym]:
                    sentiments.append(self._news_sentiment[sym][check_dt])
        return np.mean(sentiments) if sentiments else 0.0
    
    # ================================================================
    # 第一层: 宏观趋势分析 (月度)
    # ================================================================
    
    def _analyze_macro(self, dt: date) -> MacroView:
        """月度宏观分析"""
        
        vix = self._get('VIX', dt, 'close') or 20
        vix_20d_ago = self._get('VIX', dt - timedelta(days=20), 'close') or vix
        
        if vix > vix_20d_ago * 1.2:
            vix_trend = "rising"
        elif vix < vix_20d_ago * 0.8:
            vix_trend = "falling"
        else:
            vix_trend = "stable"
        
        spy_mom = self._get('SPY', dt, 'mom20') or 0
        spy_close = self._get('SPY', dt, 'close') or 0
        spy_sma50 = self._get('SPY', dt, 'sma50') or spy_close
        
        market_symbols = ['SPY', 'QQQ'] + list(SECTOR_STOCKS.keys())
        news_sentiment = self._get_news_sentiment(market_symbols, dt, 30)
        
        score = 0
        reasoning_parts = []
        
        if vix < 15:
            score += 2
            reasoning_parts.append("VIX低位(贪婪)")
        elif vix < 20:
            score += 1
            reasoning_parts.append("VIX正常")
        elif vix < 30:
            score -= 1
            reasoning_parts.append("VIX偏高(谨慎)")
        else:
            score -= 2
            reasoning_parts.append("VIX恐慌")
        
        if spy_close > spy_sma50 and spy_mom > 0.03:
            score += 2
            reasoning_parts.append("SPY强势上涨")
        elif spy_close > spy_sma50:
            score += 1
            reasoning_parts.append("SPY在均线上方")
        elif spy_close < spy_sma50 and spy_mom < -0.03:
            score -= 2
            reasoning_parts.append("SPY弱势下跌")
        else:
            score -= 1
            reasoning_parts.append("SPY在均线下方")
        
        if news_sentiment > 0.3:
            score += 1
            reasoning_parts.append("新闻情绪积极")
        elif news_sentiment < -0.3:
            score -= 1
            reasoning_parts.append("新闻情绪消极")
        
        if score >= 3:
            regime = "offensive"
            target_exposure = 0.95
        elif score >= 1:
            regime = "neutral"
            target_exposure = 0.70
        elif score >= -1:
            regime = "neutral"
            target_exposure = 0.50
        else:
            regime = "defensive"
            target_exposure = 0.30
        
        return MacroView(
            date=str(dt),
            market_regime=regime,
            target_exposure=target_exposure,
            vix_level=vix,
            vix_trend=vix_trend,
            news_sentiment=news_sentiment,
            spy_momentum=spy_mom,
            reasoning=" | ".join(reasoning_parts)
        )
    
    # ================================================================
    # 第二层: 板块轮动 (周度)
    # ================================================================
    
    def _analyze_sectors(self, dt: date, macro: MacroView) -> SectorAllocation:
        """周度板块分析"""
        
        spy_mom20 = self._get('SPY', dt, 'mom20') or 0
        
        sector_scores = {}
        for etf in SECTOR_STOCKS.keys():
            mom20 = self._get(etf, dt, 'mom20') or 0
            mom5 = self._get(etf, dt, 'mom5') or 0
            
            rs_vs_spy = mom20 - spy_mom20
            
            sector_sentiment = self._get_news_sentiment(
                SECTOR_STOCKS.get(etf, [])[:5], dt, 14
            )
            
            score = 0.4 * mom20 + 0.3 * rs_vs_spy + 0.2 * mom5 + 0.1 * sector_sentiment
            sector_scores[etf] = score
        
        if macro.market_regime == "defensive":
            for s in DEFENSIVE_SECTORS:
                if s in sector_scores:
                    sector_scores[s] += 0.05
            for s in GROWTH_SECTORS:
                if s in sector_scores:
                    sector_scores[s] -= 0.03
        elif macro.market_regime == "offensive":
            for s in GROWTH_SECTORS:
                if s in sector_scores:
                    sector_scores[s] += 0.03
        
        ranked = sorted(sector_scores.items(), key=lambda x: -x[1])
        
        if macro.market_regime == "offensive":
            top_n = 4
        elif macro.market_regime == "defensive":
            top_n = 2
        else:
            top_n = 3
        
        top_sectors = [s[0] for s in ranked[:top_n] if s[1] > -0.05]
        
        return SectorAllocation(
            date=str(dt),
            top_sectors=top_sectors,
            sector_scores=sector_scores,
            rotation_signal="hold"
        )
    
    # ================================================================
    # 第三层: 个股精选 (在板块内)
    # ================================================================
    
    def _select_stocks(self, dt: date, sectors: List[str], max_per_sector: int = 2) -> List[Tuple[str, str, float]]:
        """在指定板块内选股"""
        candidates = []
        
        for sector in sectors:
            stocks = SECTOR_STOCKS.get(sector, [])
            stock_scores = []
            
            for sym in stocks:
                if sym not in self._prices:
                    continue
                
                mom20 = self._get(sym, dt, 'mom20')
                mom5 = self._get(sym, dt, 'mom5')
                close = self._get(sym, dt, 'close')
                sma20 = self._get(sym, dt, 'sma20')
                vol_ratio = self._get(sym, dt, 'vol_ratio')
                
                if mom20 is None or close is None or sma20 is None:
                    continue
                
                if close < sma20:
                    continue
                
                if mom20 < 0:
                    continue
                
                score = mom20 * 0.5 + (mom5 or 0) * 0.3
                if vol_ratio and vol_ratio > 1.2:
                    score += 0.02
                
                stock_scores.append((sym, sector, score))
            
            stock_scores.sort(key=lambda x: -x[2])
            candidates.extend(stock_scores[:max_per_sector])
        
        candidates.sort(key=lambda x: -x[2])
        return candidates
    
    # ================================================================
    # 交易执行
    # ================================================================
    
    def _portfolio_value(self, dt: date) -> float:
        pos_val = sum(
            p.shares * (self._get(s, dt, 'close') or p.avg_cost)
            for s, p in self.positions.items()
        )
        return self.cash + pos_val
    
    def _buy(self, sym: str, sector: str, dt: date, budget: float, reason: str) -> bool:
        price = self._get(sym, dt, 'close')
        if not price or budget < 1000:
            return False
        
        shares = int(budget / price)
        if shares <= 0:
            return False
        
        cost = shares * price
        if cost > self.cash:
            return False
        
        self.cash -= cost
        
        if sym in self.positions:
            p = self.positions[sym]
            total = p.shares + shares
            p.avg_cost = (p.avg_cost * p.shares + price * shares) / total
            p.shares = total
            p.highest_price = max(p.highest_price, price)
        else:
            self.positions[sym] = Position(sym, shares, price, str(dt), sector, price)
        
        self.trades.append(Trade(str(dt), sym, "BUY", price, shares, sector, reason=reason))
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
        self.trades.append(Trade(str(dt), sym, "SELL", price, p.shares, p.sector, pnl, pnl_pct, reason))
        del self.positions[sym]
        return pnl
    
    def _check_stops(self, dt: date):
        """检查止损 - 跟踪止损 15%"""
        to_sell = []
        
        for sym, pos in self.positions.items():
            price = self._get(sym, dt, 'close')
            if not price:
                continue
            
            pos.highest_price = max(pos.highest_price, price)
            
            drawdown = (pos.highest_price - price) / pos.highest_price
            if drawdown > 0.15:
                to_sell.append((sym, f"跟踪止损({drawdown:.1%})"))
            elif price < pos.avg_cost * 0.88:
                to_sell.append((sym, "硬止损12%"))
        
        for sym, reason in to_sell:
            self._sell(sym, dt, reason)
    
    def _rebalance(self, dt: date, macro: MacroView, candidates: List[Tuple[str, str, float]]):
        """再平衡组合"""
        pv = self._portfolio_value(dt)
        current_exposure = (pv - self.cash) / pv if pv > 0 else 0
        target_exposure = macro.target_exposure
        
        current_sectors = set(p.sector for p in self.positions.values())
        target_sectors = set(self._current_sectors)
        
        for sym, pos in list(self.positions.items()):
            if pos.sector not in target_sectors:
                self._sell(sym, dt, f"板块轮出({pos.sector})")
        
        if current_exposure < target_exposure - 0.1:
            available = self.cash * 0.95
            max_positions = 8 if macro.market_regime == "offensive" else 5
            position_pct = target_exposure / max_positions
            
            for sym, sector, score in candidates:
                if len(self.positions) >= max_positions:
                    break
                if sym in self.positions:
                    continue
                
                budget = min(pv * position_pct, available)
                if self._buy(sym, sector, dt, budget, f"板块轮入({sector}, score:{score:.3f})"):
                    available -= budget
    
    # ================================================================
    # 主运行循环
    # ================================================================
    
    def run(self, start: date, end: date) -> dict:
        print("\n" + "=" * 70)
        print("V4 分层决策策略回测")
        print("=" * 70)
        print("  架构: 宏观趋势(月) → 板块轮动(周) → 个股精选(板块内)")
        
        self._load_data(start, end)
        
        if 'SPY' not in self._prices:
            raise ValueError("SPY 数据缺失")
        
        trading_days = sorted(self._prices['SPY'].index.tolist())
        trading_days = [d for d in trading_days if start <= d <= end]
        
        print(f"\n  回测区间: {start} ~ {end}")
        print(f"  交易日数: {len(trading_days)}")
        print(f"  初始资金: ${self.initial_capital:,.0f}")
        
        last_macro_month = None
        last_sector_week = None
        
        for i, dt in enumerate(trading_days):
            current_month = dt.strftime("%Y-%m")
            current_week = dt.isocalendar()[1]
            
            if current_month != last_macro_month:
                self._current_macro = self._analyze_macro(dt)
                self.macro_history.append(self._current_macro)
                last_macro_month = current_month
                
                if i % 50 == 0 or len(self.macro_history) <= 3:
                    print(f"\n  📊 [{dt}] 月度宏观: {self._current_macro.market_regime} "
                          f"(仓位:{self._current_macro.target_exposure:.0%}) - {self._current_macro.reasoning}")
            
            if current_week != last_sector_week and self._current_macro:
                sector_alloc = self._analyze_sectors(dt, self._current_macro)
                self.sector_history.append(sector_alloc)
                self._current_sectors = sector_alloc.top_sectors
                last_sector_week = current_week
            
            self._check_stops(dt)
            
            if i % 5 == 0 and self._current_macro and self._current_sectors:
                candidates = self._select_stocks(dt, self._current_sectors)
                self._rebalance(dt, self._current_macro, candidates)
            
            pv = self._portfolio_value(dt)
            spy_price = self._get('SPY', dt, 'close') or 0
            spy_base = self._get('SPY', start, 'close') or 1
            spy_val = self.initial_capital * spy_price / spy_base
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
        
        sells = [t for t in self.trades if t.action == "SELL"]
        wins = [t for t in sells if t.pnl > 0]
        win_rate = len(wins) / len(sells) if sells else 0
        
        total_win = sum(t.pnl for t in wins)
        total_loss = abs(sum(t.pnl for t in sells if t.pnl < 0))
        profit_factor = total_win / total_loss if total_loss > 0 else float('inf')
        
        regime_dist = {}
        for m in self.macro_history:
            regime_dist[m.market_regime] = regime_dist.get(m.market_regime, 0) + 1
        
        sector_counts = {}
        for t in self.trades:
            if t.action == "BUY":
                sector_counts[t.sector] = sector_counts.get(t.sector, 0) + 1
        
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
            "regime_distribution": regime_dist,
            "sector_distribution": sector_counts,
        }


def main():
    bt = HierarchicalBacktest(100000.0)
    result = bt.run(date(2023, 1, 3), date(2026, 1, 16))
    
    print("\n" + "=" * 70)
    print("V4 分层策略回测结果")
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
    print(f"  总交易: {result['total_trades']} 笔")
    
    print(f"\n  宏观状态分布:")
    for regime, count in result['regime_distribution'].items():
        print(f"    {regime}: {count} 月")
    
    print(f"\n  板块交易分布:")
    sorted_sectors = sorted(result['sector_distribution'].items(), key=lambda x: -x[1])
    for sector, count in sorted_sectors[:5]:
        print(f"    {sector}: {count} 笔")
    
    output = Path("storage/backtest_3y_v4")
    output.mkdir(parents=True, exist_ok=True)
    
    with open(output / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    
    trades_data = [
        {"date": t.date, "symbol": t.symbol, "action": t.action,
         "price": t.price, "shares": t.shares, "sector": t.sector,
         "pnl": t.pnl, "pnl_pct": t.pnl_pct, "reason": t.reason}
        for t in bt.trades
    ]
    with open(output / "trades.json", "w") as f:
        json.dump(trades_data, f, indent=2)
    
    macro_data = [
        {"date": m.date, "regime": m.market_regime, "exposure": m.target_exposure,
         "vix": m.vix_level, "sentiment": m.news_sentiment, "reasoning": m.reasoning}
        for m in bt.macro_history
    ]
    with open(output / "macro_history.json", "w") as f:
        json.dump(macro_data, f, indent=2)
    
    equity_df = pd.DataFrame(bt.equity_curve, columns=['date', 'portfolio', 'spy'])
    equity_df.to_csv(output / "equity_curve.csv", index=False)
    
    print(f"\n📁 保存到: {output}")
    
    print("\n【最大盈利交易】")
    top = sorted([t for t in bt.trades if t.action == "SELL"], key=lambda x: -x.pnl)[:5]
    for t in top:
        print(f"  {t.date} {t.symbol}({t.sector}): ${t.pnl:+,.0f} ({t.pnl_pct:+.1%}) - {t.reason}")
    
    print("\n【宏观月度回顾摘要】")
    for m in bt.macro_history[-6:]:
        print(f"  {m.date}: {m.market_regime} (VIX:{m.vix_level:.1f}, 仓位:{m.target_exposure:.0%})")


if __name__ == "__main__":
    main()
