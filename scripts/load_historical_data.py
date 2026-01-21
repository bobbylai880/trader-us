"""历史数据完整导入脚本 - 3年数据回测准备.

导入内容：
1. 市场指数：SPY, QQQ, DIA, IWM, VIX
2. 板块 ETF：11个 SPDR 板块
3. 个股：60+ 只主要股票
4. 技术指标：预计算并缓存
"""
from __future__ import annotations

import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_trader_assist.data_collector.pg_client import PostgresMarketDB, get_db
from ai_trader_assist.data_collector.yf_client import YahooFinanceClient
from ai_trader_assist.feature_engineering.indicators import (
    rsi,
    macd,
    atr,
    bollinger_bands,
)


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()


MARKET_INDICES = ["SPY", "QQQ", "DIA", "IWM", "^VIX"]

SECTOR_ETFS = [
    "XLK",  # 科技
    "XLC",  # 通讯服务
    "XLY",  # 可选消费
    "XLF",  # 金融
    "XLV",  # 医疗
    "XLE",  # 能源
    "XLI",  # 工业
    "XLP",  # 必需消费
    "XLU",  # 公用事业
    "XLB",  # 材料
    "XLRE", # 房地产
]

STOCKS = [
    # 科技巨头
    "AAPL", "MSFT", "NVDA", "AMD", "GOOGL", "GOOG", "META", "AMZN", "TSLA",
    "AVGO", "CRM", "ADBE", "CSCO", "ORCL", "INTC", "NFLX", "QCOM", "TXN",
    "AMAT", "MU", "LRCX", "KLAC", "SNPS", "CDNS",
    # 金融
    "JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "PNC", "TFC",
    "V", "MA", "AXP", "BLK", "SCHW",
    # 医疗
    "UNH", "JNJ", "LLY", "MRK", "ABBV", "PFE", "TMO", "ABT", "DHR", "BMY",
    # 能源
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY",
    # 消费
    "HD", "MCD", "NKE", "SBUX", "TJX", "LOW", "TGT", "COST", "WMT", "PG",
    "KO", "PEP", "PM", "MO",
    # 工业
    "CAT", "BA", "HON", "UNP", "GE", "RTX", "LMT", "DE", "MMM", "UPS", "FDX",
    # 通讯
    "DIS", "CMCSA", "VZ", "T", "TMUS",
    # 其他
    "BRK-B", "PYPL", "SQ", "COIN", "ABNB", "UBER", "LYFT",
]

ALL_SYMBOLS = MARKET_INDICES + SECTOR_ETFS + STOCKS


def download_symbol_data(
    symbol: str,
    start_date: date,
    end_date: date,
    yf_client: YahooFinanceClient,
    db: PostgresMarketDB,
    retry_count: int = 3,
) -> int:
    """下载单个股票数据并导入数据库"""
    for attempt in range(retry_count):
        try:
            start_dt = datetime.combine(start_date, datetime.min.time())
            end_dt = datetime.combine(end_date, datetime.min.time())
            
            df = yf_client.fetch_history(symbol, start_dt, end_dt, interval="1d", force=True)
            
            if df.empty:
                if attempt < retry_count - 1:
                    time.sleep(2)
                    continue
                return 0
            
            df = df.reset_index()
            df["symbol"] = symbol
            df = df.rename(columns={
                "Date": "trade_date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            })
            
            rows = db.upsert_daily_prices(df)
            return len(df)
            
        except Exception as e:
            if attempt < retry_count - 1:
                time.sleep(2)
            else:
                print(f"    ✗ {symbol}: {e}")
                return 0
    
    return 0


def calculate_indicators_for_symbol(
    symbol: str,
    start_date: date,
    end_date: date,
    db: PostgresMarketDB,
) -> int:
    """计算单个股票的技术指标并缓存"""
    try:
        lookback_start = start_date - timedelta(days=250)
        df = db.get_daily_prices_single(symbol, lookback_start, end_date)
        
        if df.empty or len(df) < 50:
            return 0
        
        df = df.set_index("trade_date").sort_index()
        
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]
        
        indicators_df = pd.DataFrame(index=df.index)
        indicators_df["symbol"] = symbol
        
        indicators_df["sma_20"] = sma(close, 20)
        indicators_df["sma_50"] = sma(close, 50)
        indicators_df["sma_200"] = sma(close, 200)
        indicators_df["ema_12"] = ema(close, 12)
        indicators_df["ema_26"] = ema(close, 26)
        
        indicators_df["rsi_14"] = rsi(close, 14)
        
        macd_df = macd(close)
        indicators_df["macd"] = macd_df["macd"]
        indicators_df["macd_signal"] = macd_df["signal"]
        indicators_df["macd_hist"] = macd_df["hist"]
        
        indicators_df["atr_14"] = atr(high, low, close, 14)
        
        bb_df = bollinger_bands(close, 20, 2)
        indicators_df["bb_upper"] = bb_df["upper"]
        indicators_df["bb_middle"] = bb_df["middle"]
        indicators_df["bb_lower"] = bb_df["lower"]
        
        vol_sma = sma(volume.astype(float), 20)
        indicators_df["volume_sma_20"] = vol_sma
        indicators_df["volume_ratio"] = volume / vol_sma.replace(0, 1)
        
        indicators_df["trend_slope_20"] = close.rolling(20).apply(
            lambda x: (x.iloc[-1] - x.iloc[0]) / (len(x) * x.iloc[0]) if len(x) > 0 and x.iloc[0] != 0 else 0,
            raw=False
        )
        
        indicators_df["momentum_10d"] = close.pct_change(10)
        
        indicators_df = indicators_df.reset_index()
        indicators_df = indicators_df.rename(columns={"index": "trade_date"})
        indicators_df["trade_date"] = pd.to_datetime(indicators_df["trade_date"]).dt.date
        indicators_df = indicators_df[
            (indicators_df["trade_date"] >= start_date) &
            (indicators_df["trade_date"] <= end_date)
        ]
        
        if not indicators_df.empty:
            rows = db.upsert_indicators(indicators_df)
            return len(indicators_df)
        
        return 0
        
    except Exception as e:
        print(f"    ✗ {symbol} 指标计算失败: {e}")
        return 0


def load_historical_data(
    years: int = 3,
    symbols: Optional[List[str]] = None,
    skip_existing: bool = True,
    verbose: bool = True,
) -> dict:
    """加载完整历史数据
    
    Args:
        years: 历史年数
        symbols: 股票列表（默认使用全部）
        skip_existing: 是否跳过已有数据的股票
        verbose: 是否打印进度
    """
    symbols = symbols or ALL_SYMBOLS
    end_date = date.today()
    start_date = end_date - timedelta(days=365 * years + 30)
    
    db = get_db()
    yf = YahooFinanceClient(use_pg_cache=False)
    
    stats = {
        "total_symbols": len(symbols),
        "price_success": 0,
        "price_failed": 0,
        "price_rows": 0,
        "indicator_success": 0,
        "indicator_rows": 0,
    }
    
    if verbose:
        print("=" * 70)
        print("历史数据导入 - PostgreSQL 18")
        print("=" * 70)
        print(f"股票数量: {len(symbols)}")
        print(f"日期范围: {start_date} ~ {end_date} ({years}年)")
        print()
    
    # 阶段1: 下载价格数据
    if verbose:
        print("阶段 1/2: 下载价格数据")
        print("-" * 70)
    
    for i, symbol in enumerate(symbols):
        if verbose:
            print(f"[{i+1:3d}/{len(symbols)}] {symbol:8s}", end=" ", flush=True)
        
        if skip_existing:
            existing = db.get_price_count(symbol)
            if existing > 500:
                if verbose:
                    print(f"跳过 (已有 {existing} 行)")
                stats["price_success"] += 1
                stats["price_rows"] += existing
                continue
        
        rows = download_symbol_data(symbol, start_date, end_date, yf, db)
        
        if rows > 0:
            stats["price_success"] += 1
            stats["price_rows"] += rows
            if verbose:
                print(f"✓ {rows} 行")
        else:
            stats["price_failed"] += 1
            if verbose:
                print("✗ 无数据")
        
        time.sleep(0.3)
    
    if verbose:
        print()
        print(f"价格数据: 成功 {stats['price_success']}, "
              f"失败 {stats['price_failed']}, 共 {stats['price_rows']:,} 行")
        print()
    
    # 阶段2: 计算技术指标
    if verbose:
        print("阶段 2/2: 计算技术指标")
        print("-" * 70)
    
    indicator_symbols = [s for s in symbols if not s.startswith("^")]
    
    for i, symbol in enumerate(indicator_symbols):
        if verbose:
            print(f"[{i+1:3d}/{len(indicator_symbols)}] {symbol:8s}", end=" ", flush=True)
        
        rows = calculate_indicators_for_symbol(symbol, start_date, end_date, db)
        
        if rows > 0:
            stats["indicator_success"] += 1
            stats["indicator_rows"] += rows
            if verbose:
                print(f"✓ {rows} 行")
        else:
            if verbose:
                print("✗")
    
    # 阶段3: 优化数据库
    if verbose:
        print()
        print("优化数据库...")
    
    db.vacuum_analyze()
    
    # 最终统计
    final_stats = db.get_stats()
    
    if verbose:
        print()
        print("=" * 70)
        print("导入完成!")
        print("=" * 70)
        print(f"数据库大小: {final_stats['database_size']}")
        print(f"日线数据: {final_stats['daily_prices_count']:,} 行")
        print(f"技术指标: {final_stats['indicators_count']:,} 行")
        print(f"股票数量: {final_stats['unique_symbols']} 只")
        if final_stats.get('price_date_range') and final_stats['price_date_range'].get('min'):
            print(f"日期范围: {final_stats['price_date_range']['min']} ~ "
                  f"{final_stats['price_date_range']['max']}")
    
    stats["final_stats"] = final_stats
    return stats


def verify_data_completeness(verbose: bool = True) -> dict:
    """验证数据完整性"""
    db = get_db()
    
    results = {
        "symbols_checked": 0,
        "complete": [],
        "incomplete": [],
        "missing": [],
    }
    
    if verbose:
        print("验证数据完整性...")
        print("-" * 50)
    
    expected_days = 750
    
    for symbol in ALL_SYMBOLS:
        count = db.get_price_count(symbol)
        results["symbols_checked"] += 1
        
        if count == 0:
            results["missing"].append(symbol)
            status = "缺失"
        elif count < expected_days * 0.8:
            results["incomplete"].append((symbol, count))
            status = f"不完整 ({count}行)"
        else:
            results["complete"].append(symbol)
            status = f"完整 ({count}行)"
        
        if verbose:
            print(f"  {symbol:8s}: {status}")
    
    if verbose:
        print()
        print(f"完整: {len(results['complete'])}, "
              f"不完整: {len(results['incomplete'])}, "
              f"缺失: {len(results['missing'])}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="加载历史数据到 PostgreSQL 18")
    parser.add_argument("--years", type=int, default=3, help="历史年数 (默认 3)")
    parser.add_argument("--verify", action="store_true", help="仅验证数据完整性")
    parser.add_argument("--force", action="store_true", help="强制重新下载所有数据")
    parser.add_argument("--symbols", nargs="+", help="指定股票列表")
    
    args = parser.parse_args()
    
    if args.verify:
        verify_data_completeness()
    else:
        load_historical_data(
            years=args.years,
            symbols=args.symbols,
            skip_existing=not args.force,
        )
