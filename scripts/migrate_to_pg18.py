"""数据迁移脚本：从 yfinance 获取数据并导入 PostgreSQL 18.

功能：
1. 从 yfinance 下载历史数据
2. 批量导入到 PostgreSQL
3. 计算并缓存技术指标
4. 支持增量更新
"""
from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ai_trader_assist.data_collector.pg_client import PostgresMarketDB, get_db
from ai_trader_assist.data_collector.yf_client import YahooFinanceClient
from ai_trader_assist.feature_engineering.indicators import (
    rsi,
    macd,
    atr,
    bollinger_bands,
)


def sma(series: pd.Series, window: int) -> pd.Series:
    """简单移动平均"""
    return series.rolling(window=window, min_periods=1).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    """指数移动平均"""
    return series.ewm(span=window, adjust=False).mean()


# 默认股票池
DEFAULT_SYMBOLS = [
    # 指数 ETF
    "SPY", "QQQ", "DIA", "IWM",
    # 板块 ETF
    "XLK", "XLC", "XLY", "XLF", "XLV", "XLE", "XLI", "XLP", "XLU", "XLB", "XLRE",
    # 科技股
    "AAPL", "MSFT", "NVDA", "AMD", "GOOGL", "GOOG", "META", "AMZN", "TSLA",
    "AVGO", "CRM", "ADBE", "CSCO", "ORCL", "INTC", "NFLX",
    # 金融
    "JPM", "BAC", "WFC", "GS", "MS", "V", "MA",
    # 医疗
    "UNH", "JNJ", "LLY", "MRK", "ABBV", "PFE",
    # 能源
    "XOM", "CVX", "COP",
    # 消费
    "HD", "MCD", "NKE", "SBUX", "COST", "WMT",
    # 工业
    "CAT", "BA", "HON", "UNP", "GE",
    # VIX
    "^VIX",
]


def download_and_import(
    symbols: List[str],
    start_date: date,
    end_date: date,
    db: Optional[PostgresMarketDB] = None,
    yf: Optional[YahooFinanceClient] = None,
    verbose: bool = True,
) -> dict:
    """下载数据并导入数据库
    
    Args:
        symbols: 股票列表
        start_date: 开始日期
        end_date: 结束日期
        db: 数据库客户端
        yf: Yahoo Finance 客户端
        verbose: 是否打印进度
    
    Returns:
        统计信息字典
    """
    db = db or get_db()
    yf = yf or YahooFinanceClient()
    
    stats = {
        "total_symbols": len(symbols),
        "success": 0,
        "failed": 0,
        "rows_inserted": 0,
        "errors": [],
    }
    
    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.min.time())
    
    for i, symbol in enumerate(symbols):
        if verbose:
            print(f"[{i+1}/{len(symbols)}] 处理 {symbol}...", end=" ")
        
        try:
            # 从 yfinance 获取数据
            df = yf.fetch_history(symbol, start_dt, end_dt, interval="1d")
            
            if df.empty:
                if verbose:
                    print("无数据")
                stats["failed"] += 1
                stats["errors"].append(f"{symbol}: 无数据")
                continue
            
            # 准备数据
            df = df.reset_index()
            df["symbol"] = symbol
            
            # 重命名列
            column_map = {
                "Date": "trade_date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
            df = df.rename(columns=column_map)
            
            # 确保日期格式正确
            if "trade_date" in df.columns:
                df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
            
            # 导入数据库
            rows = db.upsert_daily_prices(df)
            stats["rows_inserted"] += rows
            stats["success"] += 1
            
            if verbose:
                print(f"✓ {len(df)} 行")
            
        except Exception as e:
            stats["failed"] += 1
            stats["errors"].append(f"{symbol}: {str(e)}")
            if verbose:
                print(f"✗ {e}")
    
    return stats


def calculate_and_cache_indicators(
    symbols: List[str],
    start_date: date,
    end_date: date,
    db: Optional[PostgresMarketDB] = None,
    verbose: bool = True,
) -> dict:
    """计算技术指标并缓存到数据库
    
    Args:
        symbols: 股票列表
        start_date: 开始日期
        end_date: 结束日期
        db: 数据库客户端
        verbose: 是否打印进度
    
    Returns:
        统计信息字典
    """
    db = db or get_db()
    
    stats = {
        "total_symbols": len(symbols),
        "success": 0,
        "failed": 0,
        "rows_inserted": 0,
    }
    
    for i, symbol in enumerate(symbols):
        if verbose:
            print(f"[{i+1}/{len(symbols)}] 计算 {symbol} 指标...", end=" ")
        
        try:
            # 获取价格数据（需要额外的历史数据来计算指标）
            lookback_start = start_date - timedelta(days=250)
            df = db.get_daily_prices_single(symbol, lookback_start, end_date)
            
            if df.empty or len(df) < 50:
                if verbose:
                    print("数据不足")
                stats["failed"] += 1
                continue
            
            # 设置索引
            df = df.set_index("trade_date").sort_index()
            
            # 计算指标
            close = df["close"]
            high = df["high"]
            low = df["low"]
            volume = df["volume"]
            
            indicators_df = pd.DataFrame(index=df.index)
            indicators_df["symbol"] = symbol
            
            # 均线
            indicators_df["sma_20"] = sma(close, 20)
            indicators_df["sma_50"] = sma(close, 50)
            indicators_df["sma_200"] = sma(close, 200)
            indicators_df["ema_12"] = ema(close, 12)
            indicators_df["ema_26"] = ema(close, 26)
            
            # RSI
            indicators_df["rsi_14"] = rsi(close, 14)
            
            # MACD
            macd_line, signal_line, hist = macd(close)
            indicators_df["macd"] = macd_line
            indicators_df["macd_signal"] = signal_line
            indicators_df["macd_hist"] = hist
            
            # ATR
            indicators_df["atr_14"] = atr(high, low, close, 14)
            
            # 布林带
            bb_upper, bb_middle, bb_lower = bollinger_bands(close, 20, 2)
            indicators_df["bb_upper"] = bb_upper
            indicators_df["bb_middle"] = bb_middle
            indicators_df["bb_lower"] = bb_lower
            
            # 成交量
            indicators_df["volume_sma_20"] = sma(volume.astype(float), 20)
            vol_sma = indicators_df["volume_sma_20"]
            indicators_df["volume_ratio"] = volume / vol_sma.replace(0, 1)
            
            # 趋势斜率
            indicators_df["trend_slope_20"] = close.rolling(20).apply(
                lambda x: (x.iloc[-1] - x.iloc[0]) / (len(x) * x.iloc[0]) if len(x) > 0 and x.iloc[0] != 0 else 0,
                raw=False
            )
            
            # 动量
            indicators_df["momentum_10d"] = close.pct_change(10)
            
            # 只保留目标日期范围
            indicators_df = indicators_df.reset_index()
            indicators_df = indicators_df.rename(columns={"index": "trade_date"})
            indicators_df = indicators_df[
                (indicators_df["trade_date"] >= start_date) & 
                (indicators_df["trade_date"] <= end_date)
            ]
            
            # 导入数据库
            if not indicators_df.empty:
                rows = db.upsert_indicators(indicators_df)
                stats["rows_inserted"] += rows
                stats["success"] += 1
                
                if verbose:
                    print(f"✓ {len(indicators_df)} 行")
            else:
                if verbose:
                    print("无有效数据")
                stats["failed"] += 1
            
        except Exception as e:
            stats["failed"] += 1
            if verbose:
                print(f"✗ {e}")
    
    return stats


def full_migration(
    symbols: Optional[List[str]] = None,
    years: int = 2,
    verbose: bool = True,
) -> dict:
    """完整数据迁移
    
    Args:
        symbols: 股票列表，默认使用 DEFAULT_SYMBOLS
        years: 历史数据年数
        verbose: 是否打印进度
    
    Returns:
        统计信息字典
    """
    symbols = symbols or DEFAULT_SYMBOLS
    end_date = date.today()
    start_date = end_date - timedelta(days=365 * years)
    
    if verbose:
        print("=" * 60)
        print("PostgreSQL 18 数据迁移")
        print("=" * 60)
        print(f"股票数量: {len(symbols)}")
        print(f"日期范围: {start_date} ~ {end_date}")
        print()
    
    # 步骤 1: 下载并导入价格数据
    if verbose:
        print("步骤 1: 下载并导入价格数据")
        print("-" * 40)
    
    price_stats = download_and_import(
        symbols, start_date, end_date, verbose=verbose
    )
    
    if verbose:
        print()
        print(f"价格数据: 成功 {price_stats['success']}, 失败 {price_stats['failed']}, "
              f"共 {price_stats['rows_inserted']} 行")
        print()
    
    # 步骤 2: 计算并缓存指标
    if verbose:
        print("步骤 2: 计算并缓存技术指标")
        print("-" * 40)
    
    # 过滤掉 VIX（不需要技术指标）
    indicator_symbols = [s for s in symbols if not s.startswith("^")]
    
    indicator_stats = calculate_and_cache_indicators(
        indicator_symbols, start_date, end_date, verbose=verbose
    )
    
    if verbose:
        print()
        print(f"技术指标: 成功 {indicator_stats['success']}, 失败 {indicator_stats['failed']}, "
              f"共 {indicator_stats['rows_inserted']} 行")
        print()
    
    # 步骤 3: 执行 VACUUM ANALYZE
    if verbose:
        print("步骤 3: 优化数据库...")
    
    db = get_db()
    db.vacuum_analyze()
    
    # 获取最终统计
    final_stats = db.get_stats()
    
    if verbose:
        print()
        print("=" * 60)
        print("迁移完成!")
        print("=" * 60)
        print(f"数据库大小: {final_stats['database_size']}")
        print(f"日线数据: {final_stats['daily_prices_count']} 行")
        print(f"技术指标: {final_stats['indicators_count']} 行")
        print(f"股票数量: {final_stats['unique_symbols']} 只")
        if final_stats.get('price_date_range'):
            print(f"日期范围: {final_stats['price_date_range']['min']} ~ "
                  f"{final_stats['price_date_range']['max']}")
    
    return {
        "price_stats": price_stats,
        "indicator_stats": indicator_stats,
        "final_stats": final_stats,
    }


def incremental_update(
    symbols: Optional[List[str]] = None,
    verbose: bool = True,
) -> dict:
    """增量更新（只更新最新数据）
    
    Args:
        symbols: 股票列表
        verbose: 是否打印进度
    
    Returns:
        统计信息字典
    """
    symbols = symbols or DEFAULT_SYMBOLS
    db = get_db()
    
    # 获取数据库中的最新日期
    latest_date = db.get_latest_date("SPY")
    if latest_date is None:
        if verbose:
            print("数据库为空，执行完整迁移...")
        return full_migration(symbols, years=2, verbose=verbose)
    
    # 从最新日期的下一天开始更新
    start_date = latest_date + timedelta(days=1)
    end_date = date.today()
    
    if start_date > end_date:
        if verbose:
            print(f"数据已是最新 (最后日期: {latest_date})")
        return {"status": "up_to_date", "latest_date": str(latest_date)}
    
    if verbose:
        print(f"增量更新: {start_date} ~ {end_date}")
    
    return full_migration(symbols, years=0, verbose=verbose)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="数据迁移到 PostgreSQL 18")
    parser.add_argument("--full", action="store_true", help="完整迁移")
    parser.add_argument("--incremental", action="store_true", help="增量更新")
    parser.add_argument("--years", type=int, default=2, help="历史年数 (默认 2)")
    parser.add_argument("--symbols", nargs="+", help="指定股票列表")
    
    args = parser.parse_args()
    
    if args.full:
        full_migration(symbols=args.symbols, years=args.years)
    elif args.incremental:
        incremental_update(symbols=args.symbols)
    else:
        # 默认执行完整迁移
        full_migration(symbols=args.symbols, years=args.years)
