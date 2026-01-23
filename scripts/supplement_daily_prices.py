#!/usr/bin/env python3
"""
补充 2018-2020 年 daily_prices 数据
使用 yfinance 获取历史行情
"""

import os
import sys
from datetime import date, datetime
from typing import List

import psycopg2
import yfinance as yf
import pandas as pd

DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "192.168.10.11"),
    "port": os.getenv("POSTGRES_PORT", "5432"),
    "database": os.getenv("POSTGRES_DB", "trader"),
    "user": os.getenv("POSTGRES_USER", "trader"),
    "password": os.getenv("POSTGRES_PASSWORD", "Abc3878820@"),
}

SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "AMD", "NFLX", "CRM",
    "AVGO", "ORCL", "ADBE", "INTC", "QCOM",
    "SPY", "QQQ", "VIX",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLC", "XLB", "XLP", "XLU", "XLRE",
]


def fetch_and_insert_prices(conn, symbol: str, start: str, end: str) -> int:
    cur = conn.cursor()
    
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end, auto_adjust=False)
        
        if df.empty:
            print(f"  ⚠️ {symbol}: 无数据")
            return 0
        
        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        
        inserted = 0
        for _, row in df.iterrows():
            try:
                trade_date = row['Date']
                open_price = float(row['Open']) if pd.notna(row['Open']) else None
                high = float(row['High']) if pd.notna(row['High']) else None
                low = float(row['Low']) if pd.notna(row['Low']) else None
                close = float(row['Close']) if pd.notna(row['Close']) else None
                adj_close = float(row['Adj Close']) if pd.notna(row.get('Adj Close', row['Close'])) else close
                volume = int(row['Volume']) if pd.notna(row['Volume']) else 0
                
                if close is None or close <= 0:
                    continue
                
                daily_range = ((high - low) / close * 100) if (high and low and close) else None
                
                cur.execute("""
                    INSERT INTO daily_prices 
                    (symbol, trade_date, open, high, low, close, adj_close, volume)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (symbol, trade_date) DO NOTHING
                """, (
                    symbol, trade_date, open_price, high, low, close, adj_close, volume
                ))
                inserted += cur.rowcount
                
            except Exception as e:
                print(f"  ⚠️ {symbol} {trade_date}: {e}")
                continue
        
        conn.commit()
        return inserted
        
    except Exception as e:
        print(f"  ❌ {symbol}: {e}")
        conn.rollback()
        return 0
    finally:
        cur.close()


def main():
    print("=" * 60)
    print("补充 2018-2020 年 daily_prices 数据")
    print("=" * 60)
    
    conn = psycopg2.connect(**DB_CONFIG)
    
    start_date = "2018-01-01"
    end_date = "2021-01-01"
    
    total_inserted = 0
    
    print(f"\n下载范围: {start_date} ~ {end_date}")
    print(f"股票数量: {len(SYMBOLS)}")
    print()
    
    for i, symbol in enumerate(SYMBOLS, 1):
        print(f"[{i}/{len(SYMBOLS)}] 下载 {symbol}...", end=" ")
        sys.stdout.flush()
        
        inserted = fetch_and_insert_prices(conn, symbol, start_date, end_date)
        total_inserted += inserted
        print(f"✓ {inserted} 条")
    
    print("\n" + "=" * 60)
    print(f"✓ 总共导入 {total_inserted:,} 条价格数据")
    
    cur = conn.cursor()
    print("\n=== 导入后数据范围 ===")
    cur.execute("SELECT MIN(trade_date), MAX(trade_date), COUNT(*) FROM daily_prices")
    min_date, max_date, count = cur.fetchone()
    print(f"daily_prices: {min_date} ~ {max_date} ({count:,} 条)")
    
    print("\n各年份数据量:")
    for year in range(2018, 2027):
        cur.execute(f"SELECT COUNT(*) FROM daily_prices WHERE EXTRACT(YEAR FROM trade_date) = {year}")
        count = cur.fetchone()[0]
        print(f"  {year} 年: {count:,} 条")
    
    cur.close()
    conn.close()
    print("=" * 60)


if __name__ == "__main__":
    main()
