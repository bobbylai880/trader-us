#!/usr/bin/env python3
"""
导入客观宏观数据 (Objective Macro Data)
用于替代合成的 Alternative Data，构建真实的 V8.1 策略
"""

import os
import sys
import yfinance as yf
import pandas as pd
import psycopg2

DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "192.168.10.11"),
    "port": os.getenv("POSTGRES_PORT", "5432"),
    "database": os.getenv("POSTGRES_DB", "trader"),
    "user": os.getenv("POSTGRES_USER", "trader"),
    "password": os.getenv("POSTGRES_PASSWORD", "Abc3878820@"),
}

# 客观宏观指标列表
MACRO_SYMBOLS = {
    "^VIX": "Volatility Index (Fear Gauge)",
    "^TNX": "10-Year Treasury Yield (Inflation/Rates)",
    "^IRX": "13-Week Treasury Bill (Liquidity)",
    "HYG": "High Yield Bond ETF (Risk Appetite)",
    "TLT": "20+ Year Treasury Bond ETF (Safe Haven)",
    "DBC": "Commodity Index (Inflation Cost)",
    "UUP": "US Dollar Index Bullish (Currency)",
    "XLY": "Consumer Discretionary (Cyclical)",
    "XLP": "Consumer Staples (Defensive)",
    "RSP": "S&P 500 Equal Weight (Breadth)",
}

def fetch_and_insert(conn, symbol: str, description: str, start: str, end: str):
    cur = conn.cursor()
    print(f"正在下载 {symbol:<6} - {description}...", end=" ")
    sys.stdout.flush()
    
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end, auto_adjust=False)
        
        if df.empty:
            print("❌ 无数据")
            return
            
        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        
        inserted = 0
        for _, row in df.iterrows():
            try:
                # 处理数据: Yahoo 有时返回 NaN
                close = float(row['Close']) if pd.notna(row['Close']) else None
                if close is None: continue

                trade_date = row['Date']
                open_p = float(row['Open']) if pd.notna(row['Open']) else close
                high = float(row['High']) if pd.notna(row['High']) else close
                low = float(row['Low']) if pd.notna(row['Low']) else close
                adj_close = float(row['Adj Close']) if pd.notna(row.get('Adj Close')) else close
                volume = int(row['Volume']) if pd.notna(row['Volume']) else 0
                
                # 简单的涨跌判断
                is_green = close >= open_p
                
                cur.execute("""
                    INSERT INTO daily_prices 
                    (symbol, trade_date, open, high, low, close, adj_close, volume)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (symbol, trade_date) 
                    DO UPDATE SET
                        close = EXCLUDED.close,
                        adj_close = EXCLUDED.adj_close,
                        volume = EXCLUDED.volume
                """, (
                    symbol, trade_date, open_p, high, low, close, adj_close, volume
                ))
                inserted += cur.rowcount
            except Exception:
                continue
        
        conn.commit()
        print(f"✓ 更新 {inserted} 条记录")
        
    except Exception as e:
        conn.rollback()
        print(f"❌ 失败: {e}")
    finally:
        cur.close()

def main():
    print("=" * 70)
    print("开始导入客观宏观数据 (2018-01-01 ~ 2026-01-17)")
    print("=" * 70)
    
    conn = psycopg2.connect(**DB_CONFIG)
    start_date = "2018-01-01"
    end_date = "2026-01-17"
    
    for symbol, desc in MACRO_SYMBOLS.items():
        fetch_and_insert(conn, symbol, desc, start_date, end_date)
        
    print("\n" + "=" * 70)
    print("数据导入完成。这些数据将用于驱动真实的 V8.1 策略。")
    conn.close()

if __name__ == "__main__":
    main()
