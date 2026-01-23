#!/usr/bin/env python3
"""
创建 Alternative Data 数据表

表结构:
1. insider_transactions - 内部人交易记录
2. options_pcr - Put/Call Ratio 历史
3. fed_decisions - FOMC 决策历史
4. analyst_ratings - 分析师评级变化
5. policy_events - 政策事件 (关税等)
"""

import os
import sys
from pathlib import Path

import psycopg2

sys.path.insert(0, str(Path(__file__).parent.parent))


def get_connection():
    return psycopg2.connect(
        host=os.getenv("PG_HOST", "192.168.10.11"),
        port=os.getenv("PG_PORT", "5432"),
        database=os.getenv("PG_DATABASE", "trader"),
        user=os.getenv("PG_USER", "trader"),
        password=os.getenv("PG_PASSWORD", "")
    )


def create_tables():
    conn = get_connection()
    cur = conn.cursor()
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS insider_transactions (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(10) NOT NULL,
            filing_date DATE NOT NULL,
            trade_date DATE NOT NULL,
            insider_name VARCHAR(200),
            title VARCHAR(100),
            transaction_type CHAR(1) NOT NULL,
            shares INTEGER,
            price NUMERIC(12, 4),
            value NUMERIC(16, 2),
            ownership_type CHAR(1) DEFAULT 'D',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, filing_date, insider_name, transaction_type, shares)
        );
        
        CREATE INDEX IF NOT EXISTS idx_insider_symbol_date 
            ON insider_transactions(symbol, filing_date);
        CREATE INDEX IF NOT EXISTS idx_insider_trade_date 
            ON insider_transactions(trade_date);
    """)
    print("✅ Created: insider_transactions")
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS options_pcr (
            id SERIAL PRIMARY KEY,
            trade_date DATE NOT NULL UNIQUE,
            equity_pcr NUMERIC(6, 4),
            index_pcr NUMERIC(6, 4),
            total_pcr NUMERIC(6, 4),
            volume_put BIGINT,
            volume_call BIGINT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_pcr_date ON options_pcr(trade_date);
    """)
    print("✅ Created: options_pcr")
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS fed_decisions (
            id SERIAL PRIMARY KEY,
            meeting_date DATE NOT NULL UNIQUE,
            rate_decision VARCHAR(10) NOT NULL,
            rate_change_bps INTEGER DEFAULT 0,
            fed_funds_upper NUMERIC(5, 2),
            fed_funds_lower NUMERIC(5, 2),
            hawkish_score NUMERIC(4, 2),
            dovish_score NUMERIC(4, 2),
            statement_summary TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_fed_date ON fed_decisions(meeting_date);
    """)
    print("✅ Created: fed_decisions")
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS analyst_ratings (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(10) NOT NULL,
            rating_date DATE NOT NULL,
            analyst VARCHAR(100),
            old_rating VARCHAR(30),
            new_rating VARCHAR(30),
            old_target NUMERIC(10, 2),
            new_target NUMERIC(10, 2),
            action VARCHAR(20),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, rating_date, analyst)
        );
        
        CREATE INDEX IF NOT EXISTS idx_analyst_symbol_date 
            ON analyst_ratings(symbol, rating_date);
    """)
    print("✅ Created: analyst_ratings")
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS policy_events (
            id SERIAL PRIMARY KEY,
            event_date DATE NOT NULL,
            event_type VARCHAR(50) NOT NULL,
            topic VARCHAR(100),
            description TEXT,
            sentiment NUMERIC(4, 2),
            urgency NUMERIC(4, 2),
            sector_impacts JSONB,
            source VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(event_date, event_type, topic)
        );
        
        CREATE INDEX IF NOT EXISTS idx_policy_date ON policy_events(event_date);
        CREATE INDEX IF NOT EXISTS idx_policy_type ON policy_events(event_type);
    """)
    print("✅ Created: policy_events")
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS social_sentiment (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(10) NOT NULL,
            trade_date DATE NOT NULL,
            source VARCHAR(20) NOT NULL,
            mention_count INTEGER DEFAULT 0,
            avg_sentiment NUMERIC(5, 3),
            signal_strength NUMERIC(5, 3),
            hot_rank INTEGER,
            metadata JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, trade_date, source)
        );
        
        CREATE INDEX IF NOT EXISTS idx_social_symbol_date 
            ON social_sentiment(symbol, trade_date);
    """)
    print("✅ Created: social_sentiment")
    
    conn.commit()
    cur.close()
    conn.close()
    print("\n✅ All Alternative Data tables created successfully!")


if __name__ == "__main__":
    create_tables()
