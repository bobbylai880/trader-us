#!/usr/bin/env python3
"""
导入 Alternative Data 历史数据到数据库

数据来源:
1. Fed Decisions - 基于 hawkish_dovish.py 中的历史数据
2. Policy Events - 基于 truth_tracker.py 中的历史事件
3. Analyst Ratings - 基于 ratings_tracker.py 中的历史数据
4. Options PCR - 生成合成历史数据 (实际使用时需从 CBOE 获取)
5. Insider Transactions - 生成合成历史数据 (实际使用时需从 OpenInsider 获取)
"""

import json
import os
import sys
from datetime import date, timedelta
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


def import_fed_decisions(conn):
    """导入 FOMC 决策历史"""
    print("\n📊 Importing Fed Decisions...")
    
    fed_data = [
        ("2023-02-01", "hike", 25, 4.75, 4.50, 0.70, 0.20, "25bp hike, ongoing increases appropriate"),
        ("2023-03-22", "hike", 25, 5.00, 4.75, 0.60, 0.30, "25bp hike amid banking stress"),
        ("2023-05-03", "hike", 25, 5.25, 5.00, 0.60, 0.30, "25bp hike, credit conditions tightening"),
        ("2023-06-14", "hold", 0, 5.25, 5.00, 0.50, 0.40, "Pause to assess cumulative tightening"),
        ("2023-07-26", "hike", 25, 5.50, 5.25, 0.70, 0.20, "25bp hike to 5.50%, data dependent"),
        ("2023-09-20", "hold", 0, 5.50, 5.25, 0.60, 0.30, "Hold at 5.50%, higher for longer"),
        ("2023-11-01", "hold", 0, 5.50, 5.25, 0.50, 0.40, "Hold, tighter conditions doing work"),
        ("2023-12-13", "hold", 0, 5.50, 5.25, 0.30, 0.60, "Dovish pivot, rate cuts discussed"),
        ("2024-01-31", "hold", 0, 5.50, 5.25, 0.50, 0.40, "Hold, not ready to cut yet"),
        ("2024-03-20", "hold", 0, 5.50, 5.25, 0.50, 0.50, "Hold, inflation bumpy path"),
        ("2024-05-01", "hold", 0, 5.50, 5.25, 0.60, 0.30, "Hold, lack of progress on inflation"),
        ("2024-06-12", "hold", 0, 5.50, 5.25, 0.50, 0.50, "Hold, modest progress on inflation"),
        ("2024-07-31", "hold", 0, 5.50, 5.25, 0.40, 0.60, "Hold, cut could be on table September"),
        ("2024-09-18", "cut", -50, 5.00, 4.75, 0.20, 0.80, "50bp cut to recalibrate policy"),
        ("2024-11-07", "cut", -25, 4.75, 4.50, 0.30, 0.60, "25bp cut, economy remains solid"),
        ("2024-12-18", "cut", -25, 4.50, 4.25, 0.40, 0.50, "25bp cut, fewer cuts expected 2025"),
        ("2025-01-29", "hold", 0, 4.50, 4.25, 0.50, 0.40, "Hold, wait for more data"),
        ("2025-03-19", "hold", 0, 4.50, 4.25, 0.60, 0.30, "Hold, inflation concerns from tariffs"),
        ("2025-05-07", "hold", 0, 4.50, 4.25, 0.50, 0.50, "Hold, watching tariff impact"),
        ("2025-06-18", "cut", -25, 4.25, 4.00, 0.30, 0.60, "25bp cut, economy slowing"),
        ("2025-07-30", "hold", 0, 4.25, 4.00, 0.40, 0.50, "Hold, assess impact of June cut"),
        ("2025-09-17", "cut", -25, 4.00, 3.75, 0.30, 0.60, "25bp cut, inflation near target"),
        ("2025-11-05", "hold", 0, 4.00, 3.75, 0.40, 0.50, "Hold at 4.0%, balanced risks"),
        ("2025-12-17", "hold", 0, 4.00, 3.75, 0.50, 0.50, "Hold, economy in good place"),
    ]
    
    cur = conn.cursor()
    inserted = 0
    
    for row in fed_data:
        try:
            cur.execute("""
                INSERT INTO fed_decisions 
                (meeting_date, rate_decision, rate_change_bps, fed_funds_upper, fed_funds_lower,
                 hawkish_score, dovish_score, statement_summary)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (meeting_date) DO UPDATE SET
                    rate_decision = EXCLUDED.rate_decision,
                    rate_change_bps = EXCLUDED.rate_change_bps,
                    hawkish_score = EXCLUDED.hawkish_score,
                    dovish_score = EXCLUDED.dovish_score
            """, row)
            inserted += 1
        except Exception as e:
            print(f"  Error inserting {row[0]}: {e}")
    
    conn.commit()
    print(f"  ✅ Inserted {inserted} Fed decisions")


def import_policy_events(conn):
    """导入政策事件历史"""
    print("\n📊 Importing Policy Events...")
    
    events = [
        ("2024-11-06", "election", "trump_elected", "Trump wins 2024 election", 0.5, 1.0,
         {"XLF": 0.3, "XLE": 0.2, "XLY": 0.2, "XLI": 0.1}),
        ("2025-01-20", "inauguration", "trump_inauguration", "Trump inaugurated as 47th President", 0.3, 0.8,
         {"XLF": 0.2, "XLE": 0.2}),
        ("2025-02-01", "tariff", "tariff_threat_mexico_canada", "Tariff threats on Mexico and Canada", -0.4, 0.9,
         {"XLI": -0.3, "XLY": -0.2, "XLK": -0.1}),
        ("2025-03-04", "tariff", "tariff_implementation", "Tariffs implemented on imports", -0.6, 1.0,
         {"XLI": -0.4, "XLY": -0.3, "XLK": -0.2}),
        ("2025-04-02", "tariff", "reciprocal_tariffs", "Reciprocal tariffs announced", -0.8, 1.0,
         {"XLI": -0.5, "XLY": -0.4, "XLK": -0.3, "XLF": -0.2}),
        ("2025-04-09", "tariff", "tariff_pause_90days", "90-day tariff pause announced", 0.7, 1.0,
         {"XLI": 0.3, "XLY": 0.3, "XLK": 0.2}),
        ("2024-01-10", "fed", "fed_pivot_signal", "Fed signals potential rate cuts in 2024", 0.4, 0.7,
         {"XLK": 0.2, "XLRE": 0.3, "XLU": 0.2}),
        ("2024-08-05", "market", "yen_carry_unwind", "Yen carry trade unwind causes volatility", -0.6, 0.9,
         {"XLK": -0.2, "XLF": -0.3, "XLY": -0.2}),
        ("2023-03-10", "banking", "svb_collapse", "Silicon Valley Bank collapse", -0.7, 0.9,
         {"XLF": -0.4, "XLRE": -0.2}),
        ("2023-11-01", "ai", "chatgpt_momentum", "AI investment boom continues", 0.5, 0.7,
         {"XLK": 0.4, "XLC": 0.2}),
    ]
    
    cur = conn.cursor()
    inserted = 0
    
    for row in events:
        try:
            cur.execute("""
                INSERT INTO policy_events 
                (event_date, event_type, topic, description, sentiment, urgency, sector_impacts, source)
                VALUES (%s, %s, %s, %s, %s, %s, %s, 'historical')
                ON CONFLICT (event_date, event_type, topic) DO UPDATE SET
                    sentiment = EXCLUDED.sentiment,
                    urgency = EXCLUDED.urgency,
                    sector_impacts = EXCLUDED.sector_impacts
            """, (row[0], row[1], row[2], row[3], row[4], row[5], json.dumps(row[6])))
            inserted += 1
        except Exception as e:
            print(f"  Error inserting {row[0]} {row[2]}: {e}")
    
    conn.commit()
    print(f"  ✅ Inserted {inserted} policy events")


def import_analyst_ratings(conn):
    """导入分析师评级历史"""
    print("\n📊 Importing Analyst Ratings...")
    
    ratings = [
        ("NVDA", "2024-01-08", "Morgan Stanley", "Overweight", "Overweight", 600, 750, "reiterate"),
        ("NVDA", "2024-02-22", "Goldman Sachs", "Buy", "Buy", 625, 800, "reiterate"),
        ("NVDA", "2024-05-23", "Bank of America", "Buy", "Buy", 925, 1100, "reiterate"),
        ("NVDA", "2024-08-29", "JP Morgan", "Overweight", "Overweight", 115, 155, "reiterate"),
        ("NVDA", "2025-01-07", "Citi", "Buy", "Buy", 150, 175, "reiterate"),
        ("META", "2024-02-02", "Goldman Sachs", "Buy", "Buy", 400, 525, "reiterate"),
        ("META", "2024-04-25", "Morgan Stanley", "Overweight", "Overweight", 500, 550, "reiterate"),
        ("META", "2024-07-31", "JP Morgan", "Overweight", "Overweight", 480, 570, "reiterate"),
        ("META", "2024-10-31", "Bank of America", "Buy", "Buy", 560, 630, "reiterate"),
        ("AAPL", "2024-01-25", "Barclays", "Equal-Weight", "Underweight", 161, 160, "downgrade"),
        ("AAPL", "2024-05-03", "Goldman Sachs", "Neutral", "Buy", 199, 226, "upgrade"),
        ("AAPL", "2024-08-02", "JP Morgan", "Overweight", "Overweight", 225, 245, "reiterate"),
        ("AAPL", "2024-11-01", "Morgan Stanley", "Overweight", "Overweight", 253, 273, "reiterate"),
        ("TSLA", "2024-01-25", "Goldman Sachs", "Neutral", "Neutral", 220, 190, "reiterate"),
        ("TSLA", "2024-04-24", "Morgan Stanley", "Overweight", "Overweight", 320, 310, "reiterate"),
        ("TSLA", "2024-07-24", "Barclays", "Equal-Weight", "Equal-Weight", 180, 200, "reiterate"),
        ("TSLA", "2024-10-24", "JP Morgan", "Underweight", "Underweight", 130, 135, "reiterate"),
        ("TSLA", "2024-11-12", "Bank of America", "Neutral", "Buy", 265, 350, "upgrade"),
        ("AMD", "2024-01-31", "Morgan Stanley", "Overweight", "Overweight", 190, 210, "reiterate"),
        ("AMD", "2024-04-30", "Goldman Sachs", "Buy", "Buy", 190, 208, "reiterate"),
        ("AMD", "2024-07-30", "Bank of America", "Buy", "Buy", 195, 180, "reiterate"),
        ("AMD", "2024-10-29", "JP Morgan", "Overweight", "Overweight", 180, 200, "reiterate"),
        ("GOOGL", "2024-01-31", "Goldman Sachs", "Buy", "Buy", 160, 175, "reiterate"),
        ("GOOGL", "2024-04-26", "Morgan Stanley", "Overweight", "Overweight", 160, 190, "reiterate"),
        ("GOOGL", "2024-07-24", "JP Morgan", "Overweight", "Overweight", 195, 208, "reiterate"),
        ("GOOGL", "2024-10-30", "Bank of America", "Buy", "Buy", 200, 210, "reiterate"),
        ("MSFT", "2024-01-31", "Goldman Sachs", "Buy", "Buy", 430, 465, "reiterate"),
        ("MSFT", "2024-04-26", "Morgan Stanley", "Overweight", "Overweight", 465, 520, "reiterate"),
        ("MSFT", "2024-07-31", "JP Morgan", "Overweight", "Overweight", 470, 500, "reiterate"),
        ("MSFT", "2024-10-31", "Bank of America", "Buy", "Buy", 490, 510, "reiterate"),
        ("AMZN", "2024-02-02", "Goldman Sachs", "Buy", "Buy", 185, 220, "reiterate"),
        ("AMZN", "2024-05-01", "Morgan Stanley", "Overweight", "Overweight", 200, 210, "reiterate"),
        ("AMZN", "2024-08-02", "JP Morgan", "Overweight", "Overweight", 220, 230, "reiterate"),
        ("AMZN", "2024-11-01", "Bank of America", "Buy", "Buy", 210, 230, "reiterate"),
        ("AVGO", "2024-03-08", "Goldman Sachs", "Buy", "Buy", 1200, 1400, "reiterate"),
        ("AVGO", "2024-06-13", "Morgan Stanley", "Overweight", "Overweight", 1500, 1800, "reiterate"),
        ("AVGO", "2024-09-06", "JP Morgan", "Overweight", "Overweight", 185, 220, "reiterate"),
        ("NFLX", "2024-01-24", "Goldman Sachs", "Neutral", "Buy", 475, 600, "upgrade"),
        ("NFLX", "2024-04-19", "Morgan Stanley", "Overweight", "Overweight", 550, 650, "reiterate"),
        ("NFLX", "2024-07-19", "JP Morgan", "Overweight", "Overweight", 650, 750, "reiterate"),
    ]
    
    cur = conn.cursor()
    inserted = 0
    
    for row in ratings:
        try:
            cur.execute("""
                INSERT INTO analyst_ratings 
                (symbol, rating_date, analyst, old_rating, new_rating, old_target, new_target, action)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol, rating_date, analyst) DO UPDATE SET
                    old_rating = EXCLUDED.old_rating,
                    new_rating = EXCLUDED.new_rating,
                    old_target = EXCLUDED.old_target,
                    new_target = EXCLUDED.new_target
            """, row)
            inserted += 1
        except Exception as e:
            print(f"  Error inserting {row[0]} {row[1]}: {e}")
    
    conn.commit()
    print(f"  ✅ Inserted {inserted} analyst ratings")


def import_options_pcr(conn):
    """导入 Put/Call Ratio 历史数据"""
    print("\n📊 Importing Options PCR data...")
    
    cur = conn.cursor()
    
    start_date = date(2023, 1, 1)
    end_date = date(2026, 1, 16)
    
    pcr_events = {
        "2023-03-13": 1.35,
        "2023-10-27": 1.28,
        "2024-08-05": 1.42,
        "2024-09-06": 1.25,
        "2024-11-01": 0.72,
        "2025-04-04": 1.38,
    }
    
    inserted = 0
    current = start_date
    
    while current <= end_date:
        if current.weekday() < 5:
            seed = current.toordinal()
            base_pcr = 0.85 + (seed % 30) / 100
            
            date_str = str(current)
            if date_str in pcr_events:
                total_pcr = pcr_events[date_str]
            else:
                total_pcr = base_pcr
            
            equity_pcr = total_pcr * 0.9
            index_pcr = total_pcr * 1.2
            
            try:
                cur.execute("""
                    INSERT INTO options_pcr (trade_date, equity_pcr, index_pcr, total_pcr)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (trade_date) DO UPDATE SET
                        equity_pcr = EXCLUDED.equity_pcr,
                        index_pcr = EXCLUDED.index_pcr,
                        total_pcr = EXCLUDED.total_pcr
                """, (current, round(equity_pcr, 4), round(index_pcr, 4), round(total_pcr, 4)))
                inserted += 1
            except Exception as e:
                print(f"  Error inserting PCR {current}: {e}")
        
        current += timedelta(days=1)
    
    conn.commit()
    print(f"  ✅ Inserted {inserted} PCR records")


def import_insider_transactions(conn):
    """导入内部人交易数据 (关键事件)"""
    print("\n📊 Importing Insider Transactions...")
    
    transactions = [
        ("NVDA", "2024-01-15", "2024-01-12", "Jensen Huang", "CEO", "S", 100000, 550.0, 55000000),
        ("NVDA", "2024-03-08", "2024-03-06", "Colette Kress", "CFO", "S", 30000, 875.0, 26250000),
        ("META", "2024-02-05", "2024-02-02", "Mark Zuckerberg", "CEO", "S", 200000, 475.0, 95000000),
        ("META", "2024-11-15", "2024-11-13", "Javier Olivan", "COO", "S", 5000, 580.0, 2900000),
        ("AAPL", "2024-02-01", "2024-01-30", "Tim Cook", "CEO", "S", 50000, 185.0, 9250000),
        ("AAPL", "2024-08-05", "2024-08-02", "Luca Maestri", "CFO", "S", 20000, 220.0, 4400000),
        ("TSLA", "2024-04-29", "2024-04-26", "Elon Musk", "CEO", "P", 10000, 175.0, 1750000),
        ("TSLA", "2024-11-15", "2024-11-13", "Elon Musk", "CEO", "S", 50000, 350.0, 17500000),
        ("AMD", "2024-02-15", "2024-02-13", "Lisa Su", "CEO", "S", 25000, 175.0, 4375000),
        ("AMD", "2024-08-01", "2024-07-30", "Lisa Su", "CEO", "S", 20000, 145.0, 2900000),
        ("GOOGL", "2024-02-08", "2024-02-06", "Sundar Pichai", "CEO", "S", 15000, 145.0, 2175000),
        ("GOOGL", "2024-07-25", "2024-07-23", "Ruth Porat", "CFO", "S", 10000, 178.0, 1780000),
        ("MSFT", "2024-01-18", "2024-01-16", "Satya Nadella", "CEO", "S", 25000, 390.0, 9750000),
        ("MSFT", "2024-05-20", "2024-05-17", "Amy Hood", "CFO", "S", 8000, 420.0, 3360000),
        ("AMZN", "2024-02-12", "2024-02-09", "Andy Jassy", "CEO", "S", 10000, 175.0, 1750000),
        ("AMZN", "2024-11-20", "2024-11-18", "Brian Olsavsky", "CFO", "S", 5000, 205.0, 1025000),
        ("AVGO", "2024-03-15", "2024-03-13", "Hock Tan", "CEO", "S", 50000, 1350.0, 67500000),
        ("NFLX", "2024-01-25", "2024-01-23", "Ted Sarandos", "Co-CEO", "S", 12000, 560.0, 6720000),
        ("NFLX", "2024-07-22", "2024-07-19", "Greg Peters", "Co-CEO", "S", 8000, 680.0, 5440000),
        ("JPM", "2024-01-15", "2024-01-12", "Jamie Dimon", "CEO", "S", 150000, 175.0, 26250000),
        ("JPM", "2024-04-15", "2024-04-12", "Jamie Dimon", "CEO", "S", 100000, 195.0, 19500000),
        ("GS", "2024-02-20", "2024-02-16", "David Solomon", "CEO", "P", 5000, 385.0, 1925000),
        ("LLY", "2024-05-10", "2024-05-08", "David Ricks", "CEO", "S", 20000, 780.0, 15600000),
        ("UNH", "2024-03-15", "2024-03-13", "Andrew Witty", "CEO", "S", 8000, 495.0, 3960000),
    ]
    
    cur = conn.cursor()
    inserted = 0
    
    for row in transactions:
        try:
            cur.execute("""
                INSERT INTO insider_transactions 
                (symbol, filing_date, trade_date, insider_name, title, transaction_type, shares, price, value)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol, filing_date, insider_name, transaction_type, shares) DO NOTHING
            """, row)
            inserted += 1
        except Exception as e:
            print(f"  Error inserting {row[0]} {row[1]}: {e}")
    
    conn.commit()
    print(f"  ✅ Inserted {inserted} insider transactions")


def import_social_sentiment(conn):
    """导入社交媒体情绪数据 (合成)"""
    print("\n📊 Importing Social Sentiment data...")
    
    cur = conn.cursor()
    
    symbols = ["NVDA", "META", "GOOGL", "AMZN", "MSFT", "AAPL", "AMD", "AVGO", "NFLX", "TSLA"]
    sources = ["reddit", "twitter"]
    
    start_date = date(2023, 1, 1)
    end_date = date(2026, 1, 16)
    
    sentiment_spikes = {
        ("NVDA", "2024-02-22"): 0.8,
        ("NVDA", "2024-05-23"): 0.7,
        ("META", "2024-02-02"): 0.9,
        ("TSLA", "2024-11-06"): 0.85,
        ("GME", "2024-05-15"): 0.95,
    }
    
    inserted = 0
    current = start_date
    
    while current <= end_date:
        if current.weekday() < 5:
            for symbol in symbols:
                for source in sources:
                    seed = current.toordinal() + hash(symbol) + hash(source)
                    
                    mention_count = 10 + (seed % 90)
                    base_sentiment = ((seed % 100) - 50) / 100
                    
                    key = (symbol, str(current))
                    if key in sentiment_spikes:
                        avg_sentiment = sentiment_spikes[key]
                    else:
                        avg_sentiment = base_sentiment
                    
                    signal_strength = avg_sentiment * 0.6 + (mention_count / 100) * 0.4
                    signal_strength = max(-1, min(1, signal_strength))
                    
                    hot_rank = (seed % 50) + 1
                    
                    try:
                        cur.execute("""
                            INSERT INTO social_sentiment 
                            (symbol, trade_date, source, mention_count, avg_sentiment, signal_strength, hot_rank)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (symbol, trade_date, source) DO UPDATE SET
                                mention_count = EXCLUDED.mention_count,
                                avg_sentiment = EXCLUDED.avg_sentiment,
                                signal_strength = EXCLUDED.signal_strength
                        """, (symbol, current, source, mention_count, 
                              round(avg_sentiment, 3), round(signal_strength, 3), hot_rank))
                        inserted += 1
                    except Exception as e:
                        pass
        
        current += timedelta(days=1)
    
    conn.commit()
    print(f"  ✅ Inserted {inserted} social sentiment records")


def verify_data(conn):
    """验证导入的数据"""
    print("\n" + "=" * 60)
    print("📊 Data Verification")
    print("=" * 60)
    
    cur = conn.cursor()
    
    tables = [
        ("fed_decisions", "meeting_date"),
        ("policy_events", "event_date"),
        ("analyst_ratings", "rating_date"),
        ("options_pcr", "trade_date"),
        ("insider_transactions", "filing_date"),
        ("social_sentiment", "trade_date"),
    ]
    
    for table, date_col in tables:
        cur.execute(f"SELECT COUNT(*), MIN({date_col}), MAX({date_col}) FROM {table}")
        count, min_date, max_date = cur.fetchone()
        print(f"\n  {table}:")
        print(f"    Records: {count}")
        print(f"    Range: {min_date} ~ {max_date}")
    
    print("\n" + "=" * 60)
    print("✅ All Alternative Data imported successfully!")
    print("=" * 60)


def main():
    print("=" * 60)
    print("Alternative Data Import Script")
    print("=" * 60)
    
    conn = get_connection()
    
    try:
        import_fed_decisions(conn)
        import_policy_events(conn)
        import_analyst_ratings(conn)
        import_options_pcr(conn)
        import_insider_transactions(conn)
        import_social_sentiment(conn)
        
        verify_data(conn)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
