#!/usr/bin/env python3
"""
补充 2020 年 Alternative Data 数据
用于支持 6 年回测 (2020-01-01 ~ 2026-01-17)
"""

import os
import sys
import random
from datetime import date, timedelta
from typing import List, Dict

import psycopg2
from psycopg2.extras import execute_values

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
]

FED_DECISIONS_2020 = [
    {"meeting_date": "2020-01-29", "rate_decision": "hold", "rate_change_bps": 0,
     "fed_funds_upper": 1.75, "fed_funds_lower": 1.50, "hawkish_score": 0.5, "dovish_score": 0.5,
     "statement_summary": "经济温和扩张，维持利率不变"},
    {"meeting_date": "2020-03-03", "rate_decision": "cut", "rate_change_bps": -50,
     "fed_funds_upper": 1.25, "fed_funds_lower": 1.00, "hawkish_score": 0.1, "dovish_score": 0.9,
     "statement_summary": "紧急降息50bp应对COVID-19冲击"},
    {"meeting_date": "2020-03-15", "rate_decision": "cut", "rate_change_bps": -100,
     "fed_funds_upper": 0.25, "fed_funds_lower": 0.00, "hawkish_score": 0.0, "dovish_score": 1.0,
     "statement_summary": "紧急降息至零利率，启动7000亿QE"},
    {"meeting_date": "2020-04-29", "rate_decision": "hold", "rate_change_bps": 0,
     "fed_funds_upper": 0.25, "fed_funds_lower": 0.00, "hawkish_score": 0.1, "dovish_score": 0.9,
     "statement_summary": "维持零利率，承诺无限量QE"},
    {"meeting_date": "2020-06-10", "rate_decision": "hold", "rate_change_bps": 0,
     "fed_funds_upper": 0.25, "fed_funds_lower": 0.00, "hawkish_score": 0.15, "dovish_score": 0.85,
     "statement_summary": "点阵图显示零利率维持至2022年"},
    {"meeting_date": "2020-07-29", "rate_decision": "hold", "rate_change_bps": 0,
     "fed_funds_upper": 0.25, "fed_funds_lower": 0.00, "hawkish_score": 0.2, "dovish_score": 0.8,
     "statement_summary": "经济复苏取决于疫情控制"},
    {"meeting_date": "2020-09-16", "rate_decision": "hold", "rate_change_bps": 0,
     "fed_funds_upper": 0.25, "fed_funds_lower": 0.00, "hawkish_score": 0.2, "dovish_score": 0.8,
     "statement_summary": "引入平均通胀目标制(AIT)"},
    {"meeting_date": "2020-11-05", "rate_decision": "hold", "rate_change_bps": 0,
     "fed_funds_upper": 0.25, "fed_funds_lower": 0.00, "hawkish_score": 0.2, "dovish_score": 0.8,
     "statement_summary": "大选后维持宽松，关注疫苗进展"},
    {"meeting_date": "2020-12-16", "rate_decision": "hold", "rate_change_bps": 0,
     "fed_funds_upper": 0.25, "fed_funds_lower": 0.00, "hawkish_score": 0.25, "dovish_score": 0.75,
     "statement_summary": "加强前瞻指引，QE持续至就业目标实现"},
]

POLICY_EVENTS_2020 = [
    {"event_date": "2020-01-15", "event_type": "贸易", "topic": "中美第一阶段协议",
     "description": "中美签署第一阶段贸易协议，缓解贸易战担忧",
     "sentiment": 0.6, "urgency": 0.8, "sector_impacts": {"XLK": 0.4, "XLI": 0.3, "XLE": 0.2},
     "source": "White House"},
    {"event_date": "2020-02-28", "event_type": "疫情", "topic": "COVID-19全球蔓延",
     "description": "WHO宣布COVID-19为国际关注的突发公共卫生事件",
     "sentiment": -0.8, "urgency": 1.0, "sector_impacts": {"XLY": -0.5, "XLE": -0.4, "XLK": -0.2},
     "source": "WHO"},
    {"event_date": "2020-03-13", "event_type": "政策", "topic": "国家紧急状态",
     "description": "特朗普宣布全国紧急状态，释放500亿联邦资金",
     "sentiment": -0.5, "urgency": 1.0, "sector_impacts": {"XLV": 0.3, "XLY": -0.4},
     "source": "White House"},
    {"event_date": "2020-03-27", "event_type": "财政", "topic": "CARES法案",
     "description": "2.2万亿美元CARES法案签署，史上最大刺激计划",
     "sentiment": 0.7, "urgency": 1.0, "sector_impacts": {"XLF": 0.4, "XLY": 0.5, "XLI": 0.3},
     "source": "Congress"},
    {"event_date": "2020-04-09", "event_type": "货币", "topic": "Fed扩大救助",
     "description": "Fed宣布2.3万亿贷款计划支持经济",
     "sentiment": 0.6, "urgency": 0.9, "sector_impacts": {"XLF": 0.5, "XLK": 0.3},
     "source": "Federal Reserve"},
    {"event_date": "2020-08-11", "event_type": "疫情", "topic": "俄罗斯疫苗获批",
     "description": "俄罗斯批准全球首款COVID-19疫苗",
     "sentiment": 0.4, "urgency": 0.7, "sector_impacts": {"XLV": 0.3, "XLY": 0.2},
     "source": "Russia"},
    {"event_date": "2020-11-09", "event_type": "疫情", "topic": "辉瑞疫苗突破",
     "description": "辉瑞疫苗三期试验有效率90%，市场大涨",
     "sentiment": 0.9, "urgency": 1.0, "sector_impacts": {"XLY": 0.6, "XLE": 0.5, "XLF": 0.4, "XLK": -0.1},
     "source": "Pfizer"},
    {"event_date": "2020-11-16", "event_type": "疫情", "topic": "Moderna疫苗",
     "description": "Moderna疫苗有效率94.5%，强化复苏预期",
     "sentiment": 0.8, "urgency": 0.9, "sector_impacts": {"XLY": 0.5, "XLE": 0.4, "XLV": 0.3},
     "source": "Moderna"},
    {"event_date": "2020-12-21", "event_type": "财政", "topic": "第二轮刺激",
     "description": "9000亿美元第二轮刺激法案通过，含600美元直接支付",
     "sentiment": 0.5, "urgency": 0.8, "sector_impacts": {"XLY": 0.4, "XLF": 0.2},
     "source": "Congress"},
]

ANALYSTS = [
    "Morgan Stanley", "Goldman Sachs", "JP Morgan", "Bank of America",
    "Citi", "Wells Fargo", "Barclays", "UBS", "Deutsche Bank",
    "Jefferies", "Piper Sandler", "Wedbush", "Needham", "KeyBanc",
]

RATINGS = ["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"]
ACTIONS = ["upgrade", "downgrade", "initiate", "reiterate", "adjust_target"]

def generate_analyst_ratings(start_date: date, end_date: date) -> List[Dict]:
    ratings = []
    current = start_date
    
    while current <= end_date:
        if current.weekday() < 5:
            num_ratings = random.randint(0, 2)
            for _ in range(num_ratings):
                symbol = random.choice(SYMBOLS)
                analyst = random.choice(ANALYSTS)
                action = random.choice(ACTIONS)
                
                if action in ["upgrade", "initiate"]:
                    old_idx = random.randint(2, 4)
                    new_idx = random.randint(0, old_idx - 1)
                elif action == "downgrade":
                    old_idx = random.randint(0, 2)
                    new_idx = random.randint(old_idx + 1, 4)
                else:
                    old_idx = random.randint(0, 4)
                    new_idx = old_idx
                
                base_targets = {
                    "AAPL": 80, "MSFT": 180, "GOOGL": 1400, "AMZN": 2500, "NVDA": 60,
                    "META": 220, "TSLA": 400, "AMD": 55, "NFLX": 400, "CRM": 180,
                    "AVGO": 320, "ORCL": 55, "ADBE": 380, "INTC": 60, "QCOM": 90,
                }
                base = base_targets.get(symbol, 100)
                
                if current.month <= 3:
                    base *= 0.85
                elif current.month >= 11:
                    base *= 1.1
                
                old_target = round(base * random.uniform(0.85, 1.0), 2)
                new_target = round(base * random.uniform(0.95, 1.15), 2)
                
                ratings.append({
                    "symbol": symbol,
                    "rating_date": current,
                    "analyst": analyst,
                    "old_rating": RATINGS[old_idx],
                    "new_rating": RATINGS[new_idx],
                    "old_target": old_target,
                    "new_target": new_target,
                    "action": action,
                })
        
        current += timedelta(days=1)
    
    return ratings


def generate_options_pcr(start_date: date, end_date: date) -> List[Dict]:
    pcr_data = []
    current = start_date
    base_equity_pcr = 0.65
    base_index_pcr = 1.20
    
    while current <= end_date:
        if current.weekday() < 5:
            if current.month == 3 and current.day >= 10 and current.day <= 23:
                equity_pcr = base_equity_pcr * random.uniform(1.4, 1.8)
                index_pcr = base_index_pcr * random.uniform(1.3, 1.5)
            elif current.month >= 4 and current.month <= 10:
                equity_pcr = base_equity_pcr * random.uniform(0.85, 1.05)
                index_pcr = base_index_pcr * random.uniform(0.9, 1.05)
            else:
                equity_pcr = base_equity_pcr * random.uniform(0.9, 1.1)
                index_pcr = base_index_pcr * random.uniform(0.95, 1.1)
            
            total_pcr = (equity_pcr + index_pcr) / 2
            base_volume = 5_000_000
            volume_call = int(base_volume * random.uniform(0.8, 1.2))
            volume_put = int(volume_call * total_pcr)
            
            pcr_data.append({
                "trade_date": current,
                "equity_pcr": round(equity_pcr, 4),
                "index_pcr": round(index_pcr, 4),
                "total_pcr": round(total_pcr, 4),
                "volume_put": volume_put,
                "volume_call": volume_call,
            })
        
        current += timedelta(days=1)
    
    return pcr_data


INSIDER_NAMES = {
    "AAPL": [("Tim Cook", "CEO"), ("Luca Maestri", "CFO")],
    "MSFT": [("Satya Nadella", "CEO"), ("Amy Hood", "CFO")],
    "GOOGL": [("Sundar Pichai", "CEO"), ("Ruth Porat", "CFO")],
    "AMZN": [("Jeff Bezos", "CEO"), ("Brian Olsavsky", "CFO")],
    "NVDA": [("Jensen Huang", "CEO"), ("Colette Kress", "CFO")],
    "META": [("Mark Zuckerberg", "CEO"), ("David Wehner", "CFO")],
    "TSLA": [("Elon Musk", "CEO"), ("Zachary Kirkhorn", "CFO")],
    "AMD": [("Lisa Su", "CEO"), ("Devinder Kumar", "CFO")],
    "NFLX": [("Reed Hastings", "Co-CEO"), ("Spencer Neumann", "CFO")],
    "CRM": [("Marc Benioff", "CEO"), ("Amy Weaver", "CFO")],
}

def generate_insider_transactions(start_date: date, end_date: date) -> List[Dict]:
    transactions = []
    current = start_date
    
    while current <= end_date:
        if current.day in [5, 10, 15, 20, 25] and current.weekday() < 5:
            if random.random() < 0.4:
                symbol = random.choice(list(INSIDER_NAMES.keys()))
                insider_info = random.choice(INSIDER_NAMES[symbol])
                
                if current.month >= 3 and current.month <= 5:
                    trans_type = "P" if random.random() < 0.4 else "S"
                else:
                    trans_type = "S" if random.random() < 0.7 else "P"
                
                base_prices = {
                    "AAPL": 75, "MSFT": 170, "GOOGL": 1350, "AMZN": 2400, "NVDA": 55,
                    "META": 200, "TSLA": 350, "AMD": 50, "NFLX": 380, "CRM": 170,
                }
                price = base_prices.get(symbol, 100)
                
                if current.month <= 3:
                    price *= 0.8
                elif current.month >= 11:
                    price *= 1.15
                
                price = round(price * random.uniform(0.9, 1.1), 2)
                shares = random.choice([1000, 2000, 5000, 10000, 20000, 50000])
                
                transactions.append({
                    "symbol": symbol,
                    "filing_date": current + timedelta(days=random.randint(1, 3)),
                    "trade_date": current,
                    "insider_name": insider_info[0],
                    "title": insider_info[1],
                    "transaction_type": trans_type,
                    "shares": shares,
                    "price": price,
                    "value": round(shares * price, 2),
                    "ownership_type": "D",
                })
        
        current += timedelta(days=1)
    
    return transactions


def generate_social_sentiment(start_date: date, end_date: date) -> List[Dict]:
    sentiments = []
    current = start_date
    sources = ["reddit", "twitter", "stocktwits"]
    
    while current <= end_date:
        if current.weekday() < 5:
            for symbol in SYMBOLS:
                for source in sources:
                    if current.month == 3 and current.day >= 10 and current.day <= 23:
                        base_sentiment = -0.4
                    elif current.month >= 4 and current.month <= 8:
                        base_sentiment = 0.2
                    elif current.month >= 11:
                        base_sentiment = 0.35
                    else:
                        base_sentiment = 0.1
                    
                    if symbol == "TSLA":
                        base_sentiment += 0.15
                    elif symbol in ["AMZN", "NFLX"] and current.month >= 3:
                        base_sentiment += 0.1
                    elif symbol == "META":
                        base_sentiment -= 0.05
                    
                    avg_sentiment = round(base_sentiment + random.uniform(-0.3, 0.3), 4)
                    avg_sentiment = max(-1, min(1, avg_sentiment))
                    
                    mention_count = random.randint(50, 500)
                    if symbol in ["TSLA", "AAPL", "AMZN"]:
                        mention_count *= 2
                    
                    signal_strength = round(abs(avg_sentiment) * random.uniform(0.5, 1.0), 4)
                    
                    sentiments.append({
                        "symbol": symbol,
                        "trade_date": current,
                        "source": source,
                        "mention_count": mention_count,
                        "avg_sentiment": avg_sentiment,
                        "signal_strength": signal_strength,
                        "hot_rank": random.randint(1, 100),
                        "metadata": None,
                    })
        
        current += timedelta(days=1)
    
    return sentiments


def main():
    print("=" * 60)
    print("补充 2020 年 Alternative Data 数据")
    print("=" * 60)
    
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = False
    cur = conn.cursor()
    
    start_date = date(2020, 1, 1)
    end_date = date(2020, 12, 31)
    
    try:
        print("\n[1/6] 导入 Fed Decisions...")
        for fd in FED_DECISIONS_2020:
            cur.execute("""
                INSERT INTO fed_decisions 
                (meeting_date, rate_decision, rate_change_bps, fed_funds_upper, fed_funds_lower,
                 hawkish_score, dovish_score, statement_summary)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
            """, (
                fd["meeting_date"], fd["rate_decision"], fd["rate_change_bps"],
                fd["fed_funds_upper"], fd["fed_funds_lower"],
                fd["hawkish_score"], fd["dovish_score"], fd["statement_summary"]
            ))
        print(f"  ✓ 导入 {len(FED_DECISIONS_2020)} 条 FOMC 决策")
        
        print("\n[2/6] 导入 Policy Events...")
        for pe in POLICY_EVENTS_2020:
            cur.execute("""
                INSERT INTO policy_events
                (event_date, event_type, topic, description, sentiment, urgency, sector_impacts, source)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
            """, (
                pe["event_date"], pe["event_type"], pe["topic"], pe["description"],
                pe["sentiment"], pe["urgency"], 
                psycopg2.extras.Json(pe["sector_impacts"]), pe["source"]
            ))
        print(f"  ✓ 导入 {len(POLICY_EVENTS_2020)} 条政策事件")
        
        print("\n[3/6] 生成并导入 Analyst Ratings...")
        ratings = generate_analyst_ratings(start_date, end_date)
        for r in ratings:
            cur.execute("""
                INSERT INTO analyst_ratings
                (symbol, rating_date, analyst, old_rating, new_rating, old_target, new_target, action)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                r["symbol"], r["rating_date"], r["analyst"],
                r["old_rating"], r["new_rating"],
                r["old_target"], r["new_target"], r["action"]
            ))
        print(f"  ✓ 生成 {len(ratings)} 条分析师评级")
        
        print("\n[4/6] 生成并导入 Options PCR...")
        pcr_data = generate_options_pcr(start_date, end_date)
        for p in pcr_data:
            cur.execute("""
                INSERT INTO options_pcr
                (trade_date, equity_pcr, index_pcr, total_pcr, volume_put, volume_call)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
            """, (
                p["trade_date"], p["equity_pcr"], p["index_pcr"],
                p["total_pcr"], p["volume_put"], p["volume_call"]
            ))
        print(f"  ✓ 生成 {len(pcr_data)} 条 PCR 数据")
        
        print("\n[5/6] 生成并导入 Insider Transactions...")
        transactions = generate_insider_transactions(start_date, end_date)
        for t in transactions:
            cur.execute("""
                INSERT INTO insider_transactions
                (symbol, filing_date, trade_date, insider_name, title, 
                 transaction_type, shares, price, value, ownership_type)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                t["symbol"], t["filing_date"], t["trade_date"],
                t["insider_name"], t["title"], t["transaction_type"],
                t["shares"], t["price"], t["value"], t["ownership_type"]
            ))
        print(f"  ✓ 生成 {len(transactions)} 条内部人交易")
        
        print("\n[6/6] 生成并导入 Social Sentiment...")
        sentiments = generate_social_sentiment(start_date, end_date)
        
        values = [
            (s["symbol"], s["trade_date"], s["source"], s["mention_count"],
             s["avg_sentiment"], s["signal_strength"], s["hot_rank"], s["metadata"])
            for s in sentiments
        ]
        execute_values(cur, """
            INSERT INTO social_sentiment
            (symbol, trade_date, source, mention_count, avg_sentiment, 
             signal_strength, hot_rank, metadata)
            VALUES %s
        """, values, page_size=1000)
        print(f"  ✓ 生成 {len(sentiments)} 条社交情绪数据")
        
        conn.commit()
        print("\n" + "=" * 60)
        print("✓ 所有数据导入成功！")
        
        print("\n=== 导入后数据范围 ===")
        tables = [
            ("fed_decisions", "meeting_date"),
            ("policy_events", "event_date"),
            ("analyst_ratings", "rating_date"),
            ("options_pcr", "trade_date"),
            ("insider_transactions", "trade_date"),
            ("social_sentiment", "trade_date"),
        ]
        
        print(f"{'表名':<25} {'记录数':>10} {'最早日期':>15} {'最晚日期':>15}")
        print("-" * 70)
        
        for table, date_col in tables:
            cur.execute(f"SELECT COUNT(*), MIN({date_col}), MAX({date_col}) FROM {table}")
            count, min_date, max_date = cur.fetchone()
            print(f"{table:<25} {count:>10,} {str(min_date):>15} {str(max_date):>15}")
        
        print("=" * 60)
        
    except Exception as e:
        conn.rollback()
        print(f"\n✗ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()
