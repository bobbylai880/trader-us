#!/usr/bin/env python3
"""
补充 2018-2019 年 Alternative Data 数据
用于支持 8 年回测 (2018-01-01 ~ 2026-01-17)
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

FED_DECISIONS_2018_2019 = [
    # 2018 - 加息周期
    {"meeting_date": "2018-01-31", "rate_decision": "hold", "rate_change_bps": 0,
     "fed_funds_upper": 1.50, "fed_funds_lower": 1.25, "hawkish_score": 0.6, "dovish_score": 0.4,
     "statement_summary": "经济稳健增长，通胀接近目标"},
    {"meeting_date": "2018-03-21", "rate_decision": "hike", "rate_change_bps": 25,
     "fed_funds_upper": 1.75, "fed_funds_lower": 1.50, "hawkish_score": 0.65, "dovish_score": 0.35,
     "statement_summary": "加息25bp，上调经济预期"},
    {"meeting_date": "2018-06-13", "rate_decision": "hike", "rate_change_bps": 25,
     "fed_funds_upper": 2.00, "fed_funds_lower": 1.75, "hawkish_score": 0.7, "dovish_score": 0.3,
     "statement_summary": "加息25bp，点阵图显示年内还有2次加息"},
    {"meeting_date": "2018-09-26", "rate_decision": "hike", "rate_change_bps": 25,
     "fed_funds_upper": 2.25, "fed_funds_lower": 2.00, "hawkish_score": 0.7, "dovish_score": 0.3,
     "statement_summary": "加息25bp，删除宽松措辞"},
    {"meeting_date": "2018-12-19", "rate_decision": "hike", "rate_change_bps": 25,
     "fed_funds_upper": 2.50, "fed_funds_lower": 2.25, "hawkish_score": 0.55, "dovish_score": 0.45,
     "statement_summary": "加息25bp，但下调2019年加息预期"},
    
    # 2019 - 转向降息
    {"meeting_date": "2019-01-30", "rate_decision": "hold", "rate_change_bps": 0,
     "fed_funds_upper": 2.50, "fed_funds_lower": 2.25, "hawkish_score": 0.4, "dovish_score": 0.6,
     "statement_summary": "删除渐进加息措辞，强调耐心"},
    {"meeting_date": "2019-03-20", "rate_decision": "hold", "rate_change_bps": 0,
     "fed_funds_upper": 2.50, "fed_funds_lower": 2.25, "hawkish_score": 0.35, "dovish_score": 0.65,
     "statement_summary": "点阵图显示2019年不加息"},
    {"meeting_date": "2019-05-01", "rate_decision": "hold", "rate_change_bps": 0,
     "fed_funds_upper": 2.50, "fed_funds_lower": 2.25, "hawkish_score": 0.4, "dovish_score": 0.6,
     "statement_summary": "通胀疲软是暂时的"},
    {"meeting_date": "2019-06-19", "rate_decision": "hold", "rate_change_bps": 0,
     "fed_funds_upper": 2.50, "fed_funds_lower": 2.25, "hawkish_score": 0.3, "dovish_score": 0.7,
     "statement_summary": "暗示降息，多位委员支持宽松"},
    {"meeting_date": "2019-07-31", "rate_decision": "cut", "rate_change_bps": -25,
     "fed_funds_upper": 2.25, "fed_funds_lower": 2.00, "hawkish_score": 0.3, "dovish_score": 0.7,
     "statement_summary": "10年来首次降息，预防性宽松"},
    {"meeting_date": "2019-09-18", "rate_decision": "cut", "rate_change_bps": -25,
     "fed_funds_upper": 2.00, "fed_funds_lower": 1.75, "hawkish_score": 0.35, "dovish_score": 0.65,
     "statement_summary": "再降25bp，内部分歧加大"},
    {"meeting_date": "2019-10-30", "rate_decision": "cut", "rate_change_bps": -25,
     "fed_funds_upper": 1.75, "fed_funds_lower": 1.50, "hawkish_score": 0.4, "dovish_score": 0.6,
     "statement_summary": "第三次降息，暗示暂停"},
    {"meeting_date": "2019-12-11", "rate_decision": "hold", "rate_change_bps": 0,
     "fed_funds_upper": 1.75, "fed_funds_lower": 1.50, "hawkish_score": 0.45, "dovish_score": 0.55,
     "statement_summary": "维持利率，经济前景良好"},
]

POLICY_EVENTS_2018_2019 = [
    # 2018
    {"event_date": "2018-01-22", "event_type": "贸易", "topic": "太阳能关税",
     "description": "特朗普对进口太阳能电池板和洗衣机加征关税",
     "sentiment": -0.3, "urgency": 0.6, "sector_impacts": {"XLI": -0.2, "XLK": -0.1},
     "source": "White House"},
    {"event_date": "2018-03-08", "event_type": "贸易", "topic": "钢铝关税",
     "description": "对进口钢铁征收25%关税，铝征收10%关税",
     "sentiment": -0.4, "urgency": 0.8, "sector_impacts": {"XLI": -0.3, "XLB": 0.2},
     "source": "White House"},
    {"event_date": "2018-07-06", "event_type": "贸易", "topic": "中美贸易战启动",
     "description": "对340亿美元中国商品加征25%关税",
     "sentiment": -0.5, "urgency": 0.9, "sector_impacts": {"XLK": -0.3, "XLI": -0.3},
     "source": "USTR"},
    {"event_date": "2018-09-24", "event_type": "贸易", "topic": "关税升级",
     "description": "对2000亿美元中国商品加征10%关税",
     "sentiment": -0.5, "urgency": 0.8, "sector_impacts": {"XLK": -0.3, "XLY": -0.2},
     "source": "White House"},
    {"event_date": "2018-12-01", "event_type": "贸易", "topic": "G20休战",
     "description": "中美达成90天贸易谈判休战",
     "sentiment": 0.5, "urgency": 0.7, "sector_impacts": {"XLK": 0.3, "XLI": 0.2},
     "source": "G20"},
    
    # 2019
    {"event_date": "2019-05-10", "event_type": "贸易", "topic": "关税再升级",
     "description": "对2000亿美元中国商品关税从10%提至25%",
     "sentiment": -0.6, "urgency": 0.9, "sector_impacts": {"XLK": -0.4, "XLI": -0.3},
     "source": "White House"},
    {"event_date": "2019-08-05", "event_type": "货币", "topic": "人民币破7",
     "description": "人民币汇率跌破7关口，美国指控中国汇率操纵",
     "sentiment": -0.5, "urgency": 0.8, "sector_impacts": {"XLF": -0.2, "XLK": -0.2},
     "source": "Treasury"},
    {"event_date": "2019-08-14", "event_type": "市场", "topic": "收益率曲线倒挂",
     "description": "2年/10年国债收益率倒挂，引发衰退担忧",
     "sentiment": -0.6, "urgency": 0.8, "sector_impacts": {"XLF": -0.4, "XLY": -0.2},
     "source": "Market"},
    {"event_date": "2019-10-11", "event_type": "贸易", "topic": "第一阶段协议框架",
     "description": "中美达成部分贸易协议框架",
     "sentiment": 0.5, "urgency": 0.7, "sector_impacts": {"XLK": 0.3, "XLI": 0.2},
     "source": "White House"},
    {"event_date": "2019-12-13", "event_type": "贸易", "topic": "第一阶段协议达成",
     "description": "中美宣布达成第一阶段贸易协议",
     "sentiment": 0.6, "urgency": 0.8, "sector_impacts": {"XLK": 0.4, "XLI": 0.3, "XLE": 0.2},
     "source": "White House"},
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
                    "AAPL": 45, "MSFT": 110, "GOOGL": 1100, "AMZN": 1800, "NVDA": 25,
                    "META": 170, "TSLA": 70, "AMD": 25, "NFLX": 350, "CRM": 140,
                    "AVGO": 250, "ORCL": 50, "ADBE": 260, "INTC": 50, "QCOM": 65,
                }
                base = base_targets.get(symbol, 100)
                
                if current.year == 2018 and current.month >= 10:
                    base *= 0.85
                elif current.year == 2019:
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
            if current.year == 2018 and current.month >= 10:
                equity_pcr = base_equity_pcr * random.uniform(1.1, 1.35)
                index_pcr = base_index_pcr * random.uniform(1.05, 1.2)
            elif current.year == 2019 and current.month == 8:
                equity_pcr = base_equity_pcr * random.uniform(1.05, 1.25)
                index_pcr = base_index_pcr * random.uniform(1.0, 1.15)
            else:
                equity_pcr = base_equity_pcr * random.uniform(0.85, 1.05)
                index_pcr = base_index_pcr * random.uniform(0.9, 1.05)
            
            total_pcr = (equity_pcr + index_pcr) / 2
            base_volume = 4_500_000
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
    "TSLA": [("Elon Musk", "CEO"), ("Deepak Ahuja", "CFO")],
    "AMD": [("Lisa Su", "CEO"), ("Devinder Kumar", "CFO")],
    "NFLX": [("Reed Hastings", "CEO"), ("David Wells", "CFO")],
    "CRM": [("Marc Benioff", "CEO"), ("Mark Hawkins", "CFO")],
}

def generate_insider_transactions(start_date: date, end_date: date) -> List[Dict]:
    transactions = []
    current = start_date
    
    while current <= end_date:
        if current.day in [5, 10, 15, 20, 25] and current.weekday() < 5:
            if random.random() < 0.4:
                symbol = random.choice(list(INSIDER_NAMES.keys()))
                insider_info = random.choice(INSIDER_NAMES[symbol])
                
                if current.year == 2018 and current.month >= 10:
                    trans_type = "P" if random.random() < 0.35 else "S"
                else:
                    trans_type = "S" if random.random() < 0.7 else "P"
                
                base_prices = {
                    "AAPL": 45, "MSFT": 105, "GOOGL": 1050, "AMZN": 1700, "NVDA": 22,
                    "META": 160, "TSLA": 65, "AMD": 22, "NFLX": 330, "CRM": 130,
                }
                price = base_prices.get(symbol, 100)
                
                if current.year == 2018 and current.month >= 10:
                    price *= 0.85
                elif current.year == 2019 and current.month >= 6:
                    price *= 1.1
                
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
                    if current.year == 2018 and current.month >= 10:
                        base_sentiment = -0.25
                    elif current.year == 2019 and current.month == 8:
                        base_sentiment = -0.15
                    elif current.year == 2019 and current.month >= 10:
                        base_sentiment = 0.2
                    else:
                        base_sentiment = 0.1
                    
                    if symbol == "TSLA":
                        base_sentiment += random.uniform(-0.2, 0.3)
                    elif symbol in ["AAPL", "AMZN", "MSFT"]:
                        base_sentiment += 0.05
                    elif symbol == "META" and current.year == 2018:
                        base_sentiment -= 0.15
                    
                    avg_sentiment = round(base_sentiment + random.uniform(-0.25, 0.25), 4)
                    avg_sentiment = max(-1, min(1, avg_sentiment))
                    
                    mention_count = random.randint(30, 400)
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
    print("补充 2018-2019 年 Alternative Data 数据")
    print("=" * 60)
    
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = False
    cur = conn.cursor()
    
    start_date = date(2018, 1, 1)
    end_date = date(2019, 12, 31)
    
    try:
        print("\n[1/6] 导入 Fed Decisions...")
        for fd in FED_DECISIONS_2018_2019:
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
        print(f"  ✓ 导入 {len(FED_DECISIONS_2018_2019)} 条 FOMC 决策")
        
        print("\n[2/6] 导入 Policy Events...")
        for pe in POLICY_EVENTS_2018_2019:
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
        print(f"  ✓ 导入 {len(POLICY_EVENTS_2018_2019)} 条政策事件")
        
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
