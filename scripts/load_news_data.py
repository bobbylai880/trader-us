#!/usr/bin/env python3
"""
新闻数据导入脚本 - 从 Yahoo Finance 获取新闻并存储到 PostgreSQL

功能:
1. 遍历数据库中所有股票/ETF
2. 调用 yfinance 获取近期新闻
3. 计算情绪分数
4. 存储到 PostgreSQL news 表 (去重)

使用方法:
    PYTHONPATH=. python scripts/load_news_data.py [--days 30] [--symbols AAPL,NVDA]
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import psycopg2
from psycopg2.extras import execute_values

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import yfinance as yf
except ImportError:
    print("❌ 请安装 yfinance: pip install yfinance")
    sys.exit(1)


# 情绪分析关键词
POSITIVE_KEYWORDS = [
    "beat", "beats", "exceeded", "surge", "surges", "soar", "soars", "jump", "jumps",
    "gain", "gains", "rally", "rallies", "bullish", "upgrade", "upgrades", "outperform",
    "buy", "strong", "growth", "profit", "record", "high", "positive", "optimistic",
    "breakthrough", "innovation", "success", "boost", "boosts", "expand", "expansion",
    "recover", "recovery", "improve", "improvement", "beat expectations", "all-time high",
    "upside", "momentum", "accelerate", "accelerates", "opportunity", "opportunities",
]

NEGATIVE_KEYWORDS = [
    "miss", "misses", "missed", "fall", "falls", "drop", "drops", "plunge", "plunges",
    "decline", "declines", "crash", "crashes", "bearish", "downgrade", "downgrades",
    "underperform", "sell", "weak", "loss", "losses", "concern", "concerns", "risk",
    "risks", "warning", "warns", "cut", "cuts", "layoff", "layoffs", "recession",
    "slowdown", "slowing", "negative", "pessimistic", "fear", "fears", "trouble",
    "problem", "problems", "fail", "fails", "failure", "lawsuit", "investigation",
    "downside", "pressure", "pressures", "struggle", "struggles", "uncertainty",
]


def calculate_sentiment(title: str, summary: str = "") -> float:
    """计算情绪分数 (-1 到 1)"""
    text = f"{title} {summary}".lower()
    
    positive_count = sum(1 for word in POSITIVE_KEYWORDS if word in text)
    negative_count = sum(1 for word in NEGATIVE_KEYWORDS if word in text)
    
    total = positive_count + negative_count
    if total == 0:
        return 0.0
    
    score = (positive_count - negative_count) / total
    return round(score, 3)


def get_db_connection():
    """获取数据库连接"""
    return psycopg2.connect(
        host=os.getenv("PG_HOST", "192.168.10.11"),
        port=os.getenv("PG_PORT", "5432"),
        database=os.getenv("PG_DATABASE", "trader"),
        user=os.getenv("PG_USER", "trader"),
        password=os.getenv("PG_PASSWORD", "")
    )


def get_all_symbols(conn) -> List[str]:
    """从数据库获取所有股票代码"""
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT symbol FROM daily_prices ORDER BY symbol")
        return [row[0] for row in cur.fetchall()]


def fetch_news_from_yf(symbol: str, lookback_days: int = 30) -> List[Dict]:
    """从 Yahoo Finance 获取新闻"""
    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news
        
        if not news:
            return []
        
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        articles = []
        
        for item in news:
            # 解析发布时间
            pub_time = item.get("providerPublishTime")
            if isinstance(pub_time, (int, float)):
                try:
                    published_at = datetime.fromtimestamp(pub_time, tz=timezone.utc)
                except (OverflowError, OSError, ValueError):
                    published_at = datetime.now(timezone.utc)
            else:
                published_at = datetime.now(timezone.utc)
            
            # 过滤旧新闻
            if published_at < cutoff:
                continue
            
            # 提取内容
            content = item.get("content", {}) if isinstance(item.get("content"), dict) else {}
            
            title = item.get("title") or content.get("title") or ""
            summary = item.get("summary") or content.get("summary") or content.get("description") or ""
            publisher = item.get("publisher") or ""
            if not publisher and isinstance(item.get("provider"), dict):
                publisher = item.get("provider", {}).get("displayName", "")
            
            link = item.get("link") or ""
            if not link:
                canonical = item.get("canonicalUrl")
                if isinstance(canonical, dict):
                    link = canonical.get("url", "")
                elif isinstance(canonical, str):
                    link = canonical
            
            if not title:
                continue
            
            # 计算情绪分数
            sentiment = calculate_sentiment(title, summary)
            
            articles.append({
                "symbol": symbol,
                "title": title[:500] if title else None,
                "summary": summary[:2000] if summary else None,
                "content": None,  # 不获取全文以节省时间
                "publisher": publisher[:100] if publisher else None,
                "url": link[:500] if link else None,
                "published_at": published_at,
                "sentiment_score": sentiment,
            })
        
        return articles
    
    except Exception as e:
        print(f"  ⚠️ {symbol} 获取新闻失败: {e}")
        return []


def save_news_batch(conn, articles: List[Dict]) -> int:
    """批量保存新闻到数据库"""
    if not articles:
        return 0
    
    query = """
        INSERT INTO news (symbol, title, summary, content, publisher, url, published_at, sentiment_score)
        VALUES %s
        ON CONFLICT (url) WHERE url IS NOT NULL DO NOTHING
    """
    
    values = [
        (
            a["symbol"],
            a["title"],
            a["summary"],
            a["content"],
            a["publisher"],
            a["url"],
            a["published_at"],
            a["sentiment_score"],
        )
        for a in articles
    ]
    
    with conn.cursor() as cur:
        execute_values(cur, query, values)
        inserted = cur.rowcount
    
    conn.commit()
    return inserted


def main():
    parser = argparse.ArgumentParser(description="导入新闻数据到 PostgreSQL")
    parser.add_argument("--days", type=int, default=30, help="获取最近N天的新闻 (默认: 30)")
    parser.add_argument("--symbols", type=str, help="指定股票代码，逗号分隔 (默认: 数据库中所有股票)")
    parser.add_argument("--batch-size", type=int, default=10, help="每批处理的股票数量 (默认: 10)")
    parser.add_argument("--delay", type=float, default=0.5, help="每只股票之间的延迟秒数 (默认: 0.5)")
    args = parser.parse_args()
    
    print("=" * 70)
    print("新闻数据导入脚本")
    print("=" * 70)
    
    # 连接数据库
    print("\n【1. 连接数据库】")
    try:
        conn = get_db_connection()
        print("  ✅ 数据库连接成功")
    except Exception as e:
        print(f"  ❌ 数据库连接失败: {e}")
        sys.exit(1)
    
    # 获取股票列表
    print("\n【2. 获取股票列表】")
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
        print(f"  指定股票: {len(symbols)} 只")
    else:
        symbols = get_all_symbols(conn)
        print(f"  数据库股票: {len(symbols)} 只")
    
    # 检查现有新闻数量
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM news")
        existing_count = cur.fetchone()[0]
    print(f"  现有新闻: {existing_count} 条")
    
    # 获取新闻
    print(f"\n【3. 获取最近 {args.days} 天新闻】")
    total_articles = 0
    total_inserted = 0
    failed_symbols = []
    
    for i, symbol in enumerate(symbols):
        progress = f"[{i+1}/{len(symbols)}]"
        
        # 获取新闻
        articles = fetch_news_from_yf(symbol, args.days)
        
        if articles:
            # 保存到数据库
            inserted = save_news_batch(conn, articles)
            total_articles += len(articles)
            total_inserted += inserted
            print(f"  {progress} {symbol}: {len(articles)} 条新闻, 新增 {inserted} 条")
        else:
            print(f"  {progress} {symbol}: 无新闻")
        
        # 延迟以避免 API 限流
        if i < len(symbols) - 1:
            time.sleep(args.delay)
    
    # 统计结果
    print("\n【4. 导入结果】")
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM news")
        final_count = cur.fetchone()[0]
        
        cur.execute("""
            SELECT COUNT(DISTINCT symbol), 
                   MIN(published_at), 
                   MAX(published_at),
                   ROUND(AVG(sentiment_score)::numeric, 3)
            FROM news
        """)
        stats = cur.fetchone()
    
    print(f"  获取新闻总数: {total_articles} 条")
    print(f"  新增新闻: {total_inserted} 条")
    print(f"  数据库新闻总数: {final_count} 条")
    print(f"  覆盖股票数: {stats[0]} 只")
    print(f"  日期范围: {stats[1]} ~ {stats[2]}")
    print(f"  平均情绪分数: {stats[3]}")
    
    # 显示情绪分布
    print("\n【5. 情绪分布】")
    with conn.cursor() as cur:
        cur.execute("""
            SELECT 
                CASE 
                    WHEN sentiment_score > 0.3 THEN '强正面 (>0.3)'
                    WHEN sentiment_score > 0 THEN '正面 (0~0.3)'
                    WHEN sentiment_score = 0 THEN '中性 (0)'
                    WHEN sentiment_score > -0.3 THEN '负面 (-0.3~0)'
                    ELSE '强负面 (<-0.3)'
                END as sentiment_category,
                COUNT(*) as count
            FROM news
            GROUP BY 1
            ORDER BY 1
        """)
        for row in cur.fetchall():
            print(f"  {row[0]}: {row[1]} 条")
    
    # 显示最新新闻
    print("\n【6. 最新5条新闻】")
    with conn.cursor() as cur:
        cur.execute("""
            SELECT symbol, title, publisher, published_at, sentiment_score
            FROM news
            ORDER BY published_at DESC
            LIMIT 5
        """)
        for row in cur.fetchall():
            sentiment_icon = "🟢" if row[4] > 0 else ("🔴" if row[4] < 0 else "⚪")
            title_short = row[1][:50] + "..." if len(row[1]) > 50 else row[1]
            print(f"  {sentiment_icon} [{row[0]}] {title_short}")
            print(f"      {row[2]} | {row[3]} | 情绪: {row[4]}")
    
    conn.close()
    print("\n" + "=" * 70)
    print("✅ 新闻数据导入完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
