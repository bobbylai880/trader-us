"""PostgreSQL 18 Market Database Client.

高性能市场数据库客户端，利用 PG 18 新特性：
- UUIDv7 时间有序主键
- VIRTUAL 生成列
- AIO 异步 I/O
- Skip Scan 优化
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, Generator, List, Optional, Tuple

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values, RealDictCursor


@dataclass
class PGConfig:
    """PostgreSQL 连接配置"""
    host: str = "localhost"
    port: int = 5432
    database: str = "trader"
    user: str = "trader"
    password: str = ""
    connect_timeout: int = 10
    
    @classmethod
    def from_env(cls) -> "PGConfig":
        """从环境变量读取配置"""
        return cls(
            host=os.getenv("PG_HOST", "localhost"),
            port=int(os.getenv("PG_PORT", "5432")),
            database=os.getenv("PG_DATABASE", "trader"),
            user=os.getenv("PG_USER", "trader"),
            password=os.getenv("PG_PASSWORD", ""),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "user": self.user,
            "password": self.password,
            "connect_timeout": self.connect_timeout,
        }


class PostgresMarketDB:
    """PostgreSQL 18 市场数据库客户端
    
    提供高效的市场数据存取功能，支持：
    - 日线数据的批量读写
    - 技术指标缓存
    - 智能增量更新
    - 动态选股查询
    """
    
    def __init__(self, config: Optional[PGConfig] = None):
        self.config = config or PGConfig()
        self._conn: Optional[psycopg2.extensions.connection] = None
        self.stats = {
            "queries": 0,
            "inserts": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }
    
    @contextmanager
    def get_conn(self) -> Generator[psycopg2.extensions.connection, None, None]:
        """获取数据库连接（上下文管理器）"""
        conn = psycopg2.connect(**self.config.to_dict())
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    @contextmanager
    def get_cursor(self, dict_cursor: bool = False):
        """获取游标（上下文管理器）"""
        with self.get_conn() as conn:
            cursor_factory = RealDictCursor if dict_cursor else None
            cur = conn.cursor(cursor_factory=cursor_factory)
            try:
                yield cur
            finally:
                cur.close()
    
    # ========================================
    # 日线数据操作
    # ========================================
    
    def get_daily_prices(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """批量获取日线数据
        
        利用 PG 18 的 Skip Scan 和 AIO 特性优化多股票查询
        """
        if not symbols:
            return pd.DataFrame()
        
        query = """
            SELECT symbol, trade_date, open, high, low, close, adj_close, volume,
                   daily_range, daily_return, is_green
            FROM daily_prices
            WHERE symbol = ANY(%s)
              AND trade_date BETWEEN %s AND %s
            ORDER BY symbol, trade_date
        """
        self.stats["queries"] += 1
        
        with self.get_conn() as conn:
            df = pd.read_sql(query, conn, params=(symbols, start_date, end_date))
        
        if not df.empty:
            df["trade_date"] = pd.to_datetime(df["trade_date"])
        
        return df
    
    def get_daily_prices_single(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """获取单只股票的日线数据"""
        return self.get_daily_prices([symbol], start_date, end_date)
    
    def upsert_daily_prices(self, df: pd.DataFrame) -> int:
        """批量插入/更新日线数据
        
        使用 ON CONFLICT 实现 upsert
        """
        if df.empty:
            return 0
        
        # 确保列名正确
        required_cols = ['symbol', 'trade_date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                if col == 'adj_close' and 'Adj Close' in df.columns:
                    df['adj_close'] = df['Adj Close']
                elif col == 'trade_date' and df.index.name == 'Date':
                    df = df.reset_index()
                    df['trade_date'] = df['Date']
                else:
                    raise ValueError(f"Missing required column: {col}")
        
        # 准备数据
        records = []
        for idx in range(len(df)):
            row = df.iloc[idx]
            
            # 处理日期
            trade_date = row['trade_date']
            if hasattr(trade_date, 'date'):
                trade_date = trade_date.date()
            elif hasattr(trade_date, 'item'):
                trade_date = trade_date.item()
                if hasattr(trade_date, 'date'):
                    trade_date = trade_date.date()
            
            # 安全地提取标量值
            def safe_float(val):
                if val is None:
                    return None
                if hasattr(val, 'item'):
                    val = val.item()
                if pd.isna(val):
                    return None
                return float(val)
            
            def safe_int(val):
                if val is None:
                    return None
                if hasattr(val, 'item'):
                    val = val.item()
                if pd.isna(val):
                    return None
                return int(val)
            
            def safe_str(val):
                if hasattr(val, 'item'):
                    val = val.item()
                return str(val) if val is not None else None
            
            records.append((
                safe_str(row['symbol']),
                trade_date,
                safe_float(row['open']),
                safe_float(row['high']),
                safe_float(row['low']),
                safe_float(row['close']),
                safe_float(row['adj_close']),
                safe_int(row['volume']),
            ))
        
        query = """
            INSERT INTO daily_prices (symbol, trade_date, open, high, low, close, adj_close, volume)
            VALUES %s
            ON CONFLICT (symbol, trade_date) 
            DO UPDATE SET 
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                adj_close = EXCLUDED.adj_close,
                volume = EXCLUDED.volume,
                updated_at = CURRENT_TIMESTAMP
        """
        
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                execute_values(cur, query, records)
                row_count = cur.rowcount
        
        self.stats["inserts"] += row_count
        return row_count
    
    def get_missing_dates(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> List[date]:
        """找出缺失的交易日
        
        通过与 SPY 的交易日对比，找出需要补充的数据
        """
        query = """
            WITH trading_days AS (
                SELECT DISTINCT trade_date 
                FROM daily_prices 
                WHERE symbol = 'SPY' 
                  AND trade_date BETWEEN %s AND %s
            )
            SELECT td.trade_date
            FROM trading_days td
            LEFT JOIN daily_prices dp 
                ON dp.symbol = %s AND dp.trade_date = td.trade_date
            WHERE dp.symbol IS NULL
            ORDER BY td.trade_date
        """
        
        with self.get_cursor() as cur:
            cur.execute(query, (start_date, end_date, symbol))
            return [row[0] for row in cur.fetchall()]
    
    def get_latest_date(self, symbol: str = "SPY") -> Optional[date]:
        """获取最新交易日"""
        query = "SELECT MAX(trade_date) FROM daily_prices WHERE symbol = %s"
        
        with self.get_cursor() as cur:
            cur.execute(query, (symbol,))
            result = cur.fetchone()
            return result[0] if result and result[0] else None
    
    def get_price_count(self, symbol: Optional[str] = None) -> int:
        """获取数据行数"""
        if symbol:
            query = "SELECT COUNT(*) FROM daily_prices WHERE symbol = %s"
            params: tuple = (symbol,)
        else:
            query = "SELECT COUNT(*) FROM daily_prices"
            params = ()
        
        with self.get_cursor() as cur:
            cur.execute(query, params)
            result = cur.fetchone()
            return result[0] if result else 0
    
    # ========================================
    # 技术指标操作
    # ========================================
    
    def get_indicators(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """获取技术指标缓存"""
        if not symbols:
            return pd.DataFrame()
        
        query = """
            SELECT *
            FROM indicators
            WHERE symbol = ANY(%s)
              AND trade_date BETWEEN %s AND %s
            ORDER BY symbol, trade_date
        """
        
        with self.get_conn() as conn:
            df = pd.read_sql(query, conn, params=(symbols, start_date, end_date))
        
        self.stats["queries"] += 1
        return df
    
    def upsert_indicators(self, df: pd.DataFrame) -> int:
        """批量插入/更新技术指标"""
        if df.empty:
            return 0
        
        # 指标列映射
        indicator_cols = [
            'sma_20', 'sma_50', 'sma_200', 'ema_12', 'ema_26',
            'rsi_14', 'macd', 'macd_signal', 'macd_hist',
            'atr_14', 'bb_upper', 'bb_middle', 'bb_lower',
            'volume_sma_20', 'volume_ratio', 'trend_slope_20', 'momentum_10d'
        ]
        
        records = []
        for _, row in df.iterrows():
            trade_date = row['trade_date']
            if isinstance(trade_date, pd.Timestamp):
                trade_date = trade_date.date()
            
            record = [row['symbol'], trade_date]
            for col in indicator_cols:
                val = row.get(col)
                if pd.notna(val):
                    record.append(float(val) if col != 'volume_sma_20' else int(val))
                else:
                    record.append(None)
            records.append(tuple(record))
        
        cols = ', '.join(['symbol', 'trade_date'] + indicator_cols)
        placeholders = ', '.join(['%s'] * (2 + len(indicator_cols)))
        
        query = f"""
            INSERT INTO indicators ({cols})
            VALUES %s
            ON CONFLICT (symbol, trade_date) 
            DO UPDATE SET 
                {', '.join(f'{col} = EXCLUDED.{col}' for col in indicator_cols)},
                created_at = CURRENT_TIMESTAMP
        """
        
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                execute_values(cur, query, records)
                return cur.rowcount
    
    def has_indicators(self, symbol: str, trade_date: date) -> bool:
        """检查是否已有指标缓存"""
        query = """
            SELECT 1 FROM indicators 
            WHERE symbol = %s AND trade_date = %s
            LIMIT 1
        """
        with self.get_cursor() as cur:
            cur.execute(query, (symbol, trade_date))
            return cur.fetchone() is not None
    
    # ========================================
    # 动态选股
    # ========================================
    
    def get_stock_candidates(
        self,
        min_volume_ratio: float = 0.5,
        min_momentum_rank: float = 0.5,
        trend_state: Optional[str] = None,
        limit: int = 20,
    ) -> pd.DataFrame:
        """获取动态选股候选池
        
        利用预创建的视图 v_stock_candidates
        """
        conditions = ["volume_ratio >= %s", "momentum_rank >= %s"]
        params: List[Any] = [min_volume_ratio, min_momentum_rank]
        
        if trend_state:
            conditions.append("trend_state = %s")
            params.append(trend_state)
        
        query = f"""
            SELECT *
            FROM v_stock_candidates
            WHERE {' AND '.join(conditions)}
            ORDER BY momentum_rank DESC
            LIMIT %s
        """
        params.append(limit)
        
        with self.get_conn() as conn:
            return pd.read_sql(query, conn, params=params)
    
    def get_sector_leaders(
        self,
        top_n_sectors: int = 3,
        top_n_per_sector: int = 3,
    ) -> pd.DataFrame:
        """获取板块领先股票
        
        先找领先板块，再从每个板块选择最强个股
        """
        query = """
            WITH sector_momentum AS (
                SELECT 
                    sh.sector_etf,
                    AVG(i.momentum_10d) AS avg_momentum
                FROM sector_holdings sh
                JOIN indicators i ON sh.symbol = i.symbol
                WHERE i.trade_date = (SELECT MAX(trade_date) FROM indicators)
                GROUP BY sh.sector_etf
                ORDER BY avg_momentum DESC
                LIMIT %s
            ),
            ranked_stocks AS (
                SELECT 
                    sh.sector_etf,
                    sh.symbol,
                    dp.close,
                    i.rsi_14,
                    i.momentum_10d,
                    i.volume_ratio,
                    ROW_NUMBER() OVER (
                        PARTITION BY sh.sector_etf 
                        ORDER BY i.momentum_10d DESC NULLS LAST
                    ) AS rank_in_sector
                FROM sector_momentum sm
                JOIN sector_holdings sh ON sm.sector_etf = sh.sector_etf
                JOIN daily_prices dp ON sh.symbol = dp.symbol
                JOIN indicators i ON sh.symbol = i.symbol AND dp.trade_date = i.trade_date
                WHERE dp.trade_date = (SELECT MAX(trade_date) FROM daily_prices)
            )
            SELECT *
            FROM ranked_stocks
            WHERE rank_in_sector <= %s
            ORDER BY sector_etf, rank_in_sector
        """
        
        with self.get_conn() as conn:
            return pd.read_sql(query, conn, params=(top_n_sectors, top_n_per_sector))
    
    # ========================================
    # 持仓操作
    # ========================================
    
    def get_positions(self) -> pd.DataFrame:
        """获取当前持仓"""
        query = "SELECT * FROM positions WHERE shares > 0"
        with self.get_conn() as conn:
            return pd.read_sql(query, conn)
    
    def update_position(
        self,
        symbol: str,
        shares: int,
        avg_cost: float,
    ) -> None:
        """更新持仓（使用 PG 18 的 RETURNING OLD/NEW 记录历史）"""
        query = """
            INSERT INTO positions (symbol, shares, avg_cost)
            VALUES (%s, %s, %s)
            ON CONFLICT (symbol) 
            DO UPDATE SET 
                shares = EXCLUDED.shares,
                avg_cost = EXCLUDED.avg_cost,
                updated_at = CURRENT_TIMESTAMP
        """
        with self.get_cursor() as cur:
            cur.execute(query, (symbol, shares, avg_cost))
    
    def record_trade(
        self,
        symbol: str,
        action: str,
        shares: int,
        price: float,
        reason: Optional[str] = None,
        trade_date: Optional[date] = None,
    ) -> None:
        """记录交易历史"""
        query = """
            INSERT INTO position_history (symbol, action, shares, price, reason, trade_date)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        with self.get_cursor() as cur:
            cur.execute(query, (
                symbol, action, shares, price, reason,
                trade_date or date.today()
            ))
    
    # ========================================
    # 回测记录
    # ========================================
    
    def save_backtest_run(
        self,
        name: str,
        strategy: str,
        symbols: List[str],
        start_date: date,
        end_date: date,
        initial_capital: float,
        final_capital: float,
        total_return: float,
        max_drawdown: float,
        sharpe_ratio: float,
        win_rate: float,
        total_trades: int,
        config: Optional[Dict] = None,
        monthly_returns: Optional[Dict] = None,
    ) -> str:
        """保存回测运行记录，返回 run_id"""
        import json
        
        query = """
            INSERT INTO backtest_runs (
                name, strategy, symbols, start_date, end_date,
                initial_capital, final_capital, total_return, max_drawdown,
                sharpe_ratio, win_rate, total_trades, config, monthly_returns
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """
        
        with self.get_cursor() as cur:
            cur.execute(query, (
                name, strategy, symbols, start_date, end_date,
                initial_capital, final_capital, total_return, max_drawdown,
                sharpe_ratio, win_rate, total_trades,
                json.dumps(config) if config else None,
                json.dumps(monthly_returns) if monthly_returns else None,
            ))
            result = cur.fetchone()
            return str(result[0]) if result else ""
    
    def save_backtest_trades(
        self,
        run_id: str,
        trades: List[Dict],
    ) -> int:
        """保存回测交易记录"""
        if not trades:
            return 0
        
        records = [
            (
                run_id,
                t['symbol'],
                t['trade_date'],
                t['action'],
                t['shares'],
                t['price'],
                t.get('reason'),
                t.get('pnl'),
                t.get('pnl_pct'),
            )
            for t in trades
        ]
        
        query = """
            INSERT INTO backtest_trades 
                (run_id, symbol, trade_date, action, shares, price, reason, pnl, pnl_pct)
            VALUES %s
        """
        
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                execute_values(cur, query, records)
                return cur.rowcount
    
    # ========================================
    # 新闻和 LLM 分析
    # ========================================
    
    def save_news(
        self,
        symbol: str,
        title: str,
        summary: Optional[str] = None,
        content: Optional[str] = None,
        publisher: Optional[str] = None,
        url: Optional[str] = None,
        published_at: Optional[datetime] = None,
        sentiment_score: Optional[float] = None,
    ) -> None:
        """保存新闻"""
        query = """
            INSERT INTO news (symbol, title, summary, content, publisher, url, published_at, sentiment_score)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (url) WHERE url IS NOT NULL DO NOTHING
        """
        with self.get_cursor() as cur:
            cur.execute(query, (
                symbol, title, summary, content, publisher, url, published_at, sentiment_score
            ))
    
    def search_news(
        self,
        query_text: str,
        symbol: Optional[str] = None,
        limit: int = 20,
    ) -> pd.DataFrame:
        """全文搜索新闻"""
        conditions = ["search_vector @@ plainto_tsquery('english', %s)"]
        params: List[Any] = [query_text]
        
        if symbol:
            conditions.append("symbol = %s")
            params.append(symbol)
        
        query = f"""
            SELECT symbol, title, summary, publisher, published_at, sentiment_score,
                   ts_rank(search_vector, plainto_tsquery('english', %s)) AS relevance
            FROM news
            WHERE {' AND '.join(conditions)}
            ORDER BY relevance DESC, published_at DESC
            LIMIT %s
        """
        params.insert(0, query_text)  # For ts_rank
        params.append(limit)
        
        with self.get_conn() as conn:
            return pd.read_sql(query, conn, params=params)
    
    def save_llm_analysis(
        self,
        trade_date: date,
        stage: str,
        regime: Optional[str],
        analysis: Dict,
    ) -> None:
        """保存 LLM 分析结果"""
        import json
        
        query = """
            INSERT INTO llm_analysis (trade_date, stage, regime, analysis)
            VALUES (%s, %s, %s, %s)
        """
        with self.get_cursor() as cur:
            cur.execute(query, (trade_date, stage, regime, json.dumps(analysis)))
    
    # ========================================
    # 统计和维护
    # ========================================
    
    def get_stats(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        stats = {}
        
        with self.get_cursor() as cur:
            # 表行数统计
            tables = [
                'daily_prices', 'indicators', 'stock_meta', 'sector_holdings',
                'positions', 'position_history', 'news', 'backtest_runs'
            ]
            for table in tables:
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                result = cur.fetchone()
                stats[f"{table}_count"] = result[0] if result else 0
            
            # 日期范围
            cur.execute("SELECT MIN(trade_date), MAX(trade_date) FROM daily_prices")
            result = cur.fetchone()
            if result:
                stats["price_date_range"] = {
                    "min": str(result[0]) if result[0] else None,
                    "max": str(result[1]) if result[1] else None,
                }
            
            # 股票数量
            cur.execute("SELECT COUNT(DISTINCT symbol) FROM daily_prices")
            result = cur.fetchone()
            stats["unique_symbols"] = result[0] if result else 0
            
            # 数据库大小
            cur.execute("SELECT pg_size_pretty(pg_database_size(current_database()))")
            result = cur.fetchone()
            stats["database_size"] = result[0] if result else "unknown"
        
        # 客户端统计
        stats["client_stats"] = self.stats.copy()
        
        return stats
    
    def vacuum_analyze(self, table: Optional[str] = None) -> None:
        """执行 VACUUM ANALYZE（利用 PG 18 优化的 VACUUM）"""
        with self.get_conn() as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                if table:
                    cur.execute(f"VACUUM ANALYZE {table}")
                else:
                    cur.execute("VACUUM ANALYZE")


# 全局单例
_db_instance: Optional[PostgresMarketDB] = None


def get_db() -> PostgresMarketDB:
    """获取数据库单例"""
    global _db_instance
    if _db_instance is None:
        _db_instance = PostgresMarketDB()
    return _db_instance


def reset_db() -> None:
    """重置数据库单例（用于测试）"""
    global _db_instance
    _db_instance = None
