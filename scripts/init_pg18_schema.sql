-- ============================================
-- PostgreSQL 18 交易系统 Schema
-- AI Trader Assist - 优化版数据库结构
-- ============================================

-- 0. 清理旧表（如需重建）
-- DROP TABLE IF EXISTS news CASCADE;
-- DROP TABLE IF EXISTS llm_analysis CASCADE;
-- DROP TABLE IF EXISTS trade_signals CASCADE;
-- DROP TABLE IF EXISTS backtest_trades CASCADE;
-- DROP TABLE IF EXISTS backtest_runs CASCADE;
-- DROP TABLE IF EXISTS position_history CASCADE;
-- DROP TABLE IF EXISTS positions CASCADE;
-- DROP TABLE IF EXISTS indicators CASCADE;
-- DROP TABLE IF EXISTS daily_prices CASCADE;
-- DROP TABLE IF EXISTS sector_holdings CASCADE;
-- DROP TABLE IF EXISTS stock_meta CASCADE;

-- ============================================
-- 1. 股票元数据表
-- ============================================
CREATE TABLE IF NOT EXISTS stock_meta (
    symbol          VARCHAR(10) PRIMARY KEY,
    name            VARCHAR(200),
    sector          VARCHAR(50),
    industry        VARCHAR(100),
    market_cap      BIGINT,
    avg_volume_30d  BIGINT,
    beta            DECIMAL(6,3),
    pe_ratio        DECIMAL(10,2),
    dividend_yield  DECIMAL(6,4),
    
    -- PG 18 虚拟列：市值等级
    cap_tier VARCHAR(10) GENERATED ALWAYS AS (
        CASE 
            WHEN market_cap >= 200000000000 THEN 'MEGA'
            WHEN market_cap >= 10000000000 THEN 'LARGE'
            WHEN market_cap >= 2000000000 THEN 'MID'
            ELSE 'SMALL'
        END
    ) VIRTUAL,
    
    last_updated    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE stock_meta IS '股票元数据表 - 存储股票基本信息';

-- ============================================
-- 2. 板块成分股映射表
-- ============================================
CREATE TABLE IF NOT EXISTS sector_holdings (
    sector_etf      VARCHAR(10) NOT NULL,  -- XLK, XLF, etc.
    symbol          VARCHAR(10) NOT NULL,
    weight          DECIMAL(6,4),          -- 权重百分比
    rank            INT,                    -- 在板块中的排名
    updated_at      DATE DEFAULT CURRENT_DATE,
    PRIMARY KEY (sector_etf, symbol)
);

CREATE INDEX IF NOT EXISTS idx_sector_symbol ON sector_holdings(symbol);

COMMENT ON TABLE sector_holdings IS '板块成分股映射表 - 支持动态选股';

-- ============================================
-- 3. 日线数据表（核心表）
-- ============================================
CREATE TABLE IF NOT EXISTS daily_prices (
    symbol      VARCHAR(10) NOT NULL,
    trade_date  DATE NOT NULL,
    open        DECIMAL(12,4),
    high        DECIMAL(12,4),
    low         DECIMAL(12,4),
    close       DECIMAL(12,4),
    adj_close   DECIMAL(12,4),
    volume      BIGINT,
    
    -- PG 18 虚拟生成列（不占存储空间）
    daily_range     DECIMAL(12,4) GENERATED ALWAYS AS (high - low) VIRTUAL,
    daily_return    DECIMAL(8,6) GENERATED ALWAYS AS (
        CASE WHEN open > 0 THEN (close - open) / open ELSE 0 END
    ) VIRTUAL,
    is_green        BOOLEAN GENERATED ALWAYS AS (close >= open) VIRTUAL,
    
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, trade_date)
);

-- BRIN 索引（时序数据最优，比 B-tree 小 100x）
CREATE INDEX IF NOT EXISTS idx_prices_date_brin ON daily_prices USING BRIN (trade_date);
CREATE INDEX IF NOT EXISTS idx_prices_symbol ON daily_prices (symbol);

COMMENT ON TABLE daily_prices IS '日线行情数据表 - 核心数据存储';

-- ============================================
-- 4. 技术指标缓存表
-- ============================================
CREATE TABLE IF NOT EXISTS indicators (
    symbol      VARCHAR(10) NOT NULL,
    trade_date  DATE NOT NULL,
    
    -- 均线指标
    sma_20      DECIMAL(12,4),
    sma_50      DECIMAL(12,4),
    sma_200     DECIMAL(12,4),
    ema_12      DECIMAL(12,4),
    ema_26      DECIMAL(12,4),
    
    -- 动量指标
    rsi_14      DECIMAL(8,4),
    macd        DECIMAL(12,6),
    macd_signal DECIMAL(12,6),
    macd_hist   DECIMAL(12,6),
    
    -- 波动率指标
    atr_14      DECIMAL(12,6),
    bb_upper    DECIMAL(12,4),
    bb_middle   DECIMAL(12,4),
    bb_lower    DECIMAL(12,4),
    
    -- 成交量指标
    volume_sma_20   BIGINT,
    volume_ratio    DECIMAL(8,4),
    
    -- 趋势指标
    trend_slope_20  DECIMAL(12,8),
    momentum_10d    DECIMAL(8,6),
    
    -- PG 18 虚拟生成列：布林带位置 (0-1)
    bb_position DECIMAL(5,3) GENERATED ALWAYS AS (
        CASE WHEN bb_upper > bb_lower 
        THEN (bb_middle - bb_lower) / (bb_upper - bb_lower)
        ELSE 0.5 END
    ) VIRTUAL,
    
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, trade_date)
);

CREATE INDEX IF NOT EXISTS idx_indicators_date ON indicators (trade_date);

COMMENT ON TABLE indicators IS '技术指标缓存表 - 预计算的技术指标';

-- ============================================
-- 5. 持仓表
-- ============================================
CREATE TABLE IF NOT EXISTS positions (
    id          UUID DEFAULT uuidv7() PRIMARY KEY,  -- PG 18: UUIDv7
    symbol      VARCHAR(10) NOT NULL UNIQUE,
    shares      INT NOT NULL DEFAULT 0,
    avg_cost    DECIMAL(12,4) NOT NULL DEFAULT 0,
    
    -- PG 18 虚拟列：持仓市值（需要配合查询）
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE positions IS '当前持仓表';

-- ============================================
-- 6. 持仓历史表
-- ============================================
CREATE TABLE IF NOT EXISTS position_history (
    id          UUID DEFAULT uuidv7() PRIMARY KEY,
    symbol      VARCHAR(10) NOT NULL,
    action      VARCHAR(20) NOT NULL,  -- BUY, SELL, STOP_LOSS, TAKE_PROFIT
    shares      INT NOT NULL,
    price       DECIMAL(12,4) NOT NULL,
    total_value DECIMAL(14,2) GENERATED ALWAYS AS (shares * price) VIRTUAL,
    reason      TEXT,
    trade_date  DATE NOT NULL DEFAULT CURRENT_DATE,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_pos_history_symbol ON position_history (symbol, trade_date DESC);
CREATE INDEX IF NOT EXISTS idx_pos_history_date ON position_history (trade_date DESC);

COMMENT ON TABLE position_history IS '持仓变更历史表';

-- ============================================
-- 7. 回测运行记录表
-- ============================================
CREATE TABLE IF NOT EXISTS backtest_runs (
    id              UUID DEFAULT uuidv7() PRIMARY KEY,
    name            VARCHAR(100),
    strategy        VARCHAR(50),
    symbols         TEXT[],             -- 股票列表
    start_date      DATE,
    end_date        DATE,
    initial_capital DECIMAL(14,2),
    final_capital   DECIMAL(14,2),
    
    -- 绩效指标
    total_return    DECIMAL(10,4),
    max_drawdown    DECIMAL(10,4),
    sharpe_ratio    DECIMAL(6,3),
    win_rate        DECIMAL(5,3),
    total_trades    INT,
    
    -- 配置和详情（JSONB）
    config          JSONB,
    monthly_returns JSONB,
    
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_backtest_date ON backtest_runs (created_at DESC);

COMMENT ON TABLE backtest_runs IS '回测运行记录表';

-- ============================================
-- 8. 回测交易记录表
-- ============================================
CREATE TABLE IF NOT EXISTS backtest_trades (
    id          UUID DEFAULT uuidv7() PRIMARY KEY,
    run_id      UUID REFERENCES backtest_runs(id) ON DELETE CASCADE,
    symbol      VARCHAR(10) NOT NULL,
    trade_date  DATE NOT NULL,
    action      VARCHAR(10) NOT NULL,   -- BUY, SELL
    shares      INT NOT NULL,
    price       DECIMAL(12,4) NOT NULL,
    total_value DECIMAL(14,2) GENERATED ALWAYS AS (shares * price) VIRTUAL,
    reason      VARCHAR(100),
    pnl         DECIMAL(12,2),
    pnl_pct     DECIMAL(8,4),
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_backtest_trades_run ON backtest_trades (run_id, trade_date);

COMMENT ON TABLE backtest_trades IS '回测交易明细表';

-- ============================================
-- 9. 交易信号表
-- ============================================
CREATE TABLE IF NOT EXISTS trade_signals (
    id          UUID DEFAULT uuidv7() PRIMARY KEY,
    trade_date  DATE NOT NULL,
    symbol      VARCHAR(10) NOT NULL,
    signal      VARCHAR(10) NOT NULL,   -- BUY, SELL, HOLD
    score       DECIMAL(5,3),
    regime      VARCHAR(30),
    strategy    VARCHAR(50),
    entry_price DECIMAL(12,4),
    stop_loss   DECIMAL(12,4),
    take_profit DECIMAL(12,4),
    confidence  DECIMAL(5,3),
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_signals_date ON trade_signals (trade_date DESC);
CREATE INDEX IF NOT EXISTS idx_signals_symbol ON trade_signals (symbol, trade_date DESC);

COMMENT ON TABLE trade_signals IS '交易信号表';

-- ============================================
-- 10. 新闻表（支持全文搜索）
-- ============================================
CREATE TABLE IF NOT EXISTS news (
    id              UUID DEFAULT uuidv7() PRIMARY KEY,
    symbol          VARCHAR(10) NOT NULL,
    title           TEXT,
    summary         TEXT,
    content         TEXT,
    publisher       VARCHAR(100),
    url             TEXT,
    published_at    TIMESTAMP,
    sentiment_score DECIMAL(5,3),       -- -1 到 1
    
    -- 全文搜索向量
    search_vector   TSVECTOR GENERATED ALWAYS AS (
        setweight(to_tsvector('english', COALESCE(title, '')), 'A') ||
        setweight(to_tsvector('english', COALESCE(summary, '')), 'B')
    ) STORED,
    
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_news_symbol_date ON news (symbol, published_at DESC);
CREATE INDEX IF NOT EXISTS idx_news_search ON news USING GIN (search_vector);
CREATE UNIQUE INDEX IF NOT EXISTS idx_news_url ON news (url) WHERE url IS NOT NULL;

COMMENT ON TABLE news IS '新闻数据表 - 支持全文搜索';

-- ============================================
-- 11. LLM 分析结果表
-- ============================================
CREATE TABLE IF NOT EXISTS llm_analysis (
    id              UUID DEFAULT uuidv7() PRIMARY KEY,
    trade_date      DATE NOT NULL,
    stage           VARCHAR(30),        -- market_analyzer, sector_analyzer, etc.
    regime          VARCHAR(30),
    analysis        JSONB NOT NULL,
    
    -- PG 18 虚拟列：从 JSON 提取关键字段
    risk_level VARCHAR(20) GENERATED ALWAYS AS (
        analysis->>'risk_level'
    ) VIRTUAL,
    
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_llm_date ON llm_analysis (trade_date DESC);
CREATE INDEX IF NOT EXISTS idx_llm_json ON llm_analysis USING GIN (analysis jsonb_path_ops);

COMMENT ON TABLE llm_analysis IS 'LLM 分析结果表';

-- ============================================
-- 12. 系统配置表
-- ============================================
CREATE TABLE IF NOT EXISTS system_config (
    key         VARCHAR(100) PRIMARY KEY,
    value       JSONB,
    description TEXT,
    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE system_config IS '系统配置表';

-- ============================================
-- 辅助视图：动态选股候选池
-- ============================================
CREATE OR REPLACE VIEW v_stock_candidates AS
WITH latest_data AS (
    SELECT DISTINCT ON (dp.symbol)
        dp.symbol,
        dp.trade_date,
        dp.close,
        dp.volume,
        dp.daily_return,
        i.sma_20,
        i.sma_50,
        i.sma_200,
        i.rsi_14,
        i.macd_hist,
        i.volume_ratio,
        i.bb_position,
        i.momentum_10d,
        sm.sector,
        sm.market_cap,
        sm.cap_tier
    FROM daily_prices dp
    LEFT JOIN indicators i ON dp.symbol = i.symbol AND dp.trade_date = i.trade_date
    LEFT JOIN stock_meta sm ON dp.symbol = sm.symbol
    ORDER BY dp.symbol, dp.trade_date DESC
)
SELECT 
    symbol,
    trade_date,
    close,
    sector,
    cap_tier,
    rsi_14,
    volume_ratio,
    momentum_10d,
    -- 均线位置
    CASE WHEN sma_20 > 0 THEN (close - sma_20) / sma_20 ELSE 0 END AS pct_from_sma20,
    CASE WHEN sma_200 > 0 THEN (close - sma_200) / sma_200 ELSE 0 END AS pct_from_sma200,
    -- 趋势状态判断
    CASE 
        WHEN close > sma_50 AND sma_50 > sma_200 THEN 'UPTREND'
        WHEN close < sma_50 AND sma_50 < sma_200 THEN 'DOWNTREND'
        ELSE 'RANGE'
    END AS trend_state,
    -- 动量排名
    PERCENT_RANK() OVER (ORDER BY momentum_10d) AS momentum_rank
FROM latest_data
WHERE volume_ratio > 0.5;  -- 过滤低流动性

COMMENT ON VIEW v_stock_candidates IS '动态选股候选池视图';

-- ============================================
-- 辅助函数：获取最新交易日
-- ============================================
CREATE OR REPLACE FUNCTION get_latest_trade_date()
RETURNS DATE AS $$
    SELECT MAX(trade_date) FROM daily_prices WHERE symbol = 'SPY';
$$ LANGUAGE SQL STABLE;

COMMENT ON FUNCTION get_latest_trade_date() IS '获取最新交易日';

-- ============================================
-- 辅助函数：计算区间收益率
-- ============================================
CREATE OR REPLACE FUNCTION calc_return(
    p_symbol VARCHAR(10),
    p_start_date DATE,
    p_end_date DATE
) RETURNS DECIMAL(10,6) AS $$
DECLARE
    start_price DECIMAL(12,4);
    end_price DECIMAL(12,4);
BEGIN
    SELECT close INTO start_price 
    FROM daily_prices 
    WHERE symbol = p_symbol AND trade_date >= p_start_date 
    ORDER BY trade_date LIMIT 1;
    
    SELECT close INTO end_price 
    FROM daily_prices 
    WHERE symbol = p_symbol AND trade_date <= p_end_date 
    ORDER BY trade_date DESC LIMIT 1;
    
    IF start_price IS NULL OR start_price = 0 OR end_price IS NULL THEN
        RETURN NULL;
    END IF;
    
    RETURN (end_price - start_price) / start_price;
END;
$$ LANGUAGE plpgsql STABLE;

COMMENT ON FUNCTION calc_return(VARCHAR, DATE, DATE) IS '计算区间收益率';

-- ============================================
-- 初始化板块成分股数据
-- ============================================
INSERT INTO sector_holdings (sector_etf, symbol, weight, rank) VALUES
    -- XLK - 科技
    ('XLK', 'AAPL', 0.22, 1),
    ('XLK', 'MSFT', 0.21, 2),
    ('XLK', 'NVDA', 0.06, 3),
    ('XLK', 'AVGO', 0.05, 4),
    ('XLK', 'AMD', 0.02, 5),
    ('XLK', 'CRM', 0.03, 6),
    ('XLK', 'ADBE', 0.02, 7),
    ('XLK', 'CSCO', 0.02, 8),
    ('XLK', 'ORCL', 0.02, 9),
    ('XLK', 'INTC', 0.01, 10),
    -- XLC - 通讯服务
    ('XLC', 'META', 0.23, 1),
    ('XLC', 'GOOGL', 0.12, 2),
    ('XLC', 'GOOG', 0.11, 3),
    ('XLC', 'NFLX', 0.05, 4),
    ('XLC', 'DIS', 0.04, 5),
    ('XLC', 'CMCSA', 0.04, 6),
    ('XLC', 'VZ', 0.04, 7),
    ('XLC', 'T', 0.03, 8),
    -- XLY - 可选消费
    ('XLY', 'AMZN', 0.23, 1),
    ('XLY', 'TSLA', 0.12, 2),
    ('XLY', 'HD', 0.09, 3),
    ('XLY', 'MCD', 0.04, 4),
    ('XLY', 'NKE', 0.03, 5),
    ('XLY', 'LOW', 0.03, 6),
    ('XLY', 'SBUX', 0.03, 7),
    ('XLY', 'TJX', 0.02, 8),
    -- XLF - 金融
    ('XLF', 'BRK.B', 0.13, 1),
    ('XLF', 'JPM', 0.10, 2),
    ('XLF', 'V', 0.08, 3),
    ('XLF', 'MA', 0.07, 4),
    ('XLF', 'BAC', 0.04, 5),
    ('XLF', 'WFC', 0.03, 6),
    ('XLF', 'GS', 0.03, 7),
    ('XLF', 'MS', 0.02, 8),
    -- XLV - 医疗
    ('XLV', 'UNH', 0.10, 1),
    ('XLV', 'JNJ', 0.07, 2),
    ('XLV', 'LLY', 0.07, 3),
    ('XLV', 'MRK', 0.05, 4),
    ('XLV', 'ABBV', 0.05, 5),
    ('XLV', 'PFE', 0.04, 6),
    ('XLV', 'TMO', 0.04, 7),
    -- XLE - 能源
    ('XLE', 'XOM', 0.23, 1),
    ('XLE', 'CVX', 0.17, 2),
    ('XLE', 'COP', 0.05, 3),
    ('XLE', 'SLB', 0.04, 4),
    ('XLE', 'EOG', 0.04, 5),
    ('XLE', 'MPC', 0.04, 6),
    -- XLI - 工业
    ('XLI', 'GE', 0.05, 1),
    ('XLI', 'CAT', 0.05, 2),
    ('XLI', 'RTX', 0.04, 3),
    ('XLI', 'UNP', 0.04, 4),
    ('XLI', 'HON', 0.04, 5),
    ('XLI', 'BA', 0.03, 6),
    ('XLI', 'DE', 0.03, 7),
    -- XLP - 必需消费
    ('XLP', 'PG', 0.15, 1),
    ('XLP', 'COST', 0.11, 2),
    ('XLP', 'KO', 0.10, 3),
    ('XLP', 'PEP', 0.09, 4),
    ('XLP', 'WMT', 0.08, 5),
    ('XLP', 'PM', 0.05, 6),
    -- XLU - 公用事业
    ('XLU', 'NEE', 0.14, 1),
    ('XLU', 'SO', 0.08, 2),
    ('XLU', 'DUK', 0.07, 3),
    ('XLU', 'CEG', 0.06, 4),
    -- XLB - 材料
    ('XLB', 'LIN', 0.18, 1),
    ('XLB', 'APD', 0.08, 2),
    ('XLB', 'SHW', 0.07, 3),
    ('XLB', 'FCX', 0.06, 4),
    ('XLB', 'ECL', 0.05, 5),
    -- XLRE - 房地产
    ('XLRE', 'PLD', 0.11, 1),
    ('XLRE', 'AMT', 0.09, 2),
    ('XLRE', 'EQIX', 0.08, 3),
    ('XLRE', 'WELL', 0.06, 4),
    ('XLRE', 'SPG', 0.05, 5)
ON CONFLICT (sector_etf, symbol) DO UPDATE SET
    weight = EXCLUDED.weight,
    rank = EXCLUDED.rank,
    updated_at = CURRENT_DATE;

-- ============================================
-- 完成提示
-- ============================================
DO $$
BEGIN
    RAISE NOTICE '✅ PostgreSQL 18 Schema 初始化完成!';
    RAISE NOTICE '表: stock_meta, sector_holdings, daily_prices, indicators,';
    RAISE NOTICE '    positions, position_history, backtest_runs, backtest_trades,';
    RAISE NOTICE '    trade_signals, news, llm_analysis, system_config';
    RAISE NOTICE '视图: v_stock_candidates';
    RAISE NOTICE '函数: get_latest_trade_date(), calc_return()';
END $$;
