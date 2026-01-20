# Agent 开发指南

本文档为 AI 编程代理提供在本仓库工作的指导规范。

> **⚠️ 强制要求**：所有代理在进行任何修改前，**必须**阅读并严格遵循 `.opencode/conventions.md` 中的规范。

## 项目概述

**AI Trader Assist** 是一个基于 HKUDS/AI-Trader Base 模式实现的**半自动化美股盘前决策系统**。系统在每日美股开盘前（PT 05:30）串联以下流程：

- **数据采集**：通过 yfinance、FRED、CBOE 获取行情、宏观指标与期权数据
- **特征工程**：计算技术指标（RSI/MACD/ATR）、趋势特征、新闻情绪
- **风险评估**：VIX Z-score、Put/Call Ratio、市场宽度等
- **LLM 分析**：DeepSeek 5 阶段推理流水线
- **头寸规划**：基于 ATR 的仓位与止损计算
- **报告生成**：Markdown 与 JSON 版本的人工执行清单

> ⚠️ **免责声明**：本项目仅用于研究与教学，不连接任何券商系统，不构成投资建议。

---

## 构建、测试与运行命令

### 环境准备

```bash
# Python 版本要求
python --version  # 推荐 Python 3.12

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 填入 DEEPSEEK_API_KEY（必填）和 FRED_API_KEY（可选）
```

### 测试命令

```bash
# 运行所有测试
pytest tests -q

# 运行单个测试文件
pytest tests/test_positions.py -v

# 运行带覆盖率的测试
pytest tests --cov=ai_trader_assist --cov-report=term-missing
```

### 日常运行

```bash
# 运行每日盘前流程（默认当日）
python -m ai_trader_assist.jobs.run_daily \
  --config configs/base.json \
  --output-dir storage/daily_$(date +%F)

# 指定历史日期回测
python -m ai_trader_assist.jobs.run_daily \
  --config configs/base.json \
  --date 2025-10-27 \
  --output-dir storage/daily_2025-10-27

# 录入盘后操作记录
python -m ai_trader_assist.jobs.record_operations --config configs/base.json

# 生成持仓盈亏报告
python -m ai_trader_assist.jobs.report_portfolio --config configs/base.json --as-of 2025-10-28
```

### 代码质量检查

```bash
# 类型检查（如已安装 mypy）
mypy ai_trader_assist --ignore-missing-imports

# 代码格式检查（如已安装 ruff）
ruff check ai_trader_assist

# JSON 配置语法验证
python -c "import json; json.load(open('configs/base.json'))"
```

### Git 工作流

```bash
# 提交信息格式
git commit -m "type: description"

# Type 类型：
# feat     - 新功能
# fix      - Bug 修复
# docs     - 文档更新
# refactor - 重构（不改变外部行为）
# test     - 测试用例
# chore    - 构建/工具变更

# 示例
git commit -m "feat: add CBOE put/call ratio data source"
git commit -m "fix: handle empty news response in yf_client"
git commit -m "docs: update LLM operator configuration guide"
```

---

## 目录结构

```
trader/
├── ai_trader_assist/           # 主 Python 包
│   ├── agent/                  # 流水线编排
│   │   ├── orchestrator.py     # LLM 多阶段编排器
│   │   ├── base_agent.py       # 基础代理类
│   │   └── safe_mode.py        # LLM 失败回退策略
│   ├── agent_tools/            # 工具函数
│   │   ├── tool_math.py        # 数学计算工具
│   │   ├── tool_get_price_local.py  # 本地价格查询
│   │   └── tool_trade.py       # 交易模拟工具
│   ├── data_collector/         # 数据采集层
│   │   ├── yf_client.py        # Yahoo Finance 客户端（行情 + 新闻）
│   │   ├── fred_client.py      # FRED 宏观数据客户端
│   │   └── cboe_client.py      # CBOE Put/Call Ratio 数据
│   ├── feature_engineering/    # 特征工程
│   │   ├── pipeline.py         # 特征准备主流水线
│   │   ├── indicators.py       # 技术指标（RSI/MACD/ATR/Z-score）
│   │   └── trend_features.py   # 趋势特征（斜率/动量/均线交叉）
│   ├── decision_engine/        # 决策引擎
│   │   └── stock_scoring.py    # 板块与个股评分逻辑
│   ├── risk_engine/            # 风险引擎
│   │   └── macro_engine.py     # 宏观风险评估
│   ├── position_sizer/         # 仓位计算
│   │   └── sizer.py            # 基于 ATR 的仓位与止损
│   ├── llm/                    # LLM 客户端
│   │   ├── client.py           # DeepSeek API 封装
│   │   └── analyzer.py         # 分阶段分析器
│   ├── llm_operators/          # LLM 各阶段算子
│   │   ├── base.py             # 算子基类与校验逻辑
│   │   ├── market_analyzer.py  # 市场解读阶段
│   │   ├── sector_analyzer.py  # 板块分析阶段
│   │   ├── stock_classifier.py # 个股分类阶段
│   │   ├── exposure_planner.py # 仓位审查阶段
│   │   └── report_composer.py  # 报告整合阶段
│   ├── portfolio_manager/      # 持仓管理
│   │   ├── state.py            # 持仓状态机
│   │   └── positions.py        # 持仓快照读写
│   ├── report_tools/           # 报告工具
│   │   ├── portfolio_reporter.py  # 持仓报告生成
│   │   ├── pnl_analyzer.py     # 盈亏分析
│   │   └── history_builder.py  # 历史记录构建
│   ├── validators/             # 数据校验
│   │   ├── json_schemas.py     # JSON Schema 定义
│   │   └── pydantic_models.py  # Pydantic 模型
│   ├── jobs/                   # 调度脚本
│   │   ├── run_daily.py        # 每日主流程入口
│   │   ├── record_operations.py # 盘后操作录入
│   │   └── report_portfolio.py # 持仓盈亏报告
│   └── __init__.py
├── configs/                    # 配置文件
│   ├── base.json               # 主配置（股票池/风控参数/LLM设置）
│   └── prompts/                # LLM 提示词模板
│       ├── deepseek_base_prompt.md      # 基础系统提示
│       ├── deepseek_market_overview.md  # 市场解读
│       ├── deepseek_sector_analysis.md  # 板块分析
│       ├── deepseek_stock_actions.md    # 个股分类
│       ├── deepseek_exposure_check.md   # 仓位审查
│       └── deepseek_report_compose.md   # 报告整合
├── storage/                    # 数据存储
│   ├── operations.jsonl        # 操作日志（每行一条 JSON）
│   ├── positions.json          # 当前持仓快照
│   ├── cache/                  # 数据缓存
│   │   ├── yf/                 # Yahoo Finance 缓存
│   │   └── fred/               # FRED 数据缓存
│   ├── daily_*/                # 每日输出目录
│   └── logs/                   # 运行日志
├── tests/                      # pytest 测试用例
│   ├── test_positions.py       # 持仓计算测试
│   ├── test_sizer.py           # 仓位计算测试
│   ├── test_llm_*.py           # LLM 相关测试
│   └── ...
├── .opencode/                  # OpenCode 配置
│   └── conventions.md          # 项目规范（强制）
├── .env.example                # 环境变量模板
├── .gitignore
├── requirements.txt            # Python 依赖
├── README.md                   # 项目文档
└── AGENTS.md                   # 本文件
```

---

## 代码风格指南

### Python 文件规范

#### 类型注解（强制）

```python
# ✅ 正确：完整的类型注解
from typing import Dict, List, Optional, Mapping

def score_stocks(
    self,
    stock_features: Dict[str, Dict],
    premarket_flags: Dict[str, Dict],
) -> List[Dict]:
    ...

# ❌ 错误：缺少类型注解
def score_stocks(self, stock_features, premarket_flags):
    ...
```

#### Dataclass 使用模式

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class LLMOperatorConfig:
    prompt_file: Path
    retries: int = 0
    temperature: float = 0.2
    max_tokens: int = 8192

@dataclass
class PortfolioState:
    cash: float = 0.0
    positions: List[Position] = field(default_factory=list)
    last_updated: Optional[str] = None
```

#### Docstring 格式

```python
def fetch_history(
    self,
    symbol: str,
    start: datetime,
    end: datetime,
    interval: str = "1d",
    force: bool = False,
) -> pd.DataFrame:
    """Fetch price history, caching results locally or using a fallback.
    
    Parameters
    ----------
    symbol : str
        Ticker symbol to fetch (e.g., "AAPL", "SPY").
    start : datetime
        Start date for the history range.
    end : datetime
        End date for the history range (exclusive).
    interval : str, optional
        Data interval, default "1d".
    force : bool, optional
        If True, bypass cache and re-fetch from network.
    
    Returns
    -------
    pd.DataFrame
        OHLCV data with columns: Open, High, Low, Close, Adj Close, Volume.
    """
```

#### 异常处理模式

```python
# ✅ 正确：具体的异常类型 + 日志记录
try:
    response = requests.post(self.api_url, headers=headers, json=payload, timeout=self.timeout)
except requests.Timeout as exc:
    raise TimeoutError("DeepSeek 请求超时") from exc
except requests.RequestException as exc:
    raise RuntimeError(f"DeepSeek 请求失败: {exc}") from exc

# ❌ 错误：裸 except 或忽略异常
try:
    response = requests.post(...)
except:
    pass
```

### 命名约定

| 类型 | 格式 | 示例 |
|------|------|------|
| 模块/文件 | `snake_case` | `yf_client.py`, `stock_scoring.py` |
| 类名 | `PascalCase` | `YahooFinanceClient`, `PortfolioState` |
| 函数/方法 | `snake_case` | `fetch_history()`, `score_stocks()` |
| 常量 | `UPPER_SNAKE_CASE` | `POSITIVE_KEYWORDS`, `BASE_URL` |
| 私有方法 | `_leading_underscore` | `_parse_json_response()`, `_cache_path()` |
| 环境变量 | `UPPER_SNAKE_CASE` | `DEEPSEEK_API_KEY`, `FRED_API_KEY` |

---

## LLM 流水线架构

### 5 阶段推理流程

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Market    │ -> │   Sector    │ -> │   Stock     │ -> │  Exposure   │ -> │   Report    │
│  Analyzer   │    │  Analyzer   │    │ Classifier  │    │  Planner    │    │  Composer   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
     │                   │                   │                   │                   │
     v                   v                   v                   v                   v
 risk_level          leading/           Buy/Hold/           allocation          markdown
   bias              lagging            Reduce/Avoid          plan              report
 drivers             sectors            categories          constraints         sections
```

### 阶段说明

| 阶段 | 输入 | 输出 | 提示词模板 |
|------|------|------|----------|
| `market_analyzer` | 市场特征、VIX、宏观指标 | `risk_level`, `bias`, `drivers` | `deepseek_market_overview.md` |
| `sector_analyzer` | 板块特征、新闻 | `leading`, `lagging`, `focus_points` | `deepseek_sector_analysis.md` |
| `stock_classifier` | 个股特征、趋势、新闻 | `categories` (Buy/Hold/Reduce/Avoid) | `deepseek_stock_actions.md` |
| `exposure_planner` | 持仓状态、前阶段结果 | `allocation_plan`, `constraints` | `deepseek_exposure_check.md` |
| `report_composer` | 所有前阶段结果 | `markdown`, `sections` | `deepseek_report_compose.md` |

### Safe Mode 回退

当 LLM 调用失败或校验不通过时，系统自动进入 Safe Mode：

```python
safe_mode_config = SafeModeConfig(
    on_llm_failure="no_new_risk",  # 禁止新增风险敞口
    max_exposure_cap=0.4,          # 最大敞口上限降至 40%
)
```

---

## 数据流规范

### 主流程数据流

```
yf_client / fred_client / cboe_client
            │
            v
    prepare_feature_sets()
            │
            ├── market_features    (VIX, breadth, RS, put/call)
            ├── sector_features    (momentum, relative strength)
            ├── stock_features     (RSI, MACD, trend, news)
            ├── premarket_flags    (deviation, volume ratio)
            ├── news_bundle        (headlines, sentiment)
            ├── trend_bundle       (slopes, momentum, crosses)
            └── macro_flags        (CPI, yield curve, fed funds)
            │
            v
    StockDecisionEngine.score_stocks()
            │
            v
    LLMOrchestrator.run()
            │
            v
    HybridReportBuilder.build()
            │
            v
    storage/daily_<date>/
            ├── report.md
            ├── report.json
            ├── llm_analysis.json
            └── *_features.json
```

### 关键数据结构

```python
# 个股特征 (stock_features)
{
    "NVDA": {
        "rsi_norm": 0.65,           # RSI 归一化 (0-1)
        "macd_signal": 0.02,        # MACD 信号
        "trend_slope": 0.003,       # 价格斜率
        "volume_score": 0.15,       # 成交量相对强度
        "structure_score": 0.08,    # 均线结构得分
        "atr_pct": 0.025,           # ATR 百分比
        "price": 145.50,            # 最新价格
        "news_score": 0.3,          # 新闻情绪 (-1 ~ 1)
        "trend_strength": 0.7,      # 趋势强度
        "trend_state": "uptrend",   # uptrend/downtrend/flat
        "momentum_10d": 0.08,       # 10日动量
    }
}

# 持仓状态 (PortfolioState)
{
    "cash": 50000.0,
    "positions": [
        {"symbol": "NVDA", "shares": 100, "avg_cost": 140.0},
        {"symbol": "AAPL", "shares": 50, "avg_cost": 180.0}
    ],
    "equity_value": 165000.0,
    "exposure": 0.70
}
```

---

## 添加新组件指南

### 新增数据源

1. 在 `ai_trader_assist/data_collector/` 创建新客户端：

```python
# new_source_client.py
class NewSourceClient:
    def __init__(self, api_key: Optional[str], cache_dir: Optional[Path] = None):
        self.api_key = api_key
        self.cache_dir = cache_dir or Path("storage/cache/new_source")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch_data(self, symbol: str) -> Dict:
        # 实现数据获取逻辑
        ...
```

2. 在 `feature_engineering/pipeline.py` 中集成
3. 添加对应的测试用例 `tests/test_new_source_client.py`

### 新增技术指标

在 `feature_engineering/indicators.py` 中添加：

```python
def new_indicator(series: pd.Series, window: int = 14) -> pd.Series:
    """Calculate new indicator.
    
    Parameters
    ----------
    series : pd.Series
        Price series (typically Close).
    window : int
        Lookback window.
    
    Returns
    -------
    pd.Series
        Indicator values.
    """
    # 实现计算逻辑
    ...
```

### 新增 LLM 阶段

1. 在 `llm_operators/` 创建新算子：

```python
# new_stage_operator.py
from .base import LLMOperator

class NewStageOperator(LLMOperator):
    def _build_prompt(self, payload: Mapping) -> str:
        # 构建提示词
        ...
    
    def _validate_output(self, result: Mapping) -> None:
        # 校验输出格式
        ...
```

2. 在 `validators/json_schemas.py` 添加对应的 JSON Schema
3. 在 `agent/orchestrator.py` 的 `_init_operators()` 中注册
4. 创建提示词模板 `configs/prompts/deepseek_new_stage.md`

---

## 错误处理规范

### 网络请求失败

- **优先使用缓存**：所有数据客户端应实现本地缓存回退
- **合成数据兜底**：在完全离线时生成可追溯的合成数据
- **记录数据缺口**：在 `data_gaps` 字段中标记缺失的数据项

### LLM 调用失败

- **自动重试**：根据 `retries` 配置进行重试
- **Safe Mode 回退**：连续失败后进入保守模式
- **错误日志**：将失败的 payload 写入 `errors.jsonl`

### 持仓计算异常

- **操作日志校验**：检查时间戳顺序，跳过已处理的记录
- **备份机制**：写入前备份为 `*.bak`
- **幂等性**：重复运行不会产生重复记录

---

## 最佳实践

1. **保持模块聚焦**：每个模块只负责单一职责
2. **类型注解优先**：所有公开 API 必须有完整的类型注解
3. **缓存友好**：数据采集模块应支持离线运行
4. **日志可追溯**：关键操作使用 `log_step()` / `log_result()` / `log_ok()` 记录
5. **配置驱动**：可调参数通过 `configs/base.json` 管理，避免硬编码
6. **人工复核**：系统输出仅供参考，最终决策需人工确认

---

## Agent 化架构（OpenCode 集成）

### 架构概览

系统已重构为 OpenCode Agent/Skill/MCP Tools 架构，支持对话式交互：

```
┌─────────────────────────────────────────────────────────────┐
│                    用户对话入口                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Trading Orchestrator (主编排)                   │
│  意图识别 → 路由到 Skill 或委派 Sub-Agent                     │
└─────────────────────────────────────────────────────────────┘
          │              │              │              │
          ▼              ▼              ▼              ▼
    ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
    │  Data    │  │  Risk    │  │Portfolio │  │  Report  │
    │ Analyst  │  │ Manager  │  │ Manager  │  │ Composer │
    └──────────┘  └──────────┘  └──────────┘  └──────────┘
          │              │              │              │
          └──────────────┴──────────────┴──────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   MCP Tools (14 个工具)                      │
│  价格 │ 新闻/宏观 │ 持仓管理 │ 分析计算                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              现有 Python 模块 (data_collector, etc.)         │
└─────────────────────────────────────────────────────────────┘
```

### 目录结构（新增）

```
trader/
├── opencode.json                     # OpenCode 配置（Agent + MCP）
├── .opencode/
│   ├── agents/                       # 5 个 Agent 定义
│   │   ├── trading-orchestrator.md   # 主编排 Agent
│   │   ├── trading-data-analyst.md   # 数据分析 Agent
│   │   ├── trading-risk-manager.md   # 风险管理 Agent
│   │   ├── trading-portfolio-manager.md  # 持仓管理 Agent
│   │   └── trading-report-composer.md    # 报告生成 Agent
│   └── skills/                       # 7 个 Skills
│       ├── market-scan/SKILL.md      # 市场扫描
│       ├── sector-analysis/SKILL.md  # 板块分析
│       ├── stock-analysis/SKILL.md   # 个股分析
│       ├── position-sizing/SKILL.md  # 仓位计算
│       ├── daily-report/SKILL.md     # 生成报告
│       ├── record-operation/SKILL.md # 记录操作
│       └── show-portfolio/SKILL.md   # 查看持仓
└── ai_trader_assist/mcp_server/      # MCP Server
    ├── __init__.py
    ├── server.py                     # FastMCP 入口
    └── tools/
        ├── price_tools.py            # get_price, get_history, get_quotes
        ├── news_tools.py             # get_news, get_macro, get_pcr
        ├── portfolio_tools.py        # get_portfolio, save_operation, update_positions
        └── analysis_tools.py         # calc_indicators, score_stocks, generate_orders
```

### MCP Tools 列表

| 工具 | 说明 | 参数 |
|------|------|------|
| `get_price` | 获取最新价格 | `symbol` |
| `get_history` | 获取历史行情 | `symbol`, `days`, `interval` |
| `get_quotes` | 批量获取报价 | `symbols` |
| `get_news` | 获取相关新闻 | `symbol`, `max_items`, `lookback_days` |
| `get_macro` | 获取宏观指标 | - |
| `get_pcr` | 获取 Put/Call Ratio | - |
| `get_portfolio` | 获取当前持仓 | - |
| `save_operation` | 记录交易操作 | `symbol`, `action`, `shares`, `price`, `reason` |
| `update_positions` | 更新持仓快照 | - |
| `get_operations_history` | 获取操作历史 | `days` |
| `calc_indicators` | 计算技术指标 | `symbol`, `indicators` |
| `score_stocks` | 对股票评分 | `symbols` |
| `generate_orders` | 生成订单建议 | `symbol`, `action`, `budget`/`shares` |

### Skills 使用方式

| 用户指令 | 触发 Skill |
|---------|-----------|
| "今天市场怎么样" | `/market-scan` |
| "分析一下英伟达" | `/stock-analysis NVDA` |
| "哪些板块表现好" | `/sector-analysis` |
| "我的持仓情况" | `/show-portfolio` |
| "买 50 股应该设多少止损" | `/position-sizing` |
| "记录今天买了 100 股 AAPL" | `/record-operation` |
| "生成今日盘前报告" | `/daily-report` |

### 启动 MCP Server

```bash
# 安装 FastMCP
pip install mcp[cli]

# 启动 MCP Server
python -m ai_trader_assist.mcp_server.server
```

---

## 📖 参考文档

- **项目规范（强制）**：`.opencode/conventions.md` - 所有代理必读
- **项目说明**：`README.md` - 快速开始与使用指南
- **配置说明**：`configs/base.json` - 运行参数详解
- **Agent 定义**：`.opencode/agents/` - Agent 角色与职责
- **Skill 定义**：`.opencode/skills/` - 可用 Skills 列表
- **本指南**：`AGENTS.md` - Agent 开发详细指南
