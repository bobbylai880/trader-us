---
name: project_conventions
type: configuration
description: AI Trader Assist 项目规范与约定
version: 1.0.0
last_updated: 2026-01-20
---

# 项目规范

本文档定义 AI Trader Assist 的强制规范，所有 AI agents 和人类贡献者必须遵守。

---

## 📁 目录结构规范

### 项目标准结构

```
trader/
├── ai_trader_assist/           # 主 Python 包（所有业务逻辑）
│   ├── agent/                  # 流水线编排
│   ├── agent_tools/            # 工具函数
│   ├── data_collector/         # 数据采集层
│   ├── decision_engine/        # 决策引擎
│   ├── feature_engineering/    # 特征工程
│   ├── llm/                    # LLM 客户端
│   ├── llm_operators/          # LLM 各阶段算子
│   ├── portfolio_manager/      # 持仓管理
│   ├── position_sizer/         # 仓位计算
│   ├── report_tools/           # 报告工具
│   ├── risk_engine/            # 风险引擎
│   ├── validators/             # 数据校验
│   ├── jobs/                   # 调度脚本
│   └── utils.py                # 通用工具函数
├── configs/                    # 配置文件
│   ├── base.json               # 主配置
│   └── prompts/                # LLM 提示词模板
├── storage/                    # 数据存储（运行时生成）
│   ├── cache/                  # 数据缓存
│   ├── daily_*/                # 每日输出
│   ├── logs/                   # 运行日志
│   ├── operations.jsonl        # 操作日志
│   └── positions.json          # 持仓快照
├── tests/                      # pytest 测试用例
├── .opencode/                  # OpenCode 配置
│   └── conventions.md          # 本文件
├── .env.example                # 环境变量模板
├── requirements.txt            # Python 依赖
├── README.md                   # 项目文档
└── AGENTS.md                   # Agent 开发指南
```

### 模块职责划分

| 模块 | 职责 | 禁止事项 |
|------|------|---------|
| `data_collector/` | 外部数据获取与缓存 | 禁止业务逻辑计算 |
| `feature_engineering/` | 特征计算与转换 | 禁止直接网络请求 |
| `decision_engine/` | 评分与决策逻辑 | 禁止 I/O 操作 |
| `llm/` | LLM API 交互 | 禁止业务逻辑 |
| `llm_operators/` | LLM 阶段定义与校验 | 禁止直接 API 调用 |
| `portfolio_manager/` | 持仓状态管理 | 禁止直接修改文件 |
| `jobs/` | 流程编排与调度 | 可以协调所有模块 |

---

## 🐍 Python 代码规范

### 类型注解（强制）

所有公开 API 必须有完整的类型注解：

```python
# ✅ 正确
from typing import Dict, List, Optional, Mapping
from pathlib import Path

def fetch_history(
    self,
    symbol: str,
    start: datetime,
    end: datetime,
    interval: str = "1d",
) -> pd.DataFrame:
    ...

# ❌ 错误：缺少类型注解
def fetch_history(self, symbol, start, end, interval="1d"):
    ...
```

### Dataclass 使用模式

配置类和数据容器优先使用 `dataclass`：

```python
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path

@dataclass
class LLMOperatorConfig:
    """LLM 算子配置。"""
    prompt_file: Path
    retries: int = 0
    temperature: float = 0.2
    max_tokens: int = 8192

@dataclass
class Position:
    """单个持仓记录。"""
    symbol: str
    shares: float
    avg_cost: float
    last_price: float = 0.0

    @property
    def market_value(self) -> float:
        return self.shares * self.last_price
```

### Docstring 格式（NumPy 风格）

```python
def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> pd.Series:
    """计算 Average True Range (ATR)。

    Parameters
    ----------
    high : pd.Series
        最高价序列。
    low : pd.Series
        最低价序列。
    close : pd.Series
        收盘价序列。
    window : int, optional
        计算窗口，默认 14。

    Returns
    -------
    pd.Series
        ATR 值序列。

    Examples
    --------
    >>> atr = calculate_atr(df["High"], df["Low"], df["Close"])
    >>> atr.iloc[-1]
    2.35
    """
```

### 异常处理规范

```python
# ✅ 正确：具体异常类型 + 链式异常 + 日志
try:
    response = requests.post(url, json=payload, timeout=30)
    response.raise_for_status()
except requests.Timeout as exc:
    logger.warning("请求超时: %s", url)
    raise TimeoutError(f"API 请求超时: {url}") from exc
except requests.HTTPError as exc:
    logger.error("HTTP 错误 %d: %s", response.status_code, response.text)
    raise RuntimeError(f"API 调用失败: {exc}") from exc

# ❌ 错误：裸 except
try:
    response = requests.post(url, json=payload)
except:
    pass

# ❌ 错误：忽略异常
try:
    result = dangerous_operation()
except Exception:
    result = None  # 静默失败
```

### 禁止事项

| 禁止 | 原因 | 替代方案 |
|------|------|---------|
| `from module import *` | 污染命名空间 | 显式导入 |
| 裸 `except:` | 隐藏真实错误 | 具体异常类型 |
| 可变默认参数 | 共享状态 Bug | `field(default_factory=...)` |
| 硬编码路径 | 跨平台问题 | `Path` + 配置文件 |
| `print()` 调试 | 无法追溯 | `logging` 模块 |
| `# type: ignore` | 隐藏类型问题 | 修复类型注解 |

---

## 🤖 LLM 集成规范

### 5 阶段流水线架构

```
market_analyzer → sector_analyzer → stock_classifier → exposure_planner → report_composer
```

| 阶段 | 输入 | 输出 | 校验要求 |
|------|------|------|---------|
| `market_analyzer` | 市场特征、VIX、宏观 | `risk_level`, `bias`, `drivers` | JSON Schema |
| `sector_analyzer` | 板块特征、新闻 | `leading`, `lagging` | JSON Schema |
| `stock_classifier` | 个股特征、趋势 | `categories` | Ticker 白名单 |
| `exposure_planner` | 持仓、前阶段结果 | `allocation_plan` | JSON Schema |
| `report_composer` | 所有前阶段结果 | `markdown`, `sections` | 非空校验 |

### 提示词模板规范

所有提示词文件位于 `configs/prompts/`，命名格式：`deepseek_<stage>.md`

**必需元素**：
1. 任务描述
2. 输入数据说明
3. 输出 JSON Schema
4. 示例输出
5. 约束条件

```markdown
# 市场分析阶段

## 任务
分析当前市场风险状态...

## 输入数据
- `market`: 市场指标
- `vix_zscore`: VIX Z 分数
...

## 输出格式
```json
{
  "risk_level": "low|medium|high|extreme",
  "bias": "bullish|neutral|bearish",
  "drivers": ["driver1", "driver2"],
  "summary": "..."
}
```

## 约束
- 必须使用提供的数据
- 禁止编造 Ticker
```

### JSON Schema 校验

所有 LLM 输出必须通过 `validators/json_schemas.py` 中定义的 Schema 校验：

```python
SCHEMAS = {
    "market_analyzer": {
        "type": "object",
        "required": ["risk_level", "bias", "drivers"],
        "properties": {
            "risk_level": {"enum": ["low", "medium", "high", "extreme"]},
            "bias": {"enum": ["bullish", "neutral", "bearish"]},
            "drivers": {"type": "array", "items": {"type": "string"}},
        },
    },
    # ...
}
```

### Safe Mode 回退

当 LLM 调用失败时，系统进入 Safe Mode：

```python
@dataclass
class SafeModeConfig:
    on_llm_failure: str = "no_new_risk"  # 禁止新增风险敞口
    max_exposure_cap: float = 0.4         # 最大敞口降至 40%
```

**Safe Mode 行为**：
- 所有新买入建议变为 Hold
- 目标敞口不超过 `max_exposure_cap`
- 报告标注"Safe Mode 启用"
- 错误详情写入 `errors.jsonl`

---

## 📊 数据流规范

### 主数据流

```
[数据采集层]
yf_client.fetch_history()     → 行情数据 (pd.DataFrame)
yf_client.fetch_news()        → 新闻数据 (List[Dict])
fred_client.fetch_series()    → 宏观数据 (pd.DataFrame)
cboe_client.fetch_put_call()  → 期权数据 (Dict)
            │
            v
[特征工程层]
prepare_feature_sets()
            │
            ├── market_features    (Dict)
            ├── sector_features    (Dict[str, Dict])
            ├── stock_features     (Dict[str, Dict])
            ├── premarket_flags    (Dict[str, Dict])
            ├── news_bundle        (Dict)
            ├── trend_bundle       (Dict)
            └── macro_flags        (Dict)
            │
            v
[决策层]
StockDecisionEngine.score_stocks() → List[Dict]
            │
            v
[LLM 层]
LLMOrchestrator.run() → LLMRunResult
            │
            v
[报告层]
HybridReportBuilder.build() → report.md, report.json
```

### 特征字典规范

**stock_features 结构**：

```python
{
    "NVDA": {
        # 技术指标（必需）
        "rsi_norm": 0.65,          # float, 0-1
        "macd_signal": 0.02,       # float
        "trend_slope": 0.003,      # float
        "atr_pct": 0.025,          # float, > 0
        "price": 145.50,           # float, > 0

        # 趋势指标（必需）
        "trend_strength": 0.7,     # float, 0-1
        "trend_state": "uptrend",  # "uptrend"|"downtrend"|"flat"
        "momentum_10d": 0.08,      # float

        # 新闻情绪（可选）
        "news_score": 0.3,         # float, -1 to 1
        "recent_news": [...],      # List[Dict]

        # 持仓信息（运行时注入）
        "position_shares": 100,    # float
        "position_value": 14550,   # float
    }
}
```

### 缓存策略

| 数据类型 | 缓存位置 | TTL | 回退策略 |
|---------|---------|-----|---------|
| 行情历史 | `storage/cache/yf/*.parquet` | 1 天 | 合成数据 |
| 新闻 | `storage/cache/yf/news/*.json` | 3 小时 | 合成新闻 |
| FRED 数据 | `storage/cache/fred/*.json` | 7 天 | 上次缓存 |
| 报价 | 内存缓存 | 5 分钟 | 空字典 |

---

## 🧪 测试规范

### 测试文件命名

```
tests/
├── test_positions.py       # 测试 portfolio_manager/positions.py
├── test_sizer.py           # 测试 position_sizer/sizer.py
├── test_yf_client.py       # 测试 data_collector/yf_client.py
├── test_llm_parsing.py     # 测试 LLM 响应解析
├── test_llm_schemas.py     # 测试 JSON Schema 校验
└── ...
```

### 测试结构

```python
import pytest
from ai_trader_assist.portfolio_manager.state import PortfolioState, Position


class TestPortfolioState:
    """PortfolioState 单元测试。"""

    def test_empty_state(self):
        """空状态初始化。"""
        state = PortfolioState()
        assert state.cash == 0.0
        assert state.positions == []
        assert state.market_value == 0.0

    def test_add_position(self):
        """添加持仓。"""
        state = PortfolioState(cash=10000.0)
        state.apply_operations([
            {"symbol": "AAPL", "action": "BUY", "shares": 10, "price": 150.0}
        ])
        assert len(state.positions) == 1
        assert state.cash == 10000.0 - 10 * 150.0

    @pytest.mark.parametrize("action,expected", [
        ("BUY", 100),
        ("SELL", -100),
    ])
    def test_action_types(self, action, expected):
        """测试不同操作类型。"""
        ...
```

### Mock 与离线测试

```python
from unittest.mock import Mock, patch

def test_fetch_history_offline():
    """离线模式下返回缓存数据。"""
    with patch("yfinance.download") as mock_download:
        mock_download.side_effect = Exception("Network error")
        client = YahooFinanceClient(cache_dir=Path("/tmp/test_cache"))
        # 应该返回缓存或合成数据
        df = client.fetch_history("AAPL", start, end)
        assert not df.empty
```

### 测试覆盖率要求

- **核心模块**（`portfolio_manager/`, `position_sizer/`）：≥ 80%
- **数据采集**（`data_collector/`）：≥ 60%（网络依赖）
- **LLM 相关**（`llm/`, `llm_operators/`）：≥ 70%

---

## 📝 Git 提交规范

### Commit Message 格式

```
<type>: <description>

[optional body]

[optional footer]
```

### Type 类型

| Type | 说明 | 示例 |
|------|------|------|
| `feat` | 新功能 | `feat: add CBOE put/call ratio data source` |
| `fix` | Bug 修复 | `fix: handle empty news response in yf_client` |
| `docs` | 文档更新 | `docs: update LLM operator configuration guide` |
| `refactor` | 重构 | `refactor: extract common validation logic` |
| `test` | 测试用例 | `test: add unit tests for position sizer` |
| `chore` | 构建/工具 | `chore: update requirements.txt` |
| `perf` | 性能优化 | `perf: cache compiled regex patterns` |

### 分支规范

- `main` - 主分支（稳定版本）
- `feature/*` - 功能分支
- `fix/*` - 修复分支
- `docs/*` - 文档分支

---

## 🔐 环境变量规范

### 必需变量

| 变量 | 说明 | 示例 |
|------|------|------|
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥（必填） | `sk-xxx...` |

### 可选变量

| 变量 | 说明 | 默认值 |
|------|------|-------|
| `FRED_API_KEY` | FRED API 密钥 | 空（匿名访问） |
| `DEEPSEEK_MODEL` | DeepSeek 模型名 | `deepseek-chat` |
| `DEEPSEEK_API_URL` | API 入口地址 | `https://api.deepseek.com/v1/chat/completions` |
| `DEEPSEEK_TIMEOUT` | 请求超时（秒） | `90` |
| `DEEPSEEK_MAX_TOKENS` | 最大 Token 数 | `8192` |
| `TZ` | 时区 | `America/Los_Angeles` |

### 阶段特定模型覆盖

```bash
# 为特定 LLM 阶段使用不同模型
DEEPSEEK_MODEL_MARKET_ANALYZER=deepseek-reasoner
DEEPSEEK_MODEL_REPORT_COMPOSER=deepseek-coder
```

### .env.example 模板

```bash
# 必填
DEEPSEEK_API_KEY=

# 可选
FRED_API_KEY=
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_API_URL=https://api.deepseek.com/v1/chat/completions
DEEPSEEK_TIMEOUT=90
DEEPSEEK_MAX_TOKENS=8192
TZ=America/Los_Angeles
```

---

## ✅ 验证检查清单

在提交代码前，确保：

### 代码质量
- [ ] 所有公开函数有类型注解
- [ ] 所有公开函数有 Docstring
- [ ] 无裸 `except:` 语句
- [ ] 无 `# type: ignore` 注释
- [ ] 无硬编码路径或密钥

### 测试
- [ ] 新功能有对应测试用例
- [ ] `pytest tests -q` 全部通过
- [ ] 核心模块覆盖率 ≥ 80%

### 文档
- [ ] README.md 已更新（如有必要）
- [ ] AGENTS.md 已更新（如有必要）
- [ ] 配置字段有注释说明

### Git
- [ ] Commit message 符合规范
- [ ] 无敏感信息提交
- [ ] 无 `.env` 文件提交

---

## 📚 参考文档

- **Agent 开发指南**：`AGENTS.md`
- **项目说明**：`README.md`
- **主配置文件**：`configs/base.json`
- **LLM 提示词**：`configs/prompts/`

---

**最后更新**: 2026-01-20  
**维护者**: Project Team  
**版本**: 1.0.0  
**状态**: ✅ 强制执行
