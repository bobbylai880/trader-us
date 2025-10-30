# AI Trader Assist

AI Trader Assist 是一个参考 HKUDS/AI-Trader Base 模式实现的**半自动化盘前决策系统**。系统在每日美股开盘前（PT）串联数据采集、风险评估、板块与个股评分、头寸规划以及报告生成环节，输出 Markdown 与 JSON 版本的人工执行清单。整套流程保持“人机协同”：由系统提供建议，人工确认后再执行与复盘。

> ⚠️ **免责声明**：本项目仅用于研究与教学，不连接任何券商系统，不构成投资建议。请在人工复核后再执行所有交易。

---

## 目录结构速览

```
.
├── ai_trader_assist/       # Python 包：宏观→决策→仓位→报告主流水线
│   ├── agent/              # 风险→决策→仓位→报告 orchestrator
│   ├── agent_tools/        # 行情缓存、账本模拟、数学函数
│   ├── data_collector/     # yfinance 与 FRED 抽象层
│   ├── feature_engineering/# 指标工程 (RSI/MACD/ATR/趋势 等)
│   ├── decision_engine/    # 板块与个股评分逻辑
│   ├── position_sizer/     # 仓位与股数分配
│   ├── portfolio_manager/  # 基于人工日志维护持仓
│   ├── report_builder/     # Markdown/JSON 报告输出
│   └── jobs/               # 每日调度脚本 (run_daily 等)
├── configs/                # 运行参数、提示词模板、股票池名单
├── storage/                # 操作日志、持仓快照与每日归档
├── tests/                  # pytest 用例
├── README.md               # 使用说明 (当前文件)
└── requirements.txt        # Python 依赖列表

**默认股票池速览**

- 指数/ETF：SPY、QQQ、DIA、IWM、VIX 以及 11 支 SPDR 板块 ETF。
- 个股：AAPL、AMZN、AMD、META、MSFT、NVDA（可在配置中扩展）。
- 可在 `configs/base.json` 中调整上述列表；所有新增标的会自动纳入数据采集与评分流程。
```

---

## 环境准备

1. **Python 版本**：推荐 Python 3.12。
2. **虚拟环境**：
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows 使用 .venv\Scripts\activate
   ```
3. **安装依赖**：
   ```bash
   pip install -r requirements.txt
   ```
4. **环境变量**：
   ```bash
   cp .env.example .env
   ```
   - `FRED_API_KEY`：可选；若为空则尝试匿名访问（部分频率有限制）。
  - `DEEPSEEK_API_KEY`：DeepSeek LLM 的访问令牌，用于盘前分析提示词调用（必填）。
  - `DEEPSEEK_MODEL`：可选，自定义 DeepSeek 模型名称，默认为 `deepseek-chat`。
  - `DEEPSEEK_API_URL`：可选，自定义 API 入口地址，默认为官方 `https://api.deepseek.com/v1/chat/completions`。
  - `DEEPSEEK_TIMEOUT`：可选，覆盖请求超时时间（秒），默认 90。
  - `DEEPSEEK_MAX_TOKENS`：可选，限制每次生成的最大 Token 数，默认 8192，对齐 DeepSeek 当前接口上限以避免请求被拒绝。
   - `TZ`：默认 `America/Los_Angeles`，用于统一时区。

---

## 核心配置 `configs/base.json`

主要字段说明：

| 字段 | 说明 |
| --- | --- |
| `universe.sectors` | 参与板块评分的 SPDR ETF 列表。 |
| `universe.watchlist` | 个股候选池（结合板块领先情况共同打分）。 |
| `limits.max_exposure` | 组合最大仓位（占总权益的比例）。 |
| `limits.max_single_weight` | 单只股票的权重上限。 |
| `limits.min_ticket` | 最小加仓预算，低于该金额则跳过买入。 |
| `risk.cooling_days` | 连续亏损后冷却期天数。 |
| `risk.earnings_blackout` | 是否在财报窗口自动加入黑名单。 |
| `risk_constraints.risk_budget` | 风险预算配置（如日/周亏损上限、最大回撤），提供给 LLM 仓位审查阶段引用。 |
| `risk_constraints.var_limits` | 组合 VaR/CVaR 限制（自定义键值对，例如 `portfolio_var_95_pct`），用于提示词中的风险约束。 |
| `sizer.k1_stop/k2_target` | 止损与止盈的 ATR 系数。 |
| `trend.*` | 趋势特征窗口（近 5/20 日斜率、10 日动量、均线、波动率窗口等）。 |
| `macro.*` | FRED 指标配置与回溯天数，可在 `series` 中增删宏观序列或调整 `lookback_days`。 |
| `llm.max_stock_payload` | 向 LLM 提供的个股数量上限（默认 8），超出部分将按顺序截断，避免提示词过长。 |
| `llm.operators.*` | 定义五个分阶段推理算子的提示词文件与重试次数，例如 `market_analyzer.retries=1`。 |
| `llm.guardrails` | JSON Schema 校验与 ticker 审查等硬约束参数，异常时会触发自动重试。 |
| `llm.safe_mode` | LLM 失败时的安全回退策略（禁止新增风险、限制目标仓位等）。 |
| `logging.log_dir / operations_path / positions_path` | 日志输出与手工操作记录、持仓快照路径，可按需调整。 |
| `schedule.*` | 每日关键节点（05:30 数据/06:10 报告等）。 |

如需自定义不同市场或股票池，可复制 `base.json` 创建新的配置文件，并在运行脚本时通过环境变量或命令参数指定。

---

## 每日运行流程（建议在 PT 05:30 附近执行）

1. **手工确认昨日操作日志**：确保 `storage/operations.jsonl` 已追加上一交易日的实际成交。
2. **运行调度脚本**：
   ```bash
   python -m ai_trader_assist.jobs.run_daily \
     --config configs/base.json \
     --output-dir storage/daily_$(date +%F)
   ```
   - `--config`：可替换为自定义配置。
   - `--output-dir`：当日输出目录，默认会按日期创建。
   - 日志：默认写入 `storage/logs/run_daily_<时间戳>.log` 并同步输出到终端，可通过 `--log-dir` 自定义目录，使用 `--verbose` 输出 DEBUG 级别信息，或通过 `--quiet` 将终端输出限制为 ERROR。
   - 调度脚本会自动读取 `.env` 或系统环境变量中的 `FRED_API_KEY`，并通过 `yfinance` 与 FRED 下载最近约 1 年的行情与宏观数据。若网络受限，脚本会优先使用本地缓存并在输出目录记录异常。
3. **脚本输出**：
   - `report.md`：面向人工的 Markdown 摘要。
   - `report.json`：结构化操作清单、目标敞口与风控信息。
  - `llm_analysis.json`：分阶段 DeepSeek（或其替代方案）分析结果，含市场/板块/个股/敞口摘要与最终汇总。
  - `llm/llm_<date>/step_*`：每个 LLM 阶段的输入、输出与原始响应，便于审计及 Prompt 迭代（仅当启用分阶段编排时生成）。
  - `market_features.json`、`sector_features.json`、`stock_features.json`、`premarket_flags.json`：原始特征快照与盘前风险评估，现已包含实时 VIX 数值 (`vix_value`) 与 Z 分数 (`vix_zscore`) 等关键指标，便于复盘与调试。
  - `macro_flags.json`：从 FRED 缓存的宏观指标（如 CPI、收益率曲线、联邦基金利率、M2、失业率、工业产出）及其环比变动，供 LLM 与人工审阅市场风险信号。
  - `trend_features.json`：指数、板块、个股的趋势强度、动量、波动率趋势等量化指标（`trend_strength`、`momentum_10d`、`volatility_trend`）。
  - `news_bundle.json` 与 `news_snapshot.json`：市场/板块/个股新闻摘要及其情绪评分，供人工快速追踪事件驱动。
4. **人工复核与执行**：根据报告在券商端手动下单。
5. **收盘后/次日早晨**：将实际执行记录追加到 `storage/operations.jsonl`，并确认 `storage/positions.json` 是否更新。

> 提示：如需历史回溯，可对旧日期运行 `run_daily.py --as-of YYYY-MM-DD`（若已实现），或手动指定 `--date` 以复现当日建议。

---

## 盘后录入操作记录（CLI）

收盘后运行交互式脚本，将人工执行的真实成交写入操作日志并同步更新持仓：

```bash
python -m ai_trader_assist.jobs.record_operations --config configs/base.json
```

- 默认会在项目根目录读取 `logging.operations_path`（默认为 `storage/operations.jsonl`）并追加写入，每条操作独占一行 JSON。
- 输入内容包含日期、代码、动作（BUY/SELL/REDUCE/HOLD）、数量、价格与可选备注；系统会自动补充 `source="manual"` 与 UTC 时间戳。
- 写入前会将旧文件备份为 `*.bak`，防止误操作导致数据损坏。
- 保存后立即加载 `logging.positions_path`（默认为 `storage/positions.json`），按最新操作更新现金、均价与股数，并写回 `last_updated` 时间戳。

示例会话：

```
=== Record Today's Operations ===
Date (default: 2025-10-28):
Symbol (e.g., NVDA): NVDA
Action [BUY/SELL/REDUCE/HOLD]: BUY
Quantity: 30
Price: 1098.2
Reason (optional): trend breakout
Add another? (y/n): y

Symbol (e.g., NVDA): AAPL
Action [BUY/SELL/REDUCE/HOLD]: SELL
Quantity: 20
Price: 228.5
Reason (optional): rebalance
Add another? (y/n): n
Save and update positions? (y/n): y
✅ 2 operations appended to operations.jsonl
✅ positions.json updated successfully
```

生成的日志与持仓片段如下：

```json
{"date":"2025-10-28","symbol":"NVDA","action":"BUY","quantity":30,"price":1098.2,"reason":"trend breakout","source":"manual","timestamp":"2025-10-28T06:30:00+00:00"}
```

```json
{
  "date": "2025-10-28",
  "cash": 18240.0,
  "positions": [
    {"symbol": "NVDA", "shares": 130, "avg_cost": 1090.2},
    {"symbol": "AAPL", "shares": 80, "avg_cost": 226.5}
  ],
  "equity_value": 159846.0,
  "exposure": 0.85,
  "last_updated": "2025-10-28T06:30:00+00:00"
}
```

> 建议：若需要批量修正历史记录，可手动编辑 `operations.jsonl`，再运行 `record_operations` 或 `run_daily` 让系统重新计算持仓。

---

## 持仓复盘与盈亏报告（CLI）

每日收盘或次日盘前，可快速生成历史持仓与实时盈亏视图：

```bash
python -m ai_trader_assist.jobs.report_portfolio --config configs/base.json --as-of 2025-10-28
```

- 默认会读取 `logging.operations_path` 与 `logging.positions_path`，聚合 `operations.jsonl` 全量操作，结合最新 `positions.json` 校准现金余额。
- 价格优先来自 `storage/daily_<date>/stock_features.json` 中缓存的收盘价；若缺失且未指定 `--no-fetch`，则回落至 Yahoo Finance 历史行情。
- 结果写入 `storage/reports/<date>/`：`current_pnl.json`（逐持仓盈亏）、`history_report.json`（每日快照序列）、`portfolio_report.md`（Markdown 摘要）。
- Markdown 会列出当前仓位表、累计盈亏、平均敞口，以及历史持仓变化，便于盘后复盘与风控留档。

> 小贴士：若只想在终端查看结果，可追加 `--output-dir /tmp/report --no-fetch`，或修改 `configs/base.json` 的 `logging` 段落以统一存储位置。

---

## 数据与缓存

| 模块 | 数据源 | 说明 |
| --- | --- | --- |
| `data_collector.yf_client` | `yfinance` | 默认获取日线历史与近一日的盘前价格估计，支持本地缓存以减少请求。 |
| `data_collector.fred_client` | FRED | 拉取宏观序列（如 10Y/2Y 国债利差、CPI），频率低，建议本地缓存 JSON。 |
| `feature_engineering.indicators` | 本地计算 | 对历史价格序列计算 RSI、MACD、ATR、VWAP、z-score、斜率等。 |
| `feature_engineering.trend_features` | 本地计算 | 近 5/20 日斜率、10 日动量、波动率趋势、均线交叉等趋势指标，输出到 `trend_features.json`。 |
| `feature_engineering.pipeline` | `yfinance.news` | 聚合指数、板块、个股新闻并计算关键词情绪分数，默认缓存 3 小时。 |

### 新闻聚合与情绪分数

- 通过 `yfinance.Ticker(symbol).news` 抓取市场（SPY、QQQ）、板块 ETF 与个股的新闻，字段包含标题、摘要、发布方与时间戳。
- 数据默认缓存 3 小时；若离线运行，会生成可追溯的合成新闻以维持流程完整性。
- 指定 `--date YYYY-MM-DD` 回测时，流水线会将该日期的盘后时间戳作为 `as_of` 参数，仅保留当日及其向前 7 日的新闻，避免报告混入未来事件；如需更长窗口，可在调用 `YahooFinanceClient.fetch_news(..., lookback_days=)` 时覆盖默认值。
- `feature_engineering.pipeline.prepare_feature_sets` 会为每个层级计算关键词情绪分数（-1~1）并输出到 `news_bundle.json`，供 LLM 与人工复核。
- 报告与 LLM 摘要会引用这些新闻条目，确保量化指标与事件驱动双线结合。

### 趋势指标快照

- `feature_engineering.trend_features.compute_trend_features` 会为市场（SPY/QQQ）、板块 ETF 及个股分别计算：
  - `trend_slope_5d` / `trend_slope_20d`：近 5/20 个交易日的线性回归斜率；
  - `momentum_10d`：近 10 日累计涨幅；
  - `volatility_trend`：短期波动率与 20 日波动率之比，用于识别波动放大；
  - `moving_avg_cross`：10 日与 30 日均线是否发生金叉/死叉（1/-1/0）；
  - `trend_strength`、`trend_state`、`momentum_state`：综合趋势方向、稳定性与动量变化。
- 结果存入 `trend_features.json` 并被 `stock_features.json`、`sector_features.json` 以及 LLM 提示词消费，用于解释趋势强度与回调风险。

---

## LLM 提示词配置

LLM 推理拆分为四个分析阶段与一个终稿阶段，对应以下模板（均位于 `configs/prompts/`）：

| 步骤 | 文件 | 作用 |
| --- | --- | --- |
| 市场解读 | `deepseek_market_overview.md` | 概括市场风险等级、倾向与关键驱动因子。 |
| 板块说明 | `deepseek_sector_analysis.md` | 解释领先/落后板块与数据证据。 |
| 个股分类 | `deepseek_stock_actions.md` | 依据信号将个股划分为 Buy/Hold/Reduce/Avoid 并阐述驱动与风险。 |
| 仓位审查 | `deepseek_exposure_check.md` | 对比当前敞口与目标敞口并提出调仓方向。 |
| 报告整合 | `deepseek_report_compose.md` | 结合前述结论生成 Markdown 盘前报告并列出异常。 |

`configs/base.json` 默认引用这些模板路径；如需自定义，可在派生配置中覆盖 `llm.prompt_files` 对应键值，或直接在 `llm.operators` 下为每个阶段指定新的提示词。

- 为控制请求体大小与响应时长，系统仅向 LLM 发送按监控列表顺序截取的前 8 只个股，并将每条新闻的 `summary`/`content` 裁剪至约 400 字符；可通过 `llm.max_stock_payload` 与 `DEEPSEEK_MAX_TOKENS` 自行调整。
- `llm.guardrails.reject_on_hallucinated_tickers=true` 将拒绝非监控列表的 ticker，触发自动重试；连续失败后将进入 `llm.safe_mode`，输出“无新增风险”的保守建议并强行压低目标仓位。

### 输出结构规范

- 所有 DeepSeek 提示词现统一要求 **JSON 输出**，禁止返回纯文本或 Markdown；通用准则收录在 `deepseek_base_prompt.md` 中。
- `deepseek_market_overview.md`：返回包含 `risk_level`、`bias`、`summary`、`drivers`、`premarket_flags`、`news_sentiment`、`news_highlights`、`data_gaps` 的对象。
- `deepseek_sector_analysis.md`：输出 `leading`、`lagging`、`focus_points`、`data_gaps` 字段，每个板块条目需附带量化证据与新闻摘要。
- `deepseek_stock_actions.md`：以 `categories` 字典给出 Buy/Hold/Reduce/Avoid 列表，并列出 `drivers`、`risks`、`premarket_score`、`news_highlights`（每股最多 1 条）、`trend_change`、`momentum_strength`、`trend_explanation` 等客观指标。
- `deepseek_exposure_check.md`：返回敞口差异与 `allocation_plan`、`constraints` 建议，便于对接头寸引擎。
- `deepseek_report_compose.md`：输出包含 `markdown` 正文与 `sections` 摘要（含 `news` 列表）的对象，同时合并所有数据缺口至 `data_gaps`。

调用 DeepSeek 前，请确保 `.env` 或运行环境中设置了 `DEEPSEEK_API_KEY`。可选变量 `DEEPSEEK_MODEL` 与 `DEEPSEEK_API_URL` 用于覆盖默认模型与接口地址。若 API 返回错误、超时或输出无法解析，流水线会抛出异常（不会再回退到模拟结果），请在定时任务中捕获并记录日志。

若某些数据缺失或 API 访问失败，流水线会在报告末尾记录异常条目，确保人工注意补充或回溯。

---

## 组合与操作日志

- `storage/operations.jsonl`：按日期存放人工操作批次，每行一个 JSON 对象，包含买卖动作、价格、股数、备注等。
- `storage/positions.json`：系统自动维护的最新持仓快照，含现金、持仓股票、均价、组合敞口。
- `portfolio_manager/state.py`：核心状态机，读取旧快照 + 新操作，结合收盘价计算新仓位。

保持日志的完整性与时间顺序对仓位计算至关重要，建议每日备份。

---

## 运行测试

项目提供基础验收用例，确保关键模块在改动后仍然稳定：

```bash
pytest tests -q
```

- `test_macro_engine.py`：检查目标仓位是否落在 0.4–0.8 区间，并包含驱动因素。
- `test_sizer.py`：验证买入预算分配与期望增量资金误差 < 5%。
- `test_positions.py`：模拟操作日志，确认现金、持仓与均价计算正确。

### 端到端验证示例

若需在本地确认数据采集、新闻管线与 LLM 分析完整串联，可在命令行临时注入真实的 API Key 后运行每日作业。以下命令以 2025-10-27 为例，生成完整的盘前 artefacts 供人工校验：

```bash
FRED_API_KEY="<your_fred_key>" \
DEEPSEEK_API_KEY="<your_deepseek_key>" \
python -m ai_trader_assist.jobs.run_daily \
  --config configs/base.json \
  --output-dir storage/daily_full_check \
  --date 2025-10-27
```

执行完成后可在输出目录中找到 `report.md`、`llm_analysis.json`、`news_bundle.json`、`trend_features.json` 等文件，并在日志目录 `storage/logs/` 查看带时间戳的执行记录，验证整套流水线在真实密钥下的表现。

---

## 常见扩展指引

- **增加新股票池/板块**：在配置文件的 `universe` 区域补充，并确保数据源可覆盖。
- **接入更多数据源**：在 `data_collector/` 目录新增客户端，并在 `agent/base_agent.py` 注入。
- **引入情绪/新闻因子**：可在 `decision_engine/stock_scoring.py` 内添加新的特征权重。
- **部署到调度器（如 cron）**：创建虚拟环境后在 cron 中调用 `run_daily.py`，注意时区设定与日志输出。

---

## 故障排查

| 现象 | 可能原因 | 解决方案 |
| --- | --- | --- |
| `yfinance` 请求失败 | 网络波动或 API 限流 | 重试、启用本地缓存、减少请求频率。 |
| FRED 返回 403 | 未配置或过期的 API Key | 检查 `.env` 中的 `FRED_API_KEY` 是否有效。 |
| 报告缺少板块/个股评分 | 数据不足或指标计算失败 | 查看日志中的异常字段，补充历史数据或调整阈值。 |
| 组合敞口异常 | 操作日志顺序错误 | 确认 `operations.jsonl` 是否按日期递增，避免重复记录。 |

---

## 参考与致谢

- [HKUDS/AI-Trader](https://github.com/HKUDS/AI-Trader) Base 模式的架构理念。
- [yfinance](https://github.com/ranaroussi/yfinance)、[FRED API](https://fred.stlouisfed.org/docs/api/fred/) 免费数据源。
- 指标计算基于 `pandas`/`numpy`，如需扩展可参考 `ta` 等开源库。

欢迎根据自身策略需求扩展模型、增强风险约束或引入可视化面板。记得保持人工复核与风控纪律。

---

## 快速体验示例

以下命令演示如何在“零持仓 + 仅关注 AMD”假设下生成当日盘前报告。请在运行前确认 `.env` 中写入真实的 API Key，或通过命令行临时注入：

```bash
FRED_API_KEY="<your_fred_key>" \
DEEPSEEK_API_KEY="<your_deepseek_key>" \
python -m ai_trader_assist.jobs.run_daily \
  --config configs/base.json \
  --output-dir storage/daily_demo
```

运行完成后，可在 `storage/daily_demo/` 中找到完整的特征快照与盘前报告；若需查看逐步评分过程，可结合 `*_features.json` 与 `configs/prompts/` 手动复盘 LLM 的输入输出。若当日已无最新行情，可通过 `--date YYYY-MM-DD` 指定历史交易日。
