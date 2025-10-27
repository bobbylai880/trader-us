# AI Trader Assist

AI Trader Assist 是一个参考 HKUDS/AI-Trader Base 模式实现的**半自动化盘前决策系统**。系统在每日美股开盘前（PT）串联数据采集、风险评估、板块与个股评分、头寸规划以及报告生成环节，输出 Markdown 与 JSON 版本的人工执行清单。整套流程保持“人机协同”：由系统提供建议，人工确认后再执行与复盘。

> ⚠️ **免责声明**：本项目仅用于研究与教学，不连接任何券商系统，不构成投资建议。请在人工复核后再执行所有交易。

---

## 目录结构速览

```
ai_trader_assist/
├── agent/                  # 宏观→决策→仓位→报告的主流水线
├── agent_tools/            # 基础工具：行情缓存、账本模拟、数学函数
├── configs/                # 运行参数与黑名单配置
├── data_collector/         # yfinance 与 FRED 数据抽象层
├── feature_engineering/    # 技术指标与统计特征
├── risk_engine/            # 市场与盘前风险评估
├── decision_engine/        # 板块与个股评分逻辑
├── position_sizer/         # 仓位与股数分配
├── portfolio_manager/      # 基于人工操作日志的持仓状态机
├── report_builder/         # Markdown/JSON 报告输出
├── jobs/                   # 日常调度脚本（如 run_daily）
├── storage/                # 持仓与操作日志、每日归档目录
└── tests/                  # pytest 用例
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
   pip install -r ai_trader_assist/requirements.txt
   ```
4. **环境变量**：
   ```bash
   cd ai_trader_assist
   cp .env.example .env
   ```
   - `FRED_API_KEY`：可选；若为空则尝试匿名访问（部分频率有限制）。
   - `DEEPSEEK_API_KEY`：DeepSeek LLM 的访问令牌，用于盘前分析提示词调用。
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
| `sizer.k1_stop/k2_target` | 止损与止盈的 ATR 系数。 |
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
   - 调度脚本会自动读取 `.env` 或系统环境变量中的 `FRED_API_KEY`，并通过 `yfinance` 与 FRED 下载最近约 1 年的行情与宏观数据。若网络受限，脚本会优先使用本地缓存并在输出目录记录异常。 
3. **脚本输出**：
   - `risk_flags.json`：市场与盘前风险评估。
   - `actions.json`：建议操作清单（含股数、止损/止盈）。
   - `report.md`：面向人工的 Markdown 摘要。
4. **人工复核与执行**：根据报告在券商端手动下单。
5. **收盘后/次日早晨**：将实际执行记录追加到 `storage/operations.jsonl`，并确认 `storage/positions.json` 是否更新。

> 提示：如需历史回溯，可对旧日期运行 `run_daily.py --as-of YYYY-MM-DD`（若已实现），或手动指定 `--date` 以复现当日建议。

---

## 数据与缓存

| 模块 | 数据源 | 说明 |
| --- | --- | --- |
| `data_collector.yf_client` | `yfinance` | 默认获取日线历史与近一日的盘前价格估计，支持本地缓存以减少请求。 |
| `data_collector.fred_client` | FRED | 拉取宏观序列（如 10Y/2Y 国债利差、CPI），频率低，建议本地缓存 JSON。 |
| `feature_engineering.indicators` | 本地计算 | 对历史价格序列计算 RSI、MACD、ATR、VWAP、z-score、斜率等。 |

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

`configs/base.json` 默认引用这些模板路径；如需自定义，可在派生配置中覆盖 `llm.prompt_files` 对应键值。

调用 DeepSeek 前，请确保 `.env` 或运行环境中设置了 `DEEPSEEK_API_KEY`。

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
pytest ai_trader_assist/tests -q
```

- `test_macro_engine.py`：检查目标仓位是否落在 0.4–0.8 区间，并包含驱动因素。
- `test_sizer.py`：验证买入预算分配与期望增量资金误差 < 5%。
- `test_positions.py`：模拟操作日志，确认现金、持仓与均价计算正确。

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
