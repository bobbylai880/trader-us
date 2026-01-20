---
name: daily-report
description: 生成每日盘前分析报告，整合市场、板块、个股分析结果。触发词：生成报告、盘前报告、daily report、今日建议、每日分析
license: MIT
compatibility: opencode
metadata:
  category: trading
  workflow: daily-analysis
---

# Daily Report Skill

生成完整的每日盘前分析报告，整合市场、板块、个股分析结果。

## 触发条件

- 用户请求生成每日报告
- 完成市场、板块、个股分析后的汇总
- 每日盘前流程的最后一步

## 报告结构

### 1. 执行摘要 (Executive Summary)
### 2. 市场概览 (Market Overview)
### 3. 板块轮动 (Sector Rotation)
### 4. 个股分析 (Stock Analysis)
### 5. 操作清单 (Action Items)
### 6. 风险提示 (Risk Warnings)

## 执行步骤

### 1. 收集所有数据

```
# 市场数据
market_quotes = get_quotes(["SPY", "QQQ", "DIA", "IWM"])
macro = get_macro()
pcr = get_pcr()

# 板块数据
sector_quotes = get_quotes(["XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"])

# 个股评分
watchlist = ["NVDA", "AAPL", "MSFT", "AMZN", "META", "AMD"]  # 从配置读取
stock_scores = score_stocks(watchlist)

# 持仓状态
portfolio = get_portfolio()
```

### 2. 综合分析

基于收集的数据：
- 判断市场风险等级
- 识别领先/落后板块
- 筛选 Top 买入候选
- 识别需要减仓的持仓

### 3. 生成操作建议

对于每个操作建议，调用：
```
generate_orders(symbol, action, budget或shares)
```

### 4. 输出格式

```markdown
# 盘前分析报告

**日期**: YYYY-MM-DD
**生成时间**: HH:MM PT
**市场阶段**: 盘前/开盘/收盘

---

## 📊 执行摘要

| 项目 | 状态 |
|------|------|
| 市场风险 | 🟢低 / 🟡中 / 🔴高 |
| 市场倾向 | Risk-On / Neutral / Risk-Off |
| 建议仓位 | XX% |
| 操作数量 | X 个 |

**今日重点**:
1. [重点1]
2. [重点2]

---

## 🌍 市场概览

### 主要指数

| 指数 | 价格 | 涨跌 | 状态 |
|------|------|------|------|
| SPY | $XXX | +X.XX% | |
| QQQ | $XXX | +X.XX% | |

### 关键指标

| 指标 | 数值 | 解读 |
|------|------|------|
| VIX | XX.X | |
| PCR | X.XX | |

---

## 🏭 板块轮动

### 领先板块
- XLK (科技): +X.X%
- ...

### 落后板块
- XLE (能源): -X.X%
- ...

---

## 📈 个股分析

### 买入候选 🟢

| 股票 | 评分 | 价格 | 建议 | 理由 |
|------|------|------|------|------|
| NVDA | 85 | $XXX | BUY | ... |

### 持有观察 🟡

| 股票 | 评分 | 持仓 | 建议 |
|------|------|------|------|

### 减仓/回避 🔴

| 股票 | 评分 | 建议 | 理由 |
|------|------|------|------|

---

## ✅ 操作清单

### 买入

| 股票 | 股数 | 价格 | 金额 | 止损 | 止盈 |
|------|------|------|------|------|------|
| NVDA | XX | $XXX | $X,XXX | $XXX | $XXX |

### 卖出/减仓

| 股票 | 股数 | 价格 | 金额 | 理由 |
|------|------|------|------|------|

---

## ⚠️ 风险提示

1. [风险1]
2. [风险2]

### 数据缺口

- [如有数据获取失败，在此列出]

---

## 📋 当前持仓

| 股票 | 股数 | 成本 | 市值 | 盈亏 | 权重 |
|------|------|------|------|------|------|

**总权益**: $XXX,XXX
**当前仓位**: XX%
**现金**: $XX,XXX

---

*本报告仅供参考，不构成投资建议。请在人工复核后执行交易。*
```

## 报告保存

生成的报告保存到：
```
storage/daily_YYYY-MM-DD/
├── report.md          # Markdown 报告
├── report.json        # 结构化数据
├── market_features.json
├── sector_features.json
├── stock_features.json
└── llm_analysis.json  # 如使用 LLM
```

## 后续步骤

报告生成后：
1. 人工复核报告内容
2. 在券商端执行交易
3. `/record-operation` - 记录实际成交
4. 收盘后 `/show-portfolio` - 查看持仓
