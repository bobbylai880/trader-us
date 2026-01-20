---
name: sector-analysis
description: 分析 11 个 SPDR 板块 ETF 的相对强弱和轮动趋势。触发词：板块分析、sector analysis、哪些板块强、板块轮动、领先板块
license: MIT
compatibility: opencode
metadata:
  category: trading
  workflow: daily-analysis
---

# Sector Analysis Skill

分析 11 个 SPDR 板块 ETF 的相对强弱，识别领先和落后板块。

## 触发条件

- 用户询问板块表现
- 用户请求板块轮动分析
- 每日盘前分析流程的第二步

## 板块 ETF 列表

| ETF | 板块 | 说明 |
|-----|------|------|
| XLK | 科技 | Technology |
| XLF | 金融 | Financials |
| XLV | 医疗 | Health Care |
| XLE | 能源 | Energy |
| XLI | 工业 | Industrials |
| XLY | 消费 | Consumer Discretionary |
| XLP | 必需消费 | Consumer Staples |
| XLU | 公用事业 | Utilities |
| XLB | 材料 | Materials |
| XLRE | 房地产 | Real Estate |
| XLC | 通信 | Communication Services |

## 执行步骤

### 1. 获取板块数据

```
1. get_quotes(["XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"])
2. 对每个板块调用 calc_indicators(symbol, ["rsi", "macd", "sma_50"])
3. get_quotes(["SPY"]) - 作为基准
```

### 2. 计算相对强度

对每个板块计算相对 SPY 的表现：
- 1 日相对强度 = (板块涨跌% - SPY涨跌%)
- RSI 相对强度 = 板块 RSI vs 50 中轴
- 趋势强度 = 价格 vs SMA50 距离

### 3. 板块分类

**领先板块（Leading）**: 
- 相对强度 > 0.5%
- RSI > 55
- 价格在 SMA50 之上

**落后板块（Lagging）**:
- 相对强度 < -0.5%
- RSI < 45
- 价格在 SMA50 之下

**中性板块（Neutral）**:
- 介于两者之间

### 4. 输出格式

```markdown
## 板块分析结果

**基准**: SPY [当日涨跌%]

### 领先板块 🟢

| 板块 | 涨跌% | 相对强度 | RSI | vs SMA50 |
|------|-------|----------|-----|----------|
| XLK | +1.2% | +0.8% | 62 | +3.2% |

### 落后板块 🔴

| 板块 | 涨跌% | 相对强度 | RSI | vs SMA50 |
|------|-------|----------|-----|----------|
| XLE | -0.8% | -1.2% | 38 | -2.1% |

### 轮动信号

- 资金从 [落后板块] 流向 [领先板块]
- [观察到的轮动模式]

### 投资建议

- **超配**: [板块列表]
- **低配**: [板块列表]
```

## 板块轮动解读

| 领先板块 | 市场阶段 | 含义 |
|----------|----------|------|
| XLK, XLY | 早期扩张 | 风险偏好回升 |
| XLF, XLI | 中期扩张 | 经济复苏确认 |
| XLE, XLB | 晚期扩张 | 通胀预期上升 |
| XLU, XLP | 防御 | 避险情绪 |
| XLRE | 利率敏感 | 关注利率变化 |

## 后续步骤

完成板块分析后：
1. 根据领先板块筛选个股
2. 继续 `/stock-analysis` 深入分析
