#!/usr/bin/env python3
"""
季度主题情报采集模块

为人工季度判断提供数据支撑:
1. Fed 政策信息 (利率、会议纪要关键词)
2. 财经新闻采集 (板块热度、情绪分析)
3. ETF 资金流向 (板块偏好)
4. 政策/言论追踪 (关税、监管)
5. LLM 主题建议生成
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

import requests
import yfinance as yf


@dataclass
class FedPolicy:
    fed_funds_rate: float
    rate_change_3m: float
    rate_direction: str
    next_meeting: Optional[str]
    market_expectation: str


@dataclass
class SectorFlow:
    etf: str
    sector_name: str
    flow_1w: float
    flow_1m: float
    momentum_20d: float
    relative_strength: float


@dataclass
class NewsItem:
    title: str
    source: str
    date: str
    sentiment: float
    keywords: List[str]


@dataclass
class ThemeIntelligence:
    report_date: str
    fed_policy: FedPolicy
    sector_flows: List[SectorFlow]
    hot_topics: List[str]
    risk_factors: List[str]
    suggested_theme: str
    suggested_focus_sectors: List[str]
    suggested_focus_stocks: List[str]
    suggested_avoid_sectors: List[str]
    reasoning: str


class ThemeIntelligenceCollector:
    
    SECTOR_ETFS = {
        "XLK": "科技", "XLC": "通讯", "XLY": "可选消费",
        "XLF": "金融", "XLV": "医疗", "XLE": "能源",
        "XLI": "工业", "XLP": "必需消费", "XLU": "公用事业",
    }
    
    SECTOR_LEADERS = {
        "XLK": ["NVDA", "AAPL", "MSFT", "AVGO", "AMD", "ADBE", "CRM", "ORCL"],
        "XLC": ["META", "GOOGL", "NFLX", "DIS", "TMUS", "VZ", "T"],
        "XLY": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW"],
        "XLF": ["JPM", "BAC", "WFC", "GS", "MS", "BLK"],
        "XLV": ["UNH", "LLY", "JNJ", "PFE", "MRK", "ABBV"],
        "XLE": ["XOM", "CVX", "COP"],
        "XLI": ["CAT", "DE", "UNP", "HON", "GE"],
        "XLP": ["PG", "KO", "PEP", "COST", "WMT"],
        "XLU": ["NEE", "DUK", "SO"],
    }
    
    POLICY_KEYWORDS = {
        "tariff": ["tariff", "关税", "贸易战", "trade war", "import tax"],
        "fed": ["fed", "fomc", "interest rate", "利率", "降息", "加息", "powell"],
        "ai": ["ai", "artificial intelligence", "人工智能", "chatgpt", "gpu", "nvidia"],
        "china": ["china", "中国", "decoupling", "脱钩", "chip ban"],
        "regulation": ["regulation", "监管", "antitrust", "反垄断"],
        "energy": ["oil", "石油", "能源", "energy", "ev", "电动车"],
        "crypto": ["bitcoin", "crypto", "加密货币"],
    }
    
    def __init__(self):
        self.fred_api_key = os.getenv("FRED_API_KEY")
        self.cache_dir = Path("storage/intelligence_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def collect_fed_policy(self) -> FedPolicy:
        """采集 Fed 政策信息"""
        print("  📊 采集 Fed 政策信息...")
        
        current_rate = 4.50
        rate_3m_ago = 4.75
        rate_change = current_rate - rate_3m_ago
        
        if rate_change < -0.25:
            direction = "降息周期"
            expectation = "市场预期继续降息"
        elif rate_change > 0.25:
            direction = "加息周期"
            expectation = "市场预期维持高利率"
        else:
            direction = "利率平稳"
            expectation = "市场预期暂停调整"
        
        if self.fred_api_key:
            try:
                url = f"https://api.stlouisfed.org/fred/series/observations"
                params = {
                    "series_id": "FEDFUNDS",
                    "api_key": self.fred_api_key,
                    "file_type": "json",
                    "sort_order": "desc",
                    "limit": 6,
                }
                resp = requests.get(url, params=params, timeout=10)
                if resp.ok:
                    data = resp.json()
                    obs = data.get("observations", [])
                    if len(obs) >= 2:
                        current_rate = float(obs[0]["value"])
                        rate_3m_ago = float(obs[min(3, len(obs)-1)]["value"])
                        rate_change = current_rate - rate_3m_ago
                        if rate_change < -0.25:
                            direction = "降息周期"
                            expectation = f"已降息{abs(rate_change):.2f}%，预期继续"
                        elif rate_change > 0.25:
                            direction = "加息周期"
                            expectation = f"已加息{rate_change:.2f}%"
                        else:
                            direction = "利率平稳"
                            expectation = "暂停调整观望"
            except Exception as e:
                print(f"    ⚠️ FRED API 错误: {e}")
        
        return FedPolicy(
            fed_funds_rate=current_rate,
            rate_change_3m=rate_change,
            rate_direction=direction,
            next_meeting="2025-01-29",
            market_expectation=expectation,
        )
    
    def collect_sector_flows(self) -> List[SectorFlow]:
        """采集板块 ETF 资金流向和动量"""
        print("  📊 采集板块资金流向...")
        
        flows = []
        spy = yf.Ticker("SPY")
        spy_hist = spy.history(period="3mo")
        spy_mom = (spy_hist['Close'].iloc[-1] / spy_hist['Close'].iloc[-20] - 1) if len(spy_hist) >= 20 else 0
        
        for etf, name in self.SECTOR_ETFS.items():
            try:
                ticker = yf.Ticker(etf)
                hist = ticker.history(period="3mo")
                
                if len(hist) < 20:
                    continue
                
                mom_20d = (hist['Close'].iloc[-1] / hist['Close'].iloc[-20] - 1)
                rs = mom_20d - spy_mom
                
                vol_recent = hist['Volume'].iloc[-5:].mean()
                vol_prev = hist['Volume'].iloc[-25:-5].mean()
                flow_proxy = (vol_recent / vol_prev - 1) if vol_prev > 0 else 0
                
                flows.append(SectorFlow(
                    etf=etf,
                    sector_name=name,
                    flow_1w=flow_proxy * 0.3,
                    flow_1m=flow_proxy,
                    momentum_20d=mom_20d,
                    relative_strength=rs,
                ))
            except Exception as e:
                print(f"    ⚠️ {etf} 数据获取失败: {e}")
        
        flows.sort(key=lambda x: x.relative_strength, reverse=True)
        return flows
    
    def collect_market_news(self) -> List[NewsItem]:
        """采集市场新闻和热点 - 使用多个免费源"""
        print("  📊 采集市场新闻...")
        
        news_items = []
        
        news_items.extend(self._fetch_yfinance_news())
        news_items.extend(self._fetch_google_news_rss())
        news_items.extend(self._fetch_reuters_rss())
        
        return news_items
    
    def _fetch_yfinance_news(self) -> List[NewsItem]:
        """从 yfinance 获取个股新闻"""
        items = []
        key_tickers = ["NVDA", "AAPL", "MSFT", "TSLA", "META", "GOOGL", "JPM", "XOM"]
        
        for sym in key_tickers[:5]:
            try:
                ticker = yf.Ticker(sym)
                news = ticker.news[:3] if hasattr(ticker, 'news') else []
                
                for item in news:
                    title = item.get("title", "")
                    parsed = self._parse_news_item(title, item.get("publisher", "yfinance"))
                    if parsed:
                        parsed.date = datetime.fromtimestamp(item.get("providerPublishTime", 0)).strftime("%Y-%m-%d")
                        items.append(parsed)
            except Exception:
                pass
        
        return items
    
    def _fetch_google_news_rss(self) -> List[NewsItem]:
        """从 Google News RSS 获取财经新闻 (免费)"""
        items = []
        
        rss_urls = [
            "https://news.google.com/rss/search?q=stock+market&hl=en-US&gl=US&ceid=US:en",
            "https://news.google.com/rss/search?q=nvidia+AI&hl=en-US&gl=US&ceid=US:en",
            "https://news.google.com/rss/search?q=federal+reserve+interest+rate&hl=en-US&gl=US&ceid=US:en",
            "https://news.google.com/rss/search?q=trump+tariff&hl=en-US&gl=US&ceid=US:en",
        ]
        
        for url in rss_urls:
            try:
                resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
                if resp.ok:
                    items.extend(self._parse_rss_xml(resp.text, "Google News"))
            except Exception:
                pass
        
        return items[:15]
    
    def _fetch_reuters_rss(self) -> List[NewsItem]:
        """从 Reuters RSS 获取新闻 (免费)"""
        items = []
        
        rss_urls = [
            "https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best",
        ]
        
        for url in rss_urls:
            try:
                resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
                if resp.ok:
                    items.extend(self._parse_rss_xml(resp.text, "Reuters"))
            except Exception:
                pass
        
        return items[:10]
    
    def _parse_rss_xml(self, xml_text: str, source: str) -> List[NewsItem]:
        """解析 RSS XML"""
        items = []
        
        title_pattern = re.compile(r'<title><!\[CDATA\[(.*?)\]\]></title>|<title>(.*?)</title>', re.DOTALL)
        matches = title_pattern.findall(xml_text)
        
        for match in matches[:10]:
            title = match[0] or match[1]
            title = title.strip()
            if not title or len(title) < 10:
                continue
            if title in ["Google News", "Reuters", "Business & Finance"]:
                continue
            
            parsed = self._parse_news_item(title, source)
            if parsed:
                items.append(parsed)
        
        return items
    
    def _parse_news_item(self, title: str, source: str) -> Optional[NewsItem]:
        """解析单条新闻，提取情绪和关键词"""
        if not title or len(title) < 10:
            return None
        
        title_lower = title.lower()
        
        sentiment = 0.0
        positive = ["surge", "jump", "beat", "record", "growth", "bull", "rally", "soar", "gain", "rise", "high"]
        negative = ["fall", "drop", "miss", "cut", "bear", "crash", "fear", "plunge", "decline", "low", "warn", "risk"]
        for p in positive:
            if p in title_lower:
                sentiment += 0.25
        for n in negative:
            if n in title_lower:
                sentiment -= 0.25
        sentiment = max(-1, min(1, sentiment))
        
        keywords = []
        for topic, kws in self.POLICY_KEYWORDS.items():
            if any(kw in title_lower for kw in kws):
                keywords.append(topic)
        
        return NewsItem(
            title=title[:120],
            source=source,
            date=date.today().isoformat(),
            sentiment=sentiment,
            keywords=keywords,
        )
    
    def analyze_hot_topics(self, news: List[NewsItem]) -> List[str]:
        """分析热点话题"""
        topic_counts: Dict[str, int] = {}
        for item in news:
            for kw in item.keywords:
                topic_counts[kw] = topic_counts.get(kw, 0) + 1
        
        sorted_topics = sorted(topic_counts.items(), key=lambda x: -x[1])
        return [t[0] for t in sorted_topics[:5]]
    
    def identify_risk_factors(self, fed: FedPolicy, flows: List[SectorFlow], news: List[NewsItem]) -> List[str]:
        """识别风险因素"""
        risks = []
        
        if fed.fed_funds_rate > 5.0:
            risks.append("高利率环境持续，成长股承压")
        
        if any("tariff" in item.keywords for item in news):
            risks.append("关税政策不确定性")
        
        if any("china" in item.keywords for item in news):
            risks.append("中美关系紧张")
        
        negative_sectors = [f for f in flows if f.relative_strength < -0.05]
        if len(negative_sectors) >= 3:
            risks.append(f"多板块走弱: {', '.join(s.sector_name for s in negative_sectors[:3])}")
        
        avg_sentiment = sum(n.sentiment for n in news) / len(news) if news else 0
        if avg_sentiment < -0.2:
            risks.append("市场情绪偏悲观")
        
        return risks
    
    def generate_theme_suggestion(
        self, 
        fed: FedPolicy, 
        flows: List[SectorFlow], 
        news: List[NewsItem],
        hot_topics: List[str],
        risks: List[str]
    ) -> tuple:
        """使用 LLM 深度思考生成主题建议"""
        
        print("  🤖 LLM 深度分析中...")
        
        try:
            return self._generate_theme_with_llm(fed, flows, news, hot_topics, risks)
        except Exception as e:
            print(f"    ⚠️ LLM 分析失败: {e}，使用规则回退")
            return self._generate_theme_fallback(fed, flows, hot_topics, risks)
    
    def _generate_theme_with_llm(
        self,
        fed: FedPolicy,
        flows: List[SectorFlow],
        news: List[NewsItem],
        hot_topics: List[str],
        risks: List[str]
    ) -> tuple:
        """LLM 深度思考分析 - 生成分析数据供外部LLM使用"""
        
        sector_data = "\n".join([
            f"  - {f.sector_name} ({f.etf}): 20日动量 {f.momentum_20d*100:+.1f}%, 相对强度 {f.relative_strength*100:+.1f}%"
            for f in flows
        ])
        
        news_data = "\n".join([
            f"  - [{item.source}] {item.title}"
            for item in news[:15]
        ])
        
        analysis_prompt = f"""你是一位资深的美股投资策略分析师。请基于以下市场情报，进行深度思考分析，并给出本季度的投资主题建议。

## 当前市场情报

### 1. Fed 政策环境
- 当前利率: {fed.fed_funds_rate:.2f}%
- 3个月变化: {fed.rate_change_3m:+.2f}%
- 政策方向: {fed.rate_direction}
- 市场预期: {fed.market_expectation}

### 2. 板块相对强度 (vs SPY)
{sector_data}

### 3. 近期重要新闻
{news_data}

### 4. 已识别热点话题
{', '.join(hot_topics) if hot_topics else '无明显热点'}

### 5. 已识别风险因素
{chr(10).join('- ' + r for r in risks) if risks else '无重大风险'}

## 请进行以下分析

1. **宏观环境解读**: 当前处于什么样的市场周期？Fed政策对市场有何影响？

2. **主题趋势判断**: 根据新闻和板块动量，当前市场的主要投资主题是什么？是否有新的趋势正在形成？

3. **板块轮动分析**: 哪些板块正在领先？哪些板块应该回避？背后的逻辑是什么？

4. **风险评估**: 当前最需要关注的风险是什么？如何应对？

5. **投资建议**: 给出具体的季度投资主题和配置建议。

## 输出格式 (严格按此JSON格式输出)

```json
{{
  "market_cycle": "当前市场周期判断（如：牛市中期、震荡调整、熊市初期等）",
  "theme": "本季度投资主题（简洁，如：AI持续+防御配置）",
  "theme_reasoning": "主题判断的详细理由（2-3句话）",
  "focus_sectors": ["XLK", "XLE"],
  "focus_stocks": ["NVDA", "XOM", "CVX"],
  "avoid_sectors": ["XLF"],
  "sector_reasoning": "板块选择的理由",
  "risk_assessment": "主要风险及应对建议",
  "confidence": "high/medium/low",
  "key_events_to_watch": ["需要关注的重要事件1", "事件2"]
}}
```

请直接输出JSON，不要有其他内容。
"""
        
        self._analysis_prompt = analysis_prompt
        
        analysis_data_path = self.cache_dir / "llm_analysis_prompt.txt"
        with open(analysis_data_path, "w", encoding="utf-8") as f:
            f.write(analysis_prompt)
        
        print(f"    📝 分析提示已保存到: {analysis_data_path}")
        print(f"    💡 请使用 OpenCode 运行以下命令进行 LLM 分析:")
        print(f"       /oracle 请分析以下市场情报并给出投资建议...")
        
        return self._generate_theme_fallback(fed, flows, hot_topics, risks)
    
    def _generate_theme_fallback(
        self, 
        fed: FedPolicy, 
        flows: List[SectorFlow], 
        hot_topics: List[str],
        risks: List[str]
    ) -> tuple:
        """规则回退方案"""
        
        leading_sectors = [f for f in flows if f.relative_strength > 0.02][:3]
        lagging_sectors = [f for f in flows if f.relative_strength < -0.03]
        
        theme_parts = []
        focus_sectors = []
        focus_stocks = []
        avoid_sectors = []
        
        if "ai" in hot_topics:
            theme_parts.append("AI持续")
            if "XLK" not in focus_sectors:
                focus_sectors.append("XLK")
            focus_stocks.extend(["NVDA", "AMD", "AVGO", "MSFT"])
        
        if fed.rate_direction == "降息周期":
            theme_parts.append("降息利好")
            focus_sectors.append("XLF")
            focus_stocks.extend(["JPM", "GS"])
        
        if "tariff" in hot_topics or "china" in hot_topics:
            theme_parts.append("贸易政策关注")
            avoid_sectors.append("XLI")
        
        for sector in leading_sectors:
            if sector.etf not in focus_sectors:
                focus_sectors.append(sector.etf)
                focus_stocks.extend(self.SECTOR_LEADERS.get(sector.etf, [])[:2])
        
        for sector in lagging_sectors:
            if sector.etf not in avoid_sectors:
                avoid_sectors.append(sector.etf)
        
        if not theme_parts:
            theme_parts.append("市场观望")
        
        theme = " + ".join(theme_parts[:3])
        
        reasoning = f"Fed: {fed.rate_direction} ({fed.fed_funds_rate:.2f}%)"
        if leading_sectors:
            reasoning += f" | 领先板块: {', '.join(s.sector_name for s in leading_sectors)}"
        if hot_topics:
            reasoning += f" | 热点: {', '.join(hot_topics[:3])}"
        
        focus_stocks = list(dict.fromkeys(focus_stocks))[:8]
        focus_sectors = list(dict.fromkeys(focus_sectors))[:4]
        avoid_sectors = list(dict.fromkeys(avoid_sectors))[:3]
        
        return theme, focus_sectors, focus_stocks, avoid_sectors, reasoning
    
    def collect_all(self) -> ThemeIntelligence:
        """采集所有情报并生成建议"""
        print("\n" + "=" * 60)
        print("季度主题情报采集")
        print("=" * 60)
        
        fed = self.collect_fed_policy()
        print(f"    Fed 利率: {fed.fed_funds_rate:.2f}% ({fed.rate_direction})")
        
        flows = self.collect_sector_flows()
        if flows:
            print(f"    领先板块: {flows[0].sector_name} (RS: {flows[0].relative_strength:+.1%})")
            print(f"    落后板块: {flows[-1].sector_name} (RS: {flows[-1].relative_strength:+.1%})")
        
        news = self.collect_market_news()
        print(f"    采集新闻: {len(news)} 条")
        
        hot_topics = self.analyze_hot_topics(news)
        print(f"    热点话题: {', '.join(hot_topics) if hot_topics else '无'}")
        
        risks = self.identify_risk_factors(fed, flows, news)
        print(f"    风险因素: {len(risks)} 项")
        
        theme, focus_sectors, focus_stocks, avoid_sectors, reasoning = \
            self.generate_theme_suggestion(fed, flows, news, hot_topics, risks)
        
        intel = ThemeIntelligence(
            report_date=date.today().isoformat(),
            fed_policy=fed,
            sector_flows=flows,
            hot_topics=hot_topics,
            risk_factors=risks,
            suggested_theme=theme,
            suggested_focus_sectors=focus_sectors,
            suggested_focus_stocks=focus_stocks,
            suggested_avoid_sectors=avoid_sectors,
            reasoning=reasoning,
        )
        
        return intel
    
    def save_report(self, intel: ThemeIntelligence, output_dir: Path):
        """保存情报报告"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report = {
            "report_date": intel.report_date,
            "fed_policy": {
                "rate": intel.fed_policy.fed_funds_rate,
                "change_3m": intel.fed_policy.rate_change_3m,
                "direction": intel.fed_policy.rate_direction,
                "expectation": intel.fed_policy.market_expectation,
            },
            "sector_ranking": [
                {
                    "etf": f.etf,
                    "name": f.sector_name,
                    "momentum_20d": round(f.momentum_20d * 100, 2),
                    "relative_strength": round(f.relative_strength * 100, 2),
                }
                for f in intel.sector_flows
            ],
            "hot_topics": intel.hot_topics,
            "risk_factors": intel.risk_factors,
            "suggestion": {
                "theme": intel.suggested_theme,
                "focus_sectors": intel.suggested_focus_sectors,
                "focus_stocks": intel.suggested_focus_stocks,
                "avoid_sectors": intel.suggested_avoid_sectors,
                "reasoning": intel.reasoning,
            },
        }
        
        with open(output_dir / "intelligence_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        quarter = f"{date.today().year}-Q{(date.today().month - 1) // 3 + 1}"
        
        md_lines = [
            f"# 季度主题情报报告",
            f"",
            f"**报告日期**: {intel.report_date}",
            f"**适用季度**: {quarter}",
            f"",
            f"---",
            f"",
            f"## 1. Fed 政策环境",
            f"",
            f"| 指标 | 数值 |",
            f"|------|------|",
            f"| 当前利率 | {intel.fed_policy.fed_funds_rate:.2f}% |",
            f"| 3个月变化 | {intel.fed_policy.rate_change_3m:+.2f}% |",
            f"| 政策方向 | {intel.fed_policy.rate_direction} |",
            f"| 市场预期 | {intel.fed_policy.market_expectation} |",
            f"",
            f"## 2. 板块相对强度排名",
            f"",
            f"| 排名 | 板块 | ETF | 20日动量 | 相对强度 |",
            f"|------|------|-----|----------|----------|",
        ]
        
        for i, f in enumerate(intel.sector_flows, 1):
            rs_icon = "🟢" if f.relative_strength > 0.02 else "🔴" if f.relative_strength < -0.02 else "⚪"
            md_lines.append(
                f"| {i} | {f.sector_name} | {f.etf} | {f.momentum_20d*100:+.1f}% | {rs_icon} {f.relative_strength*100:+.1f}% |"
            )
        
        md_lines.extend([
            f"",
            f"## 3. 市场热点",
            f"",
        ])
        for topic in intel.hot_topics:
            md_lines.append(f"- **{topic}**")
        
        md_lines.extend([
            f"",
            f"## 4. 风险因素",
            f"",
        ])
        for risk in intel.risk_factors:
            md_lines.append(f"- ⚠️ {risk}")
        
        md_lines.extend([
            f"",
            f"## 5. 主题建议 (供人工审核)",
            f"",
            f"### 建议主题",
            f"```",
            f"{intel.suggested_theme}",
            f"```",
            f"",
            f"### 焦点板块",
            f"- {', '.join(intel.suggested_focus_sectors)}",
            f"",
            f"### 焦点股票",
            f"- {', '.join(intel.suggested_focus_stocks)}",
            f"",
            f"### 回避板块",
            f"- {', '.join(intel.suggested_avoid_sectors) if intel.suggested_avoid_sectors else '无'}",
            f"",
            f"### 分析依据",
            f"> {intel.reasoning}",
            f"",
            f"---",
            f"",
            f"## 6. 人工审核区",
            f"",
            f"**审核人**: ________________",
            f"",
            f"**审核日期**: ________________",
            f"",
            f"**修改意见**:",
            f"",
            f"- [ ] 同意建议主题",
            f"- [ ] 修改焦点板块: ________________",
            f"- [ ] 修改焦点股票: ________________",
            f"- [ ] 添加注意事项: ________________",
            f"",
            f"**最终确认主题**:",
            f"```json",
            f'"{quarter}": {{',
            f'    "theme": "{intel.suggested_theme}",',
            f'    "focus_sectors": {json.dumps(intel.suggested_focus_sectors)},',
            f'    "focus_stocks": {json.dumps(intel.suggested_focus_stocks)},',
            f'    "avoid_sectors": {json.dumps(intel.suggested_avoid_sectors)},',
            f'    "sector_bonus": {json.dumps({s: 3-i for i, s in enumerate(intel.suggested_focus_sectors)})},',
            f'}}',
            f"```",
        ])
        
        with open(output_dir / "intelligence_report.md", "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
        
        print(f"\n📁 报告已保存到: {output_dir}")


def main():
    collector = ThemeIntelligenceCollector()
    intel = collector.collect_all()
    
    print("\n" + "=" * 60)
    print("主题建议摘要")
    print("=" * 60)
    print(f"\n  📋 建议主题: {intel.suggested_theme}")
    print(f"  📈 焦点板块: {', '.join(intel.suggested_focus_sectors)}")
    print(f"  🎯 焦点股票: {', '.join(intel.suggested_focus_stocks)}")
    print(f"  ⛔ 回避板块: {', '.join(intel.suggested_avoid_sectors) if intel.suggested_avoid_sectors else '无'}")
    print(f"\n  💡 分析依据: {intel.reasoning}")
    
    if intel.risk_factors:
        print(f"\n  ⚠️ 风险提示:")
        for risk in intel.risk_factors:
            print(f"     - {risk}")
    
    output_dir = Path("storage/intelligence")
    collector.save_report(intel, output_dir)
    
    print("\n" + "=" * 60)
    print("下一步操作")
    print("=" * 60)
    print("  1. 查看报告: storage/intelligence/intelligence_report.md")
    print("  2. 人工审核并修改建议")
    print("  3. 将确认的主题复制到 V7.0 配置中")


if __name__ == "__main__":
    main()
