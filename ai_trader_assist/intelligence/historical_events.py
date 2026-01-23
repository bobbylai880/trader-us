#!/usr/bin/env python3
"""
历史事件时间线数据 (2023-2026)

用于回测时模拟每个季度的市场情报，让 LLM 基于历史事件生成主题配置。
"""

HISTORICAL_EVENTS = {
    "2023-Q1": {
        "quarter": "2023-Q1",
        "major_events": [
            "ChatGPT 在 2022年11月发布后持续爆发，AI概念股开始受关注",
            "SVB 硅谷银行 3月倒闭，引发区域银行危机",
            "Fed 继续加息，3月加息25bp至4.75-5.00%",
            "通胀仍然顽固，CPI 同比 6%+",
            "中国重新开放后经济复苏预期",
        ],
        "market_sentiment": "谨慎乐观，银行危机后有避险情绪",
        "fed_policy": "加息周期末段",
        "hot_topics": ["ai", "banking_crisis", "fed"],
        "leading_narratives": ["AI/ChatGPT", "银行危机避险", "科技反弹"],
    },
    "2023-Q2": {
        "quarter": "2023-Q2",
        "major_events": [
            "NVIDIA 5月财报大超预期，股价暴涨25%+，AI概念全面爆发",
            "美国债务上限危机，6月初达成协议",
            "Fed 6月暂停加息，维持5.00-5.25%",
            "Magnificent 7 概念形成，科技龙头领涨",
            "中国经济复苏不及预期",
        ],
        "market_sentiment": "乐观，AI驱动的科技牛市",
        "fed_policy": "暂停加息观望",
        "hot_topics": ["ai", "nvidia", "debt_ceiling"],
        "leading_narratives": ["AI基础设施", "GPU短缺", "科技龙头"],
    },
    "2023-Q3": {
        "quarter": "2023-Q3",
        "major_events": [
            "AI应用扩散，各大科技公司发布AI产品",
            "Fed 7月加息25bp至5.25-5.50%，为本轮最后一次",
            "美债收益率飙升，10Y突破4.5%",
            "油价上涨，WTI突破90美元",
            "中国房地产危机加剧，恒大碧桂园暴雷",
        ],
        "market_sentiment": "谨慎，利率担忧",
        "fed_policy": "利率见顶",
        "hot_topics": ["ai", "rates", "china_property"],
        "leading_narratives": ["AI扩散", "高利率担忧", "能源上涨"],
    },
    "2023-Q4": {
        "quarter": "2023-Q4",
        "major_events": [
            "10月美债收益率触顶后回落",
            "Fed 11月12月连续暂停加息，释放鸽派信号",
            "市场开始定价2024年降息",
            "年末大涨，Santa Rally行情",
            "AI概念持续，半导体周期触底",
        ],
        "market_sentiment": "乐观，降息预期",
        "fed_policy": "暂停加息，鸽派转向",
        "hot_topics": ["ai", "rate_cuts", "semiconductor"],
        "leading_narratives": ["降息预期", "AI+半导体", "年末行情"],
    },
    "2024-Q1": {
        "quarter": "2024-Q1",
        "major_events": [
            "AI算力军备竞赛，科技巨头加大投资",
            "NVIDIA 股价创新高，市值超越Google",
            "Fed 维持利率不变，降息预期推迟",
            "日本央行结束负利率",
            "中国两会，经济刺激政策有限",
        ],
        "market_sentiment": "乐观，AI驱动",
        "fed_policy": "维持高利率",
        "hot_topics": ["ai", "nvidia", "japan"],
        "leading_narratives": ["AI算力", "科技龙头", "日元贬值"],
    },
    "2024-Q2": {
        "quarter": "2024-Q2",
        "major_events": [
            "AI应用落地加速，数据中心需求爆发",
            "NVIDIA 股价继续创新高",
            "Fed 继续按兵不动，通胀粘性",
            "地缘政治：中东紧张，乌克兰战事持续",
            "美国大选初选，Trump获得共和党提名",
        ],
        "market_sentiment": "乐观但分化",
        "fed_policy": "维持高利率",
        "hot_topics": ["ai", "data_center", "election"],
        "leading_narratives": ["AI应用落地", "数据中心", "大选"],
    },
    "2024-Q3": {
        "quarter": "2024-Q3",
        "major_events": [
            "8月初日元套息交易平仓引发全球股市闪崩",
            "Fed 9月首次降息50bp至4.75-5.00%",
            "降息周期开启，资金轮动到防御板块",
            "中国推出大规模刺激政策，A股暴涨",
            "AI概念开始分化，部分质疑声音出现",
        ],
        "market_sentiment": "波动加大，防御转换",
        "fed_policy": "降息周期开启",
        "hot_topics": ["rate_cuts", "japan_carry", "china_stimulus"],
        "leading_narratives": ["降息利好", "防御配置", "中国反弹"],
    },
    "2024-Q4": {
        "quarter": "2024-Q4",
        "major_events": [
            "11月Trump赢得大选，市场预期减税+去监管",
            "Trump交易：金融、能源、特斯拉大涨",
            "Fed 11月12月继续降息，利率降至4.25-4.50%",
            "比特币创新高突破10万美元",
            "市场担忧关税政策",
        ],
        "market_sentiment": "乐观，Trump交易",
        "fed_policy": "继续降息",
        "hot_topics": ["trump", "tariff", "crypto", "deregulation"],
        "leading_narratives": ["Trump交易", "金融去监管", "加密货币"],
    },
    "2025-Q1": {
        "quarter": "2025-Q1",
        "major_events": [
            "Trump 1月20日就职，开始实施关税政策",
            "对加拿大、墨西哥、中国加征关税",
            "市场波动加大，政策不确定性上升",
            "DeepSeek发布，中国AI突破引发关注",
            "Fed 暂停降息，观察关税影响",
        ],
        "market_sentiment": "谨慎，政策不确定",
        "fed_policy": "暂停降息观望",
        "hot_topics": ["tariff", "china", "deepseek", "policy_uncertainty"],
        "leading_narratives": ["关税不确定性", "防御配置", "中国AI"],
    },
    "2025-Q2": {
        "quarter": "2025-Q2",
        "major_events": [
            "关税政策反复，市场适应新常态",
            "AI Agent 概念兴起，软件应用受关注",
            "Fed 维持利率不变",
            "中美科技脱钩加剧",
            "能源板块因地缘政治走强",
        ],
        "market_sentiment": "震荡适应",
        "fed_policy": "维持利率",
        "hot_topics": ["ai_agent", "tariff", "energy"],
        "leading_narratives": ["AI软件应用", "关税适应", "能源景气"],
    },
    "2025-Q3": {
        "quarter": "2025-Q3",
        "major_events": [
            "AI投资持续，但回报质疑声增加",
            "半导体需求强劲",
            "Fed 开始讨论是否继续降息",
            "美国经济软着陆预期",
            "中国经济刺激效果显现",
        ],
        "market_sentiment": "谨慎乐观",
        "fed_policy": "可能继续降息",
        "hot_topics": ["ai", "semiconductor", "soft_landing"],
        "leading_narratives": ["AI持续", "半导体", "软着陆"],
    },
    "2025-Q4": {
        "quarter": "2025-Q4",
        "major_events": [
            "年末资金配置",
            "科技龙头继续受青睐",
            "Fed 12月可能降息",
            "2026年展望开始",
            "AI应用渗透率提升",
        ],
        "market_sentiment": "乐观",
        "fed_policy": "继续降息预期",
        "hot_topics": ["ai", "year_end", "outlook_2026"],
        "leading_narratives": ["年末配置", "科技龙头", "AI渗透"],
    },
    "2026-Q1": {
        "quarter": "2026-Q1",
        "major_events": [
            "Trump 关税政策持续反复",
            "AI热度持续但有降温信号",
            "Anthropic CEO 批评 NVIDIA",
            "能源板块因地缘政治强势",
            "Fed 利率平稳，暂停调整",
        ],
        "market_sentiment": "震荡",
        "fed_policy": "利率平稳",
        "hot_topics": ["ai", "tariff", "china", "energy"],
        "leading_narratives": ["能源景气", "关税不确定", "AI择优"],
    },
}


def get_quarter_events(quarter: str) -> dict:
    """获取指定季度的历史事件"""
    return HISTORICAL_EVENTS.get(quarter, {})


def get_all_quarters() -> list:
    """获取所有季度列表"""
    return list(HISTORICAL_EVENTS.keys())
