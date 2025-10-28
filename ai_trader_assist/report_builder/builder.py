"""Daily report builder producing JSON + Markdown summaries."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Tuple

from ..portfolio_manager.state import PortfolioState


@dataclass
class DailyReportBuilder:
    sizer_config: Dict

    def _calc_price_levels(self, price: float, atr_pct: float) -> Tuple[float, float]:
        atr_value = price * atr_pct
        k1 = self.sizer_config.get("k1_stop", 1.5)
        k2 = self.sizer_config.get("k2_target", 2.5)
        stop = max(0.0, price - k1 * atr_value)
        target = price + k2 * atr_value
        return stop, target

    def build(
        self,
        trading_day: date,
        risk: Dict,
        sectors: List[Dict],
        stock_scores: List[Dict],
        orders: Dict[str, List[Dict]],
        portfolio_state: PortfolioState,
        news: Optional[Dict] = None,
    ) -> Tuple[Dict, str]:
        leading = [s["symbol"] for s in sectors[:3]]
        weak = [s["symbol"] for s in sectors[-3:]] if sectors else []

        score_lookup = {item["symbol"]: item for item in stock_scores}

        actions_json: List[Dict] = []
        for order in orders.get("buy", []):
            score = score_lookup.get(order["symbol"], {})
            atr_pct = float(score.get("atr_pct", 0.02))
            stop, target = self._calc_price_levels(order["price"], atr_pct)
            actions_json.append(
                {
                    "symbol": order["symbol"],
                    "action": "buy",
                    "shares": order["shares"],
                    "price": order["price"],
                    "stop": stop,
                    "target": target,
                    "confidence": order.get("confidence", score.get("confidence")),
                }
            )

        for order in orders.get("sell", []):
            actions_json.append(
                {
                    "symbol": order["symbol"],
                    "action": "sell",
                    "shares": order["shares"],
                    "price": order["price"],
                    "reason": "risk signal",
                    "confidence": order.get("confidence"),
                }
            )

        est_cash = portfolio_state.cash
        est_market = portfolio_state.market_value
        for order in orders.get("buy", []):
            est_cash -= order["notional"]
            est_market += order["notional"]
        for order in orders.get("sell", []):
            est_cash += order["notional"]
            est_market -= min(order["notional"], est_market)
        est_equity = est_cash + est_market
        exposure_after = est_market / est_equity if est_equity > 0 else 0.0

        market_news = (news or {}).get("market", {}).get("headlines", [])[:3] if news else []
        sector_news = {
            item["symbol"]: (news or {}).get("sectors", {}).get(item["symbol"], {}).get("headlines", [])[:2]
            for item in sectors[:3]
        } if news else {}

        report_json = {
            "date": trading_day.isoformat(),
            "market": {
                "risk": risk.get("risk_level"),
                "bias": risk.get("bias"),
                "target_exposure": risk.get("target_exposure"),
            },
            "sectors": {"leading": leading, "weak": weak},
            "actions": actions_json,
            "portfolio_after_est": {"exposure": exposure_after},
            "news": {
                "market_headlines": market_news,
                "sector_headlines": sector_news,
            },
        }

        markdown_lines = [
            f"📋 {trading_day.isoformat()} \u76d8\u524d\u62a5\u544a\uff08PT\uff09",
            f"[\u5e02\u573a] \u98ce\u9669={risk.get('risk_level')}, \u503e\u5411={risk.get('bias')}, \u76ee\u6807\u4ed3\u4f4d={risk.get('target_exposure', 0):.0%}",
            f"[\u677f\u5757] \u9886\u5148\uff1a{', '.join(leading) if leading else '-'}\uff1b\u504f\u5f31\uff1a{', '.join(weak) if weak else '-'}",
            "[\u64cd\u4f5c\u6e05\u5355]",
        ]

        if actions_json:
            for action in actions_json:
                if action["action"] == "buy":
                    markdown_lines.append(
                        f"- {action['symbol']}: \u4e70\u5165 {action['shares']} \u80a1 @ {action['price']:.2f}\uff0c\u6b62\u635f {action['stop']:.2f}\uff0c\u76ee\u6807 {action['target']:.2f}\uff08\u4fe1\u5fc3\u5ea6 {action['confidence']:.2f}\uff09"
                    )
                else:
                    markdown_lines.append(
                        f"- {action['symbol']}: \u51cf\u4ed3 {action['shares']} \u80a1 @ {action['price']:.2f}\uff08\u539f\u56e0\uff1a{action.get('reason','-')}\uff09"
                    )
        else:
            markdown_lines.append("- \u6682\u65e0\u65b0\u6307\u4ee4")

        markdown_lines.append(
            f"[\u9884\u8ba1\u4ed3\u4f4d] {exposure_after:.0%}"
        )
        if market_news:
            markdown_lines.append("[\u65b0\u95fb]\u8be6\u89e3")
            for item in market_news:
                title = item.get("title", "")
                publisher = item.get("publisher", "")
                markdown_lines.append(f"- {title} ({publisher})")
        high_risk = [s for s in stock_scores if s.get("premarket", 0) > 0.6]
        markdown_lines.append("[\u98ce\u63a7] \u76d8\u524d\u9ad8\u98ce\u9669\uff1a-" if not high_risk else "[\u98ce\u63a7] \u76d8\u524d\u9ad8\u98ce\u9669\uff1a" + ", ".join(h["symbol"] for h in high_risk))
        markdown_lines.append("[\u5f85\u529e] \u76d8\u4e2d\u6267\u884c\u540e\u8bf7\u5f55\u5165 operations.jsonl")

        markdown = "\n".join(markdown_lines) + "\n"
        return report_json, markdown

    @staticmethod
    def dumps_json(payload: Dict) -> str:
        import json

        return json.dumps(payload, indent=2)
