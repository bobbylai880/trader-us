"""Markdown renderer that treats report.json as the single source of truth."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
import json
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


def _decimalize(value: float | int | str | None) -> Optional[Decimal]:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except Exception:  # pragma: no cover - defensive
        return None


@dataclass(slots=True)
class MarkdownRenderConfig:
    """Configuration used by :class:`MarkdownRenderer`."""

    locale: str = "zh"
    decimals_money: int = 2
    decimals_percent: int = 1
    max_rows_per_section: int = 50
    hide_below_weight: Optional[float] = None
    show_raw_json_appendix: bool = True
    raw_json_preview_lines: int = 40
    report_json_path: Optional[str] = None
    reproduction_command: Optional[str] = None
    news_highlights_visible: int = 5

    def money(self, value: float | int | str | None, prefix: str = "$") -> str:
        number = _decimalize(value)
        if number is None:
            return "—"
        quant = Decimal("1").scaleb(-self.decimals_money)
        formatted = number.quantize(quant, rounding=ROUND_HALF_UP)
        return f"{prefix}{formatted:,.{self.decimals_money}f}" if prefix else f"{formatted:,.{self.decimals_money}f}"

    def percent(self, value: float | int | str | None) -> str:
        number = _decimalize(value)
        if number is None:
            return "—"
        quant = Decimal("1").scaleb(-self.decimals_percent)
        pct = (number * Decimal("100")).quantize(quant, rounding=ROUND_HALF_UP)
        return f"{pct:.{self.decimals_percent}f}%"


class MarkdownRenderer:
    """Render the Markdown report from the canonical JSON payload."""

    def __init__(self, config: Optional[MarkdownRenderConfig] = None) -> None:
        self.config = config or MarkdownRenderConfig()

    def render(self, report: Mapping[str, Any]) -> str:
        data = dict(report or {})
        lines: List[str] = []
        missing_sections: List[str] = []

        header = self._render_header(data)
        lines.extend(header)

        missing_sections.extend(self._collect_missing(data, ["market", "exposure", "allocation_plan", "positions"]))

        extra_section, _ = self._render_additional_sections(
            data,
            known_keys={
                "as_of",
                "date",
                "snapshot_id",
                "input_hash",
                "config_profile",
                "market",
                "exposure",
                "allocation_plan",
                "positions",
                "sectors",
                "stocks",
                "data_gaps",
                "safe_mode",
                "ai_summary",
                "appendix",
                "artefact_summary",
            },
        )
        artefact_section, _ = self._render_artefact_summary(data.get("artefact_summary"))

        toc = self._render_toc(include_additional=True, include_artefacts=True)
        lines.extend(toc)

        market_section = self._render_market(data.get("market"))
        lines.extend(market_section)

        exposure_section = self._render_exposure(data.get("exposure"))
        lines.extend(exposure_section)

        allocation_section, allocation_missing = self._render_allocation(data.get("allocation_plan", []))
        lines.extend(allocation_section)
        missing_sections.extend(allocation_missing)

        positions_section = self._render_positions(data.get("positions"))
        lines.extend(positions_section)

        sectors_section = self._render_sectors(data.get("sectors"))
        lines.extend(sectors_section)

        stocks_section = self._render_stocks(data.get("stocks"))
        lines.extend(stocks_section)

        lines.extend(extra_section)
        lines.extend(artefact_section)

        data_gaps_section = self._render_data_gaps(data.get("data_gaps"), missing_sections)
        lines.extend(data_gaps_section)

        appendix_section = self._render_appendix(data)
        lines.extend(appendix_section)

        formatting_section = self._render_formatting_rules()
        lines.extend(formatting_section)

        warning_block = self._render_schema_warning(missing_sections)
        return warning_block + "\n".join(lines).rstrip() + "\n"

    # ------------------------------------------------------------------
    # Header & Navigation
    # ------------------------------------------------------------------
    def _render_header(self, data: Mapping[str, Any]) -> List[str]:
        as_of = data.get("as_of") or data.get("date") or "—"
        title = f"# 盘前报告（{as_of}）"
        meta_bits: List[str] = []
        if data.get("snapshot_id"):
            meta_bits.append(f"snapshot_id: {data['snapshot_id']}")
        if data.get("input_hash"):
            meta_bits.append(f"input_hash: {data['input_hash']}")
        if data.get("config_profile"):
            meta_bits.append(f"config_profile: {data['config_profile']}")
        meta_line = f"> {' · '.join(meta_bits)}" if meta_bits else "> （缺少快照标识）"

        header_lines = [title, meta_line, ""]

        safe_mode = data.get("safe_mode") or {}
        if isinstance(safe_mode, Mapping) and safe_mode:
            active = bool(safe_mode.get("active", True))
            if active:
                reason = safe_mode.get("reason", "未提供原因")
                impact = safe_mode.get("impact") or safe_mode.get("policy")
                details = f"> ⚠️ **Safe Mode 启用**：{reason}"
                if impact:
                    details += f"（影响：{impact}）"
                header_lines.append(details)
                header_lines.append("")
            else:
                note = safe_mode.get("note") or safe_mode.get("reason")
                if note:
                    header_lines.append(f"> ℹ️ Safe Mode 已禁用：{note}")
                    header_lines.append("")

        ai_summary = data.get("ai_summary")
        if isinstance(ai_summary, Mapping):
            summary_lines = self._render_ai_summary(ai_summary)
            if summary_lines:
                header_lines.extend(summary_lines)

        header_lines.append("---")
        header_lines.append("")
        return header_lines

    def _render_ai_summary(self, summary: Mapping[str, Any]) -> List[str]:
        text = (summary.get("text") or "").strip()
        points = [p.strip() for p in summary.get("key_points", []) if isinstance(p, str) and p.strip()]
        if not text and not points:
            return []
        lines = ["> 🧠 AI 总结", ">", "> " + (text or "无摘要提供。")]
        for point in points:
            lines.append(f"> - {point}")
        lines.append("")
        return lines

    def _render_toc(self, *, include_additional: bool, include_artefacts: bool) -> List[str]:
        entries = [
            ("市场概览", "市场概览"),
            ("组合敞口", "组合敞口"),
            ("头寸与预算分配", "头寸与预算分配"),
            ("当前持仓", "当前持仓"),
            ("板块视图", "板块视图"),
            ("个股视图", "个股视图"),
        ]
        if include_additional:
            entries.append(("附加信息", "附加信息"))
        if include_artefacts:
            entries.append(("产出摘要", "产出摘要"))
        entries.extend(
            [
                ("数据缺口与异常", "数据缺口与异常"),
                ("附录", "附录"),
            ]
        )
        lines = ["## 目录"]
        for label, anchor in entries:
            lines.append(f"- [{label}](#{anchor})")
        lines.append("")
        lines.append("---")
        lines.append("")
        return lines

    # ------------------------------------------------------------------
    # Sections
    # ------------------------------------------------------------------
    def _render_market(self, market: Any) -> List[str]:
        lines = ["## 市场概览"]
        if not isinstance(market, Mapping):
            lines.append("> ⚠️ 缺少 market 数据")
            lines.append("")
            return lines

        risk_level = market.get("risk_level") or market.get("risk") or "—"
        bias = market.get("bias") or "—"
        summary = market.get("summary") or "—"
        drivers = market.get("drivers") or []
        flags = market.get("premarket_flags") or []
        news_sentiment = market.get("news_sentiment")

        lines.append(f"- 风险等级：**{risk_level}**")
        lines.append(f"- 倾向：**{bias}**")
        lines.append(f"- 摘要：{summary}")
        lines.append(f"- 驱动：{self._format_list(drivers)}")
        lines.append(f"- 盘前标记：{self._format_list(flags)}")
        sentiment = f"{news_sentiment:+.2f}" if isinstance(news_sentiment, (int, float)) else "—"
        lines.append(f"- 新闻情绪：{sentiment}")

        highlights = market.get("news_highlights") or []
        if highlights:
            rendered, folded = self._render_news_highlights(highlights)
            lines.extend(rendered)
            if folded:
                lines.append("")
        lines.append("")
        return lines

    def _render_exposure(self, exposure: Any) -> List[str]:
        lines = ["## 组合敞口"]
        if not isinstance(exposure, Mapping):
            lines.append("> ⚠️ 缺少 exposure 数据")
            lines.append("")
            return lines

        current = self.config.percent(exposure.get("current"))
        target = self.config.percent(exposure.get("target"))
        delta = self.config.percent(exposure.get("delta"))
        lines.append("| 当前 | 目标 | 差值 |")
        lines.append("|---:|---:|---:|")
        lines.append(f"| {current} | {target} | {delta} |")

        constraints = exposure.get("constraints") if isinstance(exposure.get("constraints"), Mapping) else {}
        constraint_bits: List[str] = []
        for key, label in ("max_exposure", "最大敞口"), ("max_single_weight", "单一权重上限"):
            if key in constraints:
                constraint_bits.append(f"{label}={self.config.percent(constraints[key])}")
        if constraint_bits:
            lines.append("约束：" + "，".join(constraint_bits))
        lines.append("")
        return lines

    def _render_allocation(self, plan: Any) -> tuple[List[str], List[str]]:
        lines = ["## 头寸与预算分配"]
        missing: List[str] = []
        if not isinstance(plan, Sequence) or not plan:
            lines.append("(无推荐调仓)")
            lines.append("")
            return lines, missing

        visible_items = list(plan)
        hidden_threshold = self.config.hide_below_weight
        hidden_items: List[Mapping[str, Any]] = []
        if hidden_threshold is not None:
            threshold = Decimal(str(hidden_threshold))
            kept: List[Mapping[str, Any]] = []
            for item in visible_items:
                weight = _decimalize(item.get("weight"))
                if weight is not None and weight.copy_abs() < threshold:
                    hidden_items.append(item)
                else:
                    kept.append(item)
            visible_items = kept
            if hidden_items:
                percent = self.config.percent(hidden_threshold)
                lines.append(f"> 已隐藏 {len(hidden_items)} 项（权重 < {percent}）")

        header = [
            "| 代码 | 动作 | 权重 | 预算($) | 股数 | 参考价 | 止损(k1) | 止盈(k2) | 理由 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]

        table_rows = [self._format_allocation_row(item) for item in visible_items]
        lines.extend(self._foldable_table(header, table_rows))

        if hidden_items:
            hidden_header = header[:]
            hidden_rows = [self._format_allocation_row(item) for item in hidden_items]
            lines.append("<details>")
            lines.append(f"<summary>已折叠 {len(hidden_items)} 项（低权重）</summary>")
            lines.append("")
            lines.extend(hidden_header)
            lines.extend(hidden_rows)
            lines.append("</details>")

        lines.append("")
        return lines, missing

    def _format_allocation_row(self, item: Mapping[str, Any]) -> str:
        symbol = item.get("symbol", "—")
        action = (item.get("action") or "").upper() or "—"
        weight = self.config.percent(item.get("weight"))
        budget = self.config.money(item.get("budget"))
        shares = item.get("shares")
        shares_str = f"{shares}" if isinstance(shares, (int, float)) else "—"
        price = self.config.money(item.get("price_ref"), prefix="")
        stops = item.get("stops") if isinstance(item.get("stops"), Mapping) else {}
        targets = item.get("targets") if isinstance(item.get("targets"), Mapping) else {}
        stop = stops.get("atr_k1")
        target = targets.get("atr_k2")
        stop_str = f"{stop:.2f}" if isinstance(stop, (int, float)) else "—"
        target_str = f"{target:.2f}" if isinstance(target, (int, float)) else "—"
        reasons = item.get("reasons") if isinstance(item.get("reasons"), Sequence) else []
        reason_text = ", ".join(str(r) for r in reasons) if reasons else "—"
        return (
            f"| {symbol} | {action} | {weight} | {budget} | {shares_str} | {price} | "
            f"{stop_str} | {target_str} | {reason_text} |"
        )

    def _render_positions(self, positions: Any) -> List[str]:
        lines = ["## 当前持仓"]
        if not isinstance(positions, Mapping):
            lines.append("> ⚠️ 缺少 positions 数据")
            lines.append("")
            return lines

        cash = self.config.money(positions.get("cash"))
        equity = self.config.money(positions.get("equity_value"))
        exposure = self.config.percent(positions.get("exposure"))
        lines.append(f"现金：{cash} · 权益：{equity} · 敞口：{exposure}")

        items = positions.get("items") if isinstance(positions.get("items"), Sequence) else []
        if items:
            lines.append("| 代码 | 股数 | 均价 |")
            lines.append("|---|---:|---:|")
            rows = []
            for item in items:
                symbol = item.get("symbol", "—") if isinstance(item, Mapping) else "—"
                shares = item.get("shares") if isinstance(item, Mapping) else None
                avg_cost = item.get("avg_cost") if isinstance(item, Mapping) else None
                share_str = f"{shares}" if isinstance(shares, (int, float)) else "—"
                avg_str = self.config.money(avg_cost, prefix="")
                rows.append(f"| {symbol} | {share_str} | {avg_str} |")
            lines.extend(self._foldable_table([], rows))
        else:
            lines.append("(无持仓记录)")
        lines.append("")
        return lines

    def _render_sectors(self, sectors: Any) -> List[str]:
        lines = ["## 板块视图"]
        if not isinstance(sectors, Sequence) or not sectors:
            lines.append("(无板块信号)")
            lines.append("")
            return lines

        header = ["| 板块 | 评分 | 状态 | 新闻摘要 |", "|---|---:|---|---|"]
        rows = []
        for item in sectors:
            if not isinstance(item, Mapping):
                continue
            symbol = item.get("symbol", "—")
            score = f"{item.get('score'):.2f}" if isinstance(item.get("score"), (int, float)) else "—"
            state = item.get("state", "—")
            highlight = item.get("news_highlight") or "—"
            rows.append(f"| {symbol} | {score} | {state} | {highlight} |")
        lines.extend(self._foldable_table(header, rows))
        lines.append("")
        return lines

    def _render_stocks(self, stocks: Any) -> List[str]:
        lines = ["## 个股视图"]
        if not isinstance(stocks, Sequence) or not stocks:
            lines.append("(无个股打分)")
            lines.append("")
            return lines

        grouped: Dict[str, List[Mapping[str, Any]]] = {}
        order: List[str] = []
        for item in stocks:
            if not isinstance(item, Mapping):
                continue
            category = str(item.get("category") or item.get("action") or "未分类").upper()
            if category not in grouped:
                grouped[category] = []
                order.append(category)
            grouped[category].append(item)

        for category in order:
            lines.append(f"### {category}")
            items = grouped[category]
            rendered, folded = self._render_stock_items(items)
            lines.extend(rendered)
            if folded:
                lines.append("")
        lines.append("")
        return lines

    def _render_stock_items(self, items: Sequence[Mapping[str, Any]]) -> tuple[List[str], bool]:
        limit = self.config.max_rows_per_section
        rendered: List[str] = []
        folded = False
        for index, item in enumerate(items):
            block = self._format_stock_block(item, index + 1)
            if index < limit:
                rendered.extend(block)
            else:
                folded = True
                break

        if folded:
            rendered.append("<details>")
            remaining = items[limit:]
            rendered.append(f"<summary>已折叠 {len(remaining)} 项</summary>")
            rendered.append("")
            for idx, item in enumerate(remaining, start=limit + 1):
                rendered.extend(self._format_stock_block(item, idx))
            rendered.append("</details>")
        return rendered, folded

    def _format_stock_block(self, item: Mapping[str, Any], index: int) -> List[str]:
        symbol = item.get("symbol", "—")
        premarket = item.get("premarket_score")
        trend_strength = item.get("trend_strength")
        momentum = item.get("momentum_10d")
        volatility = item.get("volatility_trend")
        trend_explanation = item.get("trend_explanation") or "—"
        news_highlight = item.get("news_highlight") or "—"
        risks = self._format_list(item.get("risks"), joiner="；")
        flags = self._format_list(item.get("flags"), joiner="，")

        block = [f"{index}. **{symbol}**"]
        block.append(f"   - 盘前评分：{premarket:.2f}" if isinstance(premarket, (int, float)) else "   - 盘前评分：—")
        block.append(f"   - 趋势强度：{trend_strength:.2f}" if isinstance(trend_strength, (int, float)) else "   - 趋势强度：—")
        block.append(f"   - 动量10日：{momentum:+.2f}" if isinstance(momentum, (int, float)) else "   - 动量10日：—")
        block.append(f"   - 波动趋势：{volatility:.2f}x" if isinstance(volatility, (int, float)) else "   - 波动趋势：—")
        block.append(f"   - 趋势解读：{trend_explanation}")
        block.append(f"   - 新闻摘要：{news_highlight}")
        block.append(f"   - 风险：{risks}")
        block.append(f"   - 标记：{flags}")
        return block

    def _render_data_gaps(self, data_gaps: Any, missing_sections: Iterable[str]) -> List[str]:
        lines = ["## 数据缺口与异常"]
        combined: List[str] = []
        if isinstance(data_gaps, Sequence):
            combined.extend(str(item) for item in data_gaps if item)
        combined.extend(str(item) for item in missing_sections if item)
        if combined:
            for entry in combined:
                lines.append(f"- {entry}")
        else:
            lines.append("无")
        lines.append("")
        return lines

    def _render_appendix(self, report: Mapping[str, Any]) -> List[str]:
        lines = ["## 附录"]
        if not self.config.show_raw_json_appendix:
            lines.append("(已根据配置隐藏 JSON 附录)")
            lines.append("")
            return lines

        appendix_meta = report.get("appendix") if isinstance(report.get("appendix"), Mapping) else {}
        path_info = appendix_meta.get("report_json_path") or self.config.report_json_path or "未提供路径"
        lines.append("<details>")
        lines.append("<summary>原始 JSON 预览</summary>")
        lines.append("")
        lines.append(f"路径：{path_info}")
        lines.append("````json")
        preview = json.dumps(report, indent=2, ensure_ascii=False).splitlines()
        max_lines = max(1, self.config.raw_json_preview_lines)
        for line in preview[:max_lines]:
            lines.append(line)
        if len(preview) > max_lines:
            lines.append("…")
        lines.append("````")

        command = (
            appendix_meta.get("reproduction_command")
            or self.config.reproduction_command
            or "python -m ai_trader_assist.jobs.run_daily --config configs/base.json --output-dir storage/daily_$(date +%F)"
        )
        lines.append("")
        lines.append("复现命令：")
        lines.append("```bash")
        lines.append(command)
        lines.append("```")
        lines.append("</details>")
        lines.append("")
        return lines

    def _render_formatting_rules(self) -> List[str]:
        lines = ["### 显示格式说明"]
        lines.append(f"- 货币：千分位 + {self.config.decimals_money} 位小数（ROUND_HALF_UP）")
        lines.append(f"- 百分比：保留 {self.config.decimals_percent} 位小数")
        lines.append("- 时间：ISO-8601，含时区信息（如可用）")
        lines.append("")
        return lines

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _render_schema_warning(self, missing_sections: Iterable[str]) -> str:
        missing = [item for item in missing_sections if item]
        if not missing:
            return ""
        unique = sorted(set(missing))
        bullet = "\n".join(f"> - {item}" for item in unique)
        return f"> ⚠️ 数据校验失败：以下字段缺失或为空\n{bullet}\n\n"

    def _format_list(self, values: Any, joiner: str = "、") -> str:
        if isinstance(values, Mapping):
            values = list(values.values())
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            return str(values) if values else "—"
        cleaned = [str(v) for v in values if v not in (None, "")]
        return joiner.join(cleaned) if cleaned else "—"

    def _render_news_highlights(self, highlights: Sequence[Mapping[str, Any]]) -> tuple[List[str], bool]:
        count = len(highlights)
        if count == 0:
            return [], False

        limit = max(
            1,
            min(self.config.news_highlights_visible, self.config.max_rows_per_section),
        )

        if count <= limit:
            summary = f"重点新闻（{count} 条）"
            lines = ["<details>", f"<summary>{summary}</summary>", ""]
            for item in highlights:
                lines.append(self._format_news_line(item))
            lines.append("</details>")
            return lines, False

        primary = highlights[:limit]
        folded = highlights[limit:]
        summary = f"重点新闻（{count} 条，已折叠 {len(folded)} 项）"
        lines = ["<details>", f"<summary>{summary}</summary>", ""]
        for item in primary:
            lines.append(self._format_news_line(item))
        lines.append("<details>")
        lines.append(f"<summary>展开剩余 {len(folded)} 条</summary>")
        lines.append("")
        for item in folded:
            lines.append(self._format_news_line(item))
        lines.append("</details>")
        lines.append("</details>")
        return lines, True

    def _format_news_line(self, item: Mapping[str, Any]) -> str:
        title = item.get("title") or item.get("headline") or "(未提供标题)"
        source = item.get("source") or item.get("publisher") or "未知来源"
        timestamp = item.get("ts") or item.get("timestamp")
        ts_text = f" · {timestamp}" if timestamp else ""
        return f"- {title} ({source}{ts_text})"

    def _foldable_table(self, header: List[str], rows: List[str]) -> List[str]:
        if not rows:
            return header
        limit = self.config.max_rows_per_section
        if len(rows) <= limit:
            return header + rows
        visible = rows[:limit]
        folded = rows[limit:]
        lines = header + visible
        lines.append("<details>")
        lines.append(f"<summary>已折叠 {len(folded)} 项</summary>")
        lines.append("")
        if header:
            lines.extend(header)
        lines.extend(folded)
        lines.append("</details>")
        return lines

    def _collect_missing(self, data: Mapping[str, Any], keys: Sequence[str]) -> List[str]:
        missing: List[str] = []
        for key in keys:
            if key not in data or data.get(key) in (None, {}, []):
                missing.append(f"字段缺失：{key}")
        return missing

    def _render_additional_sections(
        self,
        report: Mapping[str, Any],
        *,
        known_keys: set[str],
    ) -> tuple[List[str], bool]:
        extras: List[tuple[str, Any]] = []
        for key, value in report.items():
            if key in known_keys:
                continue
            if value in (None, "", [], {}):
                continue
            extras.append((key, value))

        lines: List[str] = ["## 附加信息", ""]
        if not extras:
            lines.append("无额外字段")
            lines.append("")
            return lines, False

        for key, value in extras:
            lines.append(f"### {key}")
            if isinstance(value, (Mapping, Sequence)) and not isinstance(value, (str, bytes)):
                pretty = json.dumps(value, indent=2, ensure_ascii=False)
                lines.append("```json")
                lines.extend(pretty.splitlines())
                lines.append("```")
            else:
                lines.append(str(value))
            lines.append("")
        return lines, True

    def _render_artefact_summary(self, artefacts: Any) -> tuple[List[str], bool]:
        lines: List[str] = ["## 产出摘要", ""]
        if not isinstance(artefacts, Sequence) or isinstance(artefacts, (str, bytes)):
            lines.append("无可用摘要")
            lines.append("")
            return lines, False

        rows: List[str] = []
        for item in artefacts:
            if not isinstance(item, Mapping):
                continue
            name = item.get("name") or item.get("label") or item.get("file") or "—"
            path = item.get("path") or item.get("location") or "—"
            entries = item.get("entries")
            if isinstance(entries, (int, float)):
                count = f"{int(entries)}"
            else:
                count = str(entries) if entries not in (None, "") else "—"
            rows.append(f"| {name} | {path} | {count} |")

        if not rows:
            lines.append("无可用摘要")
            lines.append("")
            return lines, False

        header = ["| 名称 | 路径 | 条目数 |", "|---|---|---:|"]
        lines.extend(header)
        lines.extend(rows)
        lines.append("")
        return lines, True

