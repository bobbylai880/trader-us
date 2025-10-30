"""Hybrid report builder combining rule-based data with LLM summaries."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from pydantic import BaseModel, Field, ValidationError

from ..portfolio_manager.state import PortfolioState
from .builder import DailyReportBuilder
from .markdown_renderer import MarkdownRenderConfig, MarkdownRenderer


class LLMReportSummary(BaseModel):
    """Schema for validated LLM summary payloads."""

    summary_text: str = Field(default="", description="High level natural language summary.")
    key_points: List[str] = Field(default_factory=list, description="Bullet points supporting the summary.")

    class Config:
        extra = "forbid"

    def as_payload(self) -> Dict[str, Any]:
        """Return the summary in a JSON serialisable payload."""

        cleaned_points = [point.strip() for point in self.key_points if point and point.strip()]
        return {"text": self.summary_text.strip(), "key_points": cleaned_points}

    def to_markdown_lines(self) -> List[str]:
        """Render the summary as Markdown lines."""

        lines: List[str] = ["🧠 AI总结："]
        summary_text = self.summary_text.strip()
        if summary_text:
            lines.append(summary_text)
        for point in self.key_points:
            cleaned_point = point.strip() if point else ""
            if cleaned_point:
                lines.append(f"- {cleaned_point}")
        return lines


@dataclass
class HybridReportBuilder(DailyReportBuilder):
    """Report builder that injects LLM generated insights into the daily report."""

    def build(
        self,
        trading_day: date,
        risk: Dict,
        sectors: List[Dict],
        stock_scores: List[Dict],
        orders: Dict[str, List[Dict]],
        portfolio_state: PortfolioState,
        llm_summary: Optional[Dict[str, Any]] = None,
        news: Optional[Dict] = None,
        premarket_flags: Optional[Dict[str, Dict]] = None,
        snapshot_meta: Optional[Dict[str, Any]] = None,
        safe_mode: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict, str]:
        report_json = self.build_payload(
            trading_day=trading_day,
            risk=risk,
            sectors=sectors,
            stock_scores=stock_scores,
            orders=orders,
            portfolio_state=portfolio_state,
            news=news,
            premarket_flags=premarket_flags,
            snapshot_meta=snapshot_meta,
            safe_mode=safe_mode,
            llm_summary=llm_summary,
        )
        renderer = MarkdownRenderer(MarkdownRenderConfig())
        markdown = renderer.render(report_json)
        return report_json, markdown

    def build_payload(
        self,
        *,
        trading_day: date,
        risk: Mapping[str, Any],
        sectors: Sequence[Mapping[str, Any]],
        stock_scores: Sequence[Mapping[str, Any]],
        orders: Mapping[str, Sequence[Mapping[str, Any]]],
        portfolio_state: PortfolioState,
        news: Optional[Mapping[str, Any]] = None,
        premarket_flags: Optional[Mapping[str, Mapping[str, Any]]] = None,
        snapshot_meta: Optional[Mapping[str, Any]] = None,
        safe_mode: Optional[Mapping[str, Any]] = None,
        llm_summary: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = super().build_payload(
            trading_day=trading_day,
            risk=risk,
            sectors=sectors,
            stock_scores=stock_scores,
            orders=orders,
            portfolio_state=portfolio_state,
            news=news,
            premarket_flags=premarket_flags,
            snapshot_meta=snapshot_meta,
            safe_mode=safe_mode,
            llm_summary=llm_summary,
        )

        summary_payload = {"text": "", "key_points": []}
        if llm_summary:
            try:
                model = LLMReportSummary.parse_obj(llm_summary)
            except ValidationError:
                model = None
            else:
                summary_payload = model.as_payload()
        payload["ai_summary"] = summary_payload
        return payload
