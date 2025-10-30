"""Hybrid report builder combining rule-based data with LLM summaries."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, ValidationError

from ..portfolio_manager.state import PortfolioState
from .builder import DailyReportBuilder


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
    ) -> Tuple[Dict, str]:
        report_json, markdown = super().build(
            trading_day=trading_day,
            risk=risk,
            sectors=sectors,
            stock_scores=stock_scores,
            orders=orders,
            portfolio_state=portfolio_state,
            news=news,
        )

        summary_model: Optional[LLMReportSummary] = None
        if llm_summary:
            try:
                summary_model = LLMReportSummary.parse_obj(llm_summary)
            except ValidationError:
                summary_model = None

        ai_summary_payload = {"text": "", "key_points": []}
        if summary_model:
            ai_summary_payload = summary_model.as_payload()
            if summary_model.summary_text.strip() or summary_model.key_points:
                markdown_lines = markdown.rstrip("\n").split("\n")
                insert_index = 1 if len(markdown_lines) > 1 else len(markdown_lines)
                summary_lines = [""] + summary_model.to_markdown_lines() + [""]
                for line in reversed(summary_lines):
                    markdown_lines.insert(insert_index, line)
                markdown = "\n".join(markdown_lines) + "\n"

        report_json["ai_summary"] = ai_summary_payload
        return report_json, markdown
