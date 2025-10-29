"""Structured DeepSeek analysis orchestrator."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Mapping, Optional

from .client import DeepSeekClient
from ..portfolio_manager.state import PortfolioState
from ..utils import log_ok, log_result, log_step


def _portfolio_snapshot(state: PortfolioState) -> Dict:
    return {
        "cash": state.cash,
        "equity": state.total_equity,
        "exposure": state.current_exposure,
        "positions": [
            {
                "symbol": position.symbol,
                "shares": position.shares,
                "avg_cost": position.avg_cost,
                "last_price": position.last_price,
            }
            for position in state.positions
        ],
        "last_updated": state.last_updated,
    }


@dataclass
class DeepSeekAnalyzer:
    """Executes the staged DeepSeek prompt workflow via the real API."""

    prompt_files: Mapping[str, Path]
    client: DeepSeekClient
    base_prompt: Optional[Path] = None
    logger: Optional[logging.Logger] = None
    _prompt_cache: Dict[str, str] = field(default_factory=dict, init=False)
    _base_prompt_text: Optional[str] = field(default=None, init=False)
    _usage: Dict[str, Dict] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        for stage, path in self.prompt_files.items():
            prompt_path = Path(path)
            if not prompt_path.exists():
                raise FileNotFoundError(f"Prompt 模板缺失: {stage}")
            self._prompt_cache[stage] = prompt_path.read_text(encoding="utf-8")
        if self.base_prompt:
            base_path = Path(self.base_prompt)
            if not base_path.exists():
                raise FileNotFoundError("Prompt 模板缺失: base_prompt")
            self._base_prompt_text = base_path.read_text(encoding="utf-8")

    def run(
        self,
        trading_day: date,
        risk: Dict,
        sector_scores: List[Dict],
        stock_scores: List[Dict],
        orders: Dict[str, List[Dict]],
        portfolio_state: PortfolioState,
        market_features: Dict,
        premarket_flags: Dict[str, Dict],
        news: Optional[Dict] = None,
    ) -> Dict:
        self._usage = {}
        news_bundle = news or {}
        market_view = self._invoke(
            "market_overview",
            {
                "date": trading_day.isoformat(),
                "risk": risk,
                "market": market_features,
                "premarket": premarket_flags,
                "macro_flags": risk.get("drivers", []),
                "market_headlines": news_bundle.get("market", {}).get("headlines", []),
            },
        )
        sector_view = self._invoke(
            "sector_analysis",
            {
                "sectors": sector_scores,
                "market": market_features,
                "news": news_bundle.get("sectors", {}),
            },
        )
        stock_view = self._invoke(
            "stock_actions",
            {
                "stocks": stock_scores,
                "premarket": premarket_flags,
                "orders": orders,
                "news": news_bundle.get("stocks", {}),
            },
        )
        exposure_view = self._invoke(
            "exposure_check",
            {
                "risk": risk,
                "portfolio": _portfolio_snapshot(portfolio_state),
                "orders": orders,
            },
        )
        report_view = self._invoke(
            "report_compose",
            {
                "date": trading_day.isoformat(),
                "market_view": market_view,
                "sector_view": sector_view,
                "stock_view": stock_view,
                "exposure_view": exposure_view,
                "news": news_bundle,
            },
        )

        return {
            "prompts": {key: str(path) for key, path in self.prompt_files.items()},
            "market_overview": market_view,
            "sector_analysis": sector_view,
            "stock_actions": stock_view,
            "exposure_check": exposure_view,
            "report_compose": report_view,
            "usage": self._usage,
        }

    def _invoke(self, stage: str, payload: Dict) -> Dict:
        if stage not in self._prompt_cache:
            raise FileNotFoundError(f"Prompt 模板缺失: {stage}")
        prompt_text = self._prompt_cache[stage]
        user_prompt = self._compose_prompt(prompt_text, payload)
        messages: List[Mapping[str, str]] = []
        if self._base_prompt_text:
            messages.append({"role": "system", "content": self._base_prompt_text})
        messages.append({"role": "user", "content": user_prompt})
        if self.logger:
            log_step(self.logger, f"llm:{stage}", "Invoking DeepSeek stage")
        stage_start = perf_counter()
        raw_response, usage = self.client.chat(messages=messages, stage=stage)
        try:
            parsed = self._parse_json_response(raw_response)
        except ValueError as exc:
            self._dump_failure(stage, raw_response)
            raise ValueError(
                f"DeepSeek 阶段 {stage} 返回非结构化文本: {exc}"
            ) from exc
        self._usage[stage] = usage or {}
        if self.logger:
            prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
            completion_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
            total_tokens = usage.get("total_tokens") if isinstance(usage, dict) else None
            token_summary = ", ".join(
                f"{label}={value}"
                for label, value in (
                    ("prompt", prompt_tokens),
                    ("completion", completion_tokens),
                    ("total", total_tokens),
                )
                if isinstance(value, (int, float))
            ) or "tokens=unknown"
            keys_preview = ", ".join(list(parsed.keys())[:4]) if isinstance(parsed, dict) else ""
            log_result(
                self.logger,
                f"llm:{stage}",
                f"{token_summary} | keys={keys_preview}",
            )
            log_ok(
                self.logger,
                f"llm:{stage}",
                f"Completed in {perf_counter() - stage_start:.2f}s",
            )
        return parsed

    @staticmethod
    def _compose_prompt(template: str, payload: Dict) -> str:
        serialized = json.dumps(payload, ensure_ascii=False, indent=2)
        return (
            f"{template}\n\n"
            "以下是本阶段可用的结构化输入数据 (JSON)：\n"
            f"```json\n{serialized}\n```\n"
            "请严格按照提示词定义的 JSON Schema 输出，禁止额外文本。"
        )

    @staticmethod
    def _parse_json_response(raw: str) -> Dict:
        """Robustly parse the JSON portion of a DeepSeek response."""

        if raw is None:
            raise ValueError("响应为空")
        text = raw.strip()
        if not text:
            raise ValueError("响应为空")

        # Remove common Markdown fences such as ```json ... ```
        if text.startswith("```"):
            lines = text.splitlines()
            # drop the opening fence
            lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        # Extract the first JSON object/array if extra commentary exists
        first_brace = text.find("{")
        first_bracket = text.find("[")
        candidates = [idx for idx in (first_brace, first_bracket) if idx != -1]
        if not candidates:
            raise ValueError("未找到 JSON 起始符号")
        start = min(candidates)
        closing_char = "}" if start == first_brace else "]"
        end = text.rfind(closing_char)
        if end == -1 or end < start:
            raise ValueError("未找到匹配的 JSON 结束符号")
        json_segment = text[start : end + 1]

        try:
            return json.loads(json_segment)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise ValueError("JSON 解析失败") from exc

    @staticmethod
    def _dump_failure(stage: str, raw: str) -> None:
        """Persist the raw response for offline inspection when parsing fails."""

        try:
            failure_dir = Path("storage") / "llm_failures"
            failure_dir.mkdir(parents=True, exist_ok=True)
            failure_file = failure_dir / f"{stage}_raw.txt"
            failure_file.write_text(raw or "", encoding="utf-8")
        except Exception:
            # Swallow any issues when writing debug artefacts; parsing error is
            # still surfaced to the caller. This mirrors logging best-effort.
            pass
