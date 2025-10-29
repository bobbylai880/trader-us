"""LLM-first orchestration pipeline replacing hand-tuned rules."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

from .safe_mode import SafeModeConfig, build_safe_outputs
from ..llm_operators.base import (
    LLMOperator,
    LLMOperatorConfig,
    LLMRunArtifacts,
    LLMValidationError,
    ensure_guardrails,
)
from ..llm_operators.exposure_planner import ExposurePlannerOperator
from ..llm_operators.market_analyzer import MarketAnalyzerOperator
from ..llm_operators.report_composer import ReportComposerOperator
from ..llm_operators.sector_analyzer import SectorAnalyzerOperator
from ..llm_operators.stock_classifier import StockClassifierOperator
from ..llm.client import DeepSeekClient
from ..portfolio_manager.state import PortfolioState
from ..validators.json_schemas import SCHEMAS


@dataclass
class OperatorUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_payload(cls, payload: Mapping) -> "OperatorUsage":
        if not isinstance(payload, Mapping):
            return cls()
        return cls(
            prompt_tokens=int(payload.get("prompt_tokens", 0) or 0),
            completion_tokens=int(payload.get("completion_tokens", 0) or 0),
            total_tokens=int(payload.get("total_tokens", 0) or 0),
        )

    def to_dict(self) -> Dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class LLMRunResult:
    stages: Dict[str, Mapping]
    usage: Dict[str, OperatorUsage]
    safe_mode: Optional[Mapping]
    artifacts_dir: Path

    def to_dict(self) -> Dict:
        return {
            "stages": self.stages,
            "usage": {stage: usage.to_dict() for stage, usage in self.usage.items()},
            "safe_mode": self.safe_mode,
        }


@dataclass
class LLMOrchestrator:
    client: DeepSeekClient
    operator_configs: Mapping[str, Mapping[str, object]]
    base_prompt_path: Optional[Path]
    storage_dir: Path
    guardrails: Mapping[str, object]
    safe_mode_config: SafeModeConfig
    logger: Optional[logging.Logger] = None
    _base_prompt_text: Optional[str] = field(default=None, init=False)
    _operators: Dict[str, LLMOperator] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.base_prompt_path:
            base_path = Path(self.base_prompt_path)
            if not base_path.exists():
                raise FileNotFoundError(f"Base prompt 文件缺失: {base_path}")
            self._base_prompt_text = base_path.read_text(encoding="utf-8")
        self._operators = self._init_operators()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(
        self,
        trading_day: date,
        payload: Mapping,
        portfolio_state: PortfolioState,
    ) -> LLMRunResult:
        """Execute the staged LLM workflow."""

        storage = self._prepare_storage(trading_day)
        universe = self._extract_universe(payload)
        stage_usage: Dict[str, OperatorUsage] = {}
        stage_results: Dict[str, Mapping] = {}
        safe_mode: Optional[Mapping] = None

        builders = {
            "market_analyzer": self._build_market_payload,
            "sector_analyzer": self._build_sector_payload,
            "stock_classifier": self._build_stock_payload,
            "exposure_planner": self._build_exposure_payload,
            "report_composer": self._build_report_payload,
        }

        try:
            for stage in ("market_analyzer", "sector_analyzer", "stock_classifier", "exposure_planner", "report_composer"):
                operator = self._operators[stage]
                builder = builders[stage]
                stage_payload = builder(payload, stage_results, portfolio_state)
                artefacts = LLMRunArtifacts(
                    stage_dir=storage / stage,
                    base_prompt=self._base_prompt_text,
                )
                result, usage = operator.execute(stage_payload, artefacts)
                if self.guardrails.get("reject_on_hallucinated_tickers", False):
                    ensure_guardrails(stage, stage_payload, result, universe)
                stage_results[stage] = result
                stage_usage[stage] = OperatorUsage.from_payload(usage)
        except LLMValidationError as exc:
            reason = f"{stage} validation failed: {exc}"
            if self.logger:
                self.logger.error("LLM stage %s failed: %s", stage, exc)
            safe_outputs = build_safe_outputs(payload, reason, self.safe_mode_config)
            for key, value in safe_outputs.items():
                if key in ("safe_mode",):
                    continue
                stage_results[key] = value
                stage_dir = storage / key
                stage_dir.mkdir(parents=True, exist_ok=True)
                (stage_dir / "output.json").write_text(
                    json.dumps(value, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            safe_mode = safe_outputs.get("safe_mode", {"reason": reason})
            stage_results["safe_mode"] = safe_mode
            self._append_error_log(storage, stage, stage_payload, str(exc))
        except Exception as exc:
            reason = f"{stage} execution failed: {exc}"
            if self.logger:
                self.logger.exception("LLM stage %s raised an unexpected error", stage)
            safe_outputs = build_safe_outputs(payload, reason, self.safe_mode_config)
            for key, value in safe_outputs.items():
                if key in ("safe_mode",):
                    continue
                stage_results[key] = value
                stage_dir = storage / key
                stage_dir.mkdir(parents=True, exist_ok=True)
                (stage_dir / "output.json").write_text(
                    json.dumps(value, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            safe_mode = safe_outputs.get("safe_mode", {"reason": reason})
            stage_results["safe_mode"] = safe_mode
            self._append_error_log(storage, stage, stage_payload, str(exc))
        else:
            safe_mode = None

        analysis_path = storage / "llm_analysis.json"
        analysis_path.write_text(
            json.dumps(
                {
                    "as_of": trading_day.isoformat(),
                    "results": stage_results,
                    "usage": {k: v.to_dict() for k, v in stage_usage.items()},
                    "safe_mode": safe_mode,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        report_stage = stage_results.get("report_composer")
        if isinstance(report_stage, Mapping) and "markdown" in report_stage:
            (storage / "report.md").write_text(
                report_stage.get("markdown", ""), encoding="utf-8"
            )
            (storage / "report.json").write_text(
                json.dumps(report_stage.get("sections", {}), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        return LLMRunResult(stage_results, stage_usage, safe_mode, storage)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _init_operators(self) -> Dict[str, LLMOperator]:
        operators: Dict[str, LLMOperator] = {}
        temperature = float(self.operator_configs.get("temperature", 0.2) or 0.2)
        max_tokens = int(self.operator_configs.get("max_tokens", 1200) or 1200)
        stage_settings = self.operator_configs.get("operators", {})
        max_retry = int(self.guardrails.get("max_retry", 2) or 2)

        mapping = {
            "market_analyzer": MarketAnalyzerOperator,
            "sector_analyzer": SectorAnalyzerOperator,
            "stock_classifier": StockClassifierOperator,
            "exposure_planner": ExposurePlannerOperator,
            "report_composer": ReportComposerOperator,
        }

        for stage, cls in mapping.items():
            config_dict = stage_settings.get(stage, {}) if isinstance(stage_settings, Mapping) else {}
            prompt_file = Path(config_dict.get("prompt_file"))
            retries = int(config_dict.get("retries", 0) or 0)
            retries = min(retries, max_retry)
            operator_config = LLMOperatorConfig(
                prompt_file=prompt_file,
                retries=retries,
                temperature=float(config_dict.get("temperature", temperature)),
                max_tokens=int(config_dict.get("max_tokens", max_tokens)),
            )
            operators[stage] = cls(
                config=operator_config,
                client=self.client,
                schema=SCHEMAS[stage],
                base_prompt=self._base_prompt_text,
                logger=self.logger,
            )
        return operators

    def _prepare_storage(self, trading_day: date) -> Path:
        """Prepare the artefact directory for a run.

        The caller already provides a date-specific directory (e.g.
        ``storage/daily_YYYY-MM-DD/llm``). Creating an additional nested
        ``llm_<date>`` folder made it harder to discover artefacts and
        diverged from the documented layout, so we simply reuse the provided
        path and ensure it exists.
        """

        storage = self.storage_dir
        storage.mkdir(parents=True, exist_ok=True)
        return storage

    def _build_market_payload(
        self,
        payload: Mapping,
        stage_results: Mapping[str, Mapping],
        state: PortfolioState,
    ) -> Mapping:
        return {
            "as_of": payload.get("as_of"),
            "timezone": payload.get("timezone"),
            "features": {
                "market": payload.get("features", {}).get("market", {}),
                "trend": payload.get("features", {}).get("trend", {}),
                "news": (payload.get("features", {}).get("news", {}) or {}).get("market", []),
            },
            "context": payload.get("context", {}),
            "constraints": payload.get("constraints", {}),
        }

    def _build_sector_payload(
        self,
        payload: Mapping,
        stage_results: Mapping[str, Mapping],
        state: PortfolioState,
    ) -> Mapping:
        return {
            "as_of": payload.get("as_of"),
            "features": {
                "sectors": payload.get("features", {}).get("sectors", {}),
                "market": payload.get("features", {}).get("market", {}),
                "news": (payload.get("features", {}).get("news", {}) or {}).get("sectors", {}),
            },
            "context": payload.get("context", {}),
        }

    def _build_stock_payload(
        self,
        payload: Mapping,
        stage_results: Mapping[str, Mapping],
        state: PortfolioState,
    ) -> Mapping:
        features = payload.get("features", {})
        watchlist = payload.get("universe", {}).get("watchlist", [])
        max_items = int(self.operator_configs.get("max_stock_payload", 8) or 8)
        stocks = features.get("stocks", {})
        selected = {}
        for symbol in watchlist:
            if symbol in stocks:
                selected[symbol] = stocks[symbol]
            if len(selected) >= max_items:
                break
        if len(selected) < max_items:
            for symbol, value in stocks.items():
                if symbol not in selected:
                    selected[symbol] = value
                if len(selected) >= max_items:
                    break
        stock_news = (features.get("news", {}) or {}).get("stocks", {})
        return {
            "as_of": payload.get("as_of"),
            "features": {
                "stocks": selected,
                "trend": features.get("trend", {}),
                "news": stock_news,
            },
            "context": payload.get("context", {}),
            "constraints": payload.get("constraints", {}),
            "universe": payload.get("universe", {}),
        }

    def _build_exposure_payload(
        self,
        payload: Mapping,
        stage_results: Mapping[str, Mapping],
        state: PortfolioState,
    ) -> Mapping:
        return {
            "as_of": payload.get("as_of"),
            "portfolio": state.snapshot_dict(),
            "constraints": payload.get("constraints", {}),
            "context": payload.get("context", {}),
            "stock_view": stage_results.get("stock_classifier", {}),
            "market_view": stage_results.get("market_analyzer", {}),
        }

    def _build_report_payload(
        self,
        payload: Mapping,
        stage_results: Mapping[str, Mapping],
        state: PortfolioState,
    ) -> Mapping:
        return {
            "as_of": payload.get("as_of"),
            "market_view": stage_results.get("market_analyzer", {}),
            "sector_view": stage_results.get("sector_analyzer", {}),
            "stock_view": stage_results.get("stock_classifier", {}),
            "exposure_view": stage_results.get("exposure_planner", {}),
            "news": payload.get("features", {}).get("news", {}),
        }

    def _append_error_log(
        self,
        storage: Path,
        stage: str,
        payload: Mapping,
        error: str,
    ) -> None:
        entry = {
            "stage": stage,
            "error": error,
            "payload": payload,
        }
        error_path = storage / "errors.jsonl"
        with error_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _extract_universe(self, payload: Mapping) -> Sequence[str]:
        sectors = payload.get("universe", {}).get("sectors", [])
        watchlist = payload.get("universe", {}).get("watchlist", [])
        return list({*sectors, *watchlist})
