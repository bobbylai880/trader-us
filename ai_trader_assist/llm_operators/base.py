"""Reusable DeepSeek operator base classes with schema validation and retries."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Type

import json5

try:  # pragma: no cover - exercised indirectly via orchestrator tests
    from jsonschema import Draft7Validator, ValidationError  # type: ignore
except ImportError:  # pragma: no cover - fallback executed when dependency missing
    Draft7Validator = None  # type: ignore

    class ValidationError(Exception):
        """Simplified validation error used when jsonschema is unavailable."""

try:  # pragma: no cover - fallback when pydantic is not installed
    from pydantic import BaseModel, ValidationError as PydanticValidationError
except ImportError:  # pragma: no cover
    BaseModel = None  # type: ignore

    class PydanticValidationError(Exception):
        """Placeholder error used when Pydantic is unavailable."""

from ..llm.client import DeepSeekClient
from ..utils import log_ok, log_result, log_step


class LLMStageError(RuntimeError):
    """Base class for LLM operator failures."""


class LLMValidationError(LLMStageError):
    """Raised when the response cannot be parsed or validated."""


@dataclass
class LLMOperatorConfig:
    """Configuration describing the prompt and retry behaviour for an operator."""

    prompt_file: Path
    retries: int = 0
    temperature: float = 0.2
    max_tokens: int = 1200


@dataclass
class LLMRunArtifacts:
    """Stores contextual information and allows persisting operator artefacts."""

    stage_dir: Path
    base_prompt: Optional[str] = None
    prompt_text: Optional[str] = None

    def prepare(self) -> None:
        self.stage_dir.mkdir(parents=True, exist_ok=True)
        if self.base_prompt:
            (self.stage_dir / "system_prompt.md").write_text(
                self.base_prompt, encoding="utf-8"
            )
        if self.prompt_text:
            (self.stage_dir / "task_prompt.md").write_text(
                self.prompt_text, encoding="utf-8"
            )

    def dump_payload(self, payload: Mapping) -> None:
        (self.stage_dir / "input.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def dump_response(self, response_text: str, attempt: int) -> None:
        (self.stage_dir / f"raw_response_attempt{attempt}.txt").write_text(
            response_text, encoding="utf-8"
        )

    def dump_output(self, output: Mapping) -> None:
        (self.stage_dir / "output.json").write_text(
            json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def dump_error(self, error: str, attempt: int) -> None:
        (self.stage_dir / f"error_attempt{attempt}.log").write_text(
            error, encoding="utf-8"
        )


class _PydanticValidator:
    """Minimal validator that mirrors jsonschema's interface using Pydantic."""

    def __init__(self, model_cls: Type[Any]):
        if BaseModel is None:
            raise ImportError(
                "Pydantic 未安装，无法在缺少 jsonschema 时执行 Schema 校验。"
            )
        self._model_cls = model_cls

    def validate(self, instance: Mapping[str, Any]) -> None:
        try:
            if hasattr(self._model_cls, "model_validate"):
                self._model_cls.model_validate(instance)  # type: ignore[attr-defined]
            else:  # pragma: no cover - Pydantic v1 compatibility
                self._model_cls.parse_obj(instance)  # type: ignore[attr-defined]
        except PydanticValidationError as exc:  # pragma: no cover - delegated message
            raise ValidationError(str(exc)) from exc
@dataclass
class LLMOperator:
    """Base class encapsulating prompt loading, retries and schema validation."""

    name: str
    config: LLMOperatorConfig
    client: DeepSeekClient
    schema: Mapping
    model_cls: Optional[Type[Any]] = None
    base_prompt: Optional[str] = None
    logger: Optional[logging.Logger] = None
    guardrails: Optional[Mapping[str, bool]] = None
    _prompt_text: str = field(init=False)
    _validator: Draft7Validator = field(init=False)

    def __post_init__(self) -> None:
        prompt_path = Path(self.config.prompt_file)
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt 文件缺失: {prompt_path}")
        self._prompt_text = prompt_path.read_text(encoding="utf-8")
        if Draft7Validator is not None:
            self._validator = Draft7Validator(self.schema)
        else:
            if self.model_cls is None:
                raise ImportError(
                    "jsonschema 未安装，且当前 Operator 未提供 Pydantic 模型用于回退校验。"
                )
            self._validator = _PydanticValidator(self.model_cls)

    @property
    def retries(self) -> int:
        return max(0, self.config.retries)

    def _compose_messages(self, payload: Mapping, attempt: int) -> List[Mapping[str, str]]:
        serialized_payload = json.dumps(payload, indent=2, ensure_ascii=False)
        task_prompt = (
            f"{self._prompt_text}\n\n"
            "以下为可用的结构化输入(JSON)：\n"
            f"```json\n{serialized_payload}\n```\n"
            "请严格按照提示返回 JSON，禁止输出额外文本。"
        )
        messages: List[Mapping[str, str]] = []
        if self.base_prompt:
            messages.append({"role": "system", "content": self.base_prompt})
        messages.append({"role": "user", "content": task_prompt})
        if attempt > 0:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "⚠️ 上一次响应未通过 JSON Schema 校验。"
                        "请仅返回符合要求的 JSON，禁止解释。"
                    ),
                }
            )
        return messages

    def _parse_response(self, raw: str) -> Mapping:
        text = (raw or "").strip()
        if not text:
            raise LLMValidationError("响应为空")
        if text.startswith("```"):
            lines = text.splitlines()
            lines = lines[1:]
            while lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        first_brace = text.find("{")
        first_bracket = text.find("[")
        candidates = [idx for idx in (first_brace, first_bracket) if idx != -1]
        if not candidates:
            raise LLMValidationError("未找到 JSON 起始符号")
        start = min(candidates)
        closing = "}" if start == first_brace else "]"
        end = text.rfind(closing)
        if end == -1 or end < start:
            raise LLMValidationError("未找到 JSON 结束符号")
        json_segment = text[start : end + 1]
        try:
            parsed = json.loads(json_segment)
        except json.JSONDecodeError:
            try:
                parsed = json5.loads(json_segment)
            except Exception as exc:  # pragma: no cover - json5 supplies its own error type
                raise LLMValidationError("JSON 解析失败") from exc
        if not isinstance(parsed, Mapping):
            raise LLMValidationError("响应需为 JSON 对象")
        return parsed

    def execute(
        self,
        payload: Mapping,
        artefacts: LLMRunArtifacts,
        metadata: Optional[Mapping] = None,
    ) -> Tuple[Mapping, Dict]:
        artefacts.prompt_text = self._prompt_text
        artefacts.prepare()
        artefacts.dump_payload(payload)

        last_error: Optional[str] = None
        all_errors: List[str] = []
        usage: Dict = {}
        for attempt in range(self.retries + 1):
            messages = self._compose_messages(payload, attempt)
            if self.logger:
                log_step(self.logger, f"llm.{self.name}", f"Attempt {attempt + 1}")
            start = perf_counter()
            raw_response, usage = self.client.chat(
                messages=messages,
                stage=self.name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            artefacts.dump_response(raw_response, attempt)
            try:
                parsed = self._parse_response(raw_response)
                self._validator.validate(parsed)
            except (LLMValidationError, ValidationError) as exc:
                error_message = str(exc)
                artefacts.dump_error(error_message, attempt)
                last_error = error_message
                all_errors.append(error_message)
                if self.logger:
                    log_result(
                        self.logger,
                        f"llm.{self.name}",
                        f"attempt={attempt + 1} failed: {error_message}",
                    )
                if attempt >= self.retries:
                    raise LLMValidationError(error_message) from exc
                continue

            duration = perf_counter() - start
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
                log_result(
                    self.logger,
                    f"llm.{self.name}",
                    f"{token_summary}",
                )
                log_ok(
                    self.logger,
                    f"llm.{self.name}",
                    f"Completed in {duration:.2f}s",
                )
            artefacts.dump_output(parsed)
            result = dict(parsed)
            if metadata:
                result["_meta"] = dict(metadata)
            if all_errors:
                result["_previous_errors"] = all_errors
            return result, usage

        raise LLMValidationError(last_error or "未知错误")


def ensure_guardrails(
    stage: str,
    payload: Mapping,
    result: Mapping,
    universe: Sequence[str],
) -> None:
    """Guard against hallucinated tickers not present in the configured universe."""

    allowed = set(universe)
    offending: List[str] = []
    if stage == "stock_classifier":
        categories = result.get("categories", {}) if isinstance(result, Mapping) else {}
        for bucket in ("Buy", "Hold", "Reduce", "Avoid"):
            for item in categories.get(bucket, []) or []:
                symbol = item.get("symbol") if isinstance(item, Mapping) else None
                if symbol and symbol not in allowed:
                    offending.append(symbol)
    elif stage == "exposure_planner":
        for plan in result.get("allocation_plan", []) or []:
            symbol = plan.get("symbol") if isinstance(plan, Mapping) else None
            if symbol and symbol not in allowed:
                offending.append(symbol)
    elif stage == "sector_analyzer":
        for bucket in ("leading", "lagging"):
            for item in result.get(bucket, []) or []:
                symbol = item.get("symbol") if isinstance(item, Mapping) else None
                if symbol and symbol not in allowed:
                    offending.append(symbol)
    if offending:
        raise LLMValidationError(
            f"检测到未在 universe 中的 ticker: {', '.join(sorted(set(offending)))}"
        )
