"""DeepSeek API client utilities."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional

import requests


@dataclass
class DeepSeekClient:
    """Minimal chat-completions client for the DeepSeek API."""

    api_key: str
    model: str = "deepseek-chat"
    api_url: str = "https://api.deepseek.com/v1/chat/completions"
    timeout: float = 90.0
    max_tokens: int = 8192

    @classmethod
    def from_env(cls) -> "DeepSeekClient":
        """Instantiate the client using environment variables."""

        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("缺少 DEEPSEEK_API_KEY 环境变量，无法调用 DeepSeek API")
        model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        api_url = os.getenv(
            "DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions"
        )
        timeout_env = os.getenv("DEEPSEEK_TIMEOUT")
        timeout = float(timeout_env) if timeout_env else 90.0
        max_tokens_env = os.getenv("DEEPSEEK_MAX_TOKENS")
        max_tokens = int(max_tokens_env) if max_tokens_env else 8192
        return cls(
            api_key=api_key,
            model=model,
            api_url=api_url,
            timeout=timeout,
            max_tokens=max_tokens,
        )

    def chat(
        self,
        messages: List[Mapping[str, str]],
        stage: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> tuple[str, Dict]:
        """Send a chat completion request and return the content and usage."""

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            response = requests.post(
                self.api_url, headers=headers, json=payload, timeout=self.timeout
            )
        except requests.Timeout as exc:  # pragma: no cover - requests specific
            raise TimeoutError("DeepSeek 请求超时") from exc
        except requests.RequestException as exc:  # pragma: no cover - requests specific
            raise RuntimeError(f"DeepSeek 请求失败: {exc}") from exc

        if response.status_code >= 400:
            message = response.text
            raise RuntimeError(
                f"DeepSeek 阶段 {stage} 调用失败: HTTP {response.status_code} {message}"
            )

        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            raise ValueError("DeepSeek 返回非结构化文本") from exc

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("DeepSeek 返回非结构化文本") from exc

        if not isinstance(content, str):
            raise ValueError("DeepSeek 返回非结构化文本")

        usage = data.get("usage") if isinstance(data, dict) else {}
        if usage is None:
            usage = {}

        return content, usage


def get_model_for_operator(
    operator_name: str,
    operator_config: Mapping[str, object],
    base_model: Optional[str] = None,
) -> str:
    """Resolve the DeepSeek model for a specific operator stage.

    The resolution order is:

    1. Stage-specific environment variable ``DEEPSEEK_MODEL_<STAGE>``.
    2. ``model`` defined inside the operator configuration (e.g. ``base.json``).
    3. Global ``DEEPSEEK_MODEL`` environment variable.
    4. ``base_model`` argument provided by the caller, falling back to
       ``"deepseek-chat"`` when absent.
    """

    stage_env_var = f"DEEPSEEK_MODEL_{operator_name.upper()}"
    stage_model = os.getenv(stage_env_var)
    if stage_model:
        return stage_model

    if isinstance(operator_config, Mapping):
        configured = operator_config.get("model")
        if isinstance(configured, str) and configured.strip():
            return configured.strip()

    global_model = os.getenv("DEEPSEEK_MODEL")
    if global_model:
        return global_model

    if base_model and base_model.strip():
        return base_model.strip()

    return "deepseek-chat"
