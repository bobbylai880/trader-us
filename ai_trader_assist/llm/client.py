"""DeepSeek API client utilities."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Mapping

import requests


@dataclass
class DeepSeekClient:
    """Minimal chat-completions client for the DeepSeek API."""

    api_key: str
    model: str = "deepseek-chat"
    api_url: str = "https://api.deepseek.com/v1/chat/completions"
    timeout: float = 30.0
    max_tokens: int = 1200

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
        timeout = float(timeout_env) if timeout_env else 30.0
        max_tokens_env = os.getenv("DEEPSEEK_MAX_TOKENS")
        max_tokens = int(max_tokens_env) if max_tokens_env else 1200
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
    ) -> str:
        """Send a chat completion request and return the raw content string."""

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

        return content
