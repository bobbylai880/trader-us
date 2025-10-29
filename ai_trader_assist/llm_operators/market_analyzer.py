from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

from .base import LLMOperator, LLMOperatorConfig

try:  # pragma: no cover - exercised when Pydantic is optional dependency
    from ..validators.pydantic_models import MarketAnalyzerModel
except ImportError:  # pragma: no cover - fallback when Pydantic missing
    MarketAnalyzerModel = None  # type: ignore


@dataclass
class MarketAnalyzerOperator(LLMOperator):
    """Specialised operator for the market analyzer stage."""

    def __init__(
        self,
        config: LLMOperatorConfig,
        client,
        schema: Mapping,
        base_prompt: Optional[str] = None,
        logger=None,
    ) -> None:
        super().__init__(
            name="market_analyzer",
            config=config,
            client=client,
            schema=schema,
            model_cls=MarketAnalyzerModel,
            base_prompt=base_prompt,
            logger=logger,
        )
