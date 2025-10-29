from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

from .base import LLMOperator, LLMOperatorConfig
from ..validators.pydantic_models import MarketAnalyzerModel


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
