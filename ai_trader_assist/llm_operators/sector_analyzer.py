from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

from .base import LLMOperator, LLMOperatorConfig
from ..validators.pydantic_models import SectorAnalyzerModel


@dataclass
class SectorAnalyzerOperator(LLMOperator):
    """Specialised operator for the sector analyzer stage."""

    def __init__(
        self,
        config: LLMOperatorConfig,
        client,
        schema: Mapping,
        base_prompt: Optional[str] = None,
        logger=None,
    ) -> None:
        super().__init__(
            name="sector_analyzer",
            config=config,
            client=client,
            schema=schema,
            model_cls=SectorAnalyzerModel,
            base_prompt=base_prompt,
            logger=logger,
        )
