from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

from .base import LLMOperator, LLMOperatorConfig
from ..validators.pydantic_models import ExposurePlannerModel


@dataclass
class ExposurePlannerOperator(LLMOperator):
    """Specialised operator for the exposure planner stage."""

    def __init__(
        self,
        config: LLMOperatorConfig,
        client,
        schema: Mapping,
        base_prompt: Optional[str] = None,
        logger=None,
    ) -> None:
        super().__init__(
            name="exposure_planner",
            config=config,
            client=client,
            schema=schema,
            model_cls=ExposurePlannerModel,
            base_prompt=base_prompt,
            logger=logger,
        )
