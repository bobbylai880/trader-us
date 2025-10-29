from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

from .base import LLMOperator, LLMOperatorConfig


@dataclass
class ReportComposerOperator(LLMOperator):
    """Specialised operator for the report composer stage."""

    def __init__(
        self,
        config: LLMOperatorConfig,
        client,
        schema: Mapping,
        base_prompt: Optional[str] = None,
        logger=None,
    ) -> None:
        super().__init__(
            name="report_composer",
            config=config,
            client=client,
            schema=schema,
            base_prompt=base_prompt,
            logger=logger,
        )
