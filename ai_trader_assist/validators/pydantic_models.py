"""Optional Pydantic models mirroring the JSON Schema definitions."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


class MarketDriver(BaseModel):
    factor: str
    evidence: str
    direction: str


class MarketAnalyzerModel(BaseModel):
    risk_level: str
    bias: str
    drivers: List[Union[str, MarketDriver]]
    summary: str
    data_gaps: List[str] = Field(default_factory=list)
    premarket_flags: List[str] = Field(default_factory=list)
    news_sentiment: Optional[float] = None

    @validator("risk_level")
    def validate_risk(cls, value: str) -> str:
        if value not in {"low", "medium", "high"}:
            raise ValueError("risk_level 必须为 low/medium/high")
        return value

    @validator("bias")
    def validate_bias(cls, value: str) -> str:
        if value not in {"bearish", "neutral", "bullish"}:
            raise ValueError("bias 必须为 bearish/neutral/bullish")
        return value

    @validator("news_sentiment")
    def validate_sentiment(cls, value: Optional[float]) -> Optional[float]:
        if value is None:
            return value
        if not -1 <= value <= 1:
            raise ValueError("news_sentiment 必须在 [-1, 1]")
        return value


class SectorEvidenceDetail(BaseModel):
    symbol: Optional[str] = None
    sector: Optional[str] = None
    name: Optional[str] = None
    evidence: Optional[Union[str, Dict[str, Any], List[Any]]] = None
    comment: Optional[str] = None
    composite_score: Optional[float] = None
    news_sentiment: Optional[float] = None
    news_highlights: List[Union[str, Dict[str, Any]]] = Field(default_factory=list)

    @validator("news_sentiment")
    def validate_sentiment(cls, value: Optional[float]) -> Optional[float]:
        if value is None:
            return value
        if not -1 <= value <= 1:
            raise ValueError("news_sentiment 必须在 [-1, 1]")
        return value

    @validator("name", always=True)
    def ensure_identifier(
        cls, value: Optional[str], values: Dict[str, Any]
    ) -> Optional[str]:
        if not value and not values.get("symbol") and not values.get("sector"):
            raise ValueError("sector item 需至少包含 symbol/sector/name 之一")
        return value


SectorEvidence = Union[str, SectorEvidenceDetail]


class FocusPointDetail(BaseModel):
    topic: str
    rationale: Optional[str] = None
    risk: Optional[str] = None
    action: Optional[str] = None


FocusPoint = Union[str, FocusPointDetail]


class SectorAnalyzerModel(BaseModel):
    leading: List[SectorEvidence]
    lagging: List[SectorEvidence]
    focus_points: List[FocusPoint] = Field(default_factory=list)
    data_gaps: List[str] = Field(default_factory=list)


class StockDriver(BaseModel):
    metric: str
    value: Optional[Union[float, str]] = None
    direction: Optional[str] = None
    evidence: Optional[str] = None


class StockRisk(BaseModel):
    metric: str
    value: Optional[Union[float, str]] = None
    direction: Optional[str] = None
    comment: Optional[str] = None


RiskEntry = Union[str, StockRisk]


class StockItem(BaseModel):
    symbol: str
    premarket_score: Optional[float] = None
    drivers: List[Union[str, StockDriver]]
    risks: List[RiskEntry]
    trend_change: Optional[str] = None
    momentum_strength: Optional[Union[str, float]] = None
    trend_explanation: Optional[str] = None
    news_highlights: List[Union[str, Dict[str, Any]]] = Field(default_factory=list)

    @validator("premarket_score")
    def check_score(cls, value: Optional[float]) -> Optional[float]:
        if value is None:
            return value
        if not 0 <= value <= 100:
            raise ValueError("premarket_score 必须在 [0, 100]")
        return value

    @validator("momentum_strength")
    def check_momentum(
        cls, value: Optional[Union[str, float]]
    ) -> Optional[Union[str, float]]:
        if value is None:
            return value
        if isinstance(value, str):
            if value not in {"weak", "neutral", "strong"}:
                raise ValueError("momentum_strength 必须为 weak/neutral/strong 或 0~1 数值")
            return value
        if not 0 <= value <= 1:
            raise ValueError("momentum_strength 数值必须在 [0, 1]")
        return value


class StockCategories(BaseModel):
    Buy: List[StockItem] = Field(default_factory=list)
    Hold: List[StockItem] = Field(default_factory=list)
    Reduce: List[StockItem] = Field(default_factory=list)
    Avoid: List[StockItem] = Field(default_factory=list)


class UnclassifiedItem(BaseModel):
    symbol: str
    reason: str


class StockClassifierModel(BaseModel):
    categories: StockCategories
    notes: List[str] = Field(default_factory=list)
    unclassified: List[UnclassifiedItem] = Field(default_factory=list)
    data_gaps: List[str] = Field(default_factory=list)


class AllocationPlanItem(BaseModel):
    symbol: str
    weight: float
    rationale: Optional[str] = None

    @validator("weight")
    def check_weight(cls, value: float) -> float:
        if not 0 <= value <= 1:
            raise ValueError("weight 必须在 [0, 1]")
        return value


class ExposurePlannerModel(BaseModel):
    target_exposure: float
    allocation_plan: List[AllocationPlanItem]
    constraints: List[str]
    data_gaps: List[str] = Field(default_factory=list)

    @validator("target_exposure")
    def check_target(cls, value: float) -> float:
        if not 0 <= value <= 1:
            raise ValueError("target_exposure 必须在 [0, 1]")
        return value


class ReportComposerModel(BaseModel):
    markdown: str
    sections: dict
    data_gaps: List[str] = Field(default_factory=list)
