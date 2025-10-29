"""Optional Pydantic models mirroring the JSON Schema definitions."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, validator


class MarketAnalyzerModel(BaseModel):
    risk_level: str
    bias: str
    drivers: List[str]
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


class SectorEvidence(BaseModel):
    symbol: str
    evidence: str


class SectorAnalyzerModel(BaseModel):
    leading: List[SectorEvidence]
    lagging: List[SectorEvidence]
    focus_points: List[str] = Field(default_factory=list)
    data_gaps: List[str] = Field(default_factory=list)


class StockItem(BaseModel):
    symbol: str
    premarket_score: float
    drivers: List[str]
    risks: List[str]
    trend_change: Optional[str] = None
    momentum_strength: Optional[str] = None
    trend_explanation: Optional[str] = None
    news_highlights: List[str] = Field(default_factory=list)

    @validator("premarket_score")
    def check_score(cls, value: float) -> float:
        if not 0 <= value <= 100:
            raise ValueError("premarket_score 必须在 [0, 100]")
        return value

    @validator("momentum_strength")
    def check_momentum(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        if value not in {"weak", "neutral", "strong"}:
            raise ValueError("momentum_strength 必须为 weak/neutral/strong")
        return value


class StockCategories(BaseModel):
    Buy: List[StockItem] = Field(default_factory=list)
    Hold: List[StockItem] = Field(default_factory=list)
    Reduce: List[StockItem] = Field(default_factory=list)
    Avoid: List[StockItem] = Field(default_factory=list)


class StockClassifierModel(BaseModel):
    categories: StockCategories
    notes: List[str] = Field(default_factory=list)
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
