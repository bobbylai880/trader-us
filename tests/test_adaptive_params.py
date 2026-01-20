"""Tests for AdaptiveParameterManager - 自适应参数系统测试"""
import pytest
from ai_trader_assist.risk_engine.market_regime import MarketRegime
from ai_trader_assist.risk_engine.adaptive_params import (
    RegimeParameters,
    ScoringWeights,
    AdaptiveParameterManager,
    DEFAULT_REGIME_PARAMETERS,
)


class TestScoringWeights:
    def test_default_weights_sum_to_one(self):
        weights = ScoringWeights()
        total = sum(weights.to_dict().values())
        assert abs(total - 1.0) < 0.01

    def test_from_dict(self):
        data = {"trend": 0.30, "momentum": 0.25, "mean_reversion": 0.10}
        weights = ScoringWeights.from_dict(data)
        assert weights.trend == 0.30
        assert weights.momentum == 0.25
        assert weights.mean_reversion == 0.10

    def test_to_dict_roundtrip(self):
        original = ScoringWeights(trend=0.35, momentum=0.20)
        restored = ScoringWeights.from_dict(original.to_dict())
        assert restored.trend == original.trend
        assert restored.momentum == original.momentum


class TestRegimeParameters:
    def test_default_values(self):
        params = RegimeParameters()
        assert params.max_exposure == 0.70
        assert params.buy_threshold == 0.60
        assert params.stop_atr_mult == 1.5

    def test_to_dict(self):
        params = RegimeParameters(max_exposure=0.80, buy_threshold=0.55)
        d = params.to_dict()
        assert d["max_exposure"] == 0.80
        assert d["buy_threshold"] == 0.55
        assert "scoring_weights" in d


class TestDefaultRegimeParameters:
    def test_all_regimes_defined(self):
        for regime in MarketRegime:
            assert regime in DEFAULT_REGIME_PARAMETERS

    def test_bull_trend_more_aggressive(self):
        bull = DEFAULT_REGIME_PARAMETERS[MarketRegime.BULL_TREND]
        bear = DEFAULT_REGIME_PARAMETERS[MarketRegime.BEAR_TREND]

        # bull should have higher exposure limits
        assert bull.max_exposure > bear.max_exposure
        # bull should have lower buy threshold (easier to buy)
        assert bull.buy_threshold < bear.buy_threshold
        # bull should have wider stops
        assert bull.stop_atr_mult > bear.stop_atr_mult

    def test_bear_trend_conservative(self):
        bear = DEFAULT_REGIME_PARAMETERS[MarketRegime.BEAR_TREND]
        assert bear.max_exposure <= 0.35
        assert bear.buy_threshold >= 0.75
        assert bear.scoring_weights.mean_reversion > 0.20

    def test_range_bound_balanced(self):
        rng = DEFAULT_REGIME_PARAMETERS[MarketRegime.RANGE_BOUND]
        assert 0.50 <= rng.max_exposure <= 0.70
        assert rng.scoring_weights.mean_reversion > 0.10


class TestAdaptiveParameterManager:
    @pytest.fixture
    def manager(self):
        return AdaptiveParameterManager()

    def test_default_regime_is_unknown(self, manager):
        assert manager.current_regime == MarketRegime.UNKNOWN

    def test_set_regime(self, manager):
        manager.set_regime(MarketRegime.BULL_TREND)
        assert manager.current_regime == MarketRegime.BULL_TREND

    def test_get_params_for_regime(self, manager):
        params = manager.get_params(MarketRegime.BULL_TREND)
        assert params.max_exposure == 0.90
        assert params.buy_threshold == 0.55

    def test_current_params_follows_regime(self, manager):
        # given: set to bear trend
        manager.set_regime(MarketRegime.BEAR_TREND)

        # when: get current params
        params = manager.current_params

        # then: should match bear trend params
        assert params.max_exposure == 0.30
        assert params.buy_threshold == 0.80

    def test_convenience_properties(self, manager):
        manager.set_regime(MarketRegime.BULL_TREND)
        assert manager.max_exposure == 0.90
        assert manager.buy_threshold == 0.55
        assert manager.stop_atr_mult == 2.0

    def test_override_params(self, manager):
        # given: custom override
        override = RegimeParameters(max_exposure=0.50, buy_threshold=0.70)
        manager.set_override(override)

        # when: get params (even for different regime)
        manager.set_regime(MarketRegime.BULL_TREND)

        # then: override takes precedence
        assert manager.max_exposure == 0.50
        assert manager.buy_threshold == 0.70

    def test_clear_override(self, manager):
        # given: override is set
        override = RegimeParameters(max_exposure=0.50)
        manager.set_override(override)
        manager.set_regime(MarketRegime.BULL_TREND)

        # when: clear override
        manager.clear_override()

        # then: should return to regime params
        assert manager.max_exposure == 0.90

    def test_get_summary(self, manager):
        manager.set_regime(MarketRegime.BEAR_RALLY)
        summary = manager.get_summary()

        assert summary["regime"] == "bear_rally"
        assert "max_exposure" in summary
        assert "buy_threshold" in summary
        assert summary["has_override"] is False

    def test_from_config_uses_custom_values(self):
        # given: config with custom bear trend params
        config = {
            "market_regimes": {
                "bear_trend": {
                    "max_exposure": 0.25,
                    "buy_threshold": 0.85,
                    "scoring_weights": {
                        "trend": 0.05,
                        "mean_reversion": 0.40,
                    }
                }
            }
        }

        # when: create from config
        manager = AdaptiveParameterManager.from_config(config)
        params = manager.get_params(MarketRegime.BEAR_TREND)

        # then: should use custom values
        assert params.max_exposure == 0.25
        assert params.buy_threshold == 0.85
        assert params.scoring_weights.mean_reversion == 0.40

    def test_from_config_preserves_defaults(self):
        # given: config with only partial override
        config = {
            "market_regimes": {
                "bull_trend": {
                    "max_exposure": 0.95,
                }
            }
        }

        # when: create from config
        manager = AdaptiveParameterManager.from_config(config)
        params = manager.get_params(MarketRegime.BULL_TREND)

        # then: should use custom exposure but default buy_threshold
        assert params.max_exposure == 0.95
        assert params.buy_threshold == 0.55  # default

    def test_regime_transition_updates_params(self, manager):
        # given: start in bull trend
        manager.set_regime(MarketRegime.BULL_TREND)
        bull_exposure = manager.max_exposure

        # when: transition to bear
        manager.set_regime(MarketRegime.BEAR_TREND)
        bear_exposure = manager.max_exposure

        # then: params should change
        assert bear_exposure < bull_exposure

    def test_scoring_weights_accessible(self, manager):
        manager.set_regime(MarketRegime.RANGE_BOUND)
        weights = manager.scoring_weights

        assert isinstance(weights, ScoringWeights)
        assert weights.mean_reversion > 0
