"""Tests for refactored MacroRiskEngine with market regime integration"""
import pytest
from ai_trader_assist.risk_engine.macro_engine import MacroRiskEngine
from ai_trader_assist.risk_engine.market_regime import MarketRegime


class TestMacroRiskEngineBasic:
    @pytest.fixture
    def engine(self):
        return MacroRiskEngine()

    def test_evaluate_returns_required_fields(self, engine):
        features = {
            "RS_SPY": 0.5,
            "RS_QQQ": 0.4,
            "VIX_Z": -0.3,
            "PUTCALL_Z": -0.1,
            "BREADTH": 0.55,
        }
        result = engine.evaluate(features)

        assert "target_exposure" in result
        assert "risk_level" in result
        assert "drivers" in result
        assert 0.0 <= result["target_exposure"] <= 1.0

    def test_evaluate_with_regime_features(self, engine):
        # given: features including regime detection inputs
        features = {
            "RS_SPY": 0.6,
            "RS_QQQ": 0.5,
            "VIX_Z": -0.5,
            "PUTCALL_Z": -0.2,
            "BREADTH": 0.65,
            "SPY_above_sma200": True,
            "SPY_above_sma50": True,
            "SPY_sma50_slope": 0.002,
            "market_breadth": 0.65,
            "vix_level": 14.0,
        }

        # when: evaluate
        result = engine.evaluate(features)

        # then: should include regime info
        assert "regime" in result
        assert result["regime"] in [r.value for r in MarketRegime]


class TestMacroRiskEngineBullMarket:
    @pytest.fixture
    def bull_features(self):
        return {
            "RS_SPY": 0.8,
            "RS_QQQ": 0.7,
            "VIX_Z": -1.0,
            "PUTCALL_Z": -0.5,
            "BREADTH": 0.75,
            "MOMENTUM": 0.5,
            "SMA_POSITION": 0.8,
            "SPY_above_sma200": True,
            "SPY_above_sma50": True,
            "SPY_sma50_slope": 0.003,
            "market_breadth": 0.75,
            "vix_level": 12.0,
            "vix_term_structure": "contango",
        }

    def test_bull_market_high_exposure(self, bull_features):
        engine = MacroRiskEngine()
        result = engine.evaluate(bull_features)

        # should allow reasonable exposure in bull market
        assert result["target_exposure"] >= 0.45
        assert result["risk_level"] in ["low", "medium"]

    def test_bull_market_params(self, bull_features):
        engine = MacroRiskEngine()
        result = engine.evaluate(bull_features)

        # if regime detected, should include adaptive params
        if "regime_params" in result:
            params = result["regime_params"]
            assert params["buy_threshold"] <= 0.60
            assert params["max_exposure"] >= 0.80


class TestMacroRiskEngineBearMarket:
    @pytest.fixture
    def bear_features(self):
        return {
            "RS_SPY": -0.3,
            "RS_QQQ": -0.4,
            "VIX_Z": 2.0,
            "PUTCALL_Z": 1.5,
            "BREADTH": 0.25,
            "MOMENTUM": -0.5,
            "SMA_POSITION": -0.8,
            "SPY_above_sma200": False,
            "SPY_above_sma50": False,
            "SPY_sma50_slope": -0.004,
            "market_breadth": 0.20,
            "vix_level": 35.0,
            "vix_term_structure": "backwardation",
        }

    def test_bear_market_low_exposure(self, bear_features):
        engine = MacroRiskEngine()
        result = engine.evaluate(bear_features)

        # should limit exposure in bear market
        assert result["target_exposure"] <= 0.50
        assert result["risk_level"] in ["medium", "high"]

    def test_bear_market_params(self, bear_features):
        engine = MacroRiskEngine()
        result = engine.evaluate(bear_features)

        if "regime_params" in result:
            params = result["regime_params"]
            assert params["buy_threshold"] >= 0.70
            assert params["max_exposure"] <= 0.40


class TestMacroRiskEngineConfig:
    def test_from_config_custom_weights(self):
        config = {
            "macro_risk": {
                "weights": {
                    "VIX_Z": -1.5,
                    "BREADTH": 1.0,
                }
            }
        }
        engine = MacroRiskEngine.from_config(config)

        # engine should use custom weights (stored in config)
        assert engine.config.weights.get("VIX_Z") == -1.5
        assert engine.config.weights.get("BREADTH") == 1.0

    def test_from_config_custom_thresholds(self):
        config = {
            "macro_risk": {
                "risk_thresholds": {
                    "high_risk": 0.30,
                    "medium_risk": 0.60,
                },
                "vix_thresholds": {
                    "extreme": 40.0,
                }
            }
        }
        engine = MacroRiskEngine.from_config(config)

        assert engine.config.risk_thresholds["high_risk"] == 0.30
        assert engine.config.vix_thresholds["extreme"] == 40.0


class TestMacroRiskEngineEdgeCases:
    @pytest.fixture
    def engine(self):
        return MacroRiskEngine()

    def test_empty_features(self, engine):
        result = engine.evaluate({})
        assert "target_exposure" in result
        assert "risk_level" in result

    def test_partial_features(self, engine):
        features = {"VIX_Z": 0.5, "BREADTH": 0.4}
        result = engine.evaluate(features)

        assert 0.0 <= result["target_exposure"] <= 1.0

    def test_extreme_vix(self, engine):
        features = {
            "VIX_Z": 3.0,
            "BREADTH": 0.2,
            "vix_level": 45.0,
        }
        result = engine.evaluate(features)

        # extreme VIX should trigger defensive mode
        assert result["target_exposure"] <= 0.40
        assert result["risk_level"] == "high"

    def test_drivers_populated(self, engine):
        features = {
            "RS_SPY": 0.6,
            "VIX_Z": -0.5,
            "BREADTH": 0.7,
        }
        result = engine.evaluate(features)

        assert len(result["drivers"]) > 0
        assert any("VIX" in d or "breadth" in d.lower() for d in result["drivers"])


class TestMacroRiskEngineIntegration:
    def test_full_pipeline_bull_to_bear(self):
        engine = MacroRiskEngine()

        # bull market evaluation
        bull = {
            "RS_SPY": 0.7, "VIX_Z": -0.8, "BREADTH": 0.70,
            "SPY_above_sma200": True, "SPY_above_sma50": True,
            "vix_level": 13.0,
        }
        bull_result = engine.evaluate(bull)

        # bear market evaluation
        bear = {
            "RS_SPY": -0.5, "VIX_Z": 2.0, "BREADTH": 0.25,
            "SPY_above_sma200": False, "SPY_above_sma50": False,
            "vix_level": 38.0,
        }
        bear_result = engine.evaluate(bear)

        # exposure should decrease significantly
        assert bull_result["target_exposure"] > bear_result["target_exposure"]
        assert bull_result["risk_level"] != bear_result["risk_level"] or \
               bull_result["target_exposure"] - bear_result["target_exposure"] > 0.2
