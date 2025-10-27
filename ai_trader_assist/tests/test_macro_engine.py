from ai_trader_assist.risk_engine.macro_engine import MacroRiskEngine


def test_macro_engine_exposure_range():
    engine = MacroRiskEngine()
    features = {
        "RS_SPY": 0.5,
        "RS_QQQ": 0.4,
        "VIX_Z": -0.3,
        "PUTCALL_Z": -0.1,
        "BREADTH": 0.2,
    }
    result = engine.evaluate(features)
    assert 0.4 <= result["target_exposure"] <= 0.8
    assert "drivers" in result
    assert result["risk_level"] in {"low", "medium", "high"}
