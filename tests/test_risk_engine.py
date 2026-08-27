"""Unit tests for multi-factor risk assessment and risk tier calculations."""

import pytest
from src.risk_engine import compute_risk_assessment


def test_risk_low_genuine():
    mock_explanation = {
        "indicator_count": 0,
        "detected_indicators": [],
        "risk_boost": 0.0,
        "summary": "No indicators"
    }
    res = compute_risk_assessment(ai_predicted_label=0, fraud_probability=0.08, explanation_data=mock_explanation)
    assert res["risk_level"] == "LOW"
    assert res["prediction"] == "GENUINE"
    assert res["risk_score"] < 0.35


def test_risk_medium_ambiguous():
    mock_explanation = {
        "indicator_count": 1,
        "detected_indicators": [{"severity": "MEDIUM", "weight": 0.20}],
        "risk_boost": 0.20,
        "summary": "Mild indicator"
    }
    res = compute_risk_assessment(ai_predicted_label=0, fraud_probability=0.32, explanation_data=mock_explanation)
    assert res["risk_level"] in ["MEDIUM", "HIGH"]


def test_risk_high_fraud():
    mock_explanation = {
        "indicator_count": 2,
        "detected_indicators": [
            {"severity": "HIGH", "weight": 0.30},
            {"severity": "MEDIUM", "weight": 0.20}
        ],
        "risk_boost": 0.50,
        "summary": "High risk"
    }
    res = compute_risk_assessment(ai_predicted_label=1, fraud_probability=0.78, explanation_data=mock_explanation)
    assert res["risk_level"] in ["HIGH", "CRITICAL"]
    assert res["prediction"] == "FRAUD"


def test_risk_critical_override():
    mock_explanation = {
        "indicator_count": 2,
        "detected_indicators": [
            {"severity": "CRITICAL", "weight": 0.35},
            {"severity": "HIGH", "weight": 0.30}
        ],
        "risk_boost": 0.50,
        "summary": "Critical OTP request"
    }
    res = compute_risk_assessment(ai_predicted_label=1, fraud_probability=0.88, explanation_data=mock_explanation)
    assert res["risk_level"] == "CRITICAL"
    assert "DO NOT ENGAGE" in res["recommended_action"]

