"""Unit tests for model prediction, probability estimation, and inference engine."""

import pytest
from src.predictor import FraudPredictor, get_predictor, predict_message


@pytest.fixture
def predictor():
    return FraudPredictor(model_type="baseline_logistic_regression")


def test_predictor_initialization(predictor):
    assert predictor.baseline_model is not None
    assert predictor.vectorizer is not None
    assert "Baseline" in predictor.model_name or "TF-IDF" in predictor.model_name


def test_predict_fraud_message(predictor):
    fraud_text = "URGENT: Your SBI Bank account is blocked due to missing KYC. Update PAN at http://bit.ly/sbi-kyc immediately or account will be suspended."
    result = predictor.analyze(fraud_text)

    assert result["prediction"] == "FRAUD"
    assert result["predicted_label"] == 1
    assert result["fraud_probability"] > 0.50
    assert result["risk_level"] in ["HIGH", "CRITICAL"]
    assert result["indicator_count"] > 0


def test_predict_genuine_message(predictor):
    genuine_text = "Hi Rahul, are we still meeting for coffee at Starbucks at 4 PM today? Let me know if you are running late."
    result = predictor.analyze(genuine_text)

    assert result["prediction"] == "GENUINE"
    assert result["predicted_label"] == 0
    assert result["fraud_probability"] < 0.50
    assert result["risk_level"] in ["LOW", "MEDIUM"]


def test_predict_empty_text(predictor):
    result = predictor.analyze("")
    assert result["prediction"] == "GENUINE"
    assert result["confidence_percentage"] == 100.0
    assert result["risk_level"] == "LOW"


def test_singleton_get_predictor():
    p1 = get_predictor()
    p2 = get_predictor()
    assert p1 is p2

