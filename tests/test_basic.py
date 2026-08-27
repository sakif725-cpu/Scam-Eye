"""Unit tests for explainable indicators and end-to-end system flow."""

import pytest
from src.explanation import explain_prediction
from src.predictor import predict_message


def test_detect_otp_indicator():
    text = "Please share your 6-digit OTP verification code with our customer care executive."
    exp = explain_prediction(text)
    assert exp["indicator_count"] >= 1
    categories = [ind["category"] for ind in exp["detected_indicators"]]
    assert "Authentication Hijacking" in categories


def test_detect_kyc_threat_indicator():
    text = "Dear customer, your bank account is blocked due to missing PAN KYC. Update now."
    exp = explain_prediction(text)
    assert exp["indicator_count"] >= 1
    categories = [ind["category"] for ind in exp["detected_indicators"]]
    assert "Coercive Account Threat" in categories


def test_detect_lottery_indicator():
    text = "Congratulations! You have won Rs 25,00,000 cash prize in KBC Lucky Draw."
    exp = explain_prediction(text)
    assert exp["indicator_count"] >= 1
    categories = [ind["category"] for ind in exp["detected_indicators"]]
    assert "Advance-Fee / Prize Scam" in categories


def test_detect_upi_trap_indicator():
    text = "Enter your UPI PIN to receive 5000 rupees cashback instantly."
    exp = explain_prediction(text)
    assert exp["indicator_count"] >= 1
    categories = [ind["category"] for ind in exp["detected_indicators"]]
    assert "Payment / Refund Trap" in categories


def test_detect_suspicious_url():
    text = "Update details at http://sbi-kyc-update.xyz immediately."
    exp = explain_prediction(text)
    categories = [ind["category"] for ind in exp["detected_indicators"]]
    assert "Phishing Link" in categories


def test_end_to_end_delivery_message():
    text = "Your Amazon order #402-8921892 has been shipped. Track at https://www.amazon.in/orders"
    res = predict_message(text)
    assert res["prediction"] == "GENUINE"
    assert res["risk_level"] in ["LOW", "MEDIUM"]


def test_end_to_end_electricity_bill_scam():
    text = "Dear user, electricity will be disconnected tonight at 9:30 PM. Call electricity officer at 9123456789."
    res = predict_message(text)
    assert res["prediction"] == "FRAUD"
    assert res["risk_level"] in ["HIGH", "CRITICAL"]

