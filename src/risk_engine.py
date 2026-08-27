"""Real-Time Fraud Risk Engine for AI Fraud Call and Message Detector.

Synthesizes AI model probabilities with heuristic explainability indicators
to compute a nuanced, calibrated Risk Score (0.0 to 1.0) and 4-tier Risk Level:
- LOW: Legitimate transactional or conversational message.
- MEDIUM: Promotional or low-threat ambiguous communication.
- HIGH: Strong statistical and structural fraud markers.
- CRITICAL: Extreme fraud probability combined with dangerous action triggers (OTP/KYC/UPI).
"""

from typing import Dict, Any
from config import settings


def compute_risk_assessment(
    ai_predicted_label: int,
    fraud_probability: float,
    explanation_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Computes a multi-factor risk assessment combining AI inference and heuristic indicators.

    The final decision is primarily based on the AI model's probability distribution,
    while detected behavioral indicators dynamically refine the risk level and safety recommendations.

    Args:
        ai_predicted_label: Binary label from AI classifier (1=FRAUD, 0=GENUINE).
        fraud_probability: Calibrated probability (0.0 to 1.0) of message being fraudulent.
        explanation_data: Output dictionary from src.explanation.explain_prediction.

    Returns:
        Structured dictionary containing:
        - prediction: 'FRAUD' or 'GENUINE'
        - confidence_score: Confidence percentage of the predicted label
        - fraud_probability: Raw fraud probability (0.0 to 1.0)
        - risk_score: Synthesized risk metric (0.0 to 1.0)
        - risk_level: 'LOW', 'MEDIUM', 'HIGH', or 'CRITICAL'
        - recommended_action: Tailored user safety advice
        - reasoning: Concise explanation of risk determination
    """
    indicator_count = explanation_data.get("indicator_count", 0)
    detected_indicators = explanation_data.get("detected_indicators", [])
    risk_boost = explanation_data.get("risk_boost", 0.0)

    # Check for critical triggers
    severities = [ind.get("severity") for ind in detected_indicators]
    has_critical = "CRITICAL" in severities
    has_high = "HIGH" in severities

    # Base risk score starts from the AI model's fraud probability
    # Indicator boost accounts for 20% influence while AI accounts for 80%
    synthesized_risk = (0.80 * fraud_probability) + (0.20 * (risk_boost / 0.50 if risk_boost > 0 else 0.0))

    # Safety override: if model is borderline genuine (e.g. 0.35) but explicit CRITICAL indicator (OTP/UPI trap) is found
    if fraud_probability >= 0.30 and has_critical:
        synthesized_risk = max(synthesized_risk, 0.75)

    synthesized_risk = max(0.0, min(round(synthesized_risk, 4), 1.0))

    # Determine Risk Tier
    if synthesized_risk >= 0.80 or (fraud_probability >= 0.70 and has_critical):
        risk_level = "CRITICAL"
        recommended_action = (
            "🚨 DO NOT ENGAGE: This communication exhibits critical fraud signatures. "
            "Never share OTPs, PINs, or click any attached links. Report and block the sender immediately."
        )
    elif synthesized_risk >= 0.55 or (ai_predicted_label == settings.LABEL_FRAUD):
        risk_level = "HIGH"
        recommended_action = (
            "⚠️ HIGH FRAUD PROBABILITY: The message contains strong fraud characteristics or suspicious links. "
            "Do not transfer money or share personal documents. Verify directly via the official service app."
        )
    elif synthesized_risk >= 0.30 or indicator_count > 0:
        risk_level = "MEDIUM"
        recommended_action = (
            "⚡ EXERCISE CAUTION: Some promotional or unverified elements were detected. "
            "Check the sender's official credentials before taking any action."
        )
    else:
        risk_level = "LOW"
        recommended_action = (
            "✅ APPEARS SAFE: Standard legitimate communication detected. "
            "As always, maintain standard digital hygiene and never disclose account passwords."
        )

    # Human-readable prediction and confidence
    if ai_predicted_label == settings.LABEL_FRAUD or fraud_probability >= settings.CLASSIFICATION_THRESHOLD:
        prediction_str = "FRAUD"
        confidence_pct = round(fraud_probability * 100, 2)
    else:
        prediction_str = "GENUINE"
        confidence_pct = round((1.0 - fraud_probability) * 100, 2)

    reasoning = (
        f"AI Classifier estimated {fraud_probability * 100:.1f}% fraud probability. "
        f"{indicator_count} suspicious pattern(s) identified. "
        f"Synthesized Risk Score: {synthesized_risk * 100:.1f}/100."
    )

    return {
        "prediction": prediction_str,
        "predicted_label": int(prediction_str == "FRAUD"),
        "confidence_percentage": confidence_pct,
        "fraud_probability": round(fraud_probability, 4),
        "risk_score": synthesized_risk,
        "risk_level": risk_level,
        "recommended_action": recommended_action,
        "reasoning": reasoning,
        "indicator_count": indicator_count
    }


if __name__ == "__main__":
    from src.explanation import explain_prediction
    sample_text = "Your account is deactivated. Update KYC at http://bit.ly/bank-kyc"
    exp = explain_prediction(sample_text)
    assessment = compute_risk_assessment(ai_predicted_label=1, fraud_probability=0.92, explanation_data=exp)
    print("Risk Engine Output:")
    print(f"Prediction: {assessment['prediction']}")
    print(f"Confidence: {assessment['confidence_percentage']}%")
    print(f"Risk Level: {assessment['risk_level']}")
    print(f"Action: {assessment['recommended_action']}")

