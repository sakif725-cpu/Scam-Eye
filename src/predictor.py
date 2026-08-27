"""Unified Inference Engine for AI Fraud Call and Message Detector.

Loads active selected model (Baseline Logistic Regression / Transformer DistilBERT),
processes raw text, computes fraud probability, runs explanation, and calculates risk scores.
"""

from typing import Dict, Any, Optional
import json
import logging
from pathlib import Path
import joblib
import numpy as np
import torch

from config import settings
from src.data_preprocessor import clean_text_safely
from src.explanation import explain_prediction
from src.risk_engine import compute_risk_assessment

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class FraudPredictor:
    """Unified predictor class supporting baseline ML and deep learning transformer models."""

    def __init__(self, model_type: Optional[str] = None):
        """Initializes the predictor, loading the selected or specified model.

        Args:
            model_type: Optional explicit model override ('baseline' or 'transformer').
        """
        self.model_type = model_type
        self.model_name = "Unknown"
        self.vectorizer = None
        self.baseline_model = None
        self.transformer_model = None
        self.transformer_tokenizer = None
        self.device = torch.device(settings.DEVICE)
        self.load_model()

    def get_selected_model_type(self) -> str:
        """Determines active model type from selected_model.json or filesystem."""
        if self.model_type:
            return self.model_type

        if settings.SELECTED_MODEL_PATH.exists():
            try:
                with open(settings.SELECTED_MODEL_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("active_model_type", "baseline_logistic_regression")
            except Exception as e:
                logger.warning("Could not read selected_model.json: %s. Defaulting to baseline.", e)

        # Check if transformer model exists
        if (settings.TRANSFORMER_MODEL_DIR / "model.safetensors").exists() or (settings.TRANSFORMER_MODEL_DIR / "pytorch_model.bin").exists():
            return "transformer_distilbert"

        return "baseline_logistic_regression"

    def load_model(self) -> None:
        """Loads model weights and components based on selected model configuration."""
        active_type = self.get_selected_model_type()
        logger.info("Initializing FraudPredictor with active model type: %s", active_type)

        if "transformer" in active_type.lower():
            try:
                self._load_transformer()
                self.model_name = "DistilBERT (Fine-Tuned)"
                return
            except Exception as err:
                logger.warning("Failed to load Transformer model (%s). Falling back to baseline.", err)

        # Default / Fallback: Baseline Logistic Regression
        self._load_baseline()
        self.model_name = "TF-IDF + Logistic Regression"

    def _load_baseline(self) -> None:
        """Loads baseline TF-IDF vectorizer and Logistic Regression classifier."""
        if not (settings.BASELINE_VECTORIZER_PATH.exists() and settings.BASELINE_MODEL_PATH.exists()):
            logger.info("Baseline model artifacts not found. Training baseline model now...")
            from src.train_baseline import train_and_evaluate_baseline
            train_and_evaluate_baseline()

        self.vectorizer = joblib.load(settings.BASELINE_VECTORIZER_PATH)
        self.baseline_model = joblib.load(settings.BASELINE_MODEL_PATH)
        logger.info("Successfully loaded baseline model and vectorizer.")

    def _load_transformer(self) -> None:
        """Loads fine-tuned DistilBERT transformer model and tokenizer."""
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        if not (settings.TRANSFORMER_MODEL_DIR / "config.json").exists():
            raise FileNotFoundError(f"Transformer model not found at {settings.TRANSFORMER_MODEL_DIR}")

        self.transformer_tokenizer = AutoTokenizer.from_pretrained(str(settings.TRANSFORMER_MODEL_DIR))
        self.transformer_model = AutoModelForSequenceClassification.from_pretrained(str(settings.TRANSFORMER_MODEL_DIR))
        self.transformer_model.to(self.device)
        self.transformer_model.eval()
        logger.info("Successfully loaded DistilBERT model on %s.", self.device)

    def predict_proba(self, cleaned_text: str) -> float:
        """Calculates calibrated probability that the input text is fraudulent.

        Args:
            cleaned_text: Cleaned and normalized text.

        Returns:
            Float probability value between 0.0 and 1.0 (1.0 = FRAUD).
        """
        if self.transformer_model is not None and self.transformer_tokenizer is not None:
            # Transformer inference
            inputs = self.transformer_tokenizer(
                cleaned_text,
                return_tensors="pt",
                truncation=True,
                max_length=settings.TRANSFORMER_MAX_LENGTH,
                padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.transformer_model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                # Index 1 is FRAUD
                return float(probabilities[1])

        elif self.baseline_model is not None and self.vectorizer is not None:
            # Baseline ML inference
            feat = self.vectorizer.transform([cleaned_text])
            probs = self.baseline_model.predict_proba(feat)[0]
            # Index 1 is FRAUD
            return float(probs[1])

        else:
            raise RuntimeError("No active model is loaded in FraudPredictor.")

    def analyze(self, raw_text: str) -> Dict[str, Any]:
        """Runs the complete fraud detection, explanation, and risk analysis pipeline.

        Args:
            raw_text: Raw incoming text message or call transcript.

        Returns:
            Comprehensive dictionary containing predictions, risk scores, explanations, and advice.
        """
        if not raw_text or not raw_text.strip():
            return {
                "input_text": "",
                "clean_text": "",
                "prediction": "GENUINE",
                "confidence_percentage": 100.0,
                "fraud_probability": 0.0,
                "risk_score": 0.0,
                "risk_level": "LOW",
                "active_model": self.model_name,
                "explanation": explain_prediction(""),
                "recommended_action": "Please provide message or audio text to analyze.",
                "reasoning": "Empty input."
            }

        # 1. Safe normalization
        cleaned_text = clean_text_safely(raw_text)

        # 2. AI Inference
        fraud_prob = self.predict_proba(cleaned_text)
        predicted_label = int(fraud_prob >= settings.CLASSIFICATION_THRESHOLD)

        # 3. Explainability & Suspicious Indicator Extraction
        explanation_data = explain_prediction(raw_text)

        # 4. Multi-Factor Risk Assessment
        risk_data = compute_risk_assessment(
            ai_predicted_label=predicted_label,
            fraud_probability=fraud_prob,
            explanation_data=explanation_data
        )

        return {
            "input_text": raw_text,
            "clean_text": cleaned_text,
            "prediction": risk_data["prediction"],
            "predicted_label": risk_data["predicted_label"],
            "confidence_percentage": risk_data["confidence_percentage"],
            "fraud_probability": risk_data["fraud_probability"],
            "risk_score": risk_data["risk_score"],
            "risk_level": risk_data["risk_level"],
            "active_model": self.model_name,
            "explanation": explanation_data,
            "recommended_action": risk_data["recommended_action"],
            "reasoning": risk_data["reasoning"],
            "indicator_count": risk_data["indicator_count"]
        }


# Global singleton helper for app and fast inference
_PREDICTOR_INSTANCE: Optional[FraudPredictor] = None


def get_predictor(force_reload: bool = False) -> FraudPredictor:
    """Returns a singleton instance of FraudPredictor."""
    global _PREDICTOR_INSTANCE
    if _PREDICTOR_INSTANCE is None or force_reload:
        _PREDICTOR_INSTANCE = FraudPredictor()
    return _PREDICTOR_INSTANCE


def predict_message(text: str) -> Dict[str, Any]:
    """Convenience function to analyze a message using the active predictor."""
    predictor = get_predictor()
    return predictor.analyze(text)


if __name__ == "__main__":
    test_message = "Your SBI account is blocked. Update KYC at http://bit.ly/sbi-kyc or share OTP to avoid penalty."
    res = predict_message(test_message)
    print("Inference Result:")
    print(f"Prediction: {res['prediction']} (Confidence: {res['confidence_percentage']}%)")
    print(f"Risk Level: {res['risk_level']} (Score: {res['risk_score']})")
    print(f"Active Model: {res['active_model']}")
    print(f"Advice: {res['recommended_action']}")

