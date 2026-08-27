"""Baseline Model Training Module for AI Fraud Call and Message Detector.

Trains:
- TF-IDF Vectorizer (word + character n-grams) + Logistic Regression (Primary Baseline)
- TF-IDF Vectorizer + Multinomial Naive Bayes (Lightweight Comparison)

Evaluates on Held-Out Test Set:
- Accuracy, Precision, Recall, F1-Score, Confusion Matrix, ROC-AUC
- Prioritizes Recall and F1-Score to minimize missed fraudulent communications.
"""

from typing import Dict, Any, Tuple
import json
import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report
)

from config import settings
from src.data_preprocessor import run_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def train_tfidf_vectorizer(train_texts: pd.Series) -> TfidfVectorizer:
    """Fits and returns a TF-IDF vectorizer with unigram and bigram features."""
    logger.info("Fitting TF-IDF vectorizer (max_features=%d, ngrams=%s)...",
                settings.TFIDF_MAX_FEATURES, settings.TFIDF_NGRAM_RANGE)
    vectorizer = TfidfVectorizer(
        max_features=settings.TFIDF_MAX_FEATURES,
        ngram_range=settings.TFIDF_NGRAM_RANGE,
        sublinear_tf=True,
        strip_accents="unicode"
    )
    vectorizer.fit(train_texts)
    return vectorizer


def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    """Trains a calibrated Logistic Regression classifier."""
    logger.info("Training Logistic Regression (C=%.2f, max_iter=%d, class_weight='balanced')...",
                settings.LOGISTIC_REGRESSION_C, settings.LOGISTIC_REGRESSION_MAX_ITER)
    model = LogisticRegression(
        C=settings.LOGISTIC_REGRESSION_C,
        max_iter=settings.LOGISTIC_REGRESSION_MAX_ITER,
        class_weight="balanced",
        random_state=settings.RANDOM_SEED,
        solver="lbfgs"
    )
    model.fit(X_train, y_train)
    return model


def train_naive_bayes(X_train, y_train) -> MultinomialNB:
    """Trains a Multinomial Naive Bayes classifier."""
    logger.info("Training Multinomial Naive Bayes (alpha=0.5)...")
    model = MultinomialNB(alpha=0.5)
    model.fit(X_train, y_train)
    return model


def evaluate_classifier(model, X_test, y_test, model_name: str) -> Dict[str, Any]:
    """Calculates comprehensive classification metrics with focus on Fraud Recall & F1."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    cm = confusion_matrix(y_test, y_pred).tolist()
    roc = float(roc_auc_score(y_test, y_prob)) if len(np.unique(y_test)) > 1 else 1.0

    report = classification_report(y_test, y_pred, target_names=["Genuine", "Fraud"], output_dict=True, zero_division=0)

    metrics = {
        "model_name": model_name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc, 4),
        "confusion_matrix": cm,
        "classification_report": report,
        "test_sample_count": len(y_test)
    }
    return metrics


def save_baseline_artifacts(
    vectorizer: TfidfVectorizer,
    lr_model: LogisticRegression,
    nb_model: MultinomialNB,
    metrics_lr: Dict[str, Any],
    metrics_nb: Dict[str, Any]
) -> None:
    """Serializes models, vectorizer, and evaluation metrics."""
    settings.BASELINE_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(vectorizer, settings.BASELINE_VECTORIZER_PATH)
    joblib.dump(lr_model, settings.BASELINE_MODEL_PATH)
    joblib.dump(nb_model, settings.BASELINE_NB_MODEL_PATH)
    logger.info("Saved baseline model artifacts to %s", settings.BASELINE_MODEL_DIR)

    # Load existing metrics or create fresh
    all_metrics = {}
    if settings.METRICS_PATH.exists():
        try:
            with open(settings.METRICS_PATH, "r", encoding="utf-8") as f:
                all_metrics = json.load(f)
        except Exception:
            all_metrics = {}

    all_metrics["baseline_logistic_regression"] = metrics_lr
    all_metrics["baseline_naive_bayes"] = metrics_nb

    with open(settings.METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=4)
    logger.info("Updated metrics in %s", settings.METRICS_PATH)

    # Set active selected model default to baseline LR if not already set
    selected_model_data = {
        "active_model_type": "baseline_logistic_regression",
        "model_name": "TF-IDF + Logistic Regression",
        "description": "Baseline linear classifier with n-gram TF-IDF embeddings",
        "metrics": metrics_lr
    }
    with open(settings.SELECTED_MODEL_PATH, "w", encoding="utf-8") as f:
        json.dump(selected_model_data, f, indent=4)
    logger.info("Initialized active model pointer in %s", settings.SELECTED_MODEL_PATH)


def train_and_evaluate_baseline() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Runs complete baseline training and evaluation lifecycle."""
    # Ensure processed datasets exist
    if not (settings.TRAIN_DATA_PATH.exists() and settings.TEST_DATA_PATH.exists()):
        logger.info("Processed datasets not found. Running preprocessor pipeline...")
        train_df, val_df, test_df = run_pipeline()
    else:
        train_df = pd.read_csv(settings.TRAIN_DATA_PATH)
        val_df = pd.read_csv(settings.VAL_DATA_PATH)
        test_df = pd.read_csv(settings.TEST_DATA_PATH)

    train_texts = train_df["clean_text"].fillna("").astype(str)
    train_labels = train_df["label"].astype(int)

    test_texts = test_df["clean_text"].fillna("").astype(str)
    test_labels = test_df["label"].astype(int)

    # Vectorize
    vectorizer = train_tfidf_vectorizer(train_texts)
    X_train = vectorizer.transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    # Train Logistic Regression
    lr_model = train_logistic_regression(X_train, train_labels)
    metrics_lr = evaluate_classifier(lr_model, X_test, test_labels, "TF-IDF + Logistic Regression")

    # Train Naive Bayes
    nb_model = train_naive_bayes(X_train, train_labels)
    metrics_nb = evaluate_classifier(nb_model, X_test, test_labels, "TF-IDF + Multinomial Naive Bayes")

    # Save artifacts
    save_baseline_artifacts(vectorizer, lr_model, nb_model, metrics_lr, metrics_nb)

    # Display results
    print("\n" + "=" * 65)
    print("      BASELINE MODELS EVALUATION REPORT (HELD-OUT TEST SET)")
    print("=" * 65)
    for m in [metrics_lr, metrics_nb]:
        print(f"Model: {m['model_name']}")
        print(f"  - Accuracy:  {m['accuracy'] * 100:.2f}%")
        print(f"  - Precision: {m['precision'] * 100:.2f}%")
        print(f"  - Recall:    {m['recall'] * 100:.2f}%  <-- [Crucial for Fraud Detection]")
        print(f"  - F1-Score:  {m['f1_score'] * 100:.2f}%")
        print(f"  - ROC-AUC:   {m['roc_auc']:.4f}")
        print(f"  - Confusion Matrix (TN, FP, FN, TP): {m['confusion_matrix']}")
        print("-" * 65)

    return metrics_lr, metrics_nb


if __name__ == "__main__":
    train_and_evaluate_baseline()

