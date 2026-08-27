"""Model Evaluation and Benchmark Comparison Module.

Loads held-out test predictions for baseline and transformer models, compares
Accuracy, Precision, Recall, F1-Score, and ROC-AUC, and writes models/model_comparison.json.
"""

from typing import Dict, Any, List
import json
import logging
from pathlib import Path
import pandas as pd

from config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_all_metrics() -> Dict[str, Any]:
    """Loads evaluated model metrics from metrics.json."""
    if not settings.METRICS_PATH.exists():
        logger.warning("metrics.json not found. Returning empty metrics dictionary.")
        return {}
    try:
        with open(settings.METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error("Error reading metrics.json: %s", e)
        return {}


def generate_model_comparison() -> Dict[str, Any]:
    """Builds a structured benchmark comparison table across all trained models."""
    metrics_data = load_all_metrics()
    if not metrics_data:
        logger.info("No metrics available to compare.")
        return {"models": [], "summary": "No trained models found."}

    comparison_table = []
    for model_key, m in metrics_data.items():
        comparison_table.append({
            "model_key": model_key,
            "name": m.get("model_name", model_key),
            "accuracy": m.get("accuracy", 0.0),
            "precision": m.get("precision", 0.0),
            "recall": m.get("recall", 0.0),
            "f1_score": m.get("f1_score", 0.0),
            "roc_auc": m.get("roc_auc", 0.0),
            "confusion_matrix": m.get("confusion_matrix", [[0, 0], [0, 0]]),
            "test_samples": m.get("test_sample_count", 0)
        })

    # Sort primarily by Recall (descending), then F1-score (descending)
    comparison_table.sort(key=lambda x: (x["recall"], x["f1_score"], x["accuracy"]), reverse=True)

    best_model = comparison_table[0] if comparison_table else None

    comparison_result = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "total_models_evaluated": len(comparison_table),
        "champion_model_key": best_model["model_key"] if best_model else None,
        "champion_model_name": best_model["name"] if best_model else None,
        "champion_criteria": "Prioritizing Fraud Recall (minimizing false negatives) followed by F1-score",
        "models": comparison_table
    }

    # Save to model_comparison.json
    with open(settings.MODEL_COMPARISON_PATH, "w", encoding="utf-8") as f:
        json.dump(comparison_result, f, indent=4)
    logger.info("Saved model comparison to %s", settings.MODEL_COMPARISON_PATH)

    return comparison_result


def print_comparison_report() -> None:
    """Prints a terminal-friendly comparison table."""
    comp = generate_model_comparison()
    models = comp.get("models", [])
    if not models:
        print("No models evaluated yet.")
        return

    print("\n" + "=" * 80)
    print("                AI FRAUD DETECTOR - MODEL BENCHMARK COMPARISON")
    print("=" * 80)
    print(f"{'Model Name':<32} | {'Accuracy':<10} | {'Precision':<10} | {'Recall (Fraud)':<14} | {'F1-Score':<10}")
    print("-" * 80)
    for m in models:
        prefix = "🏆 " if m["model_key"] == comp.get("champion_model_key") else "   "
        name_display = f"{prefix}{m['name']}"
        print(f"{name_display:<32} | {m['accuracy']*100:<9.2f}% | {m['precision']*100:<9.2f}% | {m['recall']*100:<13.2f}% | {m['f1_score']*100:<9.2f}%")
    print("=" * 80)
    print(f"Active Champion: {comp.get('champion_model_name')} (Selection rule: {comp.get('champion_criteria')})\n")


if __name__ == "__main__":
    print_comparison_report()

