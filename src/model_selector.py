"""Model Selector Module for AI Fraud Call and Message Detector.

Automatically selects the best-performing model based on Recall and F1-score,
persists the decision to models/selected_model.json, and supports manual overrides.
"""

from typing import Dict, Any, Optional
import json
import logging
from pathlib import Path

from config import settings
from src.evaluate_models import generate_model_comparison, load_all_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def select_best_model(force_recompute: bool = False) -> Dict[str, Any]:
    """Evaluates all candidate models and selects the champion model.

    Selection Hierarchy:
    1. Highest Recall on Fraud class (minimizing catastrophic false negatives).
    2. Highest F1-Score (harmonic balance of precision and recall).
    3. Simplicity / latency preference if scores are identical.

    Args:
        force_recompute: Re-evaluates benchmark if True.

    Returns:
        Dictionary containing selected model metadata.
    """
    comp = generate_model_comparison()
    models = comp.get("models", [])

    if not models:
        logger.warning("No candidate models found in metrics. Falling back to default baseline.")
        selected_data = {
            "active_model_type": "baseline_logistic_regression",
            "model_name": "TF-IDF + Logistic Regression",
            "description": "Default baseline linear classifier with TF-IDF features",
            "status": "fallback_default"
        }
        _persist_selection(selected_data)
        return selected_data

    # Champion is top of sorted list (sorted by recall, then F1, then accuracy)
    champion = models[0]
    all_metrics = load_all_metrics()
    champ_metrics = all_metrics.get(champion["model_key"], {})

    selected_data = {
        "active_model_type": champion["model_key"],
        "model_name": champion["name"],
        "description": f"Champion model selected by Recall ({champion['recall']*100:.1f}%) and F1 ({champion['f1_score']*100:.1f}%)",
        "selection_timestamp": comp.get("timestamp"),
        "metrics": champ_metrics
    }

    _persist_selection(selected_data)
    logger.info("Champion model selected: %s (%s)", champion["name"], champion["model_key"])
    return selected_data


def set_active_model(model_key: str) -> Dict[str, Any]:
    """Manually selects a specific model as the active inference engine.

    Args:
        model_key: Key of model (e.g. 'baseline_logistic_regression', 'transformer_distilbert', 'baseline_naive_bayes').

    Returns:
        Dictionary with updated selected model metadata.
    """
    all_metrics = load_all_metrics()
    if model_key not in all_metrics:
        raise ValueError(f"Model key '{model_key}' not found in available metrics {list(all_metrics.keys())}")

    model_info = all_metrics[model_key]
    selected_data = {
        "active_model_type": model_key,
        "model_name": model_info.get("model_name", model_key),
        "description": f"Manually set active model: {model_info.get('model_name', model_key)}",
        "metrics": model_info
    }

    _persist_selection(selected_data)
    logger.info("Active model manually updated to: %s", model_key)
    return selected_data


def _persist_selection(data: Dict[str, Any]) -> None:
    """Writes selection metadata to models/selected_model.json."""
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(settings.SELECTED_MODEL_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def get_current_selected_model() -> Dict[str, Any]:
    """Retrieves current selected model metadata from disk."""
    if not settings.SELECTED_MODEL_PATH.exists():
        return select_best_model()
    try:
        with open(settings.SELECTED_MODEL_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return select_best_model()


if __name__ == "__main__":
    best = select_best_model()
    print("Currently Selected Active Model:")
    print(json.dumps(best, indent=2))

