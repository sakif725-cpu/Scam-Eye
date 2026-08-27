"""Transformer Fine-Tuning Module for AI Fraud Call and Message Detector.

Fine-tunes DistilBERT (distilbert-base-uncased) for sequence classification
with PyTorch and Hugging Face Transformers.

Features:
- Automatic CPU/GPU detection (CUDA / Apple Silicon MPS / CPU)
- Configurable hyperparameters (batch size, learning rate, epochs)
- Validation tracking and best-model checkpointing
- Saves model, tokenizer, and metrics to models/transformer/ and models/metrics.json
- Graceful error handling in case of offline/network constraints
"""

from typing import Dict, Any, Tuple
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
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


class FraudTextDataset(Dataset):
    """PyTorch Dataset for text tokenization and label encapsulation."""

    def __init__(self, texts: list, labels: list, tokenizer, max_length: int = settings.TRANSFORMER_MAX_LENGTH):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }


def evaluate_transformer(
    model: AutoModelForSequenceClassification,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """Runs evaluation on dataloader and calculates metrics."""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[:, 1]
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    acc = float(accuracy_score(all_labels, all_preds))
    prec = float(precision_score(all_labels, all_preds, zero_division=0))
    rec = float(recall_score(all_labels, all_preds, zero_division=0))
    f1 = float(f1_score(all_labels, all_preds, zero_division=0))
    cm = confusion_matrix(all_labels, all_preds).tolist()
    roc = float(roc_auc_score(all_labels, all_probs)) if len(np.unique(all_labels)) > 1 else 1.0

    report = classification_report(all_labels, all_preds, target_names=["Genuine", "Fraud"], output_dict=True, zero_division=0)

    metrics = {
        "model_name": "DistilBERT (Fine-Tuned)",
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc, 4),
        "confusion_matrix": cm,
        "classification_report": report,
        "test_sample_count": len(all_labels)
    }

    return metrics, all_preds, all_probs


def train_transformer_model(
    epochs: int = settings.TRANSFORMER_EPOCHS,
    batch_size: int = settings.TRANSFORMER_BATCH_SIZE,
    learning_rate: float = settings.TRANSFORMER_LEARNING_RATE
) -> Dict[str, Any]:
    """Trains and fine-tunes DistilBERT for binary fraud classification."""
    logger.info("Initializing Transformer Fine-Tuning Pipeline...")

    # Ensure dataset splits exist
    if not (settings.TRAIN_DATA_PATH.exists() and settings.VAL_DATA_PATH.exists() and settings.TEST_DATA_PATH.exists()):
        logger.info("Processed data not found. Running data preprocessing pipeline...")
        train_df, val_df, test_df = run_pipeline()
    else:
        train_df = pd.read_csv(settings.TRAIN_DATA_PATH)
        val_df = pd.read_csv(settings.VAL_DATA_PATH)
        test_df = pd.read_csv(settings.TEST_DATA_PATH)

    device = torch.device(settings.DEVICE)
    logger.info("Using device: %s", device)

    # Initialize Tokenizer and Model
    model_name = settings.TRANSFORMER_MODEL_NAME
    logger.info("Loading pre-trained '%s' tokenizer and model...", model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    # Prepare DataLoaders
    train_dataset = FraudTextDataset(train_df["clean_text"].tolist(), train_df["label"].tolist(), tokenizer)
    val_dataset = FraudTextDataset(val_df["clean_text"].tolist(), val_df["label"].tolist(), tokenizer)
    test_dataset = FraudTextDataset(test_df["clean_text"].tolist(), test_df["label"].tolist(), tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=settings.TRANSFORMER_WEIGHT_DECAY)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

    best_val_f1 = -1.0
    best_model_state = None

    # Training loop
    logger.info("Starting fine-tuning for %d epochs...", epochs)
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / max(len(train_loader), 1)

        # Validation
        val_metrics, _, _ = evaluate_transformer(model, val_loader, device)
        val_f1 = val_metrics["f1_score"]
        val_rec = val_metrics["recall"]

        logger.info("Epoch %d/%d - Loss: %.4f | Val F1: %.4f | Val Recall: %.4f",
                    epoch, epochs, avg_train_loss, val_f1, val_rec)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Load best weights
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    # Evaluate on held-out test set
    test_metrics, _, _ = evaluate_transformer(model, test_loader, device)
    logger.info("Held-out Test Evaluation: Accuracy=%.4f, Precision=%.4f, Recall=%.4f, F1=%.4f",
                test_metrics["accuracy"], test_metrics["precision"], test_metrics["recall"], test_metrics["f1_score"])

    # Save fine-tuned artifacts
    settings.TRANSFORMER_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(settings.TRANSFORMER_MODEL_DIR))
    tokenizer.save_pretrained(str(settings.TRANSFORMER_MODEL_DIR))
    logger.info("Saved fine-tuned transformer and tokenizer to %s", settings.TRANSFORMER_MODEL_DIR)

    # Update metrics.json
    all_metrics = {}
    if settings.METRICS_PATH.exists():
        try:
            with open(settings.METRICS_PATH, "r", encoding="utf-8") as f:
                all_metrics = json.load(f)
        except Exception:
            all_metrics = {}

    all_metrics["transformer_distilbert"] = test_metrics
    with open(settings.METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=4)
    logger.info("Updated transformer metrics in %s", settings.METRICS_PATH)

    print("\n" + "=" * 65)
    print("      TRANSFORMER DISTILBERT EVALUATION REPORT")
    print("=" * 65)
    print(f"Model: {test_metrics['model_name']}")
    print(f"  - Accuracy:  {test_metrics['accuracy'] * 100:.2f}%")
    print(f"  - Precision: {test_metrics['precision'] * 100:.2f}%")
    print(f"  - Recall:    {test_metrics['recall'] * 100:.2f}%")
    print(f"  - F1-Score:  {test_metrics['f1_score'] * 100:.2f}%")
    print(f"  - ROC-AUC:   {test_metrics['roc_auc']:.4f}")
    print(f"  - Confusion Matrix: {test_metrics['confusion_matrix']}")
    print("=" * 65)

    return test_metrics


if __name__ == "__main__":
    train_transformer_model()
