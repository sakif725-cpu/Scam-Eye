"""Centralized configuration module for AI Fraud Call and Message Detector.

Avoids hardcoding paths, parameters, thresholds, and seeds across modules.
"""

from pathlib import Path
import os
import torch

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
BASELINE_MODEL_DIR = MODELS_DIR / "baseline"
TRANSFORMER_MODEL_DIR = MODELS_DIR / "transformer"
ASSETS_DIR = BASE_DIR / "assets"

# Ensure essential directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, BASELINE_MODEL_DIR, TRANSFORMER_MODEL_DIR, ASSETS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Data file paths
FALLBACK_DATA_PATH = DATA_DIR / "fallback_sample_data.csv"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "cleaned_dataset.csv"
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.csv"
VAL_DATA_PATH = PROCESSED_DATA_DIR / "val.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.csv"

# Model metadata and artifact paths
METRICS_PATH = MODELS_DIR / "metrics.json"
MODEL_COMPARISON_PATH = MODELS_DIR / "model_comparison.json"
SELECTED_MODEL_PATH = MODELS_DIR / "selected_model.json"
BASELINE_MODEL_PATH = BASELINE_MODEL_DIR / "logistic_regression_model.joblib"
BASELINE_VECTORIZER_PATH = BASELINE_MODEL_DIR / "tfidf_vectorizer.joblib"
BASELINE_NB_MODEL_PATH = BASELINE_MODEL_DIR / "naive_bayes_model.joblib"

# Reproducibility
RANDOM_SEED = 42

# Public Dataset URLs (UCI Machine Learning Repository SMS Spam dataset mirror)
PUBLIC_DATASET_URLS = [
    "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv",
    "https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv",
]

# Dataset Splitting Ratios (Train / Validation / Held-out Test)
TRAIN_SPLIT_RATIO = 0.70
VAL_SPLIT_RATIO = 0.15
TEST_SPLIT_RATIO = 0.15

# Baseline Training Parameters
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)
LOGISTIC_REGRESSION_C = 1.0
LOGISTIC_REGRESSION_MAX_ITER = 1000

# Transformer Model Parameters
TRANSFORMER_MODEL_NAME = "distilbert-base-uncased"
TRANSFORMER_MAX_LENGTH = 128
TRANSFORMER_BATCH_SIZE = 16
TRANSFORMER_EPOCHS = 3
TRANSFORMER_LEARNING_RATE = 2e-5
TRANSFORMER_WEIGHT_DECAY = 0.01

# Speech-to-Text Parameters (Faster-Whisper)
WHISPER_DEFAULT_MODEL_SIZE = "base"
WHISPER_ALLOWED_SIZES = ["tiny", "base", "small"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "int8"

# Confidence and Classification Thresholds
CLASSIFICATION_THRESHOLD = 0.50

# Risk Engine Thresholds (0.0 to 1.0)
RISK_LEVEL_THRESHOLDS = {
    "LOW": 0.35,       # Score < 0.35 -> LOW
    "MEDIUM": 0.65,    # 0.35 <= Score < 0.65 -> MEDIUM
    "HIGH": 0.85,      # 0.65 <= Score < 0.85 -> HIGH
    "CRITICAL": 1.00   # Score >= 0.85 -> CRITICAL
}

# Label Mapping
LABEL_FRAUD = 1
LABEL_GENUINE = 0
LABEL_MAP = {
    LABEL_FRAUD: "FRAUD",
    LABEL_GENUINE: "GENUINE"
}

# Multilingual / Hinglish Support notes
# The system cleans text while retaining Hinglish keywords and token markers.
# Limitation: DistilBERT uncased is optimized for English, but TF-IDF n-grams capture Hinglish morphemes.

