"""Data loader module for AI Fraud Call and Message Detector.

Downloads or loads public fraud/spam datasets, validates dataset structure,
normalizes labels into 1=FRAUD, 0=GENUINE, and falls back to fallback sample data
if download or loading fails.
"""

from pathlib import Path
from typing import Optional, Tuple
import logging
import pandas as pd
import requests

from config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def detect_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """Detect text and label columns dynamically across various dataset formats.

    Args:
        df: Input pandas DataFrame.

    Returns:
        Tuple of (text_column_name, label_column_name).

    Raises:
        ValueError: If suitable text or label column cannot be identified.
    """
    text_candidates = ["text", "message", "sms", "content", "body", "v2", "sms_text", "clean_text", "email_text"]
    label_candidates = ["label", "class", "target", "category", "v1", "is_spam", "spam", "fraud", "type"]

    detected_text = None
    detected_label = None

    # Check lowercase column names
    col_map = {col.lower().strip(): col for col in df.columns}

    for candidate in text_candidates:
        if candidate in col_map:
            detected_text = col_map[candidate]
            break

    for candidate in label_candidates:
        if candidate in col_map:
            detected_label = col_map[candidate]
            break

    # Fallback to positional check if 2 columns
    if not detected_text or not detected_label:
        if len(df.columns) >= 2:
            # Check string lengths in columns to guess text column
            first_col = df.columns[0]
            second_col = df.columns[1]
            if df[first_col].dtype == object and df[second_col].dtype == object:
                # If one has short values and one has long text
                len_0 = df[first_col].astype(str).str.len().mean()
                len_1 = df[second_col].astype(str).str.len().mean()
                if len_0 > len_1:
                    detected_text = first_col
                    detected_label = second_col
                else:
                    detected_text = second_col
                    detected_label = first_col

    if not detected_text or not detected_label:
        raise ValueError(f"Could not automatically detect text and label columns from: {list(df.columns)}")

    return detected_text, detected_label


def normalize_labels(series: pd.Series) -> pd.Series:
    """Normalize various label representations into 1 (FRAUD) and 0 (GENUINE).

    Args:
        series: pandas Series containing raw labels.

    Returns:
        Normalized pandas Series containing binary integer labels (0 or 1).
    """
    fraud_tokens = {"spam", "fraud", "scam", "phishing", "1", 1, 1.0, "true", True, "fake", "malicious", "threat"}
    genuine_tokens = {"ham", "genuine", "legit", "normal", "0", 0, 0.0, "false", False, "safe", "clean"}

    def map_val(val):
        if pd.isna(val):
            return None
        val_str = str(val).strip().lower()
        if val in fraud_tokens or val_str in fraud_tokens:
            return settings.LABEL_FRAUD
        if val in genuine_tokens or val_str in genuine_tokens:
            return settings.LABEL_GENUINE
        # Heuristic fallback
        if "spam" in val_str or "fraud" in val_str or "scam" in val_str:
            return settings.LABEL_FRAUD
        return settings.LABEL_GENUINE

    return series.apply(map_val)


def load_fallback_dataset() -> pd.DataFrame:
    """Load the balanced fallback sample dataset.

    Returns:
        Standardized DataFrame with ['text', 'label'] columns.
    """
    logger.warning(
        "[FALLBACK NOTICE] Using fallback sample dataset from: %s\n"
        "This dataset is designed for structural validation, end-to-end testing, "
        "and hackathon demonstration. It is not production-grade.",
        settings.FALLBACK_DATA_PATH
    )
    if not settings.FALLBACK_DATA_PATH.exists():
        raise FileNotFoundError(f"Fallback dataset missing at {settings.FALLBACK_DATA_PATH}")

    df = pd.read_csv(settings.FALLBACK_DATA_PATH)
    text_col, label_col = detect_columns(df)
    standardized_df = pd.DataFrame({
        "text": df[text_col].astype(str),
        "label": normalize_labels(df[label_col])
    })
    return standardized_df.dropna(subset=["text", "label"])


def download_public_dataset(target_path: Path) -> Optional[Path]:
    """Attempt downloading a public SMS/Fraud dataset from curated URLs.

    Args:
        target_path: Local Path where raw dataset should be stored.

    Returns:
        Path to downloaded file if successful, None otherwise.
    """
    for url in settings.PUBLIC_DATASET_URLS:
        try:
            logger.info("Attempting download of public dataset from: %s", url)
            response = requests.get(url, timeout=10)
            if response.status_code == 200 and len(response.content) > 1000:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with open(target_path, "wb") as f:
                    f.write(response.content)
                logger.info("Successfully downloaded public dataset to %s (%d bytes)", target_path, len(response.content))
                return target_path
        except Exception as err:
            logger.warning("Download failed for %s: %s", url, err)
    return None


def load_dataset(use_fallback_only: bool = False, local_file_path: Optional[str] = None) -> pd.DataFrame:
    """Load and standardize the dataset for downstream preprocessing.

    Args:
        use_fallback_only: If True, skips network download and loads fallback CSV directly.
        local_file_path: Optional specific CSV/TSV file to load.

    Returns:
        DataFrame with standardized ['text', 'label'] columns where label is 0 or 1.
    """
    if local_file_path and Path(local_file_path).exists():
        path = Path(local_file_path)
        try:
            # Try reading with sep detection
            df = pd.read_csv(path, sep=None, engine="python", on_bad_lines="skip")
            text_col, label_col = detect_columns(df)
            return pd.DataFrame({
                "text": df[text_col].astype(str),
                "label": normalize_labels(df[label_col])
            }).dropna()
        except Exception as err:
            logger.error("Error reading provided file %s: %s", local_file_path, err)

    if use_fallback_only:
        return load_fallback_dataset()

    # Try downloading public dataset
    raw_file = settings.RAW_DATA_DIR / "sms_raw_dataset.csv"
    if not raw_file.exists():
        downloaded = download_public_dataset(raw_file)
        if not downloaded:
            return load_fallback_dataset()

    try:
        # Load downloaded raw dataset
        try:
            df = pd.read_csv(raw_file, sep="\t", header=None, names=["label", "text"], on_bad_lines="skip")
            if len(df) < 50:
                raise ValueError("Downloaded dataset has insufficient rows")
        except Exception:
            df = pd.read_csv(raw_file, encoding="latin-1", on_bad_lines="skip")

        text_col, label_col = detect_columns(df)
        standardized_df = pd.DataFrame({
            "text": df[text_col].astype(str),
            "label": normalize_labels(df[label_col])
        }).dropna()

        # Check label balance
        counts = standardized_df["label"].value_counts()
        if len(counts) < 2 or (counts < 10).any():
            logger.warning("Raw dataset has severe class imbalance or missing class. Augmenting with fallback.")
            fallback_df = load_fallback_dataset()
            standardized_df = pd.concat([standardized_df, fallback_df], ignore_index=True)

        return standardized_df

    except Exception as e:
        logger.error("Failed to parse raw downloaded dataset (%s). Reverting to fallback.", e)
        return load_fallback_dataset()


if __name__ == "__main__":
    df = load_dataset()
    print("Loaded Dataset Summary:")
    print(f"Total Rows: {len(df)}")
    print(f"Class Distribution: {df['label'].value_counts().to_dict()}")
    print("Sample rows:")
    print(df.head())

