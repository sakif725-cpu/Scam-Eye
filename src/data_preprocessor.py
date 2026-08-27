"""Data preprocessor module for AI Fraud Call and Message Detector.

Implements safe text normalization (preserving URLs, phone numbers, currency, OTP,
urgency cues, and multilingual/Hinglish words), deduplication, stratification,
train/val/test splitting, and statistical reporting.
"""

from typing import Tuple, Dict, Any
import re
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

from config import settings
from src.data_loader import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def clean_text_safely(text: str) -> str:
    """Safely cleans and normalizes message text while retaining fraud-indicative cues.

    Preserves:
    - URLs (normalized to 'http_url_token' or domain structure)
    - Phone/contact numbers (normalized to 'phone_num_token')
    - Currency values (₹, Rs, INR, $, USD, numbers)
    - OTP and verification tokens
    - Urgency keywords and fear triggers
    - Multilingual and Hinglish terms (e.g., 'turant', 'khata', 'band', 'inaam', 'paise')

    Args:
        text: Raw text string.

    Returns:
        Cleaned, normalized text string.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Convert to string and handle basic whitespace
    s = text.strip()

    # Normalize URLs: retain presence marker while normalizing variant schemas
    # Example: http://bit.ly/123 -> " http_url_token "
    url_pattern = r"https?://\S+|www\.\S+|bit\.ly/\S+|t\.me/\S+"
    s = re.sub(url_pattern, " http_url_token ", s, flags=re.IGNORECASE)

    # Normalize email addresses
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    s = re.sub(email_pattern, " email_address_token ", s)

    # Normalize UPI Handles (e.g., user@upi, name@okhdfcbank, pay@paytm)
    upi_pattern = r"\b[a-zA-Z0-9.\-_]{2,256}@(upi|okhdfcbank|okaxis|oksbi|paytm|ybl|ibl|apl|axl)\b"
    s = re.sub(upi_pattern, " upi_handle_token ", s, flags=re.IGNORECASE)

    # Normalize Phone Numbers (10-12 digits, optional +, -, spaces)
    phone_pattern = r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b|\b\d{10}\b"
    s = re.sub(phone_pattern, " phone_num_token ", s)

    # Normalize Currency Amounts (e.g. Rs 5000, ₹10000, $500, 50,000 INR)
    currency_pattern = r"(?:(?:Rs\.?|INR|₹|\$|USD)\s*[\d,]+(?:\.\d+)?|[\d,]+(?:\.\d+)?\s*(?:rupees|rupaye|INR|dollars|bucks))"
    s = re.sub(currency_pattern, " currency_amount_token ", s, flags=re.IGNORECASE)

    # Normalize OTP Codes (standalone 4-8 digit numbers)
    otp_pattern = r"\b\d{4,8}\b"
    s = re.sub(otp_pattern, " digit_code_token ", s)

    # Clean excessive punctuation while keeping alphanumeric and unicode (Hinglish/Devanagari)
    # Note: We do NOT strip Devanagari (range \u0900-\u097F) or standard word characters
    s = re.sub(r"[^\w\s\u0900-\u097F]", " ", s)

    # Collapse multiple whitespaces into a single space
    s = re.sub(r"\s+", " ", s).strip()

    # Convert to lowercase for uniform TF-IDF and subword tokenization
    return s.lower()


def preprocess_dataset(
    df: pd.DataFrame,
    random_state: int = settings.RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Cleans, deduplicates, validates, shuffles, and splits dataset into Train/Val/Test.

    Args:
        df: Input DataFrame with 'text' and 'label' columns.
        random_state: Seed for reproducible random shuffling.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    logger.info("Starting dataset preprocessing pipeline...")
    initial_count = len(df)

    # 1. Ensure columns exist
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("DataFrame must contain 'text' and 'label' columns.")

    # 2. Clean text column
    df = df.copy()
    df["clean_text"] = df["text"].apply(clean_text_safely)

    # 3. Filter empty, null, or ultra-short records
    df = df[df["clean_text"].str.len() >= 3]
    df = df.dropna(subset=["clean_text", "label"])
    df["label"] = df["label"].astype(int)

    # 4. Remove duplicate cleaned texts
    df = df.drop_duplicates(subset=["clean_text"])
    post_clean_count = len(df)
    logger.info("Deduplication & cleaning: %d -> %d samples retained (%d removed)",
                initial_count, post_clean_count, initial_count - post_clean_count)

    # 5. Stratified train/val/test split
    # Split 1: Train (70%) vs Temp (30%)
    train_df, temp_df = train_test_split(
        df,
        test_size=(settings.VAL_SPLIT_RATIO + settings.TEST_SPLIT_RATIO),
        random_state=random_state,
        stratify=df["label"]
    )

    # Split 2: Val (15%) vs Test (15%) -> equal split of temp (50%/50%)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=random_state,
        stratify=temp_df["label"]
    )

    # 6. Save processed datasets locally
    settings.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(settings.PROCESSED_DATA_PATH, index=False)
    train_df.to_csv(settings.TRAIN_DATA_PATH, index=False)
    val_df.to_csv(settings.VAL_DATA_PATH, index=False)
    test_df.to_csv(settings.TEST_DATA_PATH, index=False)

    logger.info("Saved processed splits to %s", settings.PROCESSED_DATA_DIR)
    return train_df, val_df, test_df


def get_dataset_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Computes comprehensive descriptive statistics for dataset reporting.

    Args:
        df: Processed DataFrame.

    Returns:
        Dictionary containing statistical metrics.
    """
    total = len(df)
    fraud_count = int((df["label"] == settings.LABEL_FRAUD).sum())
    genuine_count = int((df["label"] == settings.LABEL_GENUINE).sum())
    fraud_ratio = round((fraud_count / total * 100) if total > 0 else 0.0, 2)
    avg_char_length = round(float(df["text"].str.len().mean()) if total > 0 else 0.0, 1)
    avg_word_count = round(float(df["text"].str.split().apply(len).mean()) if total > 0 else 0.0, 1)

    return {
        "total_samples": total,
        "fraud_count": fraud_count,
        "genuine_count": genuine_count,
        "fraud_percentage": fraud_ratio,
        "genuine_percentage": round(100.0 - fraud_ratio, 2),
        "avg_char_length": avg_char_length,
        "avg_word_count": avg_word_count
    }


def print_pipeline_summary(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Prints a formatted summary table of the data pipeline splits."""
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    full_stats = get_dataset_statistics(full_df)
    train_stats = get_dataset_statistics(train_df)
    val_stats = get_dataset_statistics(val_df)
    test_stats = get_dataset_statistics(test_df)

    print("\n" + "=" * 65)
    print("       AI FRAUD DETECTOR - DATA PIPELINE SUMMARY")
    print("=" * 65)
    print(f"{'Split':<12} | {'Total':<8} | {'Fraud (1)':<10} | {'Genuine (0)':<12} | {'Fraud %':<8}")
    print("-" * 65)
    print(f"{'Full Clean':<12} | {full_stats['total_samples']:<8} | {full_stats['fraud_count']:<10} | {full_stats['genuine_count']:<12} | {full_stats['fraud_percentage']:<8.1f}%")
    print(f"{'Training':<12} | {train_stats['total_samples']:<8} | {train_stats['fraud_count']:<10} | {train_stats['genuine_count']:<12} | {train_stats['fraud_percentage']:<8.1f}%")
    print(f"{'Validation':<12} | {val_stats['total_samples']:<8} | {val_stats['fraud_count']:<10} | {val_stats['genuine_count']:<12} | {val_stats['fraud_percentage']:<8.1f}%")
    print(f"{'Held-Out Test':<12} | {test_stats['total_samples']:<8} | {test_stats['fraud_count']:<10} | {test_stats['genuine_count']:<12} | {test_stats['fraud_percentage']:<8.1f}%")
    print("=" * 65)
    print(f"Average message length: {full_stats['avg_word_count']} words ({full_stats['avg_char_length']} chars)\n")


def run_pipeline() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """End-to-end execution of loading, preprocessing, and saving data."""
    raw_df = load_dataset()
    train_df, val_df, test_df = preprocess_dataset(raw_df)
    print_pipeline_summary(train_df, val_df, test_df)
    return train_df, val_df, test_df


if __name__ == "__main__":
    run_pipeline()

