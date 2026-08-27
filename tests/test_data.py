"""Unit tests for dataset loading, cleaning, normalization, and splitting."""

import pytest
import pandas as pd
from config import settings
from src.data_loader import detect_columns, normalize_labels, load_fallback_dataset
from src.data_preprocessor import clean_text_safely, preprocess_dataset, get_dataset_statistics


def test_clean_text_preserves_urls():
    raw = "Your account is locked. Click http://bit.ly/bank-kyc-verify immediately."
    cleaned = clean_text_safely(raw)
    assert "http_url_token" in cleaned
    assert "account" in cleaned


def test_clean_text_preserves_currency_and_numbers():
    raw = "You won Rs 25,00,000 cash prize! Claim for $500 fee."
    cleaned = clean_text_safely(raw)
    assert "currency_amount_token" in cleaned or "25" in cleaned


def test_clean_text_preserves_hinglish():
    raw = "Aapka SBI khata band ho jayega, turant OTP bhejein."
    cleaned = clean_text_safely(raw)
    assert "khata" in cleaned
    assert "band" in cleaned
    assert "turant" in cleaned


def test_clean_text_empty_input():
    assert clean_text_safely("") == ""
    assert clean_text_safely("   ") == ""
    assert clean_text_safely(None) == ""


def test_detect_columns():
    df1 = pd.DataFrame({"text": ["hello"], "label": [0]})
    t1, l1 = detect_columns(df1)
    assert t1 == "text" and l1 == "label"

    df2 = pd.DataFrame({"v2": ["spam text"], "v1": ["spam"]})
    t2, l2 = detect_columns(df2)
    assert t2 == "v2" and l2 == "v1"


def test_normalize_labels():
    raw_labels = pd.Series(["spam", "ham", "1", "0", "fraud", "legit", "scam", "normal"])
    normalized = normalize_labels(raw_labels)
    expected = pd.Series([1, 0, 1, 0, 1, 0, 1, 0])
    pd.testing.assert_series_equal(normalized, expected, check_dtype=False)


def test_load_fallback_dataset():
    df = load_fallback_dataset()
    assert len(df) >= 50
    assert "text" in df.columns
    assert "label" in df.columns
    assert set(df["label"].unique()).issubset({0, 1})


def test_preprocess_and_split():
    fallback_df = load_fallback_dataset()
    train_df, val_df, test_df = preprocess_dataset(fallback_df, random_state=settings.RANDOM_SEED)

    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0
    assert "clean_text" in train_df.columns
    assert (train_df["label"].isin([0, 1])).all()

    # Check stats
    stats = get_dataset_statistics(fallback_df)
    assert stats["total_samples"] == len(fallback_df)
    assert stats["fraud_count"] + stats["genuine_count"] == stats["total_samples"]

