import hashlib
from pathlib import Path

import pandas as pd
import pytest

from sentiment_cli.data_validation import (
    validate_dataset_separation,
    validate_labeled_data,
)


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "labeled_comments.csv"
TEST_DATA_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "independent_test_comments.csv"
)


def test_labeled_dataset_is_balanced_valid_and_unique():
    data = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

    result = validate_labeled_data(data, minimum_per_class=40)

    assert set(result["label_counts"]) == {"positive", "negative", "neutral"}
    assert all(count >= 40 for count in result["label_counts"].values())
    assert result["original_duplicates"] == 0
    assert result["cleaned_duplicates"] == 0


def test_validate_labeled_data_rejects_invalid_label():
    data = pd.DataFrame({"comment": ["很好"], "label": ["good"]})

    with pytest.raises(ValueError, match="无效标签"):
        validate_labeled_data(data)


def test_validate_labeled_data_rejects_cleaned_duplicate():
    data = pd.DataFrame(
        {
            "comment": ["体验很好！", "体验很好"],
            "label": ["positive", "positive"],
        }
    )

    with pytest.raises(ValueError, match="清洗后重复"):
        validate_labeled_data(data)


def test_fixed_test_dataset_is_valid_balanced_and_separate():
    training = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    testing = pd.read_csv(TEST_DATA_PATH, encoding="utf-8-sig")

    test_result = validate_labeled_data(testing, minimum_per_class=15)
    separation = validate_dataset_separation(training, testing)

    assert test_result["total"] == 45
    assert test_result["label_counts"] == {
        "positive": 15,
        "negative": 15,
        "neutral": 15,
    }
    assert separation["cleaned_overlap"] == 0
    assert hashlib.sha256(TEST_DATA_PATH.read_bytes()).hexdigest().upper() == (
        "30EE726D8E81D7E355B3D7982B1C9B7FB0EE4114ED251F34A4C90264E5E30227"
    )


def test_dataset_separation_rejects_cleaned_overlap():
    training = pd.DataFrame({"comment": ["体验很好！"], "label": ["positive"]})
    testing = pd.DataFrame({"comment": ["体验很好"], "label": ["positive"]})

    with pytest.raises(ValueError, match="独立测试集.*重复"):
        validate_dataset_separation(training, testing)
