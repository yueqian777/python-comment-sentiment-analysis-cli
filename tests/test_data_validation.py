from pathlib import Path

import pandas as pd
import pytest

from sentiment_cli.data_validation import validate_labeled_data


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "labeled_comments.csv"


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
