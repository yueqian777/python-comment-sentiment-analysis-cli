import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from sentiment_cli.ui_helpers import (
    SENTIMENT_COLORS,
    SENTIMENT_LABELS,
    dataframe_to_csv_bytes,
    build_summary_text,
    confidence_distribution_table,
    filter_results,
    keyword_count_table,
    low_confidence_rows,
    sentiment_count_table,
    sentiment_ratio_table,
)


@pytest.fixture
def lexicon_result():
    return pd.DataFrame(
        {
            "comment": ["外观很好", "物流太慢", "包装普通", "服务很好"],
            "cleaned_text": ["外观很好", "物流太慢", "包装普通", "服务很好"],
            "tokens": ["外观 很好", "物流 太慢", "包装 普通", "服务 很好"],
            "sentiment": ["positive", "negative", "neutral", "positive"],
            "classification_method": ["lexicon"] * 4,
            "positive_score": [1, 0, 0, 1],
            "negative_score": [0, 1, 0, 0],
        }
    )


@pytest.fixture
def ml_result():
    return pd.DataFrame(
        {
            "comment": ["体验很好", "配送很慢", "包装一般"],
            "cleaned_text": ["体验很好", "配送很慢", "包装一般"],
            "tokens": ["体验 很好", "配送 很慢", "包装 一般"],
            "sentiment": ["positive", "negative", "neutral"],
            "classification_method": ["ml"] * 3,
            "confidence": [0.91, 0.55, 0.72],
            "positive_probability": [0.91, 0.20, 0.14],
            "negative_probability": [0.04, 0.55, 0.14],
            "neutral_probability": [0.05, 0.25, 0.72],
        }
    )


def test_sentiment_mappings_are_stable_and_complete():
    assert SENTIMENT_LABELS == {
        "positive": "正面",
        "negative": "负面",
        "neutral": "中性",
    }
    assert set(SENTIMENT_COLORS) == set(SENTIMENT_LABELS)
    assert len(set(SENTIMENT_COLORS.values())) == 3


def test_sentiment_count_table_keeps_fixed_order(lexicon_result):
    table = sentiment_count_table(lexicon_result)

    assert table.columns.tolist() == ["sentiment", "label", "count"]
    assert table["sentiment"].tolist() == ["positive", "negative", "neutral"]
    assert table["label"].tolist() == ["正面", "负面", "中性"]
    assert table["count"].tolist() == [2, 1, 1]


def test_sentiment_ratio_table_sums_to_one(lexicon_result):
    table = sentiment_ratio_table(lexicon_result)

    assert table.columns.tolist() == ["sentiment", "label", "ratio"]
    assert table["ratio"].sum() == pytest.approx(1.0)
    assert table.loc[table["sentiment"] == "positive", "ratio"].item() == 0.5


def test_keyword_count_table_sorts_count_then_word(lexicon_result):
    table = keyword_count_table(lexicon_result, top_n=3)

    assert table.columns.tolist() == ["keyword", "count"]
    assert table.to_dict("records") == [
        {"keyword": "很好", "count": 2},
        {"keyword": "包装", "count": 1},
        {"keyword": "外观", "count": 1},
    ]


def test_filter_results_accepts_chinese_sentiment_labels(lexicon_result):
    filtered = filter_results(lexicon_result, sentiments=["正面", "中性"])

    assert filtered["sentiment"].tolist() == ["positive", "neutral", "positive"]


def test_filter_results_searches_original_comment(lexicon_result):
    filtered = filter_results(lexicon_result, keyword="服务")

    assert filtered["comment"].tolist() == ["服务很好"]


def test_filter_results_applies_confidence_range(ml_result):
    filtered = filter_results(ml_result, min_confidence=0.60, max_confidence=0.80)

    assert filtered["comment"].tolist() == ["包装一般"]


def test_filter_results_can_show_only_low_confidence_rows(ml_result):
    filtered = filter_results(
        ml_result,
        low_confidence_only=True,
        low_confidence_threshold=0.60,
    )

    assert filtered["comment"].tolist() == ["配送很慢"]


def test_filter_results_supports_probability_sorting(ml_result):
    filtered = filter_results(ml_result, sort_by="负面概率降序")

    assert filtered["comment"].tolist() == ["配送很慢", "包装一般", "体验很好"]


def test_filter_results_does_not_modify_original_dataframe(ml_result):
    original = ml_result.copy(deep=True)

    filter_results(
        ml_result,
        sentiments=["negative"],
        keyword="配送",
        sort_by="置信度升序",
    )

    assert_frame_equal(ml_result, original)


def test_low_confidence_rows_uses_strict_threshold(ml_result):
    rows = low_confidence_rows(ml_result, threshold=0.72)

    assert rows["confidence"].tolist() == [0.55]


def test_confidence_distribution_table_counts_fixed_bins(ml_result):
    table = confidence_distribution_table(ml_result, bins=5)

    assert table.columns.tolist() == ["range", "count"]
    assert table["count"].sum() == 3
    assert len(table) == 5
    assert table.loc[table["range"] == "0.4-0.6", "count"].item() == 1


def test_empty_results_return_stable_tables():
    empty = pd.DataFrame()

    assert sentiment_count_table(empty)["count"].tolist() == [0, 0, 0]
    assert sentiment_ratio_table(empty)["ratio"].tolist() == [0.0, 0.0, 0.0]
    assert keyword_count_table(empty).empty
    assert confidence_distribution_table(empty).empty
    assert low_confidence_rows(empty).empty
    assert filter_results(empty).empty


def test_missing_required_columns_raise_clear_errors():
    result = pd.DataFrame({"comment": ["很好"]})

    with pytest.raises(ValueError, match="sentiment"):
        sentiment_count_table(result)
    with pytest.raises(ValueError, match="tokens"):
        keyword_count_table(result)
    with pytest.raises(ValueError, match="confidence"):
        low_confidence_rows(result)


def test_dataframe_to_csv_bytes_uses_utf8_bom(lexicon_result):
    content = dataframe_to_csv_bytes(lexicon_result)

    assert content.startswith(b"\xef\xbb\xbf")
    assert "外观很好" in content.decode("utf-8-sig")


def test_lexicon_summary_contains_counts_ratios_and_keywords(lexicon_result):
    summary = build_summary_text(lexicon_result, method="lexicon", top_n=2)

    assert "数据总量：4" in summary
    assert "分类方法：词典法" in summary
    assert "正面：2 条（50.00%）" in summary
    assert "很好：2" in summary
    assert "分析结果仅供参考" in summary


def test_ml_summary_contains_confidence_information(ml_result):
    summary = build_summary_text(
        ml_result,
        method="ml",
        low_confidence_threshold=0.60,
    )

    assert "分类方法：机器学习法" in summary
    assert "平均置信度：72.67%" in summary
    assert "低置信度评论数量（低于 60.00%）：1" in summary
    assert "模型置信度不等于预测一定正确" in summary
