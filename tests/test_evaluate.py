import hashlib
import json

import pandas as pd
import pytest

from sentiment_cli.evaluate import evaluate_csv, evaluate_sentiment_methods
from sentiment_cli.ml_model import CHAR_NGRAM_RANGE, LOGISTIC_C, WORD_NGRAM_RANGE


def test_evaluate_sentiment_methods_generates_report_and_chart(tmp_path):
    data = pd.DataFrame(
        {
            "comment": [
                "味道很好值得推荐",
                "服务周到体验满意",
                "电影精彩演员优秀",
                "课程清晰收获很多",
                "房间舒服位置方便",
                "配送很快包装漂亮",
                "味道很差非常失望",
                "服务敷衍不会再买",
                "电影无聊剧情糟糕",
                "课程难懂讲解很差",
                "房间很脏体验失望",
                "配送太慢饭菜难吃",
                "今天上午收到商品",
                "电影时长两个小时",
                "课程安排在周一",
                "酒店位于市中心",
                "外卖使用纸质包装",
                "商品颜色是蓝色",
            ],
            "label": ["positive"] * 6 + ["negative"] * 6 + ["neutral"] * 6,
        }
    )
    output_dir = tmp_path / "outputs"

    result = evaluate_sentiment_methods(data, output_dir=output_dir)

    report_path = output_dir / "evaluation_report.txt"
    chart_path = output_dir / "confusion_matrix.png"
    report = report_path.read_text(encoding="utf-8")

    assert report_path.exists()
    assert chart_path.exists()
    assert chart_path.stat().st_size > 0
    assert "lexicon_accuracy" in report
    assert "ml_accuracy" in report
    assert "单次分层 train_test_split" in report
    assert "classification_report" in report
    assert "confusion_matrix" in report
    assert result["used_stratify"] is True


def test_evaluate_sentiment_methods_falls_back_without_stratify(tmp_path):
    data = pd.DataFrame(
        {
            "comment": [
                "体验满意值得推荐",
                "服务很好十分满意",
                "课程精彩内容清晰",
                "酒店舒服位置方便",
                "味道很差非常失望",
                "服务糟糕不会再买",
                "电影无聊体验很差",
                "房间很脏令人失望",
                "商品颜色是蓝色",
            ],
            "label": ["positive"] * 4 + ["negative"] * 4 + ["neutral"],
        }
    )
    output_dir = tmp_path / "outputs"

    result = evaluate_sentiment_methods(data, output_dir=output_dir)
    report = (output_dir / "evaluation_report.txt").read_text(encoding="utf-8")

    assert result["used_stratify"] is False
    assert "是否使用 stratify: 否" in report


def test_evaluate_sentiment_methods_rejects_too_few_samples(tmp_path):
    data = pd.DataFrame(
        {
            "comment": ["很好", "很差", "今天收到"],
            "label": ["positive", "negative", "neutral"],
        }
    )

    with pytest.raises(ValueError, match="样本量太少"):
        evaluate_sentiment_methods(data, output_dir=tmp_path)


def test_cross_validation_reports_fold_metrics_and_summary(tmp_path):
    data = pd.DataFrame(
        {
            "comment": [
                "味道很好值得推荐", "服务周到十分满意", "电影精彩演员优秀",
                "课程清晰收获很多", "房间舒服位置方便", "配送很快包装漂亮",
                "味道很差非常失望", "服务敷衍不会再买", "剧情混乱体验糟糕",
                "课程难懂讲解很差", "房间很脏令人失望", "配送太慢饭菜难吃",
                "今天上午收到商品", "电影时长两个小时", "课程安排在周一",
                "酒店位于市中心", "外卖使用纸质包装", "商品颜色是蓝色",
            ],
            "label": ["positive"] * 6 + ["negative"] * 6 + ["neutral"] * 6,
        }
    )

    result = evaluate_sentiment_methods(data, output_dir=tmp_path, cv_folds=3)
    report = (tmp_path / "evaluation_report.txt").read_text(encoding="utf-8")

    assert result["cv_folds"] == 3
    assert len(result["cv_results"]["lexicon"]["folds"]) == 3
    assert "交叉验证折数: 3" in report
    assert "平均值" in report
    assert "标准差" in report
    assert (tmp_path / "metrics_comparison.png").stat().st_size > 0


def test_evaluation_reports_fixed_holdout_set_v1(tmp_path):
    training = pd.DataFrame(
        {
            "comment": [
                "味道很好", "服务满意", "值得推荐", "画面漂亮",
                "味道很差", "服务糟糕", "不会再买", "剧情混乱",
                "今天送达", "型号A12", "课程周二上课", "电影九十分钟",
            ],
            "label": ["positive"] * 4 + ["negative"] * 4 + ["neutral"] * 4,
        }
    )
    independent = pd.DataFrame(
        {
            "comment": [
                "住得舒服", "讲解清楚", "饭菜难吃",
                "房间很脏", "包装含说明书", "周五提交报告",
            ],
            "label": ["positive", "positive", "negative", "negative", "neutral", "neutral"],
        }
    )

    result = evaluate_sentiment_methods(
        training,
        output_dir=tmp_path,
        cv_folds=2,
        independent_data=independent,
    )
    report = (tmp_path / "evaluation_report.txt").read_text(encoding="utf-8")

    assert result["independent_test"] is not None
    assert result["independent_test"]["total"] == 6
    assert "固定留出测试集 v1 结果" in report
    assert (tmp_path / "independent_confusion_matrix.png").stat().st_size > 0


def test_evaluate_csv_writes_reproducible_metadata(tmp_path):
    training = pd.DataFrame(
        {
            "comment": [
                "味道很好", "服务满意", "值得推荐", "画面漂亮",
                "味道很差", "服务糟糕", "不会再买", "剧情混乱",
                "今天送达", "型号A12", "课程周二上课", "电影九十分钟",
            ],
            "label": ["positive"] * 4 + ["negative"] * 4 + ["neutral"] * 4,
        }
    )
    holdout = pd.DataFrame(
        {
            "comment": [
                "住得舒服", "讲解清楚", "饭菜难吃",
                "房间很脏", "包装含说明书", "周五提交报告",
            ],
            "label": ["positive", "positive", "negative", "negative", "neutral", "neutral"],
        }
    )
    training_path = tmp_path / "training.csv"
    holdout_path = tmp_path / "holdout_v1.csv"
    output_dir = tmp_path / "outputs"
    training.to_csv(training_path, index=False, encoding="utf-8-sig")
    holdout.to_csv(holdout_path, index=False, encoding="utf-8-sig")

    result = evaluate_csv(
        training_path,
        output_dir=output_dir,
        cv_folds=2,
        test_input_path=holdout_path,
    )

    metadata_path = result["evaluation_metadata_path"]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata_path.stat().st_size > 0
    assert metadata["training_data_sha256"] == hashlib.sha256(
        training_path.read_bytes()
    ).hexdigest()
    assert metadata["fixed_holdout_data_sha256"] == hashlib.sha256(
        holdout_path.read_bytes()
    ).hexdigest()
    assert metadata["training_samples"] == 12
    assert metadata["test_samples"] == 6
    assert metadata["random_state"] == 42
    assert metadata["cv_folds"] == 2
    assert metadata["word_ngram_range"] == list(WORD_NGRAM_RANGE)
    assert metadata["char_ngram_range"] == list(CHAR_NGRAM_RANGE)
    assert metadata["logistic_regression_c"] == LOGISTIC_C
    assert metadata["preprocessing_function"] == "sentiment_cli.analyzer.clean_text"
    for path in (
        result["report_path"],
        result["confusion_matrix_path"],
        result["metrics_comparison_path"],
        result["independent_confusion_matrix_path"],
    ):
        assert path.stat().st_size > 0
