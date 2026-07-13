import pandas as pd
import pytest

from sentiment_cli.evaluate import evaluate_sentiment_methods


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
