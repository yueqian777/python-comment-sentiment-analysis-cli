from argparse import Namespace

import joblib
import pandas as pd
import pytest

from sentiment_cli.main import run
from sentiment_cli.ml_model import train_sentiment_model


def make_args(input_path, output_dir, no_chart=False, method="lexicon", model=None):
    return Namespace(
        input=str(input_path),
        column="comment",
        output=str(output_dir),
        top_n=5,
        no_chart=no_chart,
        method=method,
        model=None if model is None else str(model),
        stopwords=None,
    )


def test_run_generates_both_sentiment_charts(tmp_path):
    input_path = tmp_path / "comments.csv"
    input_path.write_text("comment\n很好很满意\n很差很失望\n今天上午收到\n", encoding="utf-8-sig")
    output_dir = tmp_path / "outputs"

    run(make_args(input_path, output_dir))

    assert (output_dir / "classified_comments.csv").exists()
    assert (output_dir / "summary.txt").exists()
    assert (output_dir / "sentiment_count.png").exists()
    assert (output_dir / "sentiment_ratio.png").exists()

    result = pd.read_csv(output_dir / "classified_comments.csv")
    summary = (output_dir / "summary.txt").read_text(encoding="utf-8")
    assert set(result["classification_method"]) == {"lexicon"}
    assert {"positive_score", "negative_score", "positive_hits"} <= set(result.columns)
    assert "分类方法：lexicon" in summary


def test_run_no_chart_skips_both_sentiment_charts(tmp_path):
    input_path = tmp_path / "comments.csv"
    input_path.write_text("comment\n很好很满意\n很差很失望\n", encoding="utf-8-sig")
    output_dir = tmp_path / "outputs"

    run(make_args(input_path, output_dir, no_chart=True))

    assert not (output_dir / "sentiment_count.png").exists()
    assert not (output_dir / "sentiment_ratio.png").exists()


def test_run_ml_mode_outputs_probabilities(tmp_path):
    training_texts = [
        "很好 满意 推荐",
        "漂亮 舒服 值得",
        "很差 失望 难用",
        "糟糕 难吃 退货",
        "今天收到 蓝色包装",
        "型号A12 下午送达",
    ]
    labels = ["positive", "positive", "negative", "negative", "neutral", "neutral"]
    model = train_sentiment_model(training_texts, labels)
    model_path = tmp_path / "model.joblib"
    joblib.dump(model, model_path)

    input_path = tmp_path / "comments.csv"
    input_path.write_text("comment\n很好值得推荐\n质量很差\n型号是A12\n", encoding="utf-8-sig")
    output_dir = tmp_path / "outputs"
    run(make_args(input_path, output_dir, method="ml", model=model_path))

    result = pd.read_csv(output_dir / "classified_comments.csv")
    probability_columns = {
        "confidence",
        "positive_probability",
        "negative_probability",
        "neutral_probability",
    }
    assert set(result["classification_method"]) == {"ml"}
    assert probability_columns <= set(result.columns)
    assert result["confidence"].between(0, 1).all()


def test_run_ml_mode_requires_model(tmp_path):
    input_path = tmp_path / "comments.csv"
    input_path.write_text("comment\n很好\n", encoding="utf-8-sig")

    with pytest.raises(ValueError, match="--model"):
        run(make_args(input_path, tmp_path / "outputs", method="ml"))


def test_run_rejects_invalid_top_n(tmp_path):
    input_path = tmp_path / "comments.csv"
    input_path.write_text("comment\n很好\n", encoding="utf-8-sig")
    args = make_args(input_path, tmp_path / "outputs")
    args.top_n = 0

    with pytest.raises(ValueError, match="top-n"):
        run(args)


def test_run_rejects_empty_csv(tmp_path):
    input_path = tmp_path / "comments.csv"
    input_path.write_text("comment\n", encoding="utf-8-sig")

    with pytest.raises(ValueError, match="CSV 为空"):
        run(make_args(input_path, tmp_path / "outputs"))
