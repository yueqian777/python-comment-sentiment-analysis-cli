from argparse import Namespace

import joblib
import pandas as pd
import pytest

import sentiment_cli.inference as inference
import sentiment_cli.main as main_module
from sentiment_cli.inference import (
    analyze_comments_by_method,
    analyze_lexicon_comments,
    analyze_ml_comments,
    load_sentiment_model,
)


class FakeModel:
    classes_ = ["negative", "neutral", "positive"]

    def __init__(self, predictions=None, probabilities=None):
        self.predictions = predictions
        self.probabilities = probabilities
        self.predict_input = None
        self.proba_input = None

    def predict(self, texts):
        self.predict_input = list(texts)
        if self.predictions is not None:
            return self.predictions
        return ["positive"] * len(self.predict_input)

    def predict_proba(self, texts):
        self.proba_input = list(texts)
        if self.probabilities is not None:
            return self.probabilities
        return [[0.1, 0.2, 0.7] for _ in self.proba_input]


def test_load_sentiment_model(tmp_path):
    model_path = tmp_path / "model.joblib"
    joblib.dump(FakeModel(), model_path)

    model = load_sentiment_model(model_path)

    assert model.predict(["很好"]) == ["positive"]


def test_load_sentiment_model_reports_missing_and_damaged_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="找不到模型文件"):
        load_sentiment_model(tmp_path / "missing.joblib")

    damaged_path = tmp_path / "damaged.joblib"
    damaged_path.write_bytes(b"not a joblib model")
    with pytest.raises(ValueError, match="模型文件无法加载"):
        load_sentiment_model(damaged_path)


def test_load_sentiment_model_rejects_object_without_predict(tmp_path):
    model_path = tmp_path / "model.joblib"
    joblib.dump({"name": "invalid"}, model_path)

    with pytest.raises(ValueError, match=r"不支持 predict\(\)"):
        load_sentiment_model(model_path)


def test_analyze_lexicon_comments_uses_existing_analyzer():
    result = analyze_lexicon_comments(["服务很好", "质量很差", None])

    assert result["sentiment"].tolist() == ["positive", "negative", "neutral"]
    assert set(result["classification_method"]) == {"lexicon"}
    assert {
        "positive_score",
        "negative_score",
        "positive_hits",
        "negative_hits",
        "negated_hits",
    } <= set(result.columns)


def test_analyze_ml_comments_returns_probabilities_and_confidence():
    model = FakeModel(
        predictions=["positive", "neutral"],
        probabilities=[[0.1, 0.2, 0.7], [0.1, 0.8, 0.1]],
    )

    result = analyze_ml_comments(["服务很好", "今天收到"], model)

    assert set(result["classification_method"]) == {"ml"}
    assert result["sentiment"].tolist() == ["positive", "neutral"]
    assert result["confidence"].tolist() == [0.7, 0.8]
    assert result.loc[0, "positive_probability"] == 0.7
    assert result.loc[0, "negative_probability"] == 0.1
    assert result.loc[0, "neutral_probability"] == 0.2


def test_analyze_ml_comments_passes_raw_text_and_keeps_cleaned_text():
    model = FakeModel()
    raw_text = "  服务很好！ https://example.com  "

    result = analyze_ml_comments(pd.Series([raw_text]), model)

    assert model.predict_input == [raw_text]
    assert model.proba_input == [raw_text]
    assert result.loc[0, "comment"] == raw_text
    assert result.loc[0, "cleaned_text"] == "服务很好"


def test_analyze_comments_by_method_dispatches_lexicon(monkeypatch):
    expected = pd.DataFrame({"sentiment": ["positive"]})
    called = {}

    def fake_analyze(comments, extra_stopwords=None):
        called["comments"] = list(comments)
        called["extra_stopwords"] = extra_stopwords
        return expected

    monkeypatch.setattr(inference, "analyze_lexicon_comments", fake_analyze)

    result = analyze_comments_by_method(["很好"], "lexicon", extra_stopwords={"服务"})

    assert result is expected
    assert called == {"comments": ["很好"], "extra_stopwords": {"服务"}}


def test_analyze_comments_by_method_dispatches_ml(monkeypatch):
    expected = pd.DataFrame({"sentiment": ["negative"]})
    model = FakeModel()
    called = {}

    def fake_analyze(comments, selected_model, extra_stopwords=None):
        called["comments"] = list(comments)
        called["model"] = selected_model
        called["extra_stopwords"] = extra_stopwords
        return expected

    monkeypatch.setattr(inference, "analyze_ml_comments", fake_analyze)

    result = analyze_comments_by_method(
        ["很差"], "ml", model=model, extra_stopwords={"质量"}
    )

    assert result is expected
    assert called == {
        "comments": ["很差"],
        "model": model,
        "extra_stopwords": {"质量"},
    }


def test_analyze_comments_by_method_rejects_invalid_method():
    with pytest.raises(ValueError, match="method 只能是 lexicon 或 ml"):
        analyze_comments_by_method(["很好"], "unknown")


def test_analyze_comments_by_method_requires_model_for_ml():
    with pytest.raises(ValueError, match="必须提供模型"):
        analyze_comments_by_method(["很好"], "ml")


def test_analyze_ml_comments_rejects_invalid_label():
    model = FakeModel(predictions=["unknown"])

    with pytest.raises(ValueError, match="模型输出了非法标签：unknown"):
        analyze_ml_comments(["很好"], model)


def test_main_uses_shared_inference_interface(tmp_path, monkeypatch):
    input_path = tmp_path / "comments.csv"
    input_path.write_text("comment\n服务很好\n", encoding="utf-8-sig")
    output_dir = tmp_path / "outputs"
    called = {}

    def fake_analyze(comments, method, model=None, extra_stopwords=None):
        called["comments"] = list(comments)
        called["method"] = method
        called["model"] = model
        called["extra_stopwords"] = extra_stopwords
        return pd.DataFrame(
            {
                "comment": ["服务很好"],
                "cleaned_text": ["服务很好"],
                "tokens": ["服务"],
                "sentiment": ["positive"],
                "classification_method": ["lexicon"],
            }
        )

    monkeypatch.setattr(main_module, "analyze_comments_by_method", fake_analyze)
    args = Namespace(
        input=str(input_path),
        column="comment",
        output=str(output_dir),
        top_n=5,
        no_chart=True,
        method="lexicon",
        model=None,
        stopwords=None,
    )

    main_module.run(args)

    assert called == {
        "comments": ["服务很好"],
        "method": "lexicon",
        "model": None,
        "extra_stopwords": None,
    }
    assert main_module._load_model is load_sentiment_model
    assert main_module._analyze_with_ml is analyze_ml_comments
