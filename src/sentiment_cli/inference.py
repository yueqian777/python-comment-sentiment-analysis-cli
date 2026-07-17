from __future__ import annotations

import pickle
from collections.abc import Iterable
from pathlib import Path

import joblib
import pandas as pd

from sentiment_cli.analyzer import (
    SENTIMENTS,
    analyze_comments,
    clean_text,
    tokenize,
)


def load_sentiment_model(model_path: str | Path | None):
    if not model_path:
        raise ValueError("method=ml 时必须提供 --model")

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到模型文件：{path}")

    try:
        model = joblib.load(path)
    except (
        OSError,
        EOFError,
        ImportError,
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        pickle.UnpicklingError,
    ) as error:
        raise ValueError(f"模型文件无法加载：{path}") from error

    if not hasattr(model, "predict"):
        raise ValueError("模型文件无法加载：对象不支持 predict()")
    return model


def analyze_lexicon_comments(
    comments: Iterable[object],
    extra_stopwords: set[str] | None = None,
) -> pd.DataFrame:
    return analyze_comments(comments, extra_stopwords=extra_stopwords)


def analyze_ml_comments(
    comments: Iterable[object],
    model,
    extra_stopwords: set[str] | None = None,
) -> pd.DataFrame:
    raw_comments = ["" if pd.isna(item) else str(item) for item in comments]
    cleaned_texts = [clean_text(item) for item in raw_comments]

    try:
        predictions = [str(item) for item in model.predict(raw_comments)]
    except (ValueError, TypeError, AttributeError) as error:
        raise ValueError("模型预测失败，请检查模型与输入数据是否兼容") from error

    invalid = sorted(set(predictions) - set(SENTIMENTS))
    if invalid:
        raise ValueError("模型输出了非法标签：" + "、".join(invalid))

    probabilities = None
    classes: list[str] = []
    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(raw_comments)
            classes = [str(item) for item in model.classes_]
        except (ValueError, TypeError, AttributeError):
            probabilities = None
            classes = []

    rows = []
    for index, (comment, cleaned, prediction) in enumerate(
        zip(raw_comments, cleaned_texts, predictions)
    ):
        probability_by_label: dict[str, float] = {}
        if probabilities is not None:
            probability_by_label = {
                label: round(float(value), 6)
                for label, value in zip(classes, probabilities[index])
                if label in SENTIMENTS
            }

        rows.append(
            {
                "comment": comment,
                "cleaned_text": cleaned,
                "tokens": " ".join(tokenize(cleaned, extra_stopwords=extra_stopwords)),
                "sentiment": prediction,
                "classification_method": "ml",
                "confidence": probability_by_label.get(prediction),
                "positive_probability": probability_by_label.get("positive"),
                "negative_probability": probability_by_label.get("negative"),
                "neutral_probability": probability_by_label.get("neutral"),
            }
        )

    return pd.DataFrame(rows)


def analyze_comments_by_method(
    comments: Iterable[object],
    method: str,
    model=None,
    extra_stopwords: set[str] | None = None,
) -> pd.DataFrame:
    if method == "lexicon":
        return analyze_lexicon_comments(
            comments,
            extra_stopwords=extra_stopwords,
        )
    if method == "ml":
        if model is None:
            raise ValueError("method=ml 时必须提供模型")
        return analyze_ml_comments(
            comments,
            model,
            extra_stopwords=extra_stopwords,
        )
    raise ValueError("method 只能是 lexicon 或 ml")
