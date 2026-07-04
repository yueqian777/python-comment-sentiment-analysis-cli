from __future__ import annotations

from collections.abc import Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sentiment_cli.analyzer import tokenize


def train_sentiment_model(texts: Sequence[str], labels: Sequence[str]) -> Pipeline:
    if len(texts) != len(labels):
        raise ValueError("texts 和 labels 的长度必须一致")
    if len(set(labels)) < 2:
        raise ValueError("至少需要两个情感类别才能训练模型")

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    tokenizer=tokenize,
                    token_pattern=None,
                    lowercase=False,
                ),
            ),
            (
                "classifier",
                LogisticRegression(max_iter=1000, random_state=42),
            ),
        ]
    )
    model.fit(list(texts), list(labels))
    return model
