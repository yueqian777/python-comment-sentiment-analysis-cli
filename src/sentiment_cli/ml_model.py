from __future__ import annotations

from collections.abc import Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from sentiment_cli.analyzer import clean_text, tokenize


WORD_NGRAM_RANGE = (1, 2)
CHAR_NGRAM_RANGE = (1, 3)
LOGISTIC_C = 3.0


def train_sentiment_model(texts: Sequence[str], labels: Sequence[str]) -> Pipeline:
    if len(texts) != len(labels):
        raise ValueError("texts 和 labels 的长度必须一致")
    if len(set(labels)) < 2:
        raise ValueError("至少需要两个情感类别才能训练模型")

    model = Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "word",
                            TfidfVectorizer(
                                tokenizer=tokenize,
                                token_pattern=None,
                                lowercase=False,
                                ngram_range=WORD_NGRAM_RANGE,
                                sublinear_tf=True,
                            ),
                        ),
                        (
                            "char",
                            TfidfVectorizer(
                                analyzer="char",
                                preprocessor=clean_text,
                                lowercase=False,
                                ngram_range=CHAR_NGRAM_RANGE,
                                sublinear_tf=True,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "classifier",
                LogisticRegression(max_iter=1000, random_state=42, C=LOGISTIC_C),
            ),
        ]
    )
    model.fit(list(texts), list(labels))
    return model
