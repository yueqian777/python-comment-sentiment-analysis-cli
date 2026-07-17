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
    # AI 辅助说明：本函数曾使用 OpenAI Codex 桌面应用辅助重构和测试设计。
    # Windows 客户端包版本 26.707.12708.0，模型标识 GPT-5 Codex 编程代理。
    # 最终逻辑由使用者审阅、修改并验证。
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
