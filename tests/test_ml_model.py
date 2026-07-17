import numpy as np

from sentiment_cli.analyzer import clean_text
from sentiment_cli.ml_model import train_sentiment_model


def test_train_sentiment_model_predicts_known_style_texts():
    texts = [
        "很好吃 很满意",
        "配送很快 推荐",
        "质量太差 失望",
        "服务很差 不会再买",
        "今天收到 包装蓝色",
        "商品一般 描述一致",
    ]
    labels = ["positive", "positive", "negative", "negative", "neutral", "neutral"]

    model = train_sentiment_model(texts, labels)

    predictions = model.predict(["味道很好 推荐", "质量很差 失望", "上午收到 包装完整"])

    assert list(predictions) == ["positive", "negative", "neutral"]


def test_train_sentiment_model_combines_word_and_character_features():
    texts = ["很好", "满意", "很差", "失望", "型号A12", "今天送达"]
    labels = ["positive", "positive", "negative", "negative", "neutral", "neutral"]

    model = train_sentiment_model(texts, labels)
    transformers = dict(model.named_steps["features"].transformer_list)

    assert set(transformers) == {"word", "char"}
    assert transformers["word"].ngram_range == (1, 2)
    assert transformers["char"].analyzer == "char"
    assert transformers["char"].preprocessor is clean_text
    assert transformers["char"].lowercase is False
    assert transformers["char"].ngram_range == (1, 3)


def test_pipeline_normalizes_raw_text_inside_feature_extraction():
    texts = [
        "服务很好 物流很快",
        "味道满意 值得推荐",
        "质量很差 不会再买",
        "包装破损 非常失望",
        "今天收到 蓝色包装",
        "型号A12 下午送达",
    ]
    labels = ["positive", "positive", "negative", "negative", "neutral", "neutral"]
    model = train_sentiment_model(texts, labels)
    pairs = [
        ("服务很好！物流也很快。", "服务很好 物流也很快"),
        ("质量很差 https://example.com 不会再买", "质量很差 不会再买"),
        ("型号A12    下午送达", "型号A12 下午送达"),
    ]
    features = model.named_steps["features"]

    for raw_text, cleaned_text in pairs:
        raw_features = features.transform([raw_text]).toarray()
        cleaned_features = features.transform([cleaned_text]).toarray()
        np.testing.assert_allclose(raw_features, cleaned_features)
        assert model.predict([raw_text])[0] == model.predict([cleaned_text])[0]
        np.testing.assert_allclose(
            model.predict_proba([raw_text]),
            model.predict_proba([cleaned_text]),
        )
