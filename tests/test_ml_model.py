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
