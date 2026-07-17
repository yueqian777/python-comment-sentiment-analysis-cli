import json

import joblib

from sentiment_cli.train import train_csv


def test_train_csv_saves_model_and_model_info(tmp_path):
    input_path = tmp_path / "labeled.csv"
    input_path.write_text(
        "comment,label\n"
        "味道很好,positive\n服务满意,positive\n值得推荐,positive\n"
        "味道很差,negative\n服务糟糕,negative\n不会再买,negative\n"
        "今天收到,neutral\n型号A12,neutral\n周二上课,neutral\n",
        encoding="utf-8-sig",
    )
    model_path = tmp_path / "models" / "sentiment_model.joblib"

    result = train_csv(input_path, model_path=model_path)
    info = json.loads(result["info_path"].read_text(encoding="utf-8"))

    assert model_path.stat().st_size > 0
    assert result["info_path"].stat().st_size > 0
    assert joblib.load(model_path).predict(["很好"])[0] in {
        "positive", "negative", "neutral"
    }
    assert info["training_samples"] == 9
    assert info["algorithm"] == "Word + Character TF-IDF + LogisticRegression"
    assert info["features"]["word_ngram_range"] == [1, 2]
    assert info["features"]["char_ngram_range"] == [1, 3]
    assert info["preprocessing"] == {
        "function": "sentiment_cli.analyzer.clean_text",
        "location": "inside sklearn pipeline",
        "external_preprocessing_required": False,
    }
    assert info["label_counts"] == {"negative": 3, "neutral": 3, "positive": 3}

    loaded_model = joblib.load(model_path)
    raw_texts = [
        "味道很好！！！",
        "味道很好",
        "服务糟糕 https://example.com",
        "服务糟糕",
        "型号A12    今天收到",
        "型号A12 今天收到",
    ]
    predictions = loaded_model.predict(raw_texts)
    probabilities = loaded_model.predict_proba(raw_texts)
    assert predictions[0] == predictions[1]
    assert predictions[2] == predictions[3]
    assert predictions[4] == predictions[5]
    assert (probabilities[0] == probabilities[1]).all()
    assert (probabilities[2] == probabilities[3]).all()
    assert (probabilities[4] == probabilities[5]).all()
