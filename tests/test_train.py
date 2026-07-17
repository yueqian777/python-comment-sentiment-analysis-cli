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
    assert info["algorithm"] == "TF-IDF + LogisticRegression"
    assert info["label_counts"] == {"negative": 3, "neutral": 3, "positive": 3}
