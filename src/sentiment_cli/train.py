from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn

from sentiment_cli.data_validation import load_labeled_csv
from sentiment_cli.ml_model import (
    CHAR_NGRAM_RANGE,
    LOGISTIC_C,
    WORD_NGRAM_RANGE,
    train_sentiment_model,
)


def train_csv(
    input_path: str | Path,
    text_column: str = "comment",
    label_column: str = "label",
    model_path: str | Path = "models/sentiment_model.joblib",
) -> dict[str, Path]:
    data, validation = load_labeled_csv(
        input_path,
        text_column=text_column,
        label_column=label_column,
    )
    texts = data[text_column].astype(str).tolist()
    labels = data[label_column].astype(str).str.strip().tolist()
    model = train_sentiment_model(texts, labels)

    destination = Path(model_path)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, destination)
    except OSError as error:
        raise ValueError(f"无法保存模型文件：{destination}") from error

    info_path = destination.parent / "model_info.json"
    info = {
        "algorithm": "Word + Character TF-IDF + LogisticRegression",
        "features": {
            "word_ngram_range": list(WORD_NGRAM_RANGE),
            "char_ngram_range": list(CHAR_NGRAM_RANGE),
            "logistic_regression_c": LOGISTIC_C,
        },
        "training_samples": len(data),
        "label_counts": dict(sorted(validation["label_counts"].items())),
        "labels": sorted(validation["label_counts"]),
        "text_column": text_column,
        "label_column": label_column,
        "random_state": 42,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
    }
    try:
        info_path.write_text(
            json.dumps(info, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as error:
        raise ValueError(f"无法保存模型信息：{info_path}") from error

    return {"model_path": destination, "info_path": info_path}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="训练并保存情感分类模型")
    parser.add_argument("-i", "--input", required=True, help="带标签 CSV 文件路径")
    parser.add_argument("-c", "--column", default="comment", help="评论文本列名")
    parser.add_argument("-l", "--label", default="label", help="标签列名")
    parser.add_argument(
        "--model",
        default="models/sentiment_model.joblib",
        help="模型保存路径",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = train_csv(
            args.input,
            text_column=args.column,
            label_column=args.label,
            model_path=args.model,
        )
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    print(f"训练完成，模型已保存到：{result['model_path']}")
    print(f"模型信息已保存到：{result['info_path']}")


if __name__ == "__main__":
    main()
