from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from sentiment_cli.analyzer import SENTIMENTS, classify_text, silence_native_output
from sentiment_cli.ml_model import train_sentiment_model


def _save_confusion_matrix_chart(
    lexicon_matrix,
    ml_matrix,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with silence_native_output():
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(1, 2, figsize=(10, 4))
        labels = list(SENTIMENTS)

        for axis, matrix, title in zip(
            axes,
            (lexicon_matrix, ml_matrix),
            ("Lexicon Method", "Machine Learning"),
        ):
            image = axis.imshow(matrix, cmap="Blues")
            axis.set_title(title)
            axis.set_xlabel("Predicted label")
            axis.set_ylabel("True label")
            axis.set_xticks(range(len(labels)), labels=labels, rotation=30, ha="right")
            axis.set_yticks(range(len(labels)), labels=labels)

            for row in range(len(labels)):
                for column in range(len(labels)):
                    axis.text(column, row, str(matrix[row, column]), ha="center", va="center")

            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

        figure.tight_layout()
        figure.savefig(output_path, dpi=150)
        plt.close(figure)


def _build_report(
    lexicon_accuracy: float,
    ml_accuracy: float,
    lexicon_report: str,
    ml_report: str,
    lexicon_matrix,
    ml_matrix,
    used_stratify: bool,
    stratify_error: str | None = None,
) -> str:
    if used_stratify:
        split_note = "是（stratify=labels）"
    else:
        split_note = f"否（普通随机划分；降级原因：{stratify_error}）"
    return "\n".join(
        [
            "网络评论情感分析评估报告",
            f"是否使用 stratify: {split_note}",
            "",
            f"lexicon_accuracy: {lexicon_accuracy:.4f}",
            f"ml_accuracy: {ml_accuracy:.4f}",
            "",
            "词典法 classification_report:",
            lexicon_report,
            "词典法 confusion_matrix:",
            str(lexicon_matrix),
            "",
            "机器学习法 classification_report:",
            ml_report,
            "机器学习法 confusion_matrix:",
            str(ml_matrix),
        ]
    )


def evaluate_sentiment_methods(
    data: pd.DataFrame,
    text_column: str = "comment",
    label_column: str = "label",
    output_dir: str | Path = "outputs",
    test_size: float = 0.3,
    random_state: int = 42,
) -> dict[str, object]:
    if text_column not in data.columns:
        raise ValueError(f"CSV 文件中找不到评论列：{text_column}")
    if label_column not in data.columns:
        raise ValueError(f"CSV 文件中找不到标签列：{label_column}")
    if len(data) < 6:
        raise ValueError("样本量太少，至少需要 6 条带标签评论才能划分训练集和测试集")
    if not 0 < test_size < 1:
        raise ValueError("test_size 必须是 0 到 1 之间的小数")

    texts = data[text_column].fillna("").astype(str)
    labels = data[label_column].fillna("").astype(str).str.strip()
    invalid_labels = sorted(set(labels) - set(SENTIMENTS))
    if invalid_labels:
        raise ValueError(
            "标签列只能包含 positive、negative、neutral，发现无效值："
            + "、".join(invalid_labels)
        )
    if labels.nunique() < 2:
        raise ValueError("至少需要两个情感类别才能进行机器学习评估")

    used_stratify = True
    stratify_error = None
    try:
        x_train, x_test, y_train, y_test = train_test_split(
            texts,
            labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels,
        )
    except ValueError as error:
        used_stratify = False
        stratify_error = str(error)
        try:
            x_train, x_test, y_train, y_test = train_test_split(
                texts,
                labels,
                test_size=test_size,
                random_state=random_state,
            )
        except ValueError as split_error:
            raise ValueError(f"无法划分训练集和测试集：{split_error}") from split_error

    if y_train.nunique() < 2:
        raise ValueError("划分后的训练集只包含一个类别，请增加样本数量或调整 test_size")

    lexicon_predictions = [classify_text(text) for text in x_test]
    model = train_sentiment_model(x_train.tolist(), y_train.tolist())
    ml_predictions = model.predict(x_test.tolist())

    metric_labels = list(SENTIMENTS)
    lexicon_accuracy = accuracy_score(y_test, lexicon_predictions)
    ml_accuracy = accuracy_score(y_test, ml_predictions)
    lexicon_report = classification_report(
        y_test,
        lexicon_predictions,
        labels=metric_labels,
        zero_division=0,
    )
    ml_report = classification_report(
        y_test,
        ml_predictions,
        labels=metric_labels,
        zero_division=0,
    )
    lexicon_matrix = confusion_matrix(y_test, lexicon_predictions, labels=metric_labels)
    ml_matrix = confusion_matrix(y_test, ml_predictions, labels=metric_labels)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "evaluation_report.txt"
    chart_path = output_dir / "confusion_matrix.png"

    report_path.write_text(
        _build_report(
            lexicon_accuracy,
            ml_accuracy,
            lexicon_report,
            ml_report,
            lexicon_matrix,
            ml_matrix,
            used_stratify=used_stratify,
            stratify_error=stratify_error,
        ),
        encoding="utf-8",
    )
    _save_confusion_matrix_chart(lexicon_matrix, ml_matrix, chart_path)

    return {
        "lexicon_accuracy": lexicon_accuracy,
        "ml_accuracy": ml_accuracy,
        "used_stratify": used_stratify,
        "report_path": report_path,
        "confusion_matrix_path": chart_path,
    }


def evaluate_csv(
    input_path: str | Path,
    text_column: str = "comment",
    label_column: str = "label",
    output_dir: str | Path = "outputs",
    test_size: float = 0.3,
    random_state: int = 42,
) -> dict[str, object]:
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件：{input_path}")

    data = pd.read_csv(input_path, encoding="utf-8-sig")
    return evaluate_sentiment_methods(
        data,
        text_column=text_column,
        label_column=label_column,
        output_dir=output_dir,
        test_size=test_size,
        random_state=random_state,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="比较词典法与机器学习法的情感分类效果")
    parser.add_argument("-i", "--input", required=True, help="带标签 CSV 文件路径")
    parser.add_argument("-c", "--column", default="comment", help="评论文本所在列名")
    parser.add_argument("-l", "--label", default="label", help="情感标签所在列名")
    parser.add_argument("-o", "--output", default="outputs", help="输出目录")
    parser.add_argument("--test-size", type=float, default=0.3, help="测试集比例")
    parser.add_argument("--random-state", type=int, default=42, help="随机种子")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        result = evaluate_csv(
            args.input,
            text_column=args.column,
            label_column=args.label,
            output_dir=args.output,
            test_size=args.test_size,
            random_state=args.random_state,
        )
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        parser.error(str(error))

    print(f"评估完成，报告已保存到：{result['report_path']}")


if __name__ == "__main__":
    main()
