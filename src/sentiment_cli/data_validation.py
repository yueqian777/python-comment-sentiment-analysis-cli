from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sentiment_cli.analyzer import SENTIMENTS, clean_text


def validate_labeled_data(
    data: pd.DataFrame,
    text_column: str = "comment",
    label_column: str = "label",
    minimum_per_class: int = 0,
) -> dict[str, object]:
    if text_column not in data.columns:
        raise ValueError(f"CSV 文件中找不到评论列：{text_column}")
    if label_column not in data.columns:
        raise ValueError(f"CSV 文件中找不到标签列：{label_column}")
    if data.empty:
        raise ValueError("CSV 为空，没有可用数据")

    texts = data[text_column].fillna("").astype(str).str.strip()
    if texts.eq("").any():
        rows = [str(index + 2) for index in texts[texts.eq("")].index]
        raise ValueError("评论列存在空文本，CSV 行号：" + "、".join(rows))

    labels = data[label_column].fillna("").astype(str).str.strip()
    invalid_labels = sorted(set(labels) - set(SENTIMENTS))
    if invalid_labels:
        raise ValueError("发现无效标签：" + "、".join(invalid_labels))

    original_duplicates = int(texts.duplicated(keep=False).sum())
    if original_duplicates:
        raise ValueError(f"原文存在重复评论，共涉及 {original_duplicates} 行")

    cleaned = texts.map(clean_text)
    if cleaned.eq("").any():
        raise ValueError("评论清洗后全部为空或包含空文本")
    cleaned_duplicates = int(cleaned.duplicated(keep=False).sum())
    if cleaned_duplicates:
        raise ValueError(f"存在清洗后重复评论，共涉及 {cleaned_duplicates} 行")

    counts = {label: int((labels == label).sum()) for label in SENTIMENTS}
    if minimum_per_class > 0:
        insufficient = [
            f"{label}={count}"
            for label, count in counts.items()
            if count < minimum_per_class
        ]
        if insufficient:
            raise ValueError(
                f"每类至少需要 {minimum_per_class} 条，当前：" + "、".join(insufficient)
            )

    nonzero_counts = [count for count in counts.values() if count > 0]
    warnings = []
    if len(nonzero_counts) != len(SENTIMENTS):
        warnings.append("数据未包含全部三种情感类别")
    elif max(nonzero_counts) > min(nonzero_counts) * 1.5:
        warnings.append("类别数量明显不平衡，评估结果可能偏向样本较多的类别")

    return {
        "total": len(data),
        "label_counts": counts,
        "original_duplicates": 0,
        "cleaned_duplicates": 0,
        "warnings": warnings,
    }


def load_labeled_csv(
    input_path: str | Path,
    text_column: str = "comment",
    label_column: str = "label",
    minimum_per_class: int = 0,
) -> tuple[pd.DataFrame, dict[str, object]]:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到输入文件：{path}")

    try:
        data = pd.read_csv(path, encoding="utf-8-sig")
    except pd.errors.EmptyDataError as error:
        raise ValueError("CSV 为空，没有表头或数据") from error
    except (pd.errors.ParserError, UnicodeError, OSError) as error:
        raise ValueError(f"CSV 无法读取：{path}") from error

    result = validate_labeled_data(
        data,
        text_column=text_column,
        label_column=label_column,
        minimum_per_class=minimum_per_class,
    )
    return data, result


def validate_dataset_separation(
    training_data: pd.DataFrame,
    test_data: pd.DataFrame,
    text_column: str = "comment",
) -> dict[str, int]:
    if text_column not in training_data.columns:
        raise ValueError(f"训练集找不到评论列：{text_column}")
    if text_column not in test_data.columns:
        raise ValueError(f"固定留出测试集找不到评论列：{text_column}")

    training_texts = {
        clean_text(text)
        for text in training_data[text_column].fillna("").astype(str)
        if clean_text(text)
    }
    test_texts = {
        clean_text(text)
        for text in test_data[text_column].fillna("").astype(str)
        if clean_text(text)
    }
    overlap = training_texts & test_texts
    if overlap:
        example = sorted(overlap)[0]
        raise ValueError(f"训练集与固定留出测试集存在清洗后重复：{example}")

    return {
        "training_total": len(training_data),
        "test_total": len(test_data),
        "cleaned_overlap": 0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="检查情感标注 CSV 数据")
    parser.add_argument("-i", "--input", required=True, help="带标签 CSV 文件路径")
    parser.add_argument("-c", "--column", default="comment", help="评论文本列名")
    parser.add_argument("-l", "--label", default="label", help="标签列名")
    parser.add_argument("--minimum-per-class", type=int, default=0, help="每类最少数量")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _, result = load_labeled_csv(
            args.input,
            text_column=args.column,
            label_column=args.label,
            minimum_per_class=args.minimum_per_class,
        )
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    print(f"数据总量：{result['total']}")
    for label, count in result["label_counts"].items():
        print(f"{label}: {count}")
    for warning in result["warnings"]:
        print(f"提示：{warning}")


if __name__ == "__main__":
    main()
