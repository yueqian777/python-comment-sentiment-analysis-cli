from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from statistics import fmean, pstdev

import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from sentiment_cli.analyzer import SENTIMENTS, classify_text, silence_native_output
from sentiment_cli.data_validation import (
    load_labeled_csv,
    validate_dataset_separation,
    validate_labeled_data,
)
from sentiment_cli.ml_model import (
    CHAR_NGRAM_RANGE,
    LOGISTIC_C,
    WORD_NGRAM_RANGE,
    train_sentiment_model,
)


METRIC_NAMES = ("accuracy", "macro_precision", "macro_recall", "macro_f1")


def _calculate_metrics(y_true, y_pred) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(SENTIMENTS),
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }


def _summarize_metrics(folds: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        name: {
            "mean": fmean(fold[name] for fold in folds),
            "std": pstdev(fold[name] for fold in folds),
        }
        for name in METRIC_NAMES
    }


def cross_validate_methods(
    texts: pd.Series,
    labels: pd.Series,
    cv_folds: int = 5,
    random_state: int = 42,
) -> dict[str, object]:
    if cv_folds < 2:
        raise ValueError("cv-folds 必须大于等于 2")

    class_counts = labels.value_counts()
    smallest_class = int(class_counts.min())
    actual_folds = min(cv_folds, smallest_class)
    if actual_folds < 2:
        raise ValueError("每个类别至少需要 2 条数据才能进行分层交叉验证")

    splitter = StratifiedKFold(
        n_splits=actual_folds,
        shuffle=True,
        random_state=random_state,
    )
    splits = list(splitter.split(texts, labels))
    lexicon_folds: list[dict[str, float]] = []
    ml_folds: list[dict[str, float]] = []
    lexicon_predictions = [""] * len(texts)
    ml_predictions = [""] * len(texts)

    for train_indexes, validation_indexes in splits:
        x_train = texts.iloc[train_indexes]
        y_train = labels.iloc[train_indexes]
        x_validation = texts.iloc[validation_indexes]
        y_validation = labels.iloc[validation_indexes]

        lexicon_fold_predictions = [classify_text(text) for text in x_validation]
        model = train_sentiment_model(x_train.tolist(), y_train.tolist())
        ml_fold_predictions = [str(item) for item in model.predict(x_validation.tolist())]

        lexicon_folds.append(_calculate_metrics(y_validation, lexicon_fold_predictions))
        ml_folds.append(_calculate_metrics(y_validation, ml_fold_predictions))

        for index, prediction in zip(validation_indexes, lexicon_fold_predictions):
            lexicon_predictions[int(index)] = prediction
        for index, prediction in zip(validation_indexes, ml_fold_predictions):
            ml_predictions[int(index)] = prediction

    metric_labels = list(SENTIMENTS)
    return {
        "requested_folds": cv_folds,
        "actual_folds": actual_folds,
        "adjusted": actual_folds != cv_folds,
        "lexicon": {
            "folds": lexicon_folds,
            "summary": _summarize_metrics(lexicon_folds),
            "matrix": confusion_matrix(labels, lexicon_predictions, labels=metric_labels),
        },
        "ml": {
            "folds": ml_folds,
            "summary": _summarize_metrics(ml_folds),
            "matrix": confusion_matrix(labels, ml_predictions, labels=metric_labels),
        },
    }


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


def _save_metrics_comparison_chart(cv_results: dict[str, object], output_path: Path) -> None:
    with silence_native_output():
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        lexicon = cv_results["lexicon"]["summary"]
        ml = cv_results["ml"]["summary"]
        labels = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]
        lexicon_values = [lexicon[name]["mean"] for name in METRIC_NAMES]
        ml_values = [ml[name]["mean"] for name in METRIC_NAMES]
        positions = list(range(len(labels)))
        width = 0.36

        figure, axis = plt.subplots(figsize=(8, 4.5))
        axis.bar([item - width / 2 for item in positions], lexicon_values, width, label="Lexicon")
        axis.bar([item + width / 2 for item in positions], ml_values, width, label="Machine Learning")
        axis.set_xticks(positions, labels=labels, rotation=15)
        axis.set_ylim(0, 1)
        axis.set_ylabel("Cross-validation mean")
        axis.set_title("Sentiment Classification Metrics")
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_path, dpi=150)
        plt.close(figure)


def _format_cv_method(name: str, result: dict[str, object]) -> list[str]:
    lines = [f"{name}各折指标："]
    for index, metrics in enumerate(result["folds"], start=1):
        values = "，".join(f"{key}={metrics[key]:.4f}" for key in METRIC_NAMES)
        lines.append(f"- 第 {index} 折：{values}")

    lines.append(f"{name}平均值和标准差：")
    for key in METRIC_NAMES:
        summary = result["summary"][key]
        lines.append(f"- {key}: 平均值={summary['mean']:.4f}，标准差={summary['std']:.4f}")
    return lines


def _build_report(
    validation: dict[str, object],
    holdout: dict[str, object],
    cv_results: dict[str, object] | None,
    independent_test: dict[str, object] | None,
) -> str:
    counts = validation["label_counts"]
    evaluation_parts = ["单次分层 train_test_split"]
    if cv_results is not None:
        evaluation_parts.append(f"{cv_results['actual_folds']} 折 StratifiedKFold")
    if independent_test is not None:
        evaluation_parts.append("固定留出测试集 v1")
    lines = [
        "网络评论情感分析评估报告",
        f"数据总量: {validation['total']}",
        "类别数量: " + "，".join(f"{label}={counts[label]}" for label in SENTIMENTS),
        "评估方式: " + " + ".join(evaluation_parts),
        "",
        "单次分层 train_test_split 结果：",
        f"是否使用 stratify: {holdout['split_note']}",
        f"lexicon_accuracy: {holdout['lexicon_accuracy']:.4f}",
        f"ml_accuracy: {holdout['ml_accuracy']:.4f}",
        "词典法单次划分指标: "
        + "，".join(
            f"{name}={holdout['lexicon_metrics'][name]:.4f}"
            for name in METRIC_NAMES
        ),
        "机器学习法单次划分指标: "
        + "，".join(
            f"{name}={holdout['ml_metrics'][name]:.4f}"
            for name in METRIC_NAMES
        ),
        "",
        "词典法 classification_report:",
        holdout["lexicon_report"],
        "词典法 confusion_matrix:",
        str(holdout["lexicon_matrix"]),
        "",
        "机器学习法 classification_report:",
        holdout["ml_report"],
        "机器学习法 confusion_matrix:",
        str(holdout["ml_matrix"]),
    ]

    if cv_results is not None:
        lines.extend(["", f"交叉验证折数: {cv_results['actual_folds']}"])
        if cv_results["adjusted"]:
            lines.append(
                f"折数调整: 请求 {cv_results['requested_folds']} 折，"
                f"因最少类别样本数限制自动调整为 {cv_results['actual_folds']} 折"
            )
        lines.extend(_format_cv_method("词典法", cv_results["lexicon"]))
        lines.extend(_format_cv_method("机器学习法", cv_results["ml"]))
        lines.extend(
            [
                "词典法完整交叉验证混淆矩阵:",
                str(cv_results["lexicon"]["matrix"]),
                "机器学习法完整交叉验证混淆矩阵:",
                str(cv_results["ml"]["matrix"]),
            ]
        )

    if independent_test is not None:
        test_counts = independent_test["validation"]["label_counts"]
        lines.extend(
            [
                "",
                "固定留出测试集 v1 结果：",
                f"测试集来源: {independent_test['source']}",
                f"测试集总量: {independent_test['total']}",
                "测试集类别数量: "
                + "，".join(f"{label}={test_counts[label]}" for label in SENTIMENTS),
                "训练集与测试集清洗后重复: 0",
                "说明: 该留出集已用于上一轮评估，本轮结果用于同一留出集上的版本对比。",
                "该集合已被开发过程查看，不能据此反复调参后再宣称未参与开发决策。",
                "词典法固定留出指标: "
                + "，".join(
                    f"{name}={independent_test['lexicon']['metrics'][name]:.4f}"
                    for name in METRIC_NAMES
                ),
                "机器学习法固定留出指标: "
                + "，".join(
                    f"{name}={independent_test['ml']['metrics'][name]:.4f}"
                    for name in METRIC_NAMES
                ),
                "词典法固定留出混淆矩阵:",
                str(independent_test["lexicon"]["matrix"]),
                "机器学习法固定留出混淆矩阵:",
                str(independent_test["ml"]["matrix"]),
            ]
        )

    lines.extend(
        [
            "",
            "方法局限说明：",
            "单次划分受随机种子和样本组成影响较大。",
            "交叉验证反映训练数据内部的平均稳定性。",
            "固定留出测试集 v1 只反映当前版本在这组特定留出样本上的表现。",
            "三组结果不一致是正常现象，不能只挑最高的一组指标进行宣传。",
            "指标仍受数据规模、标注质量、领域分布和词典覆盖范围影响。",
        ]
    )
    return "\n".join(lines)


def _evaluate_independent_test(
    training_data: pd.DataFrame,
    test_data: pd.DataFrame,
    text_column: str,
    label_column: str,
    source: str,
) -> dict[str, object]:
    validation = validate_labeled_data(test_data, text_column, label_column)
    validate_dataset_separation(training_data, test_data, text_column)

    training_texts = training_data[text_column].astype(str).tolist()
    training_labels = training_data[label_column].astype(str).str.strip().tolist()
    test_texts = test_data[text_column].astype(str).tolist()
    test_labels = test_data[label_column].astype(str).str.strip().tolist()

    lexicon_predictions = [classify_text(text) for text in test_texts]
    model = train_sentiment_model(training_texts, training_labels)
    ml_predictions = [str(item) for item in model.predict(test_texts)]
    metric_labels = list(SENTIMENTS)

    return {
        "source": source,
        "total": len(test_data),
        "validation": validation,
        "lexicon": {
            "metrics": _calculate_metrics(test_labels, lexicon_predictions),
            "matrix": confusion_matrix(
                test_labels, lexicon_predictions, labels=metric_labels
            ),
        },
        "ml": {
            "metrics": _calculate_metrics(test_labels, ml_predictions),
            "matrix": confusion_matrix(test_labels, ml_predictions, labels=metric_labels),
        },
    }


def evaluate_sentiment_methods(
    data: pd.DataFrame,
    text_column: str = "comment",
    label_column: str = "label",
    output_dir: str | Path = "outputs",
    test_size: float = 0.3,
    random_state: int = 42,
    cv_folds: int | None = None,
    independent_data: pd.DataFrame | None = None,
    independent_source: str = "固定留出测试集 v1",
) -> dict[str, object]:
    if len(data) < 6:
        raise ValueError("样本量太少，至少需要 6 条带标签评论才能划分训练集和测试集")
    if not 0 < test_size < 1:
        raise ValueError("test-size 必须是 0 到 1 之间的小数")
    if cv_folds is not None and cv_folds < 2:
        raise ValueError("cv-folds 必须大于等于 2")

    validation = validate_labeled_data(data, text_column, label_column)
    texts = data[text_column].astype(str).reset_index(drop=True)
    labels = data[label_column].astype(str).str.strip().reset_index(drop=True)
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
        raise ValueError("划分后的训练集只包含一个类别，请增加样本数量或调整 test-size")

    lexicon_predictions = [classify_text(text) for text in x_test]
    model = train_sentiment_model(x_train.tolist(), y_train.tolist())
    ml_predictions = [str(item) for item in model.predict(x_test.tolist())]
    metric_labels = list(SENTIMENTS)
    lexicon_matrix = confusion_matrix(y_test, lexicon_predictions, labels=metric_labels)
    ml_matrix = confusion_matrix(y_test, ml_predictions, labels=metric_labels)
    split_note = "是（stratify=labels）"
    if not used_stratify:
        split_note = f"否（普通随机划分；降级原因：{stratify_error}）"

    lexicon_metrics = _calculate_metrics(y_test, lexicon_predictions)
    ml_metrics = _calculate_metrics(y_test, ml_predictions)
    holdout = {
        "split_note": split_note,
        "lexicon_accuracy": lexicon_metrics["accuracy"],
        "ml_accuracy": ml_metrics["accuracy"],
        "lexicon_metrics": lexicon_metrics,
        "ml_metrics": ml_metrics,
        "lexicon_report": classification_report(
            y_test, lexicon_predictions, labels=metric_labels, zero_division=0
        ),
        "ml_report": classification_report(
            y_test, ml_predictions, labels=metric_labels, zero_division=0
        ),
        "lexicon_matrix": lexicon_matrix,
        "ml_matrix": ml_matrix,
    }

    cv_results = None
    if cv_folds is not None:
        cv_results = cross_validate_methods(texts, labels, cv_folds, random_state)

    independent_test = None
    if independent_data is not None:
        independent_test = _evaluate_independent_test(
            data,
            independent_data,
            text_column,
            label_column,
            independent_source,
        )

    destination = Path(output_dir)
    try:
        destination.mkdir(parents=True, exist_ok=True)
        report_path = destination / "evaluation_report.txt"
        report_path.write_text(
            _build_report(validation, holdout, cv_results, independent_test),
            encoding="utf-8",
        )
        chart_path = destination / "confusion_matrix.png"
        chart_matrices = (
            (cv_results["lexicon"]["matrix"], cv_results["ml"]["matrix"])
            if cv_results is not None
            else (lexicon_matrix, ml_matrix)
        )
        _save_confusion_matrix_chart(*chart_matrices, chart_path)
        metrics_path = destination / "metrics_comparison.png"
        if cv_results is not None:
            _save_metrics_comparison_chart(cv_results, metrics_path)
        independent_chart_path = destination / "independent_confusion_matrix.png"
        if independent_test is not None:
            _save_confusion_matrix_chart(
                independent_test["lexicon"]["matrix"],
                independent_test["ml"]["matrix"],
                independent_chart_path,
            )
    except OSError as error:
        raise ValueError(f"无法写入输出目录：{destination}") from error

    return {
        "lexicon_accuracy": holdout["lexicon_accuracy"],
        "ml_accuracy": holdout["ml_accuracy"],
        "used_stratify": used_stratify,
        "cv_folds": None if cv_results is None else cv_results["actual_folds"],
        "cv_results": cv_results,
        "independent_test": independent_test,
        "report_path": report_path,
        "confusion_matrix_path": chart_path,
        "metrics_comparison_path": metrics_path if cv_results is not None else None,
        "independent_confusion_matrix_path": (
            independent_chart_path if independent_test is not None else None
        ),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def _git_commit_sha() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    commit_sha = completed.stdout.strip()
    return commit_sha if completed.returncode == 0 and commit_sha else None


def _git_worktree_dirty() -> bool | None:
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if completed.returncode != 0:
        return None
    return bool(completed.stdout.strip())


def _build_evaluation_metadata(
    training_path: Path,
    training_data: pd.DataFrame,
    test_path: Path | None,
    test_data: pd.DataFrame | None,
    text_column: str,
    label_column: str,
    test_size: float,
    random_state: int,
    requested_cv_folds: int | None,
    actual_cv_folds: int | None,
) -> dict[str, object]:
    training_counts = training_data[label_column].astype(str).str.strip().value_counts()
    test_counts = (
        test_data[label_column].astype(str).str.strip().value_counts()
        if test_data is not None
        else pd.Series(dtype="int64")
    )
    return {
        "training_data_file": training_path.as_posix(),
        "training_data_sha256": _sha256_file(training_path),
        "fixed_holdout_data_file": test_path.as_posix() if test_path else None,
        "fixed_holdout_data_sha256": _sha256_file(test_path) if test_path else None,
        "fixed_holdout_version": "v1" if test_path else None,
        "training_samples": len(training_data),
        "test_samples": 0 if test_data is None else len(test_data),
        "label_counts": {
            "training": dict(sorted(training_counts.astype(int).to_dict().items())),
            "fixed_holdout": dict(sorted(test_counts.astype(int).to_dict().items())),
        },
        "text_column": text_column,
        "label_column": label_column,
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "jieba_version": _package_version("jieba"),
        "pandas_version": pd.__version__,
        "test_size": test_size,
        "random_state": random_state,
        "cv_folds_requested": requested_cv_folds,
        "cv_folds": actual_cv_folds,
        "word_ngram_range": list(WORD_NGRAM_RANGE),
        "char_ngram_range": list(CHAR_NGRAM_RANGE),
        "logistic_regression_c": LOGISTIC_C,
        "preprocessing_function": "sentiment_cli.analyzer.clean_text",
        "preprocessing_location": "inside sklearn pipeline",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_sha": _git_commit_sha(),
        "git_worktree_dirty": _git_worktree_dirty(),
    }


def evaluate_csv(
    input_path: str | Path,
    text_column: str = "comment",
    label_column: str = "label",
    output_dir: str | Path = "outputs",
    test_size: float = 0.3,
    random_state: int = 42,
    cv_folds: int | None = None,
    test_input_path: str | Path | None = None,
) -> dict[str, object]:
    data, _ = load_labeled_csv(input_path, text_column, label_column)
    independent_data = None
    if test_input_path is not None:
        independent_data, _ = load_labeled_csv(
            test_input_path,
            text_column,
            label_column,
        )
    result = evaluate_sentiment_methods(
        data,
        text_column=text_column,
        label_column=label_column,
        output_dir=output_dir,
        test_size=test_size,
        random_state=random_state,
        cv_folds=cv_folds,
        independent_data=independent_data,
        independent_source=str(test_input_path) if test_input_path is not None else "",
    )
    training_path = Path(input_path)
    holdout_path = Path(test_input_path) if test_input_path is not None else None
    metadata = _build_evaluation_metadata(
        training_path,
        data,
        holdout_path,
        independent_data,
        text_column,
        label_column,
        test_size,
        random_state,
        cv_folds,
        result["cv_folds"],
    )
    metadata_path = Path(output_dir) / "evaluation_metadata.json"
    try:
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as error:
        raise ValueError(f"无法保存评估元数据：{metadata_path}") from error
    result["evaluation_metadata_path"] = metadata_path
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="比较词典法与机器学习法的情感分类效果")
    parser.add_argument("-i", "--input", required=True, help="带标签 CSV 文件路径")
    parser.add_argument("-c", "--column", default="comment", help="评论文本所在列名")
    parser.add_argument("-l", "--label", default="label", help="情感标签所在列名")
    parser.add_argument("-o", "--output", default="outputs", help="输出目录")
    parser.add_argument("--test-size", type=float, default=0.3, help="测试集比例")
    parser.add_argument("--random-state", type=int, default=42, help="随机种子")
    parser.add_argument("--cv-folds", type=int, default=5, help="分层交叉验证折数")
    parser.add_argument("--test-input", help="固定留出测试集 v1 CSV 路径")
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
            cv_folds=args.cv_folds,
            test_input_path=args.test_input,
        )
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    print(f"评估完成，报告已保存到：{result['report_path']}")


if __name__ == "__main__":
    main()
