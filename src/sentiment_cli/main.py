from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sentiment_cli.analyzer import (
    clean_text,
    load_stopwords,
    save_sentiment_count_chart,
    save_sentiment_ratio_chart,
    save_summary_file,
    sentiment_summary,
    top_keywords,
)
from sentiment_cli.inference import (
    analyze_comments_by_method,
    analyze_ml_comments,
    load_sentiment_model,
)


_load_model = load_sentiment_model
_analyze_with_ml = analyze_ml_comments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="网络评论情感分析系统")
    parser.add_argument("-i", "--input", required=True, help="输入 CSV 文件路径")
    parser.add_argument("-c", "--column", default="comment", help="评论文本所在列名")
    parser.add_argument("-o", "--output", default="outputs", help="输出目录")
    parser.add_argument("--top-n", type=int, default=10, help="输出的关键词数量")
    parser.add_argument("--no-chart", action="store_true", help="不生成情感统计图")
    parser.add_argument(
        "--method",
        choices=("lexicon", "ml"),
        default="lexicon",
        help="分类方法，默认使用词典法",
    )
    parser.add_argument("--model", help="机器学习模式使用的 joblib 模型路径")
    parser.add_argument("--stopwords", help="额外停用词文件路径")
    return parser


def _read_comments(input_path: Path, column: str) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件：{input_path}")

    try:
        data = pd.read_csv(input_path, encoding="utf-8-sig")
    except pd.errors.EmptyDataError as error:
        raise ValueError("CSV 为空，没有表头或数据") from error
    except (pd.errors.ParserError, UnicodeError, OSError) as error:
        raise ValueError(f"CSV 无法读取：{input_path}") from error

    if data.empty:
        raise ValueError("CSV 为空，没有可分析的评论")
    if column not in data.columns:
        raise ValueError(f"CSV 文件中找不到列：{column}")

    comments = data[column].fillna("").astype(str)
    if comments.map(clean_text).eq("").all():
        raise ValueError("评论列全部为空，无法进行分析")
    return data


def run(args: argparse.Namespace) -> Path:
    method = getattr(args, "method", "lexicon")
    model_path = getattr(args, "model", None)
    stopwords_path = getattr(args, "stopwords", None)
    top_n = getattr(args, "top_n", 10)
    no_chart = getattr(args, "no_chart", False)

    if method not in {"lexicon", "ml"}:
        raise ValueError("method 只能是 lexicon 或 ml")
    if top_n < 1:
        raise ValueError("top-n 必须大于等于 1")

    input_path = Path(args.input)
    data = _read_comments(input_path, args.column)
    extra_stopwords = load_stopwords(stopwords_path) if stopwords_path else None

    model = load_sentiment_model(model_path) if method == "ml" else None
    result = analyze_comments_by_method(
        data[args.column],
        method,
        model=model,
        extra_stopwords=extra_stopwords,
    )

    summary = sentiment_summary(result["sentiment"])
    keywords = top_keywords(
        data[args.column],
        limit=top_n,
        extra_stopwords=extra_stopwords,
    )

    output_dir = Path(args.output)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ValueError(f"无法创建输出目录：{output_dir}") from error

    classified_path = output_dir / "classified_comments.csv"
    summary_path = output_dir / "summary.txt"
    count_chart_path = output_dir / "sentiment_count.png"
    ratio_chart_path = output_dir / "sentiment_ratio.png"

    try:
        result.to_csv(classified_path, index=False, encoding="utf-8-sig")
        save_summary_file(
            summary,
            keywords,
            summary_path,
            classification_method=method,
        )
        if not no_chart:
            save_sentiment_count_chart(summary, count_chart_path)
            save_sentiment_ratio_chart(summary, ratio_chart_path)
    except OSError as error:
        raise ValueError(f"无法写入输出目录：{output_dir}") from error

    return output_dir


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        output_dir = run(args)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    print(f"分析完成，结果已保存到：{output_dir}")


if __name__ == "__main__":
    main()
