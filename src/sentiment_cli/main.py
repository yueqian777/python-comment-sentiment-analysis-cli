from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sentiment_cli.analyzer import (
    analyze_comments,
    save_sentiment_chart,
    save_summary_file,
    sentiment_summary,
    top_keywords,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="网络评论情感分析系统")
    parser.add_argument("-i", "--input", required=True, help="输入 CSV 文件路径")
    parser.add_argument("-c", "--column", default="comment", help="评论文本所在列名")
    parser.add_argument("-o", "--output", default="outputs", help="输出目录")
    parser.add_argument("--top-n", type=int, default=10, help="输出的关键词数量")
    parser.add_argument("--no-chart", action="store_true", help="不生成情感统计图")
    return parser


def run(args: argparse.Namespace) -> Path:
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(input_path, encoding="utf-8-sig")
    if args.column not in data.columns:
        raise ValueError(f"CSV 文件中找不到列：{args.column}")

    result = analyze_comments(data[args.column])
    summary = sentiment_summary(result["sentiment"])
    keywords = top_keywords(data[args.column], limit=args.top_n)

    classified_path = output_dir / "classified_comments.csv"
    summary_path = output_dir / "summary.txt"
    chart_path = output_dir / "sentiment_ratio.png"

    result.to_csv(classified_path, index=False, encoding="utf-8-sig")
    save_summary_file(summary, keywords, summary_path)

    if not args.no_chart:
        save_sentiment_chart(summary, chart_path)

    return output_dir


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    output_dir = run(args)
    print(f"分析完成，结果已保存到：{output_dir}")


if __name__ == "__main__":
    main()
