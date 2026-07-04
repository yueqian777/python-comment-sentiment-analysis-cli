from __future__ import annotations

import re
import logging
import os
from contextlib import contextmanager
from collections import Counter
from pathlib import Path

import jieba
import pandas as pd


POSITIVE_WORDS = {
    "好",
    "很好",
    "不错",
    "满意",
    "喜欢",
    "推荐",
    "值得",
    "舒服",
    "方便",
    "快",
    "好吃",
    "漂亮",
    "优秀",
    "清晰",
    "实惠",
}

NEGATIVE_WORDS = {
    "差",
    "很差",
    "失望",
    "难吃",
    "慢",
    "糟糕",
    "不好",
    "垃圾",
    "生气",
    "退货",
    "难用",
    "贵",
    "敷衍",
    "不会再买",
}

STOPWORDS = {
    "的",
    "了",
    "是",
    "我",
    "也",
    "很",
    "和",
    "就",
    "都",
    "在",
    "有",
    "还",
    "这",
    "那",
    "一个",
    "今天",
    "收到",
}

SENTIMENTS = ("positive", "negative", "neutral")

jieba.setLogLevel(logging.WARNING)


@contextmanager
def silence_native_output():
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        os.close(stdout_fd)
        os.close(stderr_fd)


def clean_text(text: object) -> str:
    """Clean noisy review text while keeping Chinese, letters and numbers."""
    if text is None or pd.isna(text):
        return ""

    value = str(text)
    value = re.sub(r"https?://\S+|www\.\S+", " ", value)
    value = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def tokenize(text: object) -> list[str]:
    cleaned = clean_text(text)
    words = []

    for word in jieba.lcut(cleaned):
        word = word.strip()
        if len(word) < 2:
            continue
        if word in STOPWORDS:
            continue
        words.append(word)

    return words


def classify_text(text: object) -> str:
    cleaned = clean_text(text)
    if not cleaned:
        return "neutral"

    positive_score = sum(1 for word in POSITIVE_WORDS if word in cleaned)
    negative_score = sum(1 for word in NEGATIVE_WORDS if word in cleaned)

    if positive_score > negative_score:
        return "positive"
    if negative_score > positive_score:
        return "negative"
    return "neutral"


def analyze_comments(comments: list[str] | pd.Series) -> pd.DataFrame:
    rows = []

    for comment in comments:
        cleaned = clean_text(comment)
        words = tokenize(cleaned)
        rows.append(
            {
                "comment": "" if comment is None or pd.isna(comment) else str(comment),
                "cleaned_text": cleaned,
                "tokens": " ".join(words),
                "sentiment": classify_text(cleaned),
            }
        )

    return pd.DataFrame(rows)


def top_keywords(comments: list[str] | pd.Series, limit: int = 10) -> list[tuple[str, int]]:
    counter: Counter[str] = Counter()
    for comment in comments:
        counter.update(tokenize(comment))

    return counter.most_common(limit)


def sentiment_summary(sentiments: list[str] | pd.Series) -> dict[str, dict[str, float]]:
    labels = [item for item in sentiments if item in SENTIMENTS]
    total = len(labels)
    counter = Counter(labels)
    summary: dict[str, dict[str, float]] = {}

    for sentiment in SENTIMENTS:
        count = counter.get(sentiment, 0)
        ratio = round(count / total * 100, 2) if total else 0.0
        summary[sentiment] = {"count": count, "ratio": ratio}

    return summary


def save_summary_file(
    summary: dict[str, dict[str, float]],
    keywords: list[tuple[str, int]],
    output_path: str | Path,
) -> None:
    lines = ["情感分类统计："]
    for sentiment in SENTIMENTS:
        item = summary[sentiment]
        lines.append(f"- {sentiment}: {item['count']} 条，占比 {item['ratio']}%")

    lines.append("")
    lines.append("高频关键词：")
    for word, count in keywords:
        lines.append(f"- {word}: {count}")

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def save_sentiment_chart(summary: dict[str, dict[str, float]], output_path: str | Path) -> None:
    with silence_native_output():
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        labels = ["Positive", "Negative", "Neutral"]
        values = [
            summary["positive"]["count"],
            summary["negative"]["count"],
            summary["neutral"]["count"],
        ]

        plt.figure(figsize=(6, 4))
        plt.bar(labels, values, color=["#3BA272", "#D94E5D", "#5470C6"])
        plt.title("Comment Sentiment Count")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
