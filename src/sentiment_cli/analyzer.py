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
    "棒",
    "流畅",
    "整洁",
    "周到",
    "耐心",
    "惊喜",
    "精彩",
    "充足",
    "稳定",
    "省心",
    "新鲜",
    "准时",
    "友好",
    "合理",
    "丰富",
    "牢固",
    "顺手",
    "干净",
    "安静",
    "热情",
    "细致",
    "清楚",
    "自然",
    "及时",
    "可靠",
    "合适",
    "到位",
    "扎实",
    "好评",
}

NEGATIVE_WORDS = {
    "差",
    "很差",
    "太差",
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
    "卡顿",
    "混乱",
    "破损",
    "异味",
    "延迟",
    "太冷",
    "凉了",
    "太少",
    "脏",
    "吵",
    "模糊",
    "过时",
    "死机",
    "闪退",
    "刺耳",
    "压耳",
    "油腻",
    "难懂",
    "粗糙",
    "松散",
    "生硬",
    "无聊",
    "漏水",
    "拖延",
    "失真",
    "发热",
    "断连",
    "太长",
    "比较一般",
    "差评",
    "昂贵",
    "缓慢",
    "吵闹",
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

NEGATION_WORDS = ("不", "没", "没有", "不是", "无", "并不")
NEGATION_GAP = 3
SINGLE_CHARACTER_PREFIXES = {"很", "太", "挺", "较", "更", "真", "稍", "偏"}

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


def load_stopwords(path: str | Path) -> set[str]:
    stopwords_path = Path(path)
    if not stopwords_path.exists():
        raise FileNotFoundError(f"找不到停用词文件：{stopwords_path}")

    try:
        lines = stopwords_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValueError(f"无法读取停用词文件：{stopwords_path}") from error

    return {
        line.strip()
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    }


def tokenize(text: object, extra_stopwords: set[str] | None = None) -> list[str]:
    cleaned = clean_text(text)
    words = []
    stopwords = STOPWORDS | (extra_stopwords or set())

    for word in jieba.lcut(cleaned):
        word = word.strip()
        if len(word) < 2:
            continue
        if word in stopwords:
            continue
        words.append(word)

    return words


def _single_character_spans(text: str, words: set[str]) -> set[tuple[int, int]]:
    spans: set[tuple[int, int]] = set()

    for token, start, end in jieba.tokenize(text):
        if token in words:
            spans.add((start, end))
            continue

        word = token[-1:]
        if word not in words:
            continue

        prefix = token[:-1]
        has_modifier = prefix in SINGLE_CHARACTER_PREFIXES
        has_negation = any(
            prefix.startswith(negation) and len(prefix) - len(negation) <= 2
            for negation in NEGATION_WORDS
        )
        if has_modifier or has_negation:
            spans.add((end - 1, end))

    return spans


def find_sentiment_matches(text: object) -> list[dict[str, object]]:
    cleaned = clean_text(text)
    occupied = [False] * len(cleaned)
    matches: list[dict[str, object]] = []
    words = [(word, "positive") for word in POSITIVE_WORDS]
    words.extend((word, "negative") for word in NEGATIVE_WORDS)
    single_words = {word for word, _ in words if len(word) == 1}
    valid_single_spans = _single_character_spans(cleaned, single_words)

    for word, sentiment in sorted(words, key=lambda item: (-len(item[0]), item[0])):
        for match in re.finditer(re.escape(word), cleaned):
            start, end = match.span()
            if len(word) == 1 and (start, end) not in valid_single_spans:
                continue
            if any(occupied[start:end]):
                continue

            occupied[start:end] = [True] * (end - start)
            matches.append(
                {
                    "word": word,
                    "sentiment": sentiment,
                    "start": start,
                    "end": end,
                }
            )

    return sorted(matches, key=lambda item: (int(item["start"]), -len(str(item["word"]))))


def _find_preceding_negation(text: str, word_start: int) -> tuple[str, int] | None:
    segment_start = text.rfind(" ", 0, word_start) + 1
    prefix = text[segment_start:word_start]
    candidates: list[tuple[int, str]] = []

    for negation in NEGATION_WORDS:
        position = prefix.rfind(negation)
        if position < 0:
            continue
        gap = len(prefix) - position - len(negation)
        if gap <= NEGATION_GAP:
            candidates.append((segment_start + position, negation))

    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], len(item[1])))


def score_sentiment_text(text: object) -> dict[str, object]:
    cleaned = clean_text(text)
    result: dict[str, object] = {
        "positive_score": 0,
        "negative_score": 0,
        "positive_hits": [],
        "negative_hits": [],
        "negated_hits": [],
    }

    for match in find_sentiment_matches(cleaned):
        word = str(match["word"])
        sentiment = str(match["sentiment"])
        start = int(match["start"])
        negation = _find_preceding_negation(cleaned, start)

        if negation is not None:
            negation_start, _ = negation
            hit_text = cleaned[negation_start : int(match["end"])]
            result["negated_hits"].append(hit_text)
            hit_label = f"{word}（否定反转）"
            sentiment = "negative" if sentiment == "positive" else "positive"
        else:
            hit_label = word

        score_key = f"{sentiment}_score"
        hits_key = f"{sentiment}_hits"
        result[score_key] += 1
        result[hits_key].append(hit_label)

    return result


def classify_text(text: object) -> str:
    result = score_sentiment_text(text)
    positive_score = int(result["positive_score"])
    negative_score = int(result["negative_score"])
    if positive_score == 0 and negative_score == 0:
        return "neutral"

    if positive_score > negative_score:
        return "positive"
    if negative_score > positive_score:
        return "negative"
    return "neutral"


def analyze_comments(
    comments: list[str] | pd.Series,
    extra_stopwords: set[str] | None = None,
) -> pd.DataFrame:
    rows = []

    for comment in comments:
        cleaned = clean_text(comment)
        words = tokenize(cleaned, extra_stopwords=extra_stopwords)
        score = score_sentiment_text(cleaned)
        rows.append(
            {
                "comment": "" if comment is None or pd.isna(comment) else str(comment),
                "cleaned_text": cleaned,
                "tokens": " ".join(words),
                "sentiment": classify_text(cleaned),
                "classification_method": "lexicon",
                "positive_score": score["positive_score"],
                "negative_score": score["negative_score"],
                "positive_hits": "|".join(score["positive_hits"]),
                "negative_hits": "|".join(score["negative_hits"]),
                "negated_hits": "|".join(score["negated_hits"]),
            }
        )

    return pd.DataFrame(rows)


def top_keywords(
    comments: list[str] | pd.Series,
    limit: int = 10,
    extra_stopwords: set[str] | None = None,
) -> list[tuple[str, int]]:
    counter: Counter[str] = Counter()
    for comment in comments:
        counter.update(tokenize(comment, extra_stopwords=extra_stopwords))

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
    classification_method: str = "lexicon",
) -> None:
    lines = [f"分类方法：{classification_method}", "", "情感分类统计："]
    for sentiment in SENTIMENTS:
        item = summary[sentiment]
        lines.append(f"- {sentiment}: {item['count']} 条，占比 {item['ratio']}%")

    lines.append("")
    lines.append("高频关键词：")
    for word, count in keywords:
        lines.append(f"- {word}: {count}")

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def save_sentiment_count_chart(
    summary: dict[str, dict[str, float]], output_path: str | Path
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
        bars = plt.bar(labels, values, color=["#3BA272", "#D94E5D", "#5470C6"])
        plt.title("Comment Sentiment Count")
        plt.ylabel("Count")
        plt.bar_label(bars, labels=[str(value) for value in values], padding=3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()


def save_sentiment_ratio_chart(
    summary: dict[str, dict[str, float]], output_path: str | Path
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with silence_native_output():
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        labels = ["Positive", "Negative", "Neutral"]
        values = [
            summary["positive"]["ratio"],
            summary["negative"]["ratio"],
            summary["neutral"]["ratio"],
        ]

        plt.figure(figsize=(6, 4))
        bars = plt.bar(labels, values, color=["#3BA272", "#D94E5D", "#5470C6"])
        plt.title("Comment Sentiment Ratio")
        plt.ylabel("Percentage (%)")
        plt.ylim(0, 100)
        plt.bar_label(bars, labels=[f"{value:.2f}%" for value in values], padding=3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()


def save_sentiment_chart(summary: dict[str, dict[str, float]], output_path: str | Path) -> None:
    """Keep the original chart helper as a count-chart compatibility alias."""
    save_sentiment_count_chart(summary, output_path)
