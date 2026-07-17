from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

import pandas as pd


SENTIMENT_ORDER = ("positive", "negative", "neutral")
SENTIMENT_LABELS = {
    "positive": "正面",
    "negative": "负面",
    "neutral": "中性",
}
SENTIMENT_COLORS = {
    "positive": "#2E7D32",
    "negative": "#C62828",
    "neutral": "#607D8B",
}

METHOD_LABELS = {
    "lexicon": "词典法",
    "ml": "机器学习法",
}

_CHINESE_SENTIMENTS = {label: sentiment for sentiment, label in SENTIMENT_LABELS.items()}
_SORT_RULES = {
    "original": None,
    "原始顺序": None,
    "confidence_asc": ("confidence", True),
    "置信度升序": ("confidence", True),
    "confidence_desc": ("confidence", False),
    "置信度降序": ("confidence", False),
    "positive_probability_desc": ("positive_probability", False),
    "正面概率降序": ("positive_probability", False),
    "negative_probability_desc": ("negative_probability", False),
    "负面概率降序": ("negative_probability", False),
    "neutral_probability_desc": ("neutral_probability", False),
    "中性概率降序": ("neutral_probability", False),
}


def _check_dataframe(result: pd.DataFrame) -> None:
    if not isinstance(result, pd.DataFrame):
        raise TypeError("result 必须是 pandas DataFrame")


def _require_columns(result: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in result.columns]
    if missing:
        raise ValueError(f"结果缺少必要列：{', '.join(missing)}")


def _validate_probability(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须在 0 到 1 之间")

    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} 必须在 0 到 1 之间") from error

    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(f"{name} 必须在 0 到 1 之间")
    return number


def _validate_top_n(top_n: int) -> int:
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n <= 0:
        raise ValueError("top_n 必须是正整数")
    return top_n


def _confidence_values(result: pd.DataFrame) -> pd.Series:
    _require_columns(result, ["confidence"])
    values = pd.to_numeric(result["confidence"], errors="coerce")
    invalid_text = result["confidence"].notna() & values.isna()
    if invalid_text.any():
        raise ValueError("confidence 列包含非数字值")

    valid = values.dropna()
    if ((valid < 0) | (valid > 1)).any():
        raise ValueError("confidence 列的值必须在 0 到 1 之间")
    return values


def sentiment_count_table(result: pd.DataFrame) -> pd.DataFrame:
    _check_dataframe(result)
    counts = {sentiment: 0 for sentiment in SENTIMENT_ORDER}

    if not result.empty:
        _require_columns(result, ["sentiment"])
        if result["sentiment"].isna().any():
            raise ValueError("sentiment 列包含空值")

        values = result["sentiment"].astype(str).str.strip()
        invalid = sorted(set(values) - set(SENTIMENT_ORDER))
        if invalid:
            raise ValueError(f"sentiment 列包含非法标签：{', '.join(invalid)}")
        counts.update(values.value_counts().to_dict())

    return pd.DataFrame(
        {
            "sentiment": list(SENTIMENT_ORDER),
            "label": [SENTIMENT_LABELS[item] for item in SENTIMENT_ORDER],
            "count": [int(counts[item]) for item in SENTIMENT_ORDER],
        }
    )


def sentiment_ratio_table(result: pd.DataFrame) -> pd.DataFrame:
    counts = sentiment_count_table(result)
    total = int(counts["count"].sum())
    ratios = counts["count"] / total if total else 0.0
    return pd.DataFrame(
        {
            "sentiment": counts["sentiment"],
            "label": counts["label"],
            "ratio": ratios,
        }
    )


def keyword_count_table(result: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    _check_dataframe(result)
    _validate_top_n(top_n)
    columns = ["keyword", "count"]
    if result.empty:
        return pd.DataFrame(columns=columns)

    _require_columns(result, ["tokens"])
    counter: Counter[str] = Counter()
    for value in result["tokens"]:
        if isinstance(value, str):
            words = value.split()
        elif isinstance(value, (list, tuple, set)):
            words = [str(item).strip() for item in value]
        elif pd.isna(value):
            continue
        else:
            words = str(value).split()
        counter.update(word for word in words if word)

    ranked = sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:top_n]
    return pd.DataFrame(ranked, columns=columns)


def confidence_distribution_table(
    result: pd.DataFrame,
    bins: int = 10,
) -> pd.DataFrame:
    _check_dataframe(result)
    if isinstance(bins, bool) or not isinstance(bins, int) or bins <= 0:
        raise ValueError("bins 必须是正整数")

    columns = ["range", "count"]
    if result.empty:
        return pd.DataFrame(columns=columns)

    values = _confidence_values(result).dropna()
    if values.empty:
        return pd.DataFrame(columns=columns)

    counts = [0] * bins
    for value in values:
        index = min(int(float(value) * bins), bins - 1)
        counts[index] += 1

    rows = []
    for index, count in enumerate(counts):
        lower = round(index / bins, 10)
        upper = round((index + 1) / bins, 10)
        label = f"{lower:g}-{upper:g}"
        rows.append(
            {
                "range": label,
                "count": count,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def low_confidence_rows(
    result: pd.DataFrame,
    threshold: float = 0.6,
) -> pd.DataFrame:
    _check_dataframe(result)
    threshold = _validate_probability(threshold, "threshold")
    if result.empty:
        return result.copy(deep=True)

    confidence = _confidence_values(result)
    return result.loc[confidence < threshold].copy(deep=True)


def filter_results(
    result: pd.DataFrame,
    sentiments: Sequence[str] | None = None,
    keyword: str = "",
    min_confidence: float | None = None,
    max_confidence: float | None = None,
    low_confidence_only: bool = False,
    low_confidence_threshold: float = 0.6,
    sort_by: str = "original",
) -> pd.DataFrame:
    _check_dataframe(result)
    if sort_by not in _SORT_RULES:
        raise ValueError(f"不支持的排序方式：{sort_by}")

    minimum = (
        _validate_probability(min_confidence, "min_confidence")
        if min_confidence is not None
        else None
    )
    maximum = (
        _validate_probability(max_confidence, "max_confidence")
        if max_confidence is not None
        else None
    )
    threshold = _validate_probability(low_confidence_threshold, "low_confidence_threshold")
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError("min_confidence 不能大于 max_confidence")

    filtered = result.copy(deep=True)
    if filtered.empty:
        return filtered

    if sentiments is not None:
        _require_columns(result, ["sentiment"])
        normalized = []
        for value in sentiments:
            label = str(value).strip()
            if label in SENTIMENT_LABELS:
                normalized.append(label)
            elif label in _CHINESE_SENTIMENTS:
                normalized.append(_CHINESE_SENTIMENTS[label])
            else:
                raise ValueError(f"不支持的情感类别：{value}")
        filtered = filtered.loc[filtered["sentiment"].isin(normalized)]

    search_text = "" if keyword is None else str(keyword).strip()
    if search_text:
        _require_columns(result, ["comment"])
        comments = filtered["comment"].astype("string").fillna("")
        filtered = filtered.loc[
            comments.str.contains(search_text, case=False, regex=False, na=False)
        ]

    uses_confidence = (
        minimum is not None or maximum is not None or low_confidence_only
    )
    if uses_confidence:
        _confidence_values(result)
        confidence = pd.to_numeric(filtered["confidence"], errors="coerce")
        if minimum is not None:
            filtered = filtered.loc[confidence >= minimum]
            confidence = confidence.loc[filtered.index]
        if maximum is not None:
            filtered = filtered.loc[confidence <= maximum]
            confidence = confidence.loc[filtered.index]
        if low_confidence_only:
            filtered = filtered.loc[confidence < threshold]

    sort_rule = _SORT_RULES[sort_by]
    if sort_rule is not None:
        column, ascending = sort_rule
        _require_columns(result, [column])
        filtered = filtered.sort_values(
            column,
            ascending=ascending,
            kind="mergesort",
            na_position="last",
        )

    return filtered.copy(deep=True)


def dataframe_to_csv_bytes(result: pd.DataFrame) -> bytes:
    _check_dataframe(result)
    csv_text = result.to_csv(index=False, lineterminator="\n")
    return csv_text.encode("utf-8-sig")


def build_summary_text(
    result: pd.DataFrame,
    method: str,
    top_n: int = 10,
    low_confidence_threshold: float = 0.6,
) -> str:
    _check_dataframe(result)
    _validate_top_n(top_n)
    method_key = str(method).strip()
    if method_key in METHOD_LABELS:
        method_label = METHOD_LABELS[method_key]
    elif method_key in METHOD_LABELS.values():
        method_label = method_key
        method_key = next(
            key for key, label in METHOD_LABELS.items() if label == method_key
        )
    else:
        raise ValueError(f"不支持的分类方法：{method}")

    counts = sentiment_count_table(result)
    ratios = sentiment_ratio_table(result).set_index("sentiment")["ratio"]
    lines = [
        "网络评论情感分析摘要",
        f"数据总量：{len(result)}",
        f"分类方法：{method_label}",
        "",
        "情感分类统计：",
    ]
    for row in counts.itertuples(index=False):
        ratio = float(ratios.loc[row.sentiment])
        lines.append(f"- {row.label}：{row.count} 条（{ratio * 100:.2f}%）")

    lines.extend(["", f"Top {top_n} 高频关键词："])
    if result.empty or "tokens" not in result.columns:
        lines.append("- 无")
    else:
        keywords = keyword_count_table(result, top_n=top_n)
        if keywords.empty:
            lines.append("- 无")
        else:
            for row in keywords.itertuples(index=False):
                lines.append(f"- {row.keyword}：{row.count}")

    if method_key == "ml":
        threshold = _validate_probability(
            low_confidence_threshold,
            "low_confidence_threshold",
        )
        lines.append("")
        if result.empty or "confidence" not in result.columns:
            lines.extend(
                [
                    "平均置信度：无可用数据",
                    f"低置信度评论数量（低于 {threshold:.2%}）：无可用数据",
                ]
            )
        else:
            confidence = _confidence_values(result).dropna()
            if confidence.empty:
                lines.extend(
                    [
                        "平均置信度：无可用数据",
                        f"低置信度评论数量（低于 {threshold:.2%}）：无可用数据",
                    ]
                )
            else:
                low_count = int((confidence < threshold).sum())
                lines.extend(
                    [
                        f"平均置信度：{confidence.mean():.2%}",
                        f"低置信度评论数量（低于 {threshold:.2%}）：{low_count}",
                    ]
                )

    lines.extend(
        [
            "",
            "说明：分析结果仅供参考，模型置信度不等于预测一定正确。",
        ]
    )
    return "\n".join(lines)
