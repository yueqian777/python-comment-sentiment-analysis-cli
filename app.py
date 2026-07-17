from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from sentiment_cli.analyzer import SENTIMENTS, clean_text
from sentiment_cli.evaluate import evaluate_csv
from sentiment_cli.inference import (
    analyze_comments_by_method,
    load_sentiment_model,
)
from sentiment_cli.ml_model import CHAR_NGRAM_RANGE, LOGISTIC_C, WORD_NGRAM_RANGE
from sentiment_cli.train import train_csv
from sentiment_cli.ui_helpers import (
    SENTIMENT_COLORS,
    SENTIMENT_LABELS,
    build_summary_text,
    confidence_distribution_table,
    dataframe_to_csv_bytes,
    filter_results,
    keyword_count_table,
    low_confidence_rows,
    sentiment_count_table,
    sentiment_ratio_table,
)


ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(
    os.environ.get(
        "SENTIMENT_MODEL_PATH",
        ROOT_DIR / "models" / "sentiment_model.joblib",
    )
)
MODEL_INFO_PATH = MODEL_PATH.with_name("model_info.json")
TRAINING_DATA_PATH = ROOT_DIR / "data" / "labeled_comments.csv"
FIXED_HOLDOUT_PATH = ROOT_DIR / "data" / "independent_test_comments.csv"
EXAMPLE_EVALUATION_DIR = ROOT_DIR / "docs" / "example_outputs"
STREAMLIT_EVALUATION_DIR = ROOT_DIR / "outputs" / "streamlit_evaluation"

METHOD_LABELS = {"词典法": "lexicon", "机器学习法": "ml"}
NAVIGATION_ITEMS = ("单条评论分析", "CSV 批量分析", "模型与评估", "系统说明")
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "macro_precision": "Macro Precision",
    "macro_recall": "Macro Recall",
    "macro_f1": "Macro F1",
}


st.set_page_config(page_title="网络评论情感分析系统", layout="wide")
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


@st.cache_resource(show_spinner=False)
def load_cached_model(model_path: str, file_modified_time: float):
    del file_modified_time
    return load_sentiment_model(model_path)


def _initialize_state() -> None:
    defaults = {
        "single_text": "",
        "single_result": None,
        "single_method_label": "词典法",
        "batch_result": None,
        "batch_method": None,
        "batch_source_name": None,
        "batch_source_key": None,
        "batch_comment_column": None,
        "batch_top_n": 10,
        "evaluation_dir": None,
        "recent_training_status": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _apply_page_style() -> None:
    st.markdown(
        """
        <style>
        .block-container {max-width: 1450px; padding-top: 1.4rem; padding-bottom: 2rem;}
        h1 {margin-bottom: 1.1rem;}
        [data-testid="stMetric"] {padding: 0.25rem 0;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _format_percentage(value: object) -> str:
    if value is None or pd.isna(value):
        return "无"
    return f"{float(value) * 100:.2f}%"


def _display_text(value: object) -> str:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return "无"
    return str(value)


def _metadata_value(value: object) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _show_dataframe(data: pd.DataFrame) -> None:
    width = (
        {"width": "stretch"}
        if "width" in inspect.signature(st.dataframe).parameters
        else {"use_container_width": True}
    )
    st.dataframe(data, hide_index=True, **width)


def _show_plot(figure) -> None:
    width = (
        {"width": "stretch"}
        if "width" in inspect.signature(st.pyplot).parameters
        else {"use_container_width": True}
    )
    st.pyplot(figure, **width)


def _read_json_object(path: Path) -> tuple[dict[str, object] | None, str | None]:
    if not path.exists():
        return None, f"找不到文件：{path}"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return None, f"JSON 文件无法读取：{path.name}（{error}）"
    if not isinstance(value, dict):
        return None, f"JSON 文件格式错误：{path.name} 的顶层必须是对象"
    return value, None


def _load_default_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"模型不存在：{MODEL_PATH}")
    return load_cached_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)


def _show_model_missing_message() -> None:
    st.error("默认机器学习模型不存在。")
    st.code(str(MODEL_PATH), language=None)
    st.info("请前往“模型与评估”页面训练默认模型。")


def _clear_single_result() -> None:
    st.session_state.single_text = ""
    st.session_state._single_text = ""
    st.session_state.single_result = None


def _remember_single_text() -> None:
    st.session_state.single_text = st.session_state._single_text


def _remember_single_method() -> None:
    st.session_state.single_method_label = st.session_state._single_method_label


def _render_probability_chart(row: pd.Series) -> None:
    labels = [SENTIMENT_LABELS[label] for label in SENTIMENTS]
    raw_values = [row.get(f"{label}_probability") for label in SENTIMENTS]
    if any(value is None or pd.isna(value) for value in raw_values):
        st.info("当前模型没有提供完整的三类概率，无法绘制概率图。")
        return
    values = [float(value) for value in raw_values]
    colors = [SENTIMENT_COLORS[label] for label in SENTIMENTS]
    figure, axis = plt.subplots(figsize=(7.5, 2.6))
    axis.barh(labels, values, color=colors)
    axis.set_xlim(0, 1)
    axis.set_xlabel("概率")
    axis.set_title("三类情感概率")
    for index, value in enumerate(values):
        axis.text(min(value + 0.015, 0.95), index, f"{value:.2%}", va="center")
    figure.tight_layout()
    _show_plot(figure)
    plt.close(figure)


def _render_single_result(result: pd.DataFrame) -> None:
    if result.empty:
        return
    row = result.iloc[0]
    sentiment = str(row["sentiment"])
    method = str(row["classification_method"])
    first, second, third = st.columns(3)
    first.metric("情感类别", f"{SENTIMENT_LABELS[sentiment]} ({sentiment})")

    if method == "lexicon":
        second.metric("正面得分", int(row["positive_score"]))
        third.metric("负面得分", int(row["negative_score"]))
    else:
        second.metric("分类方法", "机器学习法")
        third.metric("置信度", _format_percentage(row.get("confidence")))

    st.markdown("#### 评论内容")
    details = pd.DataFrame(
        {
            "字段": ["原始评论", "清洗后文本", "分词结果", "分类方法"],
            "内容": [
                _display_text(row.get("comment")),
                _display_text(row.get("cleaned_text")),
                _display_text(row.get("tokens")),
                "词典法" if method == "lexicon" else "机器学习法",
            ],
        }
    )
    _show_dataframe(details)

    if method == "lexicon":
        st.markdown("#### 词典匹配")
        columns = st.columns(3)
        columns[0].write(f"**正面命中：** {_display_text(row.get('positive_hits'))}")
        columns[1].write(f"**负面命中：** {_display_text(row.get('negative_hits'))}")
        columns[2].write(f"**否定反转：** {_display_text(row.get('negated_hits'))}")
    else:
        probability_columns = st.columns(3)
        for column, sentiment_key in zip(probability_columns, SENTIMENTS):
            column.metric(
                f"{SENTIMENT_LABELS[sentiment_key]}概率",
                _format_percentage(row.get(f"{sentiment_key}_probability")),
            )
        _render_probability_chart(row)
        st.caption("模型置信度不等于预测一定正确。")


def render_single_page() -> None:
    st.title("单条评论分析")
    st.session_state._single_text = st.session_state.single_text
    comment = st.text_area(
        "评论内容",
        key="_single_text",
        on_change=_remember_single_text,
        height=130,
        placeholder="请输入需要分析的中文评论",
    )
    st.session_state._single_method_label = st.session_state.single_method_label
    method_label = st.radio(
        "分类方法",
        options=list(METHOD_LABELS),
        horizontal=True,
        key="_single_method_label",
        on_change=_remember_single_method,
    )
    analyze_column, clear_column, _ = st.columns([1, 1, 6])
    analyze_clicked = analyze_column.button("开始分析", type="primary")
    clear_column.button("清空", on_click=_clear_single_result)

    if analyze_clicked:
        st.session_state.single_text = comment
        if not clean_text(comment):
            st.warning("评论不能为空。")
        else:
            method = METHOD_LABELS[method_label]
            model = None
            if method == "ml":
                try:
                    model = _load_default_model()
                except FileNotFoundError:
                    _show_model_missing_message()
                    return
                except ValueError as error:
                    st.error(str(error))
                    return
            try:
                st.session_state.single_result = analyze_comments_by_method(
                    [comment], method=method, model=model
                )
            except (ValueError, FileNotFoundError) as error:
                st.error(str(error))

    result = st.session_state.single_result
    if isinstance(result, pd.DataFrame):
        st.divider()
        _render_single_result(result)


def _read_uploaded_csv(uploaded_file) -> pd.DataFrame | None:
    try:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, encoding="utf-8-sig")
    except pd.errors.EmptyDataError:
        st.error("CSV 为空，没有表头或数据。")
    except (pd.errors.ParserError, UnicodeError, OSError) as error:
        st.error(f"CSV 无法解析：{error}")
    return None


def _read_uploaded_stopwords(uploaded_file) -> set[str] | None:
    if uploaded_file is None:
        return None
    try:
        uploaded_file.seek(0)
        text = uploaded_file.read().decode("utf-8-sig")
    except (UnicodeError, OSError) as error:
        st.error(f"停用词文件无法读取：{error}")
        return None
    return {
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def _plot_count_chart(counts: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(6.5, 3.8))
    colors = [SENTIMENT_COLORS[item] for item in counts["sentiment"]]
    bars = axis.bar(counts["label"], counts["count"], color=colors)
    axis.set_title("情感类别数量")
    axis.set_ylabel("评论数量")
    axis.bar_label(bars, padding=3)
    figure.tight_layout()
    _show_plot(figure)
    plt.close(figure)


def _plot_ratio_chart(ratios: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(6.5, 3.8))
    colors = [SENTIMENT_COLORS[item] for item in ratios["sentiment"]]
    values = ratios["ratio"].tolist()
    if sum(values) == 0:
        axis.text(0.5, 0.5, "暂无可统计数据", ha="center", va="center")
        axis.axis("off")
    else:
        axis.pie(
            values,
            labels=ratios["label"],
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops={"width": 0.45, "edgecolor": "white"},
        )
        axis.set_title("情感类别占比")
    figure.tight_layout()
    _show_plot(figure)
    plt.close(figure)


def _plot_keyword_chart(keywords: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(8, max(3.2, len(keywords) * 0.36)))
    ordered = keywords.sort_values("count", ascending=True)
    axis.barh(ordered["keyword"], ordered["count"], color="#2878B5")
    axis.set_title("高频关键词")
    axis.set_xlabel("出现次数")
    figure.tight_layout()
    _show_plot(figure)
    plt.close(figure)


def _plot_confidence_distribution(distribution: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(8, 3.4))
    axis.bar(distribution["range"], distribution["count"], color="#6A5ACD")
    axis.set_title("模型置信度分布")
    axis.set_xlabel("置信度区间")
    axis.set_ylabel("评论数量")
    axis.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    _show_plot(figure)
    plt.close(figure)


def _display_batch_overview(result: pd.DataFrame, top_n: int) -> pd.DataFrame:
    counts = sentiment_count_table(result)
    ratios = sentiment_ratio_table(result)
    ratio_by_sentiment = ratios.set_index("sentiment")["ratio"].to_dict()
    count_by_sentiment = counts.set_index("sentiment")["count"].to_dict()

    columns = st.columns(4)
    columns[0].metric("评论总数", len(result))
    for column, sentiment in zip(columns[1:], SENTIMENTS):
        column.metric(
            SENTIMENT_LABELS[sentiment],
            f"{int(count_by_sentiment.get(sentiment, 0))} 条",
            f"{float(ratio_by_sentiment.get(sentiment, 0)):.2%}",
            delta_color="off",
        )

    left, right = st.columns(2)
    with left:
        _plot_count_chart(counts)
    with right:
        _plot_ratio_chart(ratios)

    keywords = keyword_count_table(result, top_n=top_n)
    st.markdown(f"#### Top {top_n} 高频关键词")
    if keywords.empty:
        st.info("当前结果没有可统计的关键词。")
    else:
        _plot_keyword_chart(keywords)
    return keywords


def _display_batch_ml_overview(result: pd.DataFrame, threshold: float) -> None:
    left, right = st.columns([3, 1])
    distribution = confidence_distribution_table(result)
    with left:
        if distribution.empty:
            st.info("当前结果没有可用的模型置信度。")
        else:
            _plot_confidence_distribution(distribution)
    with right:
        low_count = len(low_confidence_rows(result, threshold=threshold))
        st.metric("低置信度评论", low_count)
        st.caption(f"当前阈值：{threshold:.0%}")


def _display_batch_table(result: pd.DataFrame, method: str) -> None:
    display = result.copy()
    display["sentiment"] = display["sentiment"].map(SENTIMENT_LABELS)
    if method == "lexicon":
        columns = [
            "comment",
            "cleaned_text",
            "sentiment",
            "positive_score",
            "negative_score",
            "positive_hits",
            "negative_hits",
            "negated_hits",
        ]
        names = {
            "comment": "原始评论",
            "cleaned_text": "清洗文本",
            "sentiment": "情感类别",
            "positive_score": "正面得分",
            "negative_score": "负面得分",
            "positive_hits": "正面命中",
            "negative_hits": "负面命中",
            "negated_hits": "否定反转",
        }
    else:
        columns = [
            "comment",
            "cleaned_text",
            "sentiment",
            "confidence",
            "positive_probability",
            "negative_probability",
            "neutral_probability",
        ]
        names = {
            "comment": "原始评论",
            "cleaned_text": "清洗文本",
            "sentiment": "情感类别",
            "confidence": "置信度",
            "positive_probability": "正面概率",
            "negative_probability": "负面概率",
            "neutral_probability": "中性概率",
        }
    _show_dataframe(
        display.loc[:, [column for column in columns if column in display.columns]].rename(
            columns=names
        )
    )


def _batch_filters(result: pd.DataFrame, method: str) -> tuple[pd.DataFrame, float]:
    st.markdown("#### 评论筛选")
    first, second = st.columns(2)
    selected_chinese = first.multiselect(
        "情感类别",
        options=[SENTIMENT_LABELS[item] for item in SENTIMENTS],
        default=[SENTIMENT_LABELS[item] for item in SENTIMENTS],
    )
    keyword = second.text_input("评论关键词")
    selected_sentiments = [
        sentiment
        for sentiment in SENTIMENTS
        if SENTIMENT_LABELS[sentiment] in selected_chinese
    ]

    min_confidence = None
    max_confidence = None
    low_only = False
    low_threshold = 0.60
    if method == "ml":
        confidence_column, low_column = st.columns(2)
        min_confidence, max_confidence = confidence_column.slider(
            "置信度范围", 0.0, 1.0, (0.0, 1.0), 0.01
        )
        low_only = low_column.checkbox("只看低置信度评论")
        low_threshold = low_column.slider("低置信度阈值", 0.0, 1.0, 0.60, 0.05)

    sort_options = {"原始顺序": "original"}
    if method == "ml":
        sort_options.update(
            {
                "置信度升序": "confidence_asc",
                "置信度降序": "confidence_desc",
                "正面概率降序": "positive_probability_desc",
                "负面概率降序": "negative_probability_desc",
                "中性概率降序": "neutral_probability_desc",
            }
        )
    sort_label = st.selectbox("排序方式", options=list(sort_options))
    filtered = filter_results(
        result,
        sentiments=selected_sentiments,
        keyword=keyword,
        min_confidence=min_confidence,
        max_confidence=max_confidence,
        low_confidence_only=low_only,
        low_confidence_threshold=low_threshold,
        sort_by=sort_options[sort_label],
    )
    return filtered, low_threshold


def _display_downloads(
    result: pd.DataFrame,
    filtered: pd.DataFrame,
    method: str,
    top_n: int,
    low_threshold: float,
) -> None:
    st.markdown("#### 结果下载")
    columns = st.columns(3)
    columns[0].download_button(
        "下载完整分类结果",
        data=dataframe_to_csv_bytes(result),
        file_name="classified_comments.csv",
        mime="text/csv",
    )
    columns[1].download_button(
        "下载当前筛选结果",
        data=dataframe_to_csv_bytes(filtered),
        file_name="filtered_comments.csv",
        mime="text/csv",
    )
    columns[2].download_button(
        "下载分析摘要",
        data=build_summary_text(
            result,
            method=method,
            top_n=top_n,
            low_confidence_threshold=low_threshold,
        ),
        file_name="sentiment_summary.txt",
        mime="text/plain",
    )


def _render_batch_result(result: pd.DataFrame, method: str, top_n: int) -> None:
    st.divider()
    st.subheader("批量分析结果")
    st.caption(f"数据来源：{st.session_state.batch_source_name}")
    _display_batch_overview(result, top_n)
    if method == "ml":
        _display_batch_ml_overview(result, threshold=0.60)

    filtered, low_threshold = _batch_filters(result, method)
    if filtered.empty:
        st.warning("当前筛选条件下没有评论。")
    else:
        _display_batch_table(filtered, method)
    _display_downloads(result, filtered, method, top_n, low_threshold)


def render_batch_page() -> None:
    st.title("CSV 批量分析")
    uploaded = st.file_uploader("上传 CSV 文件", type=["csv"])
    if uploaded is None:
        result = st.session_state.batch_result
        method = st.session_state.batch_method
        if isinstance(result, pd.DataFrame) and method in {"lexicon", "ml"}:
            st.info("当前显示本次会话最近一次批量分析结果。上传新 CSV 可重新分析。")
            _render_batch_result(
                result,
                method,
                int(st.session_state.batch_top_n),
            )
        else:
            st.info("上传 CSV 后可选择评论列并开始分析。")
        return

    source_hash = hashlib.sha256(uploaded.getvalue()).hexdigest()
    source_key = f"{uploaded.name}:{source_hash}"
    if st.session_state.batch_source_key != source_key:
        st.session_state.batch_result = None
        st.session_state.batch_method = None
        st.session_state.batch_source_name = uploaded.name
        st.session_state.batch_source_key = source_key
        st.session_state.batch_comment_column = None
        st.session_state.batch_top_n = 10

    data = _read_uploaded_csv(uploaded)
    if data is None:
        return
    if data.empty:
        st.warning("CSV 没有可分析的数据行。")
        return

    st.markdown("#### 数据预览")
    _show_dataframe(data.head(10))
    comment_column = st.selectbox("评论列", options=list(data.columns))
    if (
        st.session_state.batch_comment_column is not None
        and st.session_state.batch_comment_column != comment_column
    ):
        st.session_state.batch_result = None
        st.session_state.batch_method = None
    method_label = st.radio(
        "分类方法",
        options=list(METHOD_LABELS),
        horizontal=True,
        key="batch_method_label",
    )
    top_n = st.slider("Top N 关键词", min_value=5, max_value=30, value=10)
    stopwords_file = st.file_uploader("可选停用词文件", type=["txt"])

    if st.button("开始批量分析", type="primary"):
        comments = data[comment_column].fillna("").astype(str)
        if comments.map(clean_text).eq("").all():
            st.warning("选定的评论列全部为空。")
        else:
            extra_stopwords = _read_uploaded_stopwords(stopwords_file)
            if stopwords_file is not None and extra_stopwords is None:
                return
            method = METHOD_LABELS[method_label]
            model = None
            if method == "ml":
                try:
                    model = _load_default_model()
                except FileNotFoundError:
                    _show_model_missing_message()
                    return
                except ValueError as error:
                    st.error(str(error))
                    return
            try:
                with st.spinner("正在分析评论..."):
                    st.session_state.batch_result = analyze_comments_by_method(
                        comments,
                        method=method,
                        model=model,
                        extra_stopwords=extra_stopwords,
                    )
                st.session_state.batch_method = method
                st.session_state.batch_source_name = uploaded.name
                st.session_state.batch_comment_column = comment_column
                st.session_state.batch_top_n = top_n
            except (ValueError, FileNotFoundError) as error:
                st.error(str(error))

    result = st.session_state.batch_result
    method = st.session_state.batch_method
    if not isinstance(result, pd.DataFrame) or method not in {"lexicon", "ml"}:
        return
    st.session_state.batch_top_n = top_n
    _render_batch_result(result, method, top_n)


def _model_info_rows(info: dict[str, object]) -> pd.DataFrame:
    features = info.get("features") if isinstance(info.get("features"), dict) else {}
    preprocessing = (
        info.get("preprocessing") if isinstance(info.get("preprocessing"), dict) else {}
    )
    label_counts = info.get("label_counts")
    category_count = len(label_counts) if isinstance(label_counts, dict) else "未知"
    values = [
        info.get("algorithm", "未知"),
        info.get("training_samples", "未知"),
        category_count,
        info.get("python_version", "未知"),
        info.get("sklearn_version", "未知"),
        features.get("word_ngram_range", "未知"),
        features.get("char_ngram_range", "未知"),
        features.get("logistic_regression_c", "未知"),
        preprocessing.get("location", "未知"),
    ]
    return pd.DataFrame(
        {
            "项目": [
                "算法名称",
                "训练样本数",
                "类别数量",
                "Python 版本",
                "sklearn 版本",
                "词级 n-gram",
                "字符级 n-gram",
                "LogisticRegression C",
                "预处理位置",
            ],
            "值": [_metadata_value(value) for value in values],
        }
    )


def _render_model_status() -> None:
    st.subheader("模型状态和训练")
    model_exists = MODEL_PATH.exists()
    info_exists = MODEL_INFO_PATH.exists()
    first, second, third = st.columns(3)
    first.metric("模型状态", "可用" if model_exists else "不存在")
    second.metric(
        "模型文件大小",
        f"{MODEL_PATH.stat().st_size / 1024:.1f} KB" if model_exists else "无",
    )
    third.metric("模型元数据", "可用" if info_exists else "缺少")
    st.code(str(MODEL_PATH), language=None)

    if info_exists:
        info, error = _read_json_object(MODEL_INFO_PATH)
        if error:
            st.warning(error)
        elif info is not None:
            _show_dataframe(_model_info_rows(info))
    elif model_exists:
        st.warning("模型存在，但缺少模型元数据。")

    status = st.session_state.recent_training_status
    if status:
        st.success(status)
        st.session_state.recent_training_status = None

    st.warning("训练会覆盖默认模型文件。")
    if st.button("使用默认标注数据训练模型"):
        if not TRAINING_DATA_PATH.exists():
            st.error(f"默认训练数据不存在：{TRAINING_DATA_PATH}")
            return
        try:
            with st.spinner("正在训练模型..."):
                train_csv(TRAINING_DATA_PATH, model_path=MODEL_PATH)
            load_cached_model.clear()
            st.session_state.recent_training_status = "模型训练完成。"
            st.rerun()
        except (FileNotFoundError, ValueError, OSError) as error:
            st.error(f"模型训练失败：{error}")


def _metric_values(block: dict[str, object], cross_validation: bool) -> dict[str, dict]:
    values: dict[str, dict] = {}
    for method in ("lexicon", "ml"):
        method_block = block.get(method)
        if not isinstance(method_block, dict):
            raise ValueError(f"评估结果缺少 {method} 数据")
        metrics = method_block.get("mean") if cross_validation else method_block
        if not isinstance(metrics, dict):
            raise ValueError(f"评估结果中的 {method} 指标格式错误")
        values[method] = metrics
    return values


def _plot_metric_comparison(block: dict[str, object], cross_validation: bool) -> None:
    values = _metric_values(block, cross_validation)
    metric_names = list(METRIC_LABELS)
    positions = list(range(len(metric_names)))
    width = 0.36
    figure, axis = plt.subplots(figsize=(8.5, 4))
    axis.bar(
        [position - width / 2 for position in positions],
        [float(values["lexicon"][name]) for name in metric_names],
        width,
        label="词典法",
        color="#2878B5",
    )
    axis.bar(
        [position + width / 2 for position in positions],
        [float(values["ml"][name]) for name in metric_names],
        width,
        label="机器学习法",
        color="#D95F59",
    )
    axis.set_xticks(positions, [METRIC_LABELS[name] for name in metric_names], rotation=12)
    axis.set_ylim(0, 1)
    axis.set_ylabel("指标值")
    axis.set_title("词典法与机器学习法指标对比")
    axis.legend()
    figure.tight_layout()
    _show_plot(figure)
    plt.close(figure)


def _plot_confusion_matrix(axis, matrix: list[list[int]], title: str) -> None:
    image = axis.imshow(matrix, cmap="Blues")
    labels = [SENTIMENT_LABELS[item] for item in SENTIMENTS]
    axis.set_title(title)
    axis.set_xlabel("预测类别")
    axis.set_ylabel("真实类别")
    axis.set_xticks(range(len(labels)), labels=labels)
    axis.set_yticks(range(len(labels)), labels=labels)
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            axis.text(column_index, row_index, str(value), ha="center", va="center")
    return image


def _render_confusion_matrices(block: dict[str, object]) -> None:
    matrices = []
    for method in ("lexicon", "ml"):
        method_block = block.get(method)
        matrix = method_block.get("confusion_matrix") if isinstance(method_block, dict) else None
        if not isinstance(matrix, list):
            st.warning("当前评估结果缺少可绘制的混淆矩阵。")
            return
        matrices.append(matrix)
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 4))
    _plot_confusion_matrix(axes[0], matrices[0], "词典法")
    _plot_confusion_matrix(axes[1], matrices[1], "机器学习法")
    figure.tight_layout()
    _show_plot(figure)
    plt.close(figure)
    st.caption("混淆矩阵：行表示真实类别，列表示预测类别。")


def _render_evaluation_block(
    title: str,
    block: dict[str, object],
    cross_validation: bool = False,
) -> None:
    st.markdown(f"#### {title}")
    values = _metric_values(block, cross_validation)
    columns = st.columns(4)
    columns[0].metric("词典法 Accuracy", f"{float(values['lexicon']['accuracy']):.4f}")
    columns[1].metric("词典法 Macro F1", f"{float(values['lexicon']['macro_f1']):.4f}")
    columns[2].metric("ML Accuracy", f"{float(values['ml']['accuracy']):.4f}")
    columns[3].metric("ML Macro F1", f"{float(values['ml']['macro_f1']):.4f}")
    if cross_validation:
        folds = block.get("folds", "未知")
        st.caption(f"交叉验证折数：{folds}；指标卡显示各折平均值。")
        standard_deviation = pd.DataFrame(
            {
                "指标": [METRIC_LABELS[name] for name in METRIC_LABELS],
                "词典法标准差": [
                    block["lexicon"]["std"][name] for name in METRIC_LABELS
                ],
                "机器学习法标准差": [
                    block["ml"]["std"][name] for name in METRIC_LABELS
                ],
            }
        )
        _show_dataframe(standard_deviation)
    _plot_metric_comparison(block, cross_validation)
    _render_confusion_matrices(block)


def _select_evaluation_directory() -> Path:
    session_directory = st.session_state.evaluation_dir
    if session_directory:
        candidate = Path(session_directory)
        if (candidate / "evaluation_results.json").exists():
            return candidate
    return EXAMPLE_EVALUATION_DIR


def _run_default_evaluation() -> None:
    if not TRAINING_DATA_PATH.exists():
        st.error(f"默认训练数据不存在：{TRAINING_DATA_PATH}")
        return
    if not FIXED_HOLDOUT_PATH.exists():
        st.error(f"默认固定留出数据不存在：{FIXED_HOLDOUT_PATH}")
        return
    try:
        with st.spinner("正在运行默认评估..."):
            evaluate_csv(
                TRAINING_DATA_PATH,
                output_dir=STREAMLIT_EVALUATION_DIR,
                cv_folds=5,
                test_input_path=FIXED_HOLDOUT_PATH,
            )
        st.session_state.evaluation_dir = str(STREAMLIT_EVALUATION_DIR)
        st.rerun()
    except (FileNotFoundError, ValueError, OSError) as error:
        st.error(f"评估运行失败：{error}")


def _render_evaluation_results() -> None:
    st.subheader("评估结果")
    st.info("固定留出集 v1 已用于版本比较，不是新的盲测集。")
    if st.button("重新运行默认评估"):
        _run_default_evaluation()

    directory = _select_evaluation_directory()
    results_path = directory / "evaluation_results.json"
    metadata_path = directory / "evaluation_metadata.json"
    results, error = _read_json_object(results_path)
    if error:
        st.warning(error)
        return
    if results is None:
        return

    st.caption(f"当前结果目录：{directory}")
    try:
        holdout = results.get("holdout")
        cross_validation = results.get("cross_validation")
        fixed_holdout = results.get("fixed_holdout_v1")
        if not isinstance(holdout, dict):
            raise ValueError("评估结果缺少单次划分数据")
        _render_evaluation_block("单次分层 train_test_split", holdout)
        if isinstance(cross_validation, dict):
            _render_evaluation_block(
                "StratifiedKFold 交叉验证", cross_validation, cross_validation=True
            )
        else:
            st.info("当前结果没有交叉验证数据。")
        if isinstance(fixed_holdout, dict):
            _render_evaluation_block("固定留出测试集 v1", fixed_holdout)
        else:
            st.info("当前结果没有固定留出测试集数据。")
    except (KeyError, TypeError, ValueError) as error:
        st.error(f"评估结果格式错误：{error}")
        return

    metadata, metadata_error = _read_json_object(metadata_path)
    with st.expander("复现信息"):
        if metadata_error:
            st.warning(metadata_error)
        elif metadata is not None:
            st.json(metadata)

    st.warning(
        "单次划分受随机性影响；交叉验证反映训练数据内部稳定性；"
        "固定留出集 v1 已被开发过程查看。三组结果不能只选择最高值宣传，"
        "且当前数据规模有限。"
    )


def render_model_page() -> None:
    st.title("模型与评估")
    model_tab, evaluation_tab = st.tabs(["模型状态和训练", "评估结果"])
    with model_tab:
        _render_model_status()
    with evaluation_tab:
        _render_evaluation_results()


def _system_parameter_rows() -> pd.DataFrame:
    model_info, _ = _read_json_object(MODEL_INFO_PATH)
    evaluation_metadata, _ = _read_json_object(
        _select_evaluation_directory() / "evaluation_metadata.json"
    )
    features = (
        model_info.get("features")
        if isinstance(model_info, dict) and isinstance(model_info.get("features"), dict)
        else {}
    )
    preprocessing = (
        model_info.get("preprocessing")
        if isinstance(model_info, dict)
        and isinstance(model_info.get("preprocessing"), dict)
        else {}
    )
    training_samples = (
        model_info.get("training_samples")
        if isinstance(model_info, dict)
        else None
    )
    if training_samples is None and isinstance(evaluation_metadata, dict):
        training_samples = evaluation_metadata.get("training_samples")
    label_counts = model_info.get("label_counts") if isinstance(model_info, dict) else None
    category_count = len(label_counts) if isinstance(label_counts, dict) else "未知"
    holdout_samples = (
        evaluation_metadata.get("test_samples")
        if isinstance(evaluation_metadata, dict)
        else "未知"
    )
    values = [
        features.get("word_ngram_range", list(WORD_NGRAM_RANGE)),
        features.get("char_ngram_range", list(CHAR_NGRAM_RANGE)),
        features.get("logistic_regression_c", LOGISTIC_C),
        training_samples if training_samples is not None else "未知",
        category_count,
        holdout_samples,
        preprocessing.get("location", "inside sklearn pipeline"),
    ]
    return pd.DataFrame(
        {
            "参数": [
                "词级 n-gram",
                "字符级 n-gram",
                "LogisticRegression C",
                "训练样本数量",
                "类别数量",
                "固定留出集数量",
                "模型预处理位置",
            ],
            "当前值": [_metadata_value(value) for value in values],
        }
    )


def render_system_page() -> None:
    st.title("系统说明")
    st.subheader("系统流程")
    st.markdown(
        "原始评论 → 文本清洗 → 分类方法选择 → 词典法或机器学习法 "
        "→ `positive` / `negative` / `neutral` → 统计、图表和结果导出"
    )
    left, right = st.columns(2)
    with left:
        st.markdown("#### 词典法流程")
        st.markdown(
            "文本清洗 → 最长情感词匹配 → 单字词保护 → 否定词反转 "
            "→ 正负计分 → 三分类"
        )
    with right:
        st.markdown("#### 机器学习流程")
        st.markdown(
            "原始文本 → `sklearn Pipeline` → `clean_text` → 词级 TF-IDF "
            "+ 字符级 TF-IDF → `FeatureUnion` → `LogisticRegression` → 三类概率"
        )

    st.subheader("当前项目参数")
    _show_dataframe(_system_parameter_rows())
    st.subheader("使用边界")
    st.markdown(
        "- confidence 是模型内部概率，不是正确率保证。\n"
        "- 词典法难以处理复杂语境和反讽。\n"
        "- 机器学习效果受数据规模和样本分布影响。\n"
        "- 系统目前主要处理中文短评论。\n"
        "- 固定留出集 v1 仅用于版本比较。"
    )


def main() -> None:
    _initialize_state()
    _apply_page_style()
    st.sidebar.title("网络评论情感分析系统")
    page = st.sidebar.radio("功能导航", NAVIGATION_ITEMS)
    if page == "单条评论分析":
        render_single_page()
    elif page == "CSV 批量分析":
        render_batch_page()
    elif page == "模型与评估":
        render_model_page()
    else:
        render_system_page()


if __name__ == "__main__":
    main()
