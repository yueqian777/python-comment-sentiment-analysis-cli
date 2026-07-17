from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
PAGES = (
    "单条评论分析",
    "CSV 批量分析",
    "模型与评估",
    "系统说明",
)


def _load_app(page: str = "单条评论分析") -> AppTest:
    app = AppTest.from_file(APP_PATH, default_timeout=20).run()
    assert not app.exception
    if page != "单条评论分析":
        app.sidebar.radio[0].set_value(page).run(timeout=20)
    assert not app.exception
    return app


def test_app_loads_with_title_and_sidebar_navigation():
    app = _load_app()

    assert "单条评论分析" in [item.value for item in app.title]
    assert app.sidebar.radio[0].label == "功能导航"
    assert app.sidebar.radio[0].options == list(PAGES)


@pytest.mark.parametrize("page", PAGES)
def test_each_navigation_page_renders(page):
    app = _load_app(page)

    assert page in [item.value for item in app.title]


def test_single_lexicon_analysis_runs_through_streamlit():
    app = _load_app()
    app.text_area[0].input("外观不错，但是续航太差").run(timeout=20)
    app.button[0].click().run(timeout=20)

    assert not app.exception
    metric_values = [item.value for item in app.metric]
    assert "中性 (neutral)" in metric_values


def test_single_method_selection_survives_page_navigation():
    app = _load_app()
    app.text_area[0].input("服务很好").run(timeout=20)
    method = next(item for item in app.radio if item.label == "分类方法")
    method.set_value("机器学习法").run(timeout=20)

    app.sidebar.radio[0].set_value("系统说明").run(timeout=20)
    app.sidebar.radio[0].set_value("单条评论分析").run(timeout=20)

    method = next(item for item in app.radio if item.label == "分类方法")
    assert method.value == "机器学习法"
    assert app.text_area[0].value == "服务很好"


def test_model_page_renders_when_default_model_is_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("SENTIMENT_MODEL_PATH", str(tmp_path / "missing.joblib"))

    app = _load_app("模型与评估")

    assert not app.exception
    assert "不存在" in [item.value for item in app.metric]


def test_batch_page_renders_without_upload():
    app = _load_app("CSV 批量分析")

    assert not app.exception
    assert any("上传 CSV" in item.value for item in app.info)


def test_single_ml_mode_shows_missing_model_message(tmp_path, monkeypatch):
    monkeypatch.setenv("SENTIMENT_MODEL_PATH", str(tmp_path / "missing.joblib"))
    app = _load_app()
    method = next(item for item in app.radio if item.label == "分类方法")
    method.set_value("机器学习法")
    app.text_area[0].input("服务很好").run(timeout=20)
    app.button[0].click().run(timeout=20)

    assert not app.exception
    assert any("模型不存在" in item.value for item in app.error)


def test_batch_lexicon_analysis_renders_metrics_and_downloads():
    app = _load_app("CSV 批量分析")
    csv_content = "comment\n服务很好\n质量很差\n今天收到蓝色包装\n".encode("utf-8-sig")
    app.file_uploader[0].upload("comments.csv", csv_content, "text/csv").run(timeout=20)
    app.button[0].click().run(timeout=30)

    assert not app.exception
    metric_values = [item.value for item in app.metric]
    assert "3" in metric_values
    assert len(app.download_button) == 3


def test_batch_result_survives_page_navigation():
    app = _load_app("CSV 批量分析")
    csv_content = "comment\n服务很好\n质量很差\n今天收到蓝色包装\n".encode("utf-8-sig")
    app.file_uploader[0].upload("comments.csv", csv_content, "text/csv").run(timeout=20)
    app.button[0].click().run(timeout=30)

    app.sidebar.radio[0].set_value("系统说明").run(timeout=20)
    app.sidebar.radio[0].set_value("CSV 批量分析").run(timeout=20)

    assert not app.exception
    assert any("最近一次批量分析结果" in item.value for item in app.info)
    assert "3" in [item.value for item in app.metric]
    assert len(app.download_button) == 3
