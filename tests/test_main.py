from argparse import Namespace

from sentiment_cli.main import run


def make_args(input_path, output_dir, no_chart=False):
    return Namespace(
        input=str(input_path),
        column="comment",
        output=str(output_dir),
        top_n=5,
        no_chart=no_chart,
    )


def test_run_generates_both_sentiment_charts(tmp_path):
    input_path = tmp_path / "comments.csv"
    input_path.write_text("comment\n很好很满意\n很差很失望\n今天上午收到\n", encoding="utf-8-sig")
    output_dir = tmp_path / "outputs"

    run(make_args(input_path, output_dir))

    assert (output_dir / "classified_comments.csv").exists()
    assert (output_dir / "summary.txt").exists()
    assert (output_dir / "sentiment_count.png").exists()
    assert (output_dir / "sentiment_ratio.png").exists()


def test_run_no_chart_skips_both_sentiment_charts(tmp_path):
    input_path = tmp_path / "comments.csv"
    input_path.write_text("comment\n很好很满意\n很差很失望\n", encoding="utf-8-sig")
    output_dir = tmp_path / "outputs"

    run(make_args(input_path, output_dir, no_chart=True))

    assert not (output_dir / "sentiment_count.png").exists()
    assert not (output_dir / "sentiment_ratio.png").exists()
