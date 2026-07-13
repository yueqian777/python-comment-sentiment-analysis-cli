from sentiment_cli.analyzer import (
    analyze_comments,
    classify_text,
    clean_text,
    save_sentiment_count_chart,
    save_sentiment_ratio_chart,
    sentiment_summary,
    tokenize,
    top_keywords,
)


def test_clean_text_removes_url_and_noisy_symbols():
    text = "  这个外卖！！！真的不错 https://example.com #推荐#  "

    assert clean_text(text) == "这个外卖 真的不错 推荐"


def test_tokenize_returns_words_without_stopwords():
    words = tokenize("手机 很好 手机 物流 很快")

    assert "手机" in words
    assert "物流" in words
    assert "很" not in words


def test_classify_text_supports_three_sentiments():
    assert classify_text("这个商品很好，很满意") == "positive"
    assert classify_text("这个商品很差，非常失望") == "negative"
    assert classify_text("包装是蓝色的，今天上午收到") == "neutral"


def test_classify_text_handles_negated_sentiment_words():
    assert classify_text("不是很好，不推荐") == "negative"
    assert classify_text("这个商品不满意") != "positive"
    assert classify_text("价格不贵，体验不错") != "negative"


def test_classify_text_does_not_carry_negation_across_punctuation():
    assert classify_text("价格不贵，好吃") == "positive"


def test_analyze_comments_returns_rows_with_sentiment():
    result = analyze_comments(["很好吃，下次还来", "一般般", "质量太差"])

    assert list(result["sentiment"]) == ["positive", "neutral", "negative"]
    assert "tokens" in result.columns
    assert len(result) == 3


def test_top_keywords_counts_common_words():
    keywords = top_keywords(["手机 很好 手机", "手机 物流 快"], limit=2)

    assert keywords[0] == ("手机", 3)
    assert len(keywords) == 2


def test_sentiment_summary_counts_and_ratios():
    summary = sentiment_summary(["positive", "positive", "negative", "neutral"])

    assert summary["positive"]["count"] == 2
    assert summary["positive"]["ratio"] == 50.0
    assert summary["negative"]["count"] == 1
    assert summary["neutral"]["ratio"] == 25.0


def test_save_sentiment_count_chart_creates_image(tmp_path):
    summary = sentiment_summary(["positive", "positive", "negative", "neutral"])
    output_path = tmp_path / "sentiment_count.png"

    save_sentiment_count_chart(summary, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_save_sentiment_ratio_chart_creates_image(tmp_path):
    summary = sentiment_summary(["positive", "positive", "negative", "neutral"])
    output_path = tmp_path / "sentiment_ratio.png"

    save_sentiment_ratio_chart(summary, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
