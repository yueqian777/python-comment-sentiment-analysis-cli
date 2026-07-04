# Comment Sentiment CLI Design

## Goal

Build a small command-line Python program for the course topic "网络评论情感分析系统". It reads review text, cleans and tokenizes it, classifies each comment as positive, negative, or neutral, counts keywords, and writes readable result files.

## Scope

- Input is a CSV file with a comment column.
- Output is a classified CSV, a text summary, and a sentiment ratio chart.
- The default classifier is a simple word-score model so the logic is easy to understand.
- A small sklearn training example is included as an optional improvement path.
- The structure stays flat and readable, close to how a normal course project is usually written.

## Architecture

- `src/sentiment_cli/analyzer.py` contains cleaning, tokenizing, scoring, keyword counting, and report helpers.
- `src/sentiment_cli/ml_model.py` contains the optional sklearn classifier example.
- `src/sentiment_cli/main.py` contains the command-line interface.
- `tests/` covers the main behavior.
- `学习说明.md` explains the program logic and study points.

## Testing

Use pytest for unit tests and run the CLI once against `data/sample_comments.csv` to verify end-to-end behavior.
