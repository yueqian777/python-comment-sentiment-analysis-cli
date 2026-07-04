# Comment Sentiment CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a simple command-line sentiment analysis project for network comments.

**Architecture:** Keep the program flat and readable. Put the main logic in `analyzer.py`, optional sklearn logic in `ml_model.py`, and command-line parsing in `main.py`.

**Tech Stack:** Python, jieba, pandas, sklearn, matplotlib, pytest.

---

### Task 1: Tests First

**Files:**
- Create: `tests/test_analyzer.py`

- [ ] Write tests for text cleaning, tokenization, sentiment classification, keyword counting, and summary calculation.
- [ ] Run `python -m pytest -q` and confirm the tests fail because the package is not implemented yet.

### Task 2: Core Analyzer

**Files:**
- Create: `src/sentiment_cli/analyzer.py`
- Create: `src/sentiment_cli/__init__.py`

- [ ] Implement simple, readable functions: `clean_text`, `tokenize`, `classify_text`, `analyze_comments`, `top_keywords`, and `sentiment_summary`.
- [ ] Run `python -m pytest -q` and confirm the tests pass.

### Task 3: Command Line Program

**Files:**
- Create: `src/sentiment_cli/main.py`
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `data/sample_comments.csv`

- [ ] Add an argparse CLI that accepts input CSV, text column, output directory, and chart switch.
- [ ] Use pandas to read and write CSV files.
- [ ] Generate `classified_comments.csv`, `summary.txt`, and `sentiment_ratio.png`.
- [ ] Run the CLI against sample data.

### Task 4: Optional sklearn Example

**Files:**
- Create: `src/sentiment_cli/ml_model.py`
- Create: `tests/test_ml_model.py`

- [ ] Add a small sklearn pipeline using TF-IDF plus logistic regression.
- [ ] Test that it can train on sample texts and predict labels.

### Task 5: Learning Document and GitHub

**Files:**
- Create: `学习说明.md`
- Create: `README.md`
- Create: `.gitignore`

- [ ] Explain program logic, used Python knowledge, NLP basics, pandas, jieba, sklearn, testing, and how to run the project.
- [ ] Run all tests and the CLI smoke test.
- [ ] Commit the project.
- [ ] Create a public GitHub repository and push the code.
