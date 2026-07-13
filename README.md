# 网络评论情感分析系统

这是一个命令行版 Python 课程作业项目，用来对商品评论、电影评论、外卖评价等文本进行基础情感分析。程序会完成文本清洗、`jieba` 分词、关键词统计，并把评论分为 `positive`、`negative`、`neutral` 三类。

默认分析命令使用容易解释的词典法；项目另外提供 TF-IDF + LogisticRegression 的机器学习评估命令，用同一个测试集比较两种方法。

## 项目结构

```text
src/sentiment_cli/
  analyzer.py      # 清洗、分词、词典法分类、关键词和图表
  main.py          # 基础分析命令行入口
  ml_model.py      # sklearn 模型训练
  evaluate.py      # 词典法与机器学习法评估
data/
  sample_comments.csv
  labeled_comments.csv
tests/
README.md
学习说明.md
```

## 安装依赖

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

如果直连 PyPI 较慢，可以使用清华源：

```powershell
.\.venv\Scripts\python.exe -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

## 基础情感分析

```powershell
python -m sentiment_cli.main --input data/sample_comments.csv --column comment --output outputs
```

运行后生成：

- `outputs/classified_comments.csv`：每条评论的原文、清洗文本、分词结果和情感分类。
- `outputs/summary.txt`：三类评论的数量、占比和高频关键词。
- `outputs/sentiment_count.png`：positive / negative / neutral 数量柱状图。
- `outputs/sentiment_ratio.png`：直接标出百分比的情感占比柱状图。

如果不需要图表，可以增加 `--no-chart`，此时两个 PNG 都不会生成：

```powershell
python -m sentiment_cli.main --input data/sample_comments.csv --column comment --output outputs --no-chart
```

## 机器学习评估

```powershell
python -m sentiment_cli.evaluate --input data/labeled_comments.csv --column comment --label label --output outputs
```

评估命令会自动划分训练集和测试集，在同一个测试集上比较词典法与 TF-IDF + LogisticRegression 模型，并生成：

- `outputs/evaluation_report.txt`：包含 `lexicon_accuracy`、`ml_accuracy`、两种方法的 `classification_report`、`confusion_matrix`，以及是否成功使用分层抽样。
- `outputs/confusion_matrix.png`：并排展示词典法和机器学习法的混淆矩阵。

默认参数是 `test_size=0.3`、`random_state=42`。程序优先使用 `stratify=labels`；如果某个类别样本过少导致分层划分失败，会退回普通随机划分，并在报告中说明。

## 两种分类方法的区别

- 词典法：根据正面词、负面词和简单否定规则打分。优点是逻辑清楚、容易答辩；缺点是词典覆盖有限，难以处理反讽和复杂上下文。
- 机器学习法：从带标签评论中学习 TF-IDF 文本特征，再用逻辑回归分类。它有机会识别词典中没有的表达，但效果依赖标注数据的数量、质量和领域一致性。

`data/labeled_comments.csv` 只是便于演示评估流程的小型样例，不能代表真实业务数据。本项目仍属于课程作业级别，不应根据这份小样本夸大模型准确率。

## 运行测试

```powershell
python -m pytest -q
```
