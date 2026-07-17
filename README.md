# 网络评论情感分析系统

这是一个命令行版 Python 课程项目，用于分析商品、外卖、酒店、电影、课程和数码产品评论。系统支持可解释的词典法，以及词级与字符级 TF-IDF + LogisticRegression 机器学习法，最终把评论分为 `positive`、`negative`、`neutral` 三类。

## 系统流程

```mermaid
flowchart LR
    A[CSV 评论] --> B[文本清洗]
    B --> C[jieba 分词与停用词过滤]
    C --> D{分类方法}
    D -->|lexicon| E[最长词匹配与否定处理]
    D -->|ml| F[加载词级与字符级 TF-IDF 模型]
    E --> G[分类结果与解释字段]
    F --> G
    G --> H[数量、占比和关键词汇总]
    H --> I[CSV、文本报告和图表]
```

## 主要功能

- 清洗网址、标点和多余空白。
- 使用 `jieba` 分词，支持内置和外部停用词。
- 词典法按最长词优先匹配，避免“很好”同时命中“很好”和“好”。
- 否定词只影响同一分句内、距离较近的情感词。
- 词典模式输出得分、命中词和否定反转说明。
- 机器学习模式输出预测概率和 `confidence`。
- 支持模型训练、joblib 保存和模型信息记录。
- 支持随机训练/测试划分及分层交叉验证。
- 输出准确率、macro precision、macro recall、macro F1 和混淆矩阵。

## 项目结构

```text
src/sentiment_cli/
  analyzer.py          文本处理、词典分类、关键词和图表
  data_validation.py   标注数据检查
  ml_model.py          词级与字符级 TF-IDF + LogisticRegression
  train.py             最终模型训练与保存
  main.py              lexicon / ml 实际分析入口
  evaluate.py          holdout 与交叉验证评估
data/
  sample_comments.csv
  labeled_comments.csv
  independent_test_comments.csv
  stopwords.txt
docs/example_outputs/  固定演示结果
tests/                  pytest 测试
.github/workflows/      GitHub Actions
```

## 安装

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

`joblib` 是 scikit-learn 使用的模型序列化工具，项目已在依赖中显式声明。

## 检查标注数据

```powershell
python -m sentiment_cli.data_validation `
  --input data/labeled_comments.csv `
  --column comment `
  --label label `
  --minimum-per-class 40
```

当前数据共 126 条，`positive`、`negative`、`neutral` 各 42 条。校验器会检查必要列、空文本、非法标签、原文重复、清洗后重复和类别不平衡。

## 使用词典法

原有命令仍然可用，默认方法是 `lexicon`：

```powershell
python -m sentiment_cli.main `
  --input data/sample_comments.csv `
  --column comment `
  --method lexicon `
  --stopwords data/stopwords.txt `
  --output outputs/lexicon
```

不需要图表时添加 `--no-chart`。使用 `--top-n 15` 可以调整关键词数量。不指定 `--stopwords` 时仍使用代码内置停用词。

词典模式的 `classified_comments.csv` 包含：

- `classification_method`
- `positive_score`、`negative_score`
- `positive_hits`、`negative_hits`
- `negated_hits`

多个命中词使用 `|` 连接，不会把 Python 列表直接写入单元格。

单字情感词不会再做任意子串匹配。“爱好”不会命中“好”，“差异”不会命中“差”，“快递”不会命中“快”。单字只有被 jieba 识别为独立词，或带有合法程度词、否定词前缀时才参与打分。

## 训练并保存模型

```powershell
python -m sentiment_cli.train `
  --input data/labeled_comments.csv `
  --column comment `
  --label label `
  --model models/sentiment_model.joblib
```

该命令使用全部合法标注数据训练用于实际推理的最终模型，并生成：

- `models/sentiment_model.joblib`：完整 sklearn Pipeline。
- `models/model_info.json`：算法、词/字符 n-gram 范围、样本量、类别数量、列名、随机种子、生成时间，以及 Python 和 sklearn 版本。

训练命令负责生成最终推理模型，不把这些训练数据描述成独立测试结果。

## 使用机器学习法

```powershell
python -m sentiment_cli.main `
  --input data/sample_comments.csv `
  --column comment `
  --method ml `
  --model models/sentiment_model.joblib `
  --output outputs/ml
```

机器学习模式输出：

- `classification_method`
- `confidence`
- `positive_probability`
- `negative_probability`
- `neutral_probability`

`confidence` 是模型对本次预测给出的最大概率，不表示预测一定正确，也不能代替真实标签验证。

## 评估两种方法

```powershell
python -m sentiment_cli.evaluate `
  --input data/labeled_comments.csv `
  --test-input data/independent_test_comments.csv `
  --column comment `
  --label label `
  --output outputs/evaluation `
  --cv-folds 5
```

评估命令保留原来的 `train_test_split`，同时使用同一组 `StratifiedKFold` 折叠比较两种方法。机器学习模型在每一折中只使用训练折训练，不会提前看到验证折。固定独立集只在模型参数和词典冻结后用于最终测试，并检查它与训练集不存在清洗后重复。

生成文件包括：

- `evaluation_report.txt`
- `confusion_matrix.png`
- `metrics_comparison.png`
- `independent_confusion_matrix.png`

### 当前真实评估结果

以下数据来自 126 条样本、5 折分层交叉验证：

| 方法 | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| 词典法平均值 | 0.9286 | 0.9314 | 0.9269 | 0.9268 |
| 词典法标准差 | 0.0530 | 0.0518 | 0.0557 | 0.0554 |
| 机器学习平均值 | 0.5157 | 0.5187 | 0.5167 | 0.5071 |
| 机器学习标准差 | 0.0406 | 0.0920 | 0.0482 | 0.0680 |

组合字符特征后，机器学习交叉验证 accuracy 从上一版本的 `0.3643` 提高到 `0.5157`，macro F1 从 `0.3528` 提高到 `0.5071`。训练集交叉验证中词典法仍然更高，但它可能受词典与训练数据共同开发的影响。

固定独立测试集 `v1` 共 45 条，三类各 15 条，结果如下：

| 方法 | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| 词典法 | 0.6889 | 0.8062 | 0.6889 | 0.6941 |
| 机器学习法 | 0.8222 | 0.8323 | 0.8222 | 0.8234 |

独立测试集未用于训练、交叉验证、词典修改或模型参数选择。测试集只有 45 条，结果仍有较大不确定性，不能直接代表真实平台性能。冻结规则和 SHA-256 见 [`data/INDEPENDENT_TESTSET.md`](data/INDEPENDENT_TESTSET.md)。

`accuracy` 是全部样本中预测正确的比例。`macro F1` 先分别计算每个类别的 F1，再等权平均，更适合观察模型是否兼顾三类评论。

## 固定示例结果

[`docs/example_outputs`](docs/example_outputs) 保存了一次稳定运行产生的演示文件。它们只用于查看结果格式，不会在每次普通运行时自动更新。实际运行结果写入被 `.gitignore` 忽略的 `outputs/`，训练模型写入 `models/`。

## 测试与 CI

```powershell
python -m pytest -q
```

GitHub Actions 在 push 和 pull request 时使用 Python 3.10 安装依赖并运行测试。CI 设置 `MPLBACKEND=Agg`，生成图表时不依赖桌面环境。

## 使用边界

这份数据是课程演示规模，来自人工整理的多领域示例，不能代表真实平台上的语言分布。词典法难以处理反讽和复杂语境；机器学习法受样本数量和特征稀疏影响明显。项目没有使用爬虫、云端情感服务、在线大模型 API 或深度学习。

AI 工具名称、可核验客户端版本、模型标识和参与范围已如实记录在 [`AI_USAGE.md`](AI_USAGE.md)。本次使用的是 OpenAI Codex 桌面应用，Windows 包版本 `26.707.12708.0`；可确认模型标识为 GPT-5 Codex 编程代理，精确服务端快照未公开，不进行猜测。课程提交时仍应遵守学校的 AI 使用说明要求。
