# 网络评论情感分析系统

这是一个支持命令行和 Streamlit 图形界面的 Python 课程项目，用于分析商品、外卖、酒店、电影、课程和数码产品评论。系统支持可解释的词典法，以及词级与字符级 TF-IDF + LogisticRegression 机器学习法，最终把评论分为 `positive`、`negative`、`neutral` 三类。

## 系统流程

```mermaid
flowchart LR
    A[CSV 原始评论] --> D{分类方法}
    D -->|lexicon| B[文本清洗与 jieba 分词]
    B --> E[最长词匹配与否定处理]
    D -->|ml| F[sklearn Pipeline]
    F --> C[clean_text 统一清洗]
    C --> K[词级与字符级 TF-IDF]
    K --> L[LogisticRegression]
    E --> G[分类结果与解释字段]
    L --> G
    G --> H[数量、占比和关键词汇总]
    H --> I[CSV、文本报告和图表]
```

## 主要功能

- 清洗网址、标点和多余空白。
- 使用 `jieba` 分词，支持内置和外部停用词。
- 词典法按最长词优先匹配，避免“很好”同时命中“很好”和“好”。
- 词典法每个最长情感短语统一计 1 分，不设置正负不对称权重。
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
  inference.py         CLI 与界面共用的推理接口
  ui_helpers.py        图表、筛选和下载数据处理
  ml_model.py          词级与字符级 TF-IDF + LogisticRegression
  train.py             最终模型训练与保存
  main.py              lexicon / ml 实际分析入口
  evaluate.py          holdout 与交叉验证评估
app.py                 Streamlit 图形化分析平台
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

## 图形化界面

安装依赖和当前项目后启动：

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
python -m streamlit run app.py
```

界面包含四个页面：

- **单条评论分析**：输入一条评论，切换词典法或机器学习法，查看命中词、得分、概率和置信度。
- **CSV 批量分析**：上传 CSV、选择评论列和可选停用词，查看数量、占比、关键词、置信度及筛选明细。
- **模型与评估**：查看默认模型状态，在界面中训练模型或重新运行默认评估。
- **系统说明**：查看两种分类流程、当前模型参数和使用边界。

首次使用机器学习法时，可在“模型与评估”页面点击“使用默认标注数据训练模型”，也可以继续使用原有命令行训练：

```powershell
python -m sentiment_cli.train `
  --input data/labeled_comments.csv `
  --model models/sentiment_model.joblib
```

推荐答辩演示流程：启动应用，分析一条评论并切换两种方法，上传 `data/sample_comments.csv`，展示情感占比和关键词，筛选负面或低置信度评论，查看三组评估对比，最后下载 CSV 和摘要。

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
- `models/model_info.json`：算法、Pipeline 内预处理位置、词/字符 n-gram 范围、样本量、类别数量、列名、随机种子、生成时间，以及 Python 和 sklearn 版本。

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

ML 模式把原始评论直接传给模型。词级分支由 `tokenize()` 调用 `clean_text()`，字符级分支通过 `preprocessor=clean_text` 调用同一函数；`cleaned_text` 只用于结果展示。外部 `--stopwords` 只影响输出分词和关键词统计，不改变已训练模型的预测特征。

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

评估命令同时给出单次分层 `train_test_split`、5 折 `StratifiedKFold` 和固定留出测试集 v1。机器学习模型在每一折中只使用训练折训练，不会提前看到验证折。程序还检查留出集与训练集不存在清洗后重复。

生成文件包括：

- `evaluation_report.txt`
- `confusion_matrix.png`
- `metrics_comparison.png`
- `independent_confusion_matrix.png`
- `evaluation_metadata.json`
- `evaluation_results.json`

### 当前真实评估结果

以下数值均来自本轮统一 Pipeline 预处理后重新执行的报告。单次分层划分结果为：

| 方法 | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| 词典法 | 0.9474 | 0.9487 | 0.9487 | 0.9477 |
| 机器学习法 | 0.3947 | 0.4333 | 0.3932 | 0.4085 |

126 条训练数据的 5 折分层交叉验证结果为：

| 方法 | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| 词典法平均值 | 0.9209 | 0.9248 | 0.9185 | 0.9186 |
| 词典法标准差 | 0.0559 | 0.0542 | 0.0590 | 0.0585 |
| 机器学习平均值 | 0.5157 | 0.5187 | 0.5167 | 0.5071 |
| 机器学习标准差 | 0.0406 | 0.0920 | 0.0482 | 0.0680 |

单次划分受随机种子和样本组成影响较大；交叉验证更适合观察训练数据内部的平均稳定性。训练数据交叉验证中词典法更高，但词典与这批数据存在共同开发背景，不能把该结果直接外推到真实平台。

固定留出测试集 v1 共 45 条，三类各 15 条，结果如下：

| 方法 | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| 词典法 | 0.6889 | 0.8062 | 0.6889 | 0.6941 |
| 机器学习法 | 0.8000 | 0.8062 | 0.8000 | 0.8008 |

在当前固定留出测试集 v1 上，机器学习方法取得了更高的 accuracy 和 macro F1。但该集合已用于上一轮评估，本轮属于同一留出集上的版本对比，不能再声称它从未参与开发决策。三组评估不一致是正常现象，不能只挑最高指标宣传。版本说明和 SHA-256 见 [`data/INDEPENDENT_TESTSET.md`](data/INDEPENDENT_TESTSET.md)，本轮环境和参数见 [`docs/example_outputs/evaluation_metadata.json`](docs/example_outputs/evaluation_metadata.json)。

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
