# 网络评论情感分析系统

这是一个命令行版 Python 课程作业项目，用来对商品评论、电影评论、外卖评价等文本进行简单情感分析。

程序会完成：

- 文本清洗
- `jieba` 分词
- 关键词统计
- 正面、负面、中性分类
- 情感占比统计
- CSV、文本报告和图表输出
- 可选的 `sklearn` 机器学习训练示例

## 项目结构

```text
src/sentiment_cli/
  analyzer.py      # 清洗、分词、分类、关键词和统计
  main.py          # 命令行入口
  ml_model.py      # sklearn 训练示例
data/
  sample_comments.csv
outputs/
  classified_comments.csv
  summary.txt
  sentiment_ratio.png
tests/
  test_analyzer.py
  test_ml_model.py
学习说明.md
```

## 安装依赖

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

如果直连 PyPI 慢，可以使用清华源：

```powershell
.\.venv\Scripts\python.exe -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

## 运行程序

```powershell
.\.venv\Scripts\python.exe -m sentiment_cli.main --input data\sample_comments.csv --column comment --output outputs
```

运行后会生成：

- `outputs/classified_comments.csv`：每条评论的清洗结果、分词结果、情感分类
- `outputs/summary.txt`：情感数量、占比和高频关键词
- `outputs/sentiment_ratio.png`：情感分类柱状图

## 运行测试

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
