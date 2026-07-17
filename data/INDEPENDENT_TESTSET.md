# 固定独立测试集说明

- 文件：`data/independent_test_comments.csv`
- 版本：`v1`
- 冻结日期：`2026-07-17`
- 数据量：45 条
- 类别分布：positive 15、negative 15、neutral 15
- SHA-256：`30EE726D8E81D7E355B3D7982B1C9B7FB0EE4114ED251F34A4C90264E5E30227`

该文件与 `data/labeled_comments.csv` 的清洗后文本交集为 0。训练数据用于模型训练和交叉验证；本文件只用于最终独立测试，不参与：

- 情感词典增加、删除或权重调整；
- TF-IDF 特征范围和逻辑回归参数选择；
- 交叉验证折数或随机种子选择；
- 模型训练。

本轮模型配置在查看该独立测试集指标前确定为：词级 1-2 gram、字符级 1-3 gram、`LogisticRegression(C=3.0)`。如未来修改独立测试集，必须提高版本号、重新记录哈希，并把新结果与 `v1` 分开报告。
