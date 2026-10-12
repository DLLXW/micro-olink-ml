# micro-olink-ml(中文说明)

基于**属水平微生物 + 临床因子**与 **Olink 蛋白**谱,对疾病**良性 vs 恶性**(Ad-BBD=0,Ad-MBD=1)做二分类的严谨、无泄漏机器学习管线。

整个管线是一个独立的 Python 脚本(`rigorous_pipeline.py`),不含 notebook;所有指标均由代码在固定随机种子、预先确定的评估协议下从数据直接计算。

## 方法要点

- **评估协议**:5 折分层交叉验证,汇总 out-of-fold(OOF)预测后计算指标,同时报告逐折 AUC 以呈现折间波动。
- **无泄漏**:缺失值插补(median)与 `RobustScaler` 置于 sklearn `Pipeline` 内,只在训练折上拟合。
- **诚实的不确定性**:AUC 95%CI 由对 OOF 预测的 bootstrap(2000 次,2.5/97.5 分位)计算,ROC 置信带同理;任何 CI 都不由人工指定。
- **嵌套式 Top-K 特征分析**:每个外折内部仅用该折训练数据重新做 LightGBM gain 重要度排序并取 top-K,评估样本不参与特征选择;top-K 模型与全特征模型在同一协议下对比。
- **模型与名称一一对应**:Logistic Regression、SVM、Naive Bayes、MLP、Random Forest、GBDT(scikit-learn 的 GradientBoostingClassifier)、LightGBM;超参数事先固定,不做任何按结果挑选的搜索。
- **统计检验**:top 特征做 Mann–Whitney U 检验 + Benjamini–Hochberg FDR 校正。
- **可解释性**:跨折平均的 LightGBM gain 重要度;SHAP 基于单一 80/20 分层划分,结果仅作参考并明确标注。

## 目录结构

```
micro-olink-ml/
├── rigorous_pipeline.py   # 完整管线(数据 -> 指标 -> 图表)
├── data/                  # 放置输入 Excel(不入库)
│   └── README.md          # 输入数据 schema 说明
├── requirements.txt
├── LICENSE
└── .gitignore             # 排除患者数据与运行输出
```

## 环境要求

Python 3.7+(实测:scikit-learn 1.0.2、LightGBM 3.2.1、SHAP 0.42.1)。

```bash
pip install -r requirements.txt
```

## 数据准备

将两份 Excel 放入 `--data-dir`(默认 `./data`)。**患者数据不随本仓库分发**,格式要求见 [data/README.md](data/README.md):

- `BC分组V2-微生物+临床因子.xlsx`,sheet `微生物信息`:每行一个样本,含 `sample`、`Group`(Ad-BBD / Ad-MBD)及各微生物属丰度与临床因子列。
- `BC-蛋白质+临床因子.xlsx`,sheet `Sheet2`:每行一个样本,含 `sample` 及各 Olink 蛋白强度列。

两表按 `sample` 列 inner join。

## 运行

```bash
python rigorous_pipeline.py --data-dir ./data --out-dir ./outputs
```

可选参数:`--seed`(默认 2026)、`--n-splits`(默认 5)、`--top-k`(默认 6)。

## 输出(写入 `--out-dir`,已被 gitignore)

| 文件 | 内容 |
|---|---|
| `model_metrics.csv` | 各模型:OOF AUC + 95%CI、逐折均值±标准差、AP、0.5 阈值与 Youden 阈值下的 F1/P/R/特异度 |
| `topk_metrics.csv` | 嵌套 top-K LightGBM 的同类指标 |
| `feature_importance_top20.csv` | 跨折平均 LightGBM gain 重要度 top20 |
| `stat_tests_top6.csv` | top 特征 Mann–Whitney U P 值 + BH 校正 |
| `run_summary.txt` | 种子、协议、样本量、库版本 |
| `figures/01–08`(PDF+PNG) | 多模型 ROC(bootstrap 置信带)、PR 曲线、混淆矩阵(Youden 阈值)、特征重要度、top-6 小提琴(含 FDR P 值)、Spearman 热图、SHAP summary、top-K vs 全特征 ROC |

## 局限性(使用结果前请阅读)

1. 约 88 例样本、约 500 个特征,属高维小样本问题;CI 较宽是正常现象,如实报告。
2. 交叉验证属内部验证,不能替代独立外部验证。
3. Youden 阈值在同一份 OOF 预测上选取,阈值相关指标略偏乐观;请结合默认 0.5 阈值结果对照阅读。
4. 微生物特征按绝对丰度使用;若为组成性数据,建议扩展 CLR 变换与协变量调整。

## 许可

MIT,见 [LICENSE](LICENSE)。
