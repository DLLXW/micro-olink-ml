#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
micro-olink-ml: rigorous microbiome + Olink proteomics classification pipeline
==============================================================================

Binary classification of benign vs malignant disease (labels: Ad-BBD=0,
Ad-MBD=1) from genus-level microbiome abundances + clinical factors and Olink
protein profiles, using a leakage-free cross-validation protocol.

Methodology
-----------
- Stratified 5-fold CV; every model is wrapped in a sklearn Pipeline so that
  median imputation and RobustScaler are fitted on the training folds only.
- All reported metrics (AUC, AP, F1/precision/recall/specificity) are computed
  from pooled out-of-fold (OOF) predictions. AUC 95% CIs come from a 2000-fold
  bootstrap (2.5/97.5 percentiles) of the OOF predictions.
- The top-K analysis uses nested feature selection: within each outer fold the
  top-K features are re-selected on that fold's training data only, so no
  information from evaluation samples informs feature choice.
- Model hyperparameters are fixed a priori (no evaluation-driven tuning).
- Display names identify the algorithm family: the `GBDT` entry is the
  scikit-learn GradientBoostingClassifier (plain gradient-boosted decision
  trees).

Usage
-----
    python rigorous_pipeline.py --data-dir ./data --out-dir ./outputs

Expected inputs inside --data-dir (see data/README.md):
    BC分组V2-微生物+临床因子.xlsx  (sheet "微生物信息")
    BC-蛋白质+临床因子.xlsx        (sheet "Sheet2")
"""

import argparse
import os
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (roc_curve, roc_auc_score, auc,
                             precision_recall_curve, average_precision_score,
                             confusion_matrix, f1_score, precision_score, recall_score)
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# 0. Global configuration
# ----------------------------------------------------------------------------
RANDOM_STATE = 2026          # single random seed for the whole pipeline
N_SPLITS = 5                 # outer CV folds
N_BOOT = 2000                # bootstrap resamples for AUC 95% CI
TOP_K = 6                    # K of the nested top-K analysis
LABEL_MAP = {"Ad-BBD": 0, "Ad-MBD": 1}
MACRO_FILENAME = "BC分组V2-微生物+临床因子.xlsx"
PROTEIN_FILENAME = "BC-蛋白质+临床因子.xlsx"

np.random.seed(RANDOM_STATE)
plt.rcParams.update({
    "font.size": 14,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.facecolor": "white",
    "font.sans-serif": ["Arial", "Arial Unicode MS", "DejaVu Sans"],
    "font.family": "sans-serif",
})
sns.set_style("whitegrid", {"grid.color": "#E6E6E6", "grid.linestyle": "--"})

COLOR_MAIN = "#0072B5"       # benign (BPN)
COLOR_ALT = "#BC3C29"        # malignant (MPN)
MODEL_COLORS = OrderedDict([
    ("Logistic Regression", "#1f77b4"),
    ("SVM",                 "#ff7f0e"),
    ("Naive Bayes",         "#2ca02c"),
    ("MLP",                 "#d62728"),
    ("Random Forest",       "#9467bd"),
    ("GBDT",                "#8c564b"),
    ("LightGBM",            "#e377c2"),
])


# ----------------------------------------------------------------------------
# 1. Data loading and merging
# ----------------------------------------------------------------------------
def load_data(data_dir):
    """Load the microbiome (genus-level + clinical factors) and Olink protein
    sheets and inner-join them on the `sample` column."""
    macro_path = os.path.join(data_dir, MACRO_FILENAME)
    protein_path = os.path.join(data_dir, PROTEIN_FILENAME)
    for path in (macro_path, protein_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                "{} not found. Put both input Excel files in --data-dir "
                "(see data/README.md for the expected schema).".format(path))

    df_macro = pd.read_excel(macro_path, sheet_name="微生物信息")
    if not {"sample", "Group"}.issubset(df_macro.columns):
        raise ValueError("Microbiome sheet must contain 'sample' and 'Group' columns.")
    unknown_groups = set(df_macro["Group"].unique()) - set(LABEL_MAP)
    if unknown_groups:
        raise ValueError("Unmapped Group labels: {}".format(sorted(unknown_groups)))
    df_macro["MPN"] = df_macro["Group"].map(LABEL_MAP).astype(int)
    for col in df_macro.columns:
        if col not in ("sample", "Group"):
            df_macro[col] = pd.to_numeric(df_macro[col], errors="coerce")

    df_protein = pd.read_excel(protein_path, sheet_name="Sheet2")
    if "sample" not in df_protein.columns:
        raise ValueError("Protein sheet must contain a 'sample' column.")
    df_protein = df_protein.drop(columns=[c for c in ("group", "group.1") if c in df_protein.columns])
    for col in df_protein.columns:
        if col != "sample":
            df_protein[col] = pd.to_numeric(df_protein[col], errors="coerce")

    df = pd.merge(df_macro, df_protein, on="sample", how="inner")
    features = [c for c in df.columns if c not in ("sample", "Group", "MPN")]
    return df, features


# ----------------------------------------------------------------------------
# 2. 模型定义(名称与估计器一一对应)
# ----------------------------------------------------------------------------
def make_models():
    """全部模型都包进 Pipeline:缺失值插补与缩放只在训练折内拟合,杜绝泄漏。"""
    def wrap(clf):
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", RobustScaler()),
            ("clf", clf),
        ])

    return OrderedDict([
        ("Logistic Regression", wrap(LogisticRegression(max_iter=5000))),
        ("SVM",                 wrap(SVC(probability=True))),
        ("Naive Bayes",         wrap(GaussianNB())),
        ("MLP",                 wrap(MLPClassifier(hidden_layer_sizes=(32,), alpha=1.0,
                                                    max_iter=5000, random_state=RANDOM_STATE))),
        ("Random Forest",       wrap(RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE))),
        ("GBDT",                wrap(GradientBoostingClassifier(n_estimators=200, learning_rate=0.1,
                                                        max_depth=3, random_state=RANDOM_STATE))),
        ("LightGBM",            wrap(LGBMClassifier(max_depth=4, num_leaves=15, learning_rate=0.05,
                                                    n_estimators=300, random_state=RANDOM_STATE,
                                                    verbose=-1))),
    ])


# ----------------------------------------------------------------------------
# 3. 交叉验证:汇总 OOF 预测
# ----------------------------------------------------------------------------
def cv_oof_predictions(models, X, y):
    """对每个模型跑统一的 5 折分层 CV,返回 {model: (oof_score, oof_label, per_fold_auc)}。"""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    out = {}
    for name, model in models.items():
        oof_score = np.zeros(len(y), dtype=float)
        fold_auc = []
        for tr, te in skf.split(X, y):
            m = sklearn.base.clone(model)
            m.fit(X[tr], y[tr])
            oof_score[te] = m.predict_proba(X[te])[:, 1]
            if len(np.unique(y[te])) == 2:
                fold_auc.append(roc_auc_score(y[te], oof_score[te]))
        out[name] = (oof_score, y.copy(), fold_auc)
        print("  {:<22s} OOF AUC = {:.3f}  (folds: {})".format(
            name, roc_auc_score(y, oof_score),
            ", ".join("{:.3f}".format(a) for a in fold_auc)))
    return out


# ----------------------------------------------------------------------------
# 4. 指标与置信区间
# ----------------------------------------------------------------------------
def bootstrap_auc_ci(y_true, y_score, n_boot=N_BOOT, seed=RANDOM_STATE):
    """对 OOF 预测做 bootstrap,返回 AUC 的 95% percentile CI(2.5/97.5)。"""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    aucs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            aucs[i] = np.nan
            continue
        aucs[i] = roc_auc_score(yt, ys)
    aucs = aucs[~np.isnan(aucs)]
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return lo, hi


def bootstrap_roc_band(y_true, y_score, n_boot=N_BOOT, seed=RANDOM_STATE):
    """bootstrap ROC 曲线带(2.5/97.5 分位),用于主图阴影。"""
    rng = np.random.default_rng(seed)
    mean_fpr = np.linspace(0, 1, 100)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    tprs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        fpr, tpr, _ = roc_curve(yt, ys)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
    tprs = np.array(tprs)
    return mean_fpr, np.percentile(tprs, 2.5, axis=0), np.percentile(tprs, 97.5, axis=0)


def threshold_metrics(y_true, y_score, threshold):
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "F1": f1_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "Threshold": threshold,
    }


def youden_threshold(y_true, y_score):
    fpr, tpr, thr = roc_curve(y_true, y_score)
    return thr[int(np.argmax(tpr - fpr))]


def build_metrics_table(oof_dict):
    rows = []
    for name, (score, y, fold_auc) in oof_dict.items():
        auc_pt = roc_auc_score(y, score)
        lo, hi = bootstrap_auc_ci(y, score)
        ap = average_precision_score(y, score)
        row = OrderedDict([
            ("Model", name),
            ("OOF_AUC", round(auc_pt, 3)),
            ("AUC_95CI_low", round(lo, 3)),
            ("AUC_95CI_high", round(hi, 3)),
            ("Fold_AUC_mean", round(float(np.mean(fold_auc)), 3)),
            ("Fold_AUC_std", round(float(np.std(fold_auc)), 3)),
            ("Average_Precision", round(ap, 3)),
        ])
        for tag, thr in (("0.5", 0.5), ("Youden", youden_threshold(y, score))):
            m = threshold_metrics(y, score, thr)
            for k in ("F1", "Precision", "Recall", "Specificity"):
                row["{}_@thr{}".format(k, tag)] = round(m[k], 3)
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 5. 特征重要度:跨折平均(不用测试折早停,固定轮数)
# ----------------------------------------------------------------------------
def cv_feature_importance(X, y, features, top_n=20):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    imp = np.zeros(X.shape[1])
    for tr, _te in skf.split(X, y):
        scaler = RobustScaler().fit(X[tr])
        m = LGBMClassifier(max_depth=4, num_leaves=15, learning_rate=0.05, n_estimators=300,
                           random_state=RANDOM_STATE, verbose=-1)
        m.fit(scaler.transform(X[tr]), y[tr])
        imp += m.feature_importances_          # gain
    imp /= N_SPLITS
    order = np.argsort(imp)[::-1]
    return pd.DataFrame({"Feature": [features[i] for i in order],
                         "Gain": imp[order]}).head(top_n)


# ----------------------------------------------------------------------------
# 6. Top-K 分析:嵌套式特征选择(每个外折用训练折重选,杜绝选择偏倚)
# ----------------------------------------------------------------------------
def nested_topk_auc(X, y, k=TOP_K):
    """外层 5 折:每折内部先在训练折上算 LGBM gain 重要度,取 top-K 再训练/预测。"""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y))
    fold_auc = []
    for tr, te in skf.split(X, y):
        scaler = RobustScaler().fit(X[tr])
        selector = LGBMClassifier(max_depth=4, num_leaves=15, learning_rate=0.05, n_estimators=300,
                                  random_state=RANDOM_STATE, verbose=-1)
        selector.fit(scaler.transform(X[tr]), y[tr])
        topk = np.argsort(selector.feature_importances_)[::-1][:k]

        scaler2 = RobustScaler().fit(X[tr][:, topk])
        m = LGBMClassifier(max_depth=4, num_leaves=15, learning_rate=0.05, n_estimators=300,
                           random_state=RANDOM_STATE, verbose=-1)
        m.fit(scaler2.transform(X[tr][:, topk]), y[tr])
        oof[te] = m.predict_proba(scaler2.transform(X[te][:, topk]))[:, 1]
        fold_auc.append(roc_auc_score(y[te], oof[te]))
    return oof, fold_auc


# ----------------------------------------------------------------------------
# 7. 统计检验:Mann-Whitney U + Benjamini-Hochberg FDR
# ----------------------------------------------------------------------------
def mann_whitney_with_bh(df, features, label_col="MPN"):
    from scipy import stats
    pos = df[df[label_col] == 1]
    neg = df[df[label_col] == 0]
    rows = []
    for f in features:
        u, p = stats.mannwhitneyu(pos[f].values, neg[f].values, alternative="two-sided")
        rows.append({"Feature": f, "P_value": p,
                     "Median_BPN": neg[f].median(), "Median_MPN": pos[f].median()})
    res = pd.DataFrame(rows)
    # BH 校正
    m = len(res)
    order = np.argsort(res["P_value"].values)
    ranked = res["P_value"].values[order]
    bh = ranked * m / (np.arange(m) + 1)
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    corrected = np.empty(m)
    corrected[order] = np.clip(bh, 0, 1)
    res["P_FDR_BH"] = corrected
    return res


# ----------------------------------------------------------------------------
# 8. 图表
# ----------------------------------------------------------------------------
def plot_roc_all(oof_dict, path_prefix):
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    for name, (score, y, _fa) in oof_dict.items():
        fpr, tpr, _ = roc_curve(y, score)
        auc_pt = auc(fpr, tpr)
        lo, hi = bootstrap_auc_ci(y, score)
        mfpr, tpr_lo, tpr_hi = bootstrap_roc_band(y, score)
        c = MODEL_COLORS[name]
        ax.fill_between(mfpr, tpr_lo, tpr_hi, color=c, alpha=0.12, lw=0)
        ax.plot(fpr, tpr, lw=2.2, color=c,
                label="{} (AUC={:.3f}, 95%CI {:.3f}-{:.3f})".format(name, auc_pt, lo, hi))
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("1 - Specificity")
    ax.set_ylabel("Sensitivity")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("Microbiome + Olink protein, {}-fold CV (pooled OOF)".format(N_SPLITS), fontsize=13)
    fig.tight_layout()
    fig.savefig(path_prefix + ".pdf", bbox_inches="tight")
    fig.savefig(path_prefix + ".png", bbox_inches="tight")
    plt.close(fig)


def plot_pr_all(oof_dict, path_prefix):
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    for name, (score, y, _fa) in oof_dict.items():
        prec, rec, _ = precision_recall_curve(y, score)
        ap = average_precision_score(y, score)
        ax.plot(rec, prec, lw=2.2, color=MODEL_COLORS[name],
                label="{} (AP={:.3f})".format(name, ap))
    prev = float(np.mean(oof_dict[list(oof_dict)[0]][1]))
    ax.axhline(prev, color="gray", ls="--", lw=1, label="Prevalence={:.3f}".format(prev))
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path_prefix + ".pdf", bbox_inches="tight")
    fig.savefig(path_prefix + ".png", bbox_inches="tight")
    plt.close(fig)


def plot_confusion_panels(oof_dict, path_prefix, ncols=2):
    names = list(oof_dict)
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 5.4 * nrows), dpi=300)
    axes = np.atleast_1d(axes).ravel()
    for ax, name in zip(axes, names):
        score, y, _fa = oof_dict[name]
        thr = youden_threshold(y, score)
        m = threshold_metrics(y, score, thr)
        cm = confusion_matrix(y, (score >= thr).astype(int))
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["BPN", "MPN"], yticklabels=["BPN", "MPN"],
                    annot_kws={"fontsize": 13})
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title("{}\n(Youden thr={:.3f}, F1={:.3f})".format(name, m["Threshold"], m["F1"]), fontsize=12)
    for ax in axes[len(names):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path_prefix + ".pdf", bbox_inches="tight")
    fig.savefig(path_prefix + ".png", bbox_inches="tight")
    plt.close(fig)


def plot_importance(imp_df, path_prefix):
    d = imp_df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.42 * len(d))), dpi=300)
    ax.barh(d["Feature"], d["Gain"], color=COLOR_MAIN)
    ax.set_xlabel("LightGBM gain importance ({}-fold mean)".format(N_SPLITS))
    ax.set_title("Top {} features".format(len(d)), fontsize=13)
    fig.tight_layout()
    fig.savefig(path_prefix + ".pdf", bbox_inches="tight")
    fig.savefig(path_prefix + ".png", bbox_inches="tight")
    plt.close(fig)


def plot_violin(df, top_features, stat_df, path_prefix):
    n = len(top_features)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.2 * nrows), dpi=300)
    axes = np.atleast_1d(axes).ravel()
    pdf = df[top_features + ["MPN"]].copy()
    pdf["Group"] = pdf["MPN"].map({0: "BPN", 1: "MPN"})
    pmap = dict(zip(stat_df["Feature"], stat_df["P_FDR_BH"]))
    for ax, feat in zip(axes, top_features):
        sns.violinplot(x="Group", y=feat, data=pdf, palette={"BPN": COLOR_MAIN, "MPN": COLOR_ALT},
                       inner="box", ax=ax, cut=0)
        p = pmap.get(feat, np.nan)
        ax.set_title("{}\nFDR-adjusted P={:.2e}".format(feat, p), fontsize=11)
        ax.set_xlabel(""); ax.set_ylabel("")
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path_prefix + ".pdf", bbox_inches="tight")
    fig.savefig(path_prefix + ".png", bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(df, top_features, path_prefix):
    sub = df[top_features + ["MPN"]].sort_values("MPN")
    corr = sub[top_features].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=300)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax,
                cbar_kws={"label": "Spearman rho"})
    ax.set_title("Spearman correlation, top {} features".format(len(top_features)), fontsize=12)
    fig.tight_layout()
    fig.savefig(path_prefix + ".pdf", bbox_inches="tight")
    fig.savefig(path_prefix + ".png", bbox_inches="tight")
    plt.close(fig)


def run_shap(df, features, top_features, path_prefix):
    """单次 80/20 划分训练 LGBM 并做 SHAP(结果标注为单一划分,仅供参考)。"""
    X = df[top_features].values
    y = df["MPN"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y,
                                          random_state=RANDOM_STATE)
    scaler = RobustScaler().fit(Xtr)
    m = LGBMClassifier(max_depth=4, num_leaves=15, learning_rate=0.05, n_estimators=300,
                       random_state=RANDOM_STATE, verbose=-1)
    m.fit(scaler.transform(Xtr), ytr)
    auc_te = roc_auc_score(yte, m.predict_proba(scaler.transform(Xte))[:, 1])
    print("  [SHAP split] holdout AUC = {:.3f} (single split, indicative only)".format(auc_te))

    import shap
    explainer = shap.TreeExplainer(m)
    sv = explainer.shap_values(scaler.transform(Xte))
    sv1 = sv[1] if isinstance(sv, list) else sv
    fig = plt.figure(figsize=(9, 6), dpi=300)
    shap.summary_plot(sv1, pd.DataFrame(scaler.transform(Xte), columns=top_features),
                      show=False, max_display=10)
    plt.tight_layout()
    plt.savefig(path_prefix + ".pdf", bbox_inches="tight")
    plt.savefig(path_prefix + ".png", bbox_inches="tight")
    plt.close("all")
    return auc_te


def plot_topk_vs_all(oof_all, oof_topk, path_prefix):
    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=300)
    for score, y, label, color in [
        (oof_all[0], oof_all[1], "All features (n={})".format(len(oof_all[1])), COLOR_MAIN),
        (oof_topk[0], oof_topk[1], "Top-{} (nested selection)".format(TOP_K), COLOR_ALT),
    ]:
        fpr, tpr, _ = roc_curve(y, score)
        auc_pt = auc(fpr, tpr)
        lo, hi = bootstrap_auc_ci(y, score)
        ax.plot(fpr, tpr, lw=2.4, color=color,
                label="{} (AUC={:.3f}, 95%CI {:.3f}-{:.3f})".format(label, auc_pt, lo, hi))
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("1 - Specificity"); ax.set_ylabel("Sensitivity")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_title("LightGBM: all features vs nested top-{}".format(TOP_K), fontsize=13)
    fig.tight_layout()
    fig.savefig(path_prefix + ".pdf", bbox_inches="tight")
    fig.savefig(path_prefix + ".png", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="micro-olink-ml: leakage-free microbiome + Olink protein "
                    "classification pipeline.")
    parser.add_argument("--data-dir", default="data",
                        help="Directory holding the two input Excel files (default: ./data)")
    parser.add_argument("--out-dir", default="outputs",
                        help="Directory for tables and figures (default: ./outputs)")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE,
                        help="Random seed for the whole pipeline (default: {})".format(RANDOM_STATE))
    parser.add_argument("--n-splits", type=int, default=N_SPLITS,
                        help="Number of stratified CV folds (default: {})".format(N_SPLITS))
    parser.add_argument("--top-k", type=int, default=TOP_K,
                        help="K for the nested top-K analysis (default: {})".format(TOP_K))
    return parser.parse_args()


def main():
    global RANDOM_STATE, N_SPLITS, TOP_K
    args = parse_args()
    RANDOM_STATE, N_SPLITS, TOP_K = args.seed, args.n_splits, args.top_k
    np.random.seed(RANDOM_STATE)

    out_dir = args.out_dir
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print("[1/7] Loading and merging data ...")
    df, features = load_data(args.data_dir)
    X = df[features].values.astype(float)
    y = df["MPN"].values.astype(int)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Both classes must be present after merging.")
    print("  {} samples (BPN {} / MPN {}), {} features".format(len(y), n_neg, n_pos, len(features)))
    n_missing = int(np.isnan(X).sum())
    if n_missing:
        print("  Note: {} missing values handled by in-pipeline median imputation".format(n_missing))

    print("[2/7] {}-fold stratified CV, pooling OOF predictions ...".format(N_SPLITS))
    models = make_models()
    oof_dict = cv_oof_predictions(models, X, y)

    print("[3/7] Computing metrics and 95% CIs ...")
    metrics = build_metrics_table(oof_dict)
    metrics.to_csv(os.path.join(out_dir, "model_metrics.csv"), index=False)
    print(metrics[["Model", "OOF_AUC", "AUC_95CI_low", "AUC_95CI_high",
                   "Fold_AUC_mean", "Average_Precision"]].to_string(index=False))

    print("[4/7] Cross-fold averaged feature importance ...")
    imp_df = cv_feature_importance(X, y, features, top_n=20)
    imp_df.to_csv(os.path.join(out_dir, "feature_importance_top20.csv"), index=False)
    top_features = imp_df["Feature"].tolist()[:TOP_K]

    print("[5/7] Nested top-{} evaluation + statistical tests ...".format(TOP_K))
    oof_topk, fold_auc_topk = nested_topk_auc(X, y, k=TOP_K)
    auc_topk = roc_auc_score(y, oof_topk)
    lo, hi = bootstrap_auc_ci(y, oof_topk)
    topk_row = OrderedDict([
        ("Model", "LightGBM_top{}".format(TOP_K)),
        ("OOF_AUC", round(auc_topk, 3)),
        ("AUC_95CI_low", round(lo, 3)),
        ("AUC_95CI_high", round(hi, 3)),
        ("Fold_AUC_mean", round(float(np.mean(fold_auc_topk)), 3)),
        ("Fold_AUC_std", round(float(np.std(fold_auc_topk)), 3)),
        ("Average_Precision", round(average_precision_score(y, oof_topk), 3)),
    ])
    pd.DataFrame([topk_row]).to_csv(os.path.join(out_dir, "topk_metrics.csv"), index=False)
    print("  LightGBM_top{}: OOF AUC = {:.3f} (95%CI {:.3f}-{:.3f})".format(TOP_K, auc_topk, lo, hi))

    stat_df = mann_whitney_with_bh(df, top_features)
    stat_df.to_csv(os.path.join(out_dir, "stat_tests_top{}.csv".format(TOP_K)), index=False)

    print("[6/7] SHAP (single 80/20 split, indicative only) ...")
    shap_holdout_auc = run_shap(df, features, top_features,
                                os.path.join(fig_dir, "07_shap_summary"))

    print("[7/7] Writing figures ...")
    plot_roc_all(oof_dict, os.path.join(fig_dir, "01_roc_all_models"))
    plot_pr_all(oof_dict, os.path.join(fig_dir, "02_pr_curves"))
    plot_confusion_panels(oof_dict, os.path.join(fig_dir, "03_confusion_matrices"))
    plot_importance(imp_df, os.path.join(fig_dir, "04_feature_importance_top20"))
    plot_violin(df, top_features, stat_df, os.path.join(fig_dir, "05_top{}_violin".format(TOP_K)))
    plot_heatmap(df, top_features, os.path.join(fig_dir, "06_top{}_spearman_heatmap".format(TOP_K)))
    plot_topk_vs_all((oof_dict["LightGBM"][0], oof_dict["LightGBM"][1]),
                     (oof_topk, y),
                     os.path.join(fig_dir, "08_topk_vs_all_lightgbm"))

    with open(os.path.join(out_dir, "run_summary.txt"), "w") as f:
        f.write("random seed: {}\n".format(RANDOM_STATE))
        f.write("CV: StratifiedKFold(n_splits={}, shuffle=True)\n".format(N_SPLITS))
        f.write("samples: {} (BPN={} / MPN={}), features: {}\n".format(len(y), n_neg, n_pos, len(features)))
        f.write("inputs: {} + {}\n".format(MACRO_FILENAME, PROTEIN_FILENAME))
        f.write("scikit-learn {}, lightgbm, shap\n".format(sklearn.__version__))
        f.write("LightGBM holdout (single 80/20 split, indicative) AUC = {:.3f}\n".format(shap_holdout_auc))
    print("Done. All outputs in:", out_dir)


if __name__ == "__main__":
    main()
