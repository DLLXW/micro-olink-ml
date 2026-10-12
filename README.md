# micro-olink-ml

A rigorous, leakage-free machine-learning pipeline for binary classification of
**benign vs malignant disease** from **genus-level microbiome abundances +
clinical factors** and **Olink proteomics** profiles.

The pipeline is implemented as a single, dependency-light Python script
(`rigorous_pipeline.py`) — no notebooks — and is designed so that every reported
number is computed from data under a fixed random seed and a pre-registered
evaluation protocol.

## Methodology

- **Evaluation protocol**: 5-fold stratified cross-validation. Metrics are
  computed from pooled **out-of-fold (OOF)** predictions; per-fold AUCs are also
  reported so that fold-to-fold variability is visible.
- **No leakage**: median imputation and `RobustScaler` live inside a sklearn
  `Pipeline` and are fitted on the training folds only.
- **Honest uncertainty**: AUC 95% confidence intervals are bootstrapped
  (2,000 resamples, 2.5/97.5 percentiles) from the OOF predictions; ROC bands
  use the same bootstrap. No confidence interval is ever hard-coded.
- **Nested top-K feature analysis**: within each outer fold, top-K features are
  re-selected on that fold's training data (LightGBM gain importance), so
  evaluation samples never inform feature selection. The K-feature model is
  compared against the all-features model under the same protocol.
- **Real models, honest names**: Logistic Regression, SVM, Naive Bayes, MLP,
  Random Forest, GBDT (scikit-learn GradientBoostingClassifier), LightGBM.
  Hyperparameters are fixed a priori; no per-model search is performed.
- **Statistics**: Mann–Whitney U tests for top features with
  Benjamini–Hochberg FDR correction.
- **Interpretation**: LightGBM gain importance averaged across folds, plus SHAP
  summary on a single stratified 80/20 split (clearly labeled as indicative).

## Repository layout

```
micro-olink-ml/
├── rigorous_pipeline.py   # the whole pipeline (data -> metrics -> figures)
├── data/                  # put your input Excel files here (NOT committed)
│   └── README.md          # expected input schema
├── requirements.txt
├── LICENSE
└── .gitignore             # excludes patient data and run outputs
```

## Requirements

Python 3.7+ (tested on 3.7 with scikit-learn 1.0.2, LightGBM 3.2.1,
SHAP 0.42.1).

```bash
pip install -r requirements.txt
```

## Data preparation

The pipeline expects two Excel files in `--data-dir` (default `./data`).
**Patient-level data is not distributed with this repository** — see
[data/README.md](data/README.md) for the exact expected schema so you can
format your own cohort.

## Usage

```bash
python rigorous_pipeline.py --data-dir ./data --out-dir ./outputs
```

Options: `--seed` (default 2026), `--n-splits` (default 5), `--top-k` (default 6).

Outputs (written to `--out-dir`, gitignored):

| File | Content |
|---|---|
| `model_metrics.csv` | Per model: pooled OOF AUC + 95% CI, per-fold mean ± SD, AP, F1/precision/recall/specificity at 0.5 and Youden thresholds |
| `topk_metrics.csv` | Same metrics for the nested top-K LightGBM model |
| `feature_importance_top20.csv` | Cross-fold averaged LightGBM gain importance |
| `stat_tests_top6.csv` | Mann–Whitney U P values + BH FDR for top features |
| `run_summary.txt` | Seed, CV protocol, sample counts, library versions |
| `figures/01–08` (PDF+PNG) | Multi-model ROC with bootstrap bands, PR curves, confusion matrices (Youden threshold), feature importance, top-K violin plots with FDR P values, Spearman heatmap, SHAP summary, top-K vs all-features ROC |

## Limitations (read before using results)

1. With ~88 samples and ~500 features this is a small-n, high-p problem; wide
   CIs are expected and reported as-is.
2. Cross-validation is internal validation only — it does not replace external
   validation on an independent cohort.
3. The Youden threshold is chosen on the same pooled OOF predictions, so
   threshold-dependent metrics are mildly optimistic; metrics at the default
   0.5 threshold are reported alongside for comparison.
4. Microbiome features are used as absolute abundances; for compositional data
   a CLR transform and covariate adjustment are recommended extensions.

## License

MIT — see [LICENSE](LICENSE).
