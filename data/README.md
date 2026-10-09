# Data

**This directory must not contain patient-level data.** All `*.xlsx` / `*.xls`
files are git-ignored. Download or prepare your own cohort and place the two
input files here (or point `--data-dir` at another location).

## Expected inputs

### 1. `BC分组V2-微生物+临床因子.xlsx` — sheet `微生物信息`

One row per sample; microbiome genus abundances plus clinical covariates.

| Column | Type | Description |
|---|---|---|
| `sample` | str | Sample ID, join key |
| `Group` | str | `Ad-BBD` (benign) or `Ad-MBD` (malignant) |
| `g__*` | float | Genus-level relative abundance columns |
| `clin__*` | float | Clinical covariates (e.g. age, BMI, smoking, drinking, family history, anxiety) |

### 2. `BC-蛋白质+临床因子.xlsx` — sheet `Sheet2`

One row per sample; Olink protein intensities (wide format).

| Column | Type | Description |
|---|---|---|
| `sample` | str | Sample ID, join key (must overlap with the microbiome sheet) |
| all other columns | float | Olink protein intensities (columns named `group` / `group.1` are dropped automatically) |

The two tables are inner-joined on `sample`. Rows without both microbiome and
protein measurements are discarded; labels are mapped `Ad-BBD -> 0`,
`Ad-MBD -> 1`.

## Reference cohort (not distributed)

The pipeline was developed on a cohort of 88 matched samples
(42 benign / 46 malignant) with ~500 combined features.
