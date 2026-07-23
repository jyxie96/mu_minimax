# Efficient machine unlearning with minimax optimality

This repository contains research code for studying **machine unlearning** (post-removal estimation). Given a model pretrained on a full dataset (retain U forget), we compare approximate unlearning estimators against retrain-from-scratch on the retain set only.

Experiments include:

- Numerical simulations (squared loss and cross-entropy)
- A Yelp review prediction task
- A UK Biobank hospital episode outlier removal task

## Repository Structure

| File | Description |
|------|-------------|
| `simulation_gd.py` | Synthetic **linear regression** Monte Carlo study with gradient-descent solvers |
| `simulation_gd_ce.py` | Synthetic **logistic regression** (cross-entropy) Monte Carlo study |
| `yelp_data_process.py` | Preprocess Yelp reviews: sample a 200k pool, forget = longest 5% by character length |
| `yelp_estimate.py` | Unlearning experiments on Yelp (bag-of-words star rating prediction) |
| `ukb_data_analysis.py` | EDA for UK Biobank HES inpatient data (`hesin`) |
| `ukb_estimate.py` | Unlearning experiments on UKB 2022 episodes (outlier removal) |

## File Descriptions

### `simulation_gd.py`

- **Retain data:** identity covariance
- **Forget data:** AR(1)-like Toeplitz covariance (`0.3^{|i-j|}`)
- **Estimators:** Pretrain (closed-form OLS), Retrain (GD OLS), Retain-Subsample OLS, **ULS**, and **TL** (transfer-learning-style anchored regularization with CV for λ)
- Reports estimation error

### `simulation_gd_ce.py`

Same experimental design as `simulation_gd.py`, but with **binary classification** and **cross-entropy** loss.

- **Estimators:** CE pretrain / retrain / retain-subsample, **UCE** (unlearning under CE), and **TL**
- Reports estimation error

### `yelp_data_process.py`

1. Load `yelp_academic_dataset_review.json`
2. Sample 200k reviews for training
3. Split by review word count: **retain** = 10th–90th percentile, **forget** = the rest
4. Write train / retain / forget / vocabulary JSONL files (plus a small metadata JSON)

Update `dataset_path` (and output directory) before running.

### `yelp_estimate.py`

Runs unlearning estimation on Yelp review data.

- **Task:** predict `stars` from review `text` with a bag-of-words representation (top 1,500 tokens + intercept)
- **Estimators (default):** Pre-train, Retrain, Retain-Subsample, **ULS**, **TL** (optional ridge CV)
- Evaluates prediction error over multiple repeats and subsample ratios

### `ukb_data_analysis.py`

Exploratory analysis of the UK Biobank HES inpatient table (`hesin.csv`):

- Null / quality / variance reports
- Per-column summaries
- Episode start-date analysis and yearly subsets
- Exports cleaned CSV and summary tables

### `ukb_estimate.py`

Unlearning experiments on the UK Biobank **2022** hospital episode slice.

- **Task:** predict log-transformed episode duration (`epidur`)
- **Features:** mixed OneHot / Target encoding with Lasso-based one-hot feature selection
- **Forget set:** long-duration outliers (`epidur > Q3 + 1.5 × IQR`)
- **Estimators:** Pre-train, Retrain, Retain-Subsample, **ULS**, **TL**

## Datasets

### Yelp Dataset

The Yelp data comes from the [Yelp Open Dataset](https://www.yelp.com/dataset) (formerly the Yelp Academic Dataset).

**How to obtain it:**

1. Visit [https://www.yelp.com/dataset](https://www.yelp.com/dataset).
2. Click **Download Dataset** and agree to the Yelp Dataset License Agreement.
3. Extract the archive. The relevant file is `yelp_academic_dataset_review.json` (JSONL: one JSON object per line).
4. Point `dataset_path` in `yelp_data_process.py` to that file (and set `OUTPUT_DIR` if needed).
5. Run `python yelp_data_process.py` to generate the 200k pool and longest-5% forget split used by `yelp_estimate.py`.

Each review record includes fields such as `review_id`, `user_id`, `business_id`, `stars`, `text`, and `date`. This project uses `stars` as the target and `text` for bag-of-words features. The forget set consists of the longest reviews (by character length) within the sampled pool.


### UK Biobank (UKB) Dataset

The UK Biobank is a large-scale biomedical resource with genetic and health data from about 500,000 UK participants. This project uses the linked **Hospital Episode Statistics (HES) inpatient** table (`hesin`).

**Key fields used:**

| Field | Description |
|-------|-------------|
| `admimeth_uni` | Admission method (unified) |
| `classpat_uni` | Patient classification (unified) |
| `intmanag_uni` | Intended management (unified) |
| `operstat` | Operative status |
| `epitype` | Episode type |
| `tretspef_uni` | Treatment specialty (unified) |
| `epidur` | Episode duration (regression target; log-transformed) |
| `epistart` / `epiend` | Episode start / end dates (used in EDA) |

**How to access:**

UK Biobank data is **not publicly downloadable**. Access requires an approved research application:

1. Register at [https://www.ukbiobank.ac.uk/](https://www.ukbiobank.ac.uk/).
2. Submit a research application describing the intended use.
3. After approval, access data via the Access Management System (AMS) or Research Analysis Platform (RAP).
4. Export the `hesin` table (hospital inpatient data) as CSV for local scripts.

See also the [UK Biobank Data Showcase](https://biobank.ndph.ox.ac.uk/showcase/) and [HES documentation](https://biobank.ndph.ox.ac.uk/showcase/label.cgi?id=2000).

## Dependencies

Python 3.9+ with:

```bash
pip install numpy pandas scikit-learn matplotlib tqdm
```

`scikit-learn` should be recent enough to provide `TargetEncoder` and `OneHotEncoder(..., sparse_output=False)` (roughly ≥ 1.2).

## Notes

- Data and output paths in the scripts are hard-coded absolute paths (e.g. `/data/xiejingyi/...`). Update them for your environment before running.
- These scripts are standalone research experiments, not an installable package.
