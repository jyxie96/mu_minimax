# Efficient machine unlearning with minimax optimality

This repository contains research code for studying **machine unlearning** (post-removal estimation). Given a model pretrained on a full dataset, we compare several approximate unlearning estimators against the retrain from scratch on the retain set only. Experiments span numerical simulations, a Yelp review prediction task, and a UK Biobank hospital episode outlier removal task.

## File Descriptions

### `simulation.py`

Monte Carlo simulation study using synthetic linear regression data. Generates retain data with identity covariance and forget data with Toeplitz covariance, then compares closed-form and gradient-descent-based estimators (OLS, ULS, GradDiff, ULS+) along with confidence interval coverage. Results are saved as CSVs across a grid of sample sizes and dimensions.

### `yelp_data_process.py`

Data preprocessing pipeline for the Yelp Academic Dataset. Reads the raw review JSONL file, samples 200k reviews for training, and splits them into retain (reviews with word counts between the 10th and 90th percentiles) and forget (the rest) subsets. Outputs train, retain, forget, and vocabulary files in JSONL format.

### `yelp_estimate.py`

Runs the unlearning estimation experiments on the Yelp dataset. Builds a bag-of-words feature matrix (top 1,500 words + intercept) to predict star ratings. Compares Pre-train, Retrain, Retain-Subsample, ULS, GradDiff, ULS+, and PRU (Projective Residual Update) estimators over 20 folds, reporting both prediction error and parameter error.

### `ukb_data_analysis.py`

Exploratory data analysis script for the UK Biobank Hospital Episode Statistics (HES) inpatient table (`hesin.csv`). Prints data quality reports, column variance summaries, and date analyses. Exports a cleaned version of the full dataset as well as yearly slices (2021, 2022) and cross-tabulation tables.

### `ukb_estimate.py`

Runs the unlearning estimation experiments on the UK Biobank 2022 hospital episode data. Predicts log-transformed episode duration (`epidur`) using mixed OneHotEncoder and TargetEncoder features, with Lasso-based feature selection. The forget set is defined as long-duration outlier episodes (above Q3 + 1.5×IQR). Compares the same family of estimators as the Yelp experiments and additionally performs asymptotic inference on a specific coefficient (`operstat`).

## Datasets

### Yelp Dataset

The Yelp dataset used in this project comes from the [Yelp Open Dataset](https://www.yelp.com/dataset) (formerly known as the Yelp Academic Dataset).

**How to obtain it:**

1. Visit [https://www.yelp.com/dataset](https://www.yelp.com/dataset).
2. Click **"Download Dataset"** and agree to the Yelp Dataset License Agreement.
3. After downloading, extract the archive. The relevant file is `yelp_academic_dataset_review.json` (a JSONL file with one JSON object per line).
4. Place the file in your data directory and update the `dataset_path` variable in `yelp_data_process.py` to point to it.
5. Run `yelp_data_process.py` to generate the train/retain/forget splits and vocabulary file used by `yelp_estimate.py`.

Alternatively, the Yelp review dataset is also available on Hugging Face: [https://huggingface.co/datasets/Yelp/yelp_review_full](https://huggingface.co/datasets/Yelp/yelp_review_full).

Each line in the review file contains fields such as `review_id`, `user_id`, `business_id`, `stars`, `text`, `date`, etc. This project uses the `stars` (1–5 rating) as the prediction target and `text` as the input for bag-of-words feature extraction.

### UK Biobank (UKB) Dataset

The UK Biobank (UKB) is a large-scale biomedical database containing in-depth genetic and health information from approximately 500,000 participants across the United Kingdom. This project specifically uses the **Hospital Episode Statistics (HES) inpatient** table (`hesin`), which records hospital admission episodes linked to UKB participants.

**Key fields used:**

| Field | Description |
|-------|-------------|
| `eid` | Participant ID |
| `ins_index` | Record instance index |
| `dsource` | Data source |
| `admimeth_uni` | Admission method (unified) |
| `admisorc_uni` | Admission source (unified) |
| `classpat_uni` | Patient classification (unified) |
| `intmanag_uni` | Intended management (unified) |
| `operstat` | Operative status |
| `epitype` | Episode type |
| `tretspef_uni` | Treatment specialty (unified) |
| `epistart` | Episode start date |
| `epiend` | Episode end date |
| `epidur` | Episode duration (target variable, log-transformed for regression) |

**How to access:**

The UK Biobank dataset is **not publicly downloadable**. Access requires an approved research application:

1. Register as a researcher at [https://www.ukbiobank.ac.uk/](https://www.ukbiobank.ac.uk/).
2. Submit a research application describing your intended use.
3. Once approved, data can be accessed through the UK Biobank Access Management System (AMS) or the Research Analysis Platform (RAP).
4. The `hesin` table is part of the linked hospital inpatient data and can be exported as CSV.

For more information, see the [UK Biobank Data Showcase](https://biobank.ndph.ox.ac.uk/showcase/) and the [HES data documentation](https://biobank.ndph.ox.ac.uk/showcase/label.cgi?id=2000).

## Dependencies

This project requires Python 3.9+ and the following packages:

- `numpy`
- `pandas`
- `scikit-learn` 
- `matplotlib`
- `tqdm`

Install with:

```bash
pip install numpy pandas scikit-learn matplotlib tqdm
```

## Notes

- All data paths in the scripts are hard-coded to absolute paths (e.g., `/data/xiejingyi/...`). Update them to match your local environment before running.
- The scripts are designed as standalone research experiments, not as an installable package.
