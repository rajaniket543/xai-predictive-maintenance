# Explainable AI-Based Predictive Maintenance System for Manufacturing

**Using the SECOM Dataset, SHAP, LIME, and DiCE**

A final-year Computer Science project: an end-to-end machine learning system that predicts
semiconductor manufacturing failures from sensor data and explains every prediction with three
complementary explainable-AI techniques, presented through an interactive Streamlit dashboard.

> All metrics, figures, and example explanations in this repository are generated from the real
> [UCI SECOM dataset](https://archive.ics.uci.edu/dataset/179/secom) by running `train_pipeline.py`
> end to end — nothing is fabricated or hand-typed.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Objectives](#3-objectives)
4. [Dataset](#4-dataset)
5. [Methodology](#5-methodology)
6. [Architecture](#6-architecture)
7. [Preprocessing](#7-preprocessing)
8. [Feature Selection](#8-feature-selection)
9. [Class Imbalance](#9-class-imbalance)
10. [Models](#10-models)
11. [Evaluation](#11-evaluation)
12. [Explainable AI](#12-explainable-ai)
13. [SHAP](#13-shap)
14. [LIME](#14-lime)
15. [DiCE](#15-dice)
16. [Dashboard](#16-dashboard)
17. [Installation](#17-installation)
18. [Usage](#18-usage)
19. [Project Structure](#19-project-structure)
20. [Limitations](#20-limitations)
21. [Future Scope](#21-future-scope)
22. [Conclusion](#22-conclusion)
23. [Testing Checklist](#23-testing-checklist)

---

## 1. Project Overview

Manufacturing lines generate large volumes of sensor/process data. Predictive maintenance uses
this data to flag likely failures *before* they happen, so inspection or intervention can be
scheduled proactively instead of reactively. Black-box predictions, however, are hard for a
process engineer to act on or trust. This project builds a **failure classifier** on the SECOM
semiconductor manufacturing dataset and pairs every prediction with three explanations —
**SHAP**, **LIME**, and **DiCE** — so a user sees not just *what* the model predicts, but *why*,
and *what would need to change*.

## 2. Problem Statement

Given 590 anonymized sensor/process measurements from a semiconductor manufacturing line, predict
whether a given production run will **fail**, and provide a human-understandable explanation for
that prediction. The dataset is small (1,567 examples), high-dimensional, has structured missing
values, and is **severely imbalanced** (~6.6% failures) — all of which must be handled explicitly
and honestly rather than assumed away.

## 3. Objectives

- Build a reproducible, leakage-free ML pipeline from raw SECOM data to a deployed prediction.
- Handle missing data, feature redundancy, and class imbalance with justified, evidence-based
  choices rather than defaults.
- Compare multiple models on metrics appropriate for imbalanced classification (not accuracy alone).
- Explain every prediction with SHAP (global + local attribution), LIME (local surrogate), and
  DiCE (counterfactuals), and be explicit about what each method does and does not prove.
- Present the whole system through a professional, interactive dashboard suitable for a live
  final-year project demonstration.

## 4. Dataset

[SECOM](https://archive.ics.uci.edu/dataset/179/secom) (UCI Machine Learning Repository): 1,567
examples from a real semiconductor manufacturing process, 590 anonymized sensor/process
measurements per example, and a pass/fail label with a timestamp.

**Actual measured characteristics of this dataset** (see `outputs/metrics/dataset_diagnostics.json`
and `notebooks/exploratory_analysis.ipynb` for the full computation):

| Property | Value |
|---|---|
| Rows | 1,567 |
| Raw features | 590 |
| Failures | 104 (6.64%) — imbalance ratio ≈ 14.1 : 1 |
| Duplicate rows | 0 |
| Missing cells | 41,951 (4.54% of all cells) |
| Columns with any missing values | 538 / 590 |
| Columns with >55% missing | 24 |
| Constant (zero-variance) columns | 116 |

### Dataset placement

```text
data/
    secom.data          # feature matrix, whitespace-separated, "NaN" = missing
    secom_labels.data    # "<label> <quoted timestamp>" per line, label ∈ {-1, 1}
```

If these files are not present, `src/data_loader.py` raises a clear `FileNotFoundError` with
download instructions. To obtain them:

```bash
mkdir -p data
curl -o data/secom.data https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data
curl -o data/secom_labels.data https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data
```

Paths are configurable via `src/utils.py: SECOM_DATA_PATH` / `SECOM_LABELS_PATH`, or by passing
explicit paths to `load_secom_dataset(data_path=..., labels_path=...)`.

## 5. Methodology

```text
SECOM raw data (590 sensors)
        │
        ▼
Drop columns >55% missing → median-impute → drop constant columns → standard-scale
        │
        ▼
Correlation filtering (|r| > 0.95) → mutual-information SelectKBest (top 40)
        │
        ▼
Class-imbalance strategy comparison (class-weighting vs. SMOTE vs. SMOTEENN)
        │
        ▼
SMOTEENN oversampling (training folds only)
        │
        ▼
Model comparison (6 algorithms, 5-fold stratified CV) → RandomizedSearchCV tuning (top 3 by PR-AUC)
        │
        ▼
Final model selection (composite score: 0.4·recall + 0.3·F1 + 0.3·PR-AUC)
        │
        ▼
Evaluation (confusion matrix, ROC, PR curve, threshold analysis)
        │
        ▼
SHAP · LIME · DiCE explanations → risk banding → Streamlit dashboard
```

Every step that *learns* from data (imputation medians, scaler mean/std, correlation filter,
mutual-information ranking, SMOTEENN, the classifier itself) is fit **only on the training split**,
inside scikit-learn/imbalanced-learn `Pipeline` objects, and reused unchanged at prediction time.
This is enforced structurally, not just by convention — see [§7](#7-preprocessing).

## 6. Architecture

```text
xai-predictive-maintenance/
├── data/                       # secom.data, secom_labels.data
├── notebooks/
│   └── exploratory_analysis.ipynb
├── src/
│   ├── utils.py                 # paths, config, logging, risk-level logic
│   ├── data_loader.py           # load + diagnose the raw dataset
│   ├── preprocessing.py         # missingness/constant-column drop, impute, scale
│   ├── feature_selection.py     # correlation filter + mutual-information SelectKBest
│   ├── train.py                 # pipeline assembly, model/imbalance comparison, tuning
│   ├── evaluate.py               # metrics, confusion matrix, ROC/PR, threshold analysis
│   ├── prediction.py            # single reusable inference pipeline + validation
│   ├── explain_shap.py          # SHAP global + local explanations
│   ├── explain_lime.py          # LIME local explanations
│   └── explain_dice.py          # DiCE counterfactual explanations
├── models/                      # trained artifacts (generated by train_pipeline.py)
├── outputs/
│   ├── figures/                 # EDA + evaluation plots (.png)
│   ├── metrics/                 # metrics, comparisons (.csv / .json)
│   └── explanations/            # example SHAP/LIME/DiCE outputs
├── app/
│   ├── app.py                   # Streamlit entry point (sidebar + navigation)
│   ├── state.py                 # cached model/data loaders, session state
│   ├── components.py            # reusable UI widgets (gauge, cards, bar charts)
│   ├── styles.py                # CSS + theme constants
│   └── views/                   # one file per dashboard page (7 pages)
├── train_pipeline.py            # orchestrates the full pipeline end to end
├── requirements.txt
└── README.md
```

## 7. Preprocessing

Implemented in `src/preprocessing.py`, as a `Pipeline` (pandas-in/pandas-out via
`set_output(transform="pandas")`, so feature names survive every step for SHAP/LIME/DiCE):

1. **`HighMissingnessDropper`** (threshold 55%, fit on train only) — columns missing this often
   carry too little signal to impute reliably; imputing them would mostly inject a constant.
   Drops 24/590 columns on this dataset.
2. **`SimpleImputer(strategy="median")`** — median is robust to the skew/outliers typical of raw
   sensor data, unlike the mean.
3. **`ConstantColumnDropper`** — a sensor that never changes in the training data carries zero
   predictive information and cannot be meaningfully explained by SHAP/LIME. Drops 116 columns.
4. **`StandardScaler`** — needed for Logistic Regression and for SHAP's additivity assumptions;
   harmless for the tree-based models.

**Data leakage:** all four steps are fit exclusively on the training split (or the training fold,
inside cross-validation) and only ever `.transform()` the test/validation data.

## 8. Feature Selection

Implemented in `src/feature_selection.py`:

1. **`CorrelationFilter`** (|r| > 0.95, fit on train only) — greedily drops one feature from every
   near-duplicate pair. Manufacturing sensor data often has clusters of sensors tracking the same
   underlying process step; keeping every one just splits SHAP importance across near-identical
   features and makes explanations harder to read.
2. **`SelectKBest(mutual_info_classif, k=40)`** — mutual information captures non-linear
   relationships (unlike a plain correlation filter) without needing a fitted model first.

**PCA was deliberately not used.** Its components mix all 590 original sensors into synthetic
axes; every SHAP/LIME/DiCE explanation would then read as "component_7 increased failure risk" —
uninterpretable to a process engineer. Keeping named, original features is what makes the
explanations in this project actionable.

## 9. Class Imbalance

`train_pipeline.py` (Phase 5) empirically compares three strategies via 5-fold stratified CV on a
Random Forest probe model, rather than assuming SMOTE upfront (see
`outputs/metrics/imbalance_strategy_comparison.csv`):

| Strategy | Recall | Precision | F1 | PR-AUC |
|---|---|---|---|---|
| class_weight="balanced" | 0.024 | 0.167 | 0.042 | 0.131 |
| SMOTE | 0.037 | 0.078 | 0.049 | 0.117 |
| **SMOTEENN** | **0.265** | 0.115 | 0.159 | 0.124 |

**SMOTEENN was selected** — it combines SMOTE oversampling with Edited Nearest Neighbours
cleaning, which removes ambiguous majority-class points near the decision boundary. SECOM's
failure class heavily overlaps the normal class in feature space; ENN's cleaning step addresses
that overlap in a way plain oversampling alone does not, visible here as a large recall gain.
`class_weight="balanced"` was evaluated too but not used as the pipeline-wide strategy since not
every candidate model exposes an equivalent parameter (e.g. XGBoost uses `scale_pos_weight`
instead of `class_weight`) — a sampler composes uniformly across every model in the comparison.

Oversampling is applied **only inside the training folds**, via an `imbalanced-learn` `Pipeline`
(which — unlike a plain scikit-learn `Pipeline` — applies its sampler step only during `.fit()`,
never during `.transform()`/`.predict()`), so validation and test data are never resampled.

## 10. Models

Compared in `src/train.py`: a `DummyClassifier` (majority-class) baseline, Logistic Regression,
Decision Tree, Random Forest, HistGradientBoosting, and XGBoost.

**Why recall on the failure class matters more than accuracy here:** with a 6.6% failure rate, a
model that always predicts "normal" already scores ~93% accuracy while catching zero real
failures. In predictive maintenance, a **false negative** (a real failure predicted as normal) is
typically far costlier than a **false positive** (one unnecessary inspection) — a missed failure
can mean scrapped product, unplanned downtime, or a safety incident, while a false alarm only
costs one avoidable inspection. Accuracy alone would reward exactly the wrong behavior here.

**Hyperparameter tuning** (`src/train.py: tune_model`): `RandomizedSearchCV` with `StratifiedKFold`
(5 folds) on the top 3 models by baseline PR-AUC, optimizing **PR-AUC (average precision)** rather
than accuracy or plain ROC-AUC — PR-AUC is far more sensitive to how well the model ranks the rare
positive class, which is exactly what matters for failure detection under this imbalance.

**Final model selection** uses a documented composite score rather than "highest accuracy":

```
composite_score = 0.4 × recall + 0.3 × F1 + 0.3 × PR-AUC
```

deliberately weighting failure recall above raw accuracy or precision, consistent with the
false-negative cost argument above.

### Actual results on this run

Tuned comparison (5-fold stratified CV on the training set; see
`outputs/metrics/final_model_ranking.csv`):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Composite |
|---|---|---|---|---|---|---|---|
| **XGBoost (selected)** | 0.793 | 0.141 | **0.409** | **0.209** | 0.685 | **0.173** | **0.278** |
| HistGradientBoosting | 0.819 | 0.126 | 0.288 | 0.174 | 0.695 | 0.171 | 0.218 |
| Random Forest | 0.779 | 0.100 | 0.288 | 0.147 | 0.667 | 0.133 | 0.199 |

**XGBoost** was selected. Tuned hyperparameters: `max_depth=4, n_estimators=364,
learning_rate=0.068, subsample=0.606, colsample_bytree=0.977` (see `models/metadata.json`).

## 11. Evaluation

Full evaluation on the **held-out test set** (314 examples, never used for training or tuning),
at the F1-optimal decision threshold of **0.61**:

| Metric | Value |
|---|---|
| Accuracy | 0.863 |
| Precision (failure) | 0.211 |
| Recall (failure) | 0.381 |
| F1 (failure) | 0.271 |
| ROC-AUC | 0.735 |
| PR-AUC | 0.201 |

**Confusion matrix:** TN=263, FP=30, FN=13, **TP=8** — of 21 real failures in the test set, the
model correctly flags 8.

**Why PR-AUC over ROC-AUC here:** with only 6.7% of test examples being real failures, ROC-AUC can
look deceptively strong even when the model misses most failures, because the false-positive rate
stays low almost by default when negatives vastly outnumber positives. PR-AUC (0.201, vs. a
0.067 no-skill baseline) is the more honest signal of how well the model ranks true failures
above false alarms.

**Threshold is not assumed to be 0.5.** `src/evaluate.py: threshold_analysis` sweeps thresholds
from 0.05–0.95 and reports precision/recall/F1 at each (`outputs/metrics/threshold_analysis.csv`,
`outputs/figures/threshold_analysis_xgboost.png`, and interactively on the dashboard's Model
Performance page). On this run: F1-optimal threshold = **0.61**; the threshold achieving ≥80%
recall with the best available precision is **0.06** — reflecting a real precision/recall
trade-off: catching 80% of failures on this model means accepting many more false alarms.

These numbers are modest in absolute terms — expected for SECOM, a widely used but genuinely hard
benchmark (noisy, small, high-dimensional, severely imbalanced). They are reported as measured,
not adjusted to look better.

## 12. Explainable AI

Three complementary methods, each answering a different question about the same prediction:

| Method | Question Answered | Scope |
|---|---|---|
| **SHAP** | Which features contributed to this prediction, and by how much? | Global + local |
| **LIME** | Which features locally influenced this specific prediction? | Local only |
| **DiCE** | What would need to change for this prediction to flip? | Local, counterfactual |

**All three describe model behavior, not verified real-world causality.** SHAP and LIME
attributions reflect statistical patterns the model learned — not a proven physical
cause-and-effect relationship between a sensor reading and an actual failure. DiCE counterfactuals
are what-if scenarios produced by search, not guaranteed or physically validated maintenance
instructions. Every explanation surface in the dashboard repeats this disclaimer explicitly.

## 13. SHAP

`src/explain_shap.py`. The explainer type is chosen automatically from the final model:
`TreeExplainer` for tree-based models (used here, since XGBoost was selected — fast and exact),
`LinearExplainer` for Logistic Regression, and a generic permutation-based `Explainer` otherwise.

- **Global**: mean |SHAP value| per feature across a sample of training data → ranks which sensors
  matter most to the model overall (`outputs/explanations/shap_global_importance.png`), plus the
  standard SHAP beeswarm summary plot (`outputs/explanations/shap_summary.png`).
- **Local**: for one instance, the top contributing features with signed SHAP values and plain-
  language direction ("Feature_103 = 1.26 → increases failure probability"), generated only from
  the model's real output — see `outputs/explanations/shap_example_explanation.txt` for an actual
  generated example.

## 14. LIME

`src/explain_lime.py`, via `lime.lime_tabular.LimeTabularExplainer` with
`discretize_continuous=True` (LIME's recommended setting for tabular data — more stable,
human-readable conditions like `Feature_12 <= 0.31` than raw linear coefficients on differently-
scaled sensor readings). For one instance, LIME perturbs the input, observes how the model's
prediction changes, and fits a local linear surrogate — its coefficients are the reported
contributions, valid only in the immediate neighborhood of that one instance (not a global rule).
See `outputs/explanations/lime_example_explanation.txt` / `.png` for a real generated example.

## 15. DiCE

`src/explain_dice.py`, via `dice_ml` with `method="random"` (works with any scikit-learn classifier
exposing `predict_proba`, including tree ensembles — unlike the gradient-based method, which
needs a differentiable model). To keep suggestions realistic:

- **`permitted_range`** bounds every feature to its **1st–99th percentile** of observed training
  values (not raw min/max — a single sensor outlier in the training data would otherwise license
  an unrealistically large suggested change).
- **`features_to_vary`** defaults to a caller-supplied subset (in practice, the top SHAP-ranked
  features) rather than all 40 selected features, keeping counterfactuals sparse.
- Returned counterfactuals are sorted by **number of features changed**, surfacing the sparsest
  (most actionable) alternative first.

See `outputs/explanations/dice_example_explanation.txt` / `.json` for a real generated example —
on this run, a genuine true-positive failure prediction (98% probability) is flipped to a 44%
(non-failure) prediction by changing just 2 of the 40 selected features.

## 16. Dashboard

Streamlit app (`app/app.py`, run with `streamlit run app/app.py`), seven pages via
`st.navigation`:

1. **Dashboard** — project overview, dataset/model headline metrics, current session's prediction summary.
2. **Predict Failure** — score an instance via three input modes (pick a real historical test
   example, upload a CSV, or manually adjust the top SHAP-ranked sensors); shows a probability
   gauge, risk badge, and recommended action; keeps a downloadable session prediction history.
3. **SHAP Explanation** — global importance + beeswarm plot, and local explanation for whatever
   instance was most recently predicted.
4. **LIME Explanation** — local surrogate explanation for the current instance, with an
   adjustable number of features.
5. **Counterfactual Analysis** — DiCE what-if scenarios for the current instance, with sliders for
   how many counterfactuals to generate and how many top-influence features may vary.
6. **Model Performance** — confusion matrix, classification report, ROC/PR curves, an interactive
   threshold-analysis chart, and the full model/imbalance-strategy comparison.
7. **About / Methodology** — the content of this README's methodology section, in-app.

The sidebar exposes the **decision threshold** and the **risk-level thresholds** (Low / Medium /
High / Critical) as live, adjustable controls — explicitly labeled as configurable project demo
thresholds, not scientifically validated maintenance standards, per Phase 14 of the project brief.

## 17. Installation

Requires Python 3.11+.

```bash
git clone <this-repository>
cd xai-predictive-maintenance

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Place `secom.data` and `secom_labels.data` in `data/` (see [§4](#4-dataset) if you need to
download them).

## 18. Usage

```bash
# 1. Run the full training pipeline (data → preprocessing → models → tuning →
#    evaluation → SHAP/LIME/DiCE examples → saved artifacts). Takes ~2 minutes.
python train_pipeline.py

# 2. Launch the dashboard
streamlit run app/app.py
```

Re-running `train_pipeline.py` is fully reproducible (`RANDOM_SEED=42` throughout — splits,
SMOTEENN, cross-validation folds, `RandomizedSearchCV`, and every model's `random_state`) and
regenerates every artifact under `models/` and `outputs/`.

To explore the raw dataset interactively:

```bash
jupyter lab notebooks/exploratory_analysis.ipynb
```

## 19. Project Structure

See [§6 Architecture](#6-architecture) above for the annotated directory tree.

## 20. Limitations

- SECOM is an older benchmark dataset; anonymized feature names (`Feature_103`, ...) do not map to
  known physical sensors, which limits how actionable an explanation can be for a real engineer.
- Severe class imbalance (6.6% failures) makes simultaneously high precision *and* high recall
  hard to achieve on this dataset — visible directly in the reported test metrics (recall 0.381,
  precision 0.211).
- DiCE counterfactuals are statistically plausible, not necessarily physically achievable in a
  real process — they are what-if search results, not engineering instructions.
- SHAP/LIME explain **model behavior**, not necessarily the real physical cause of a failure.
- This is a decision-support prototype, not a certified industrial control system. Real deployment
  would need real-time sensor integration and every recommendation validated by a process
  domain expert before acting on it.
- The dataset represents one specific manufacturing line at one point in time; it does not
  generalize to every manufacturing environment.

## 21. Future Scope

Real-time IoT sensor integration and streaming predictions; edge deployment; automated alerting
integrated with maintenance-management systems; time-series-aware modeling (this project treats
each run independently); deep learning approaches; digital twins; domain-specific causal analysis
conducted jointly with process engineers; automated maintenance scheduling.

## 22. Conclusion

This project demonstrates a complete, honest, reproducible predictive-maintenance pipeline: real
data with real missingness and real imbalance, preprocessing and feature selection chosen for
genuine interpretability (not just accuracy), multiple models compared on failure-relevant
metrics, and every prediction explained three complementary ways with their limitations stated
explicitly rather than hidden. The measured results are modest — consistent with SECOM's known
difficulty — and are reported as measured. The value of the project is in the rigor of the
pipeline and the transparency of the explanations, not in an inflated accuracy number.

## 23. Testing Checklist

- [x] `python train_pipeline.py` runs end to end with no errors, using the real SECOM data (verified: ~128s, final model XGBoost).
- [x] Data-leakage check: every fitted transformer (imputer, scaler, correlation filter, feature selector, sampler) is fit only on training folds.
- [x] `streamlit run app/app.py` launches with no exceptions; all 7 pages verified to render in a real browser (Playwright).
- [x] Predict Failure → SHAP / LIME / Counterfactual pages verified to pick up the same selected instance and render live explanations with no console/page errors.
- [x] SHAP, LIME, and DiCE all verified against the real trained model (not mocked).
- [x] `notebooks/exploratory_analysis.ipynb` executes end to end with 0 cell errors.
- [x] No hard-coded absolute paths — all paths derive from `src/utils.py: PROJECT_ROOT`.
- [x] Missing dataset / missing model artifacts raise clear, actionable errors instead of crashing.
