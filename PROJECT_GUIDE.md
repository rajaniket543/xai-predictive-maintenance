# Project Guide: Explainable AI Predictive Maintenance

## 1. Project at a glance

This repository is an end-to-end research and demonstration system for predicting manufacturing failures from the SECOM semiconductor process dataset. It combines data inspection, leakage-aware preprocessing, model comparison and training, failure-risk scoring, local and global explanations, explanation-quality experiments, and an interactive Streamlit dashboard.

The complete workflow is:

```text
SECOM sensor and label files
  → load and inspect
  → split into training and held-out test data
  → fit preprocessing and feature selection on training data
  → compare imbalance strategies and classifiers
  → choose a model and training-only decision threshold
  → save model and experiment artifacts
  → predict and explain through the Streamlit application
  → evaluate explanations and run controlled what-if experiments
```

This is a demonstration and retrospective evaluation, not a deployed plant control system. Sensor names are anonymous, test data is small, and generated explanations or counterfactuals do not establish physical causes or maintenance actions.

## 2. Dataset and prediction target

The `data/` directory contains the two original SECOM text files:

- `secom.data` contains one manufacturing run per row and 590 whitespace-delimited sensor/process measurements. Missing values are encoded as `NaN`.
- `secom_labels.data` contains the result and timestamp for each corresponding run. Original labels `-1` and `+1` are mapped to `0` (normal/pass) and `1` (failure/fail).

The checked-in dataset has 1,567 runs, including 104 failures. This severe class imbalance makes accuracy alone misleading: a model that always predicts normal would appear accurate while detecting no failures. Timestamps are retained for data understanding but excluded from model features. There are no protected demographic attributes in this dataset, so demographic fairness cannot be evaluated directly.

`src/data_loader.py` reads both files, checks that their row counts match, names measurements `Feature_0` through `Feature_589`, parses timestamps, maps labels, and computes descriptive diagnostics. Its diagnostics include class balance, duplicate feature rows, missingness, constant and near-constant columns, and data types. Loading and diagnostics do not fit transformations.

## 3. Repository layout

### Source code: `src/`

- `data_loader.py` loads SECOM and calculates raw-data diagnostics.
- `utils.py` centralizes project paths, random seed, class labels, risk-level defaults, JSON helpers, and logging.
- `preprocessing.py` defines training-fitted cleaning and scaling steps.
- `feature_selection.py` defines correlation filtering and mutual-information selection.
- `train.py` creates splits and full pipelines, defines candidate estimators, compares imbalance strategies and models, tunes parameters, and ranks candidates.
- `evaluate.py` calculates classification metrics, threshold sweeps, confusion matrices, ROC curves, and precision-recall curves. PR-AUC is average precision.
- `prediction.py` loads the saved artifacts and provides the shared raw-input validation, transformation, probability, class, and risk-level interface used by the dashboard.
- `explain_shap.py` provides tree or linear SHAP explanations, global feature ranking, and per-row contributions.
- `explain_lime.py` creates a local interpretable surrogate around one observation.
- `explain_dice.py` searches for alternative feature vectors that cross a requested model decision threshold.
- `model_runs.py` archives the original model and trains or loads additional saved runs in isolated directories.
- `xai_evaluation.py` evaluates explanation faithfulness, SHAP/LIME agreement, stability, robustness, and confidence summaries on a reproducible sample.

### Application code: `app/`

- `app.py` configures Streamlit, builds the navigation and active-model controls, and dispatches to each page.
- `state.py` manages session state, selected observations, active model runs, and cached model/data/explainer/report loading. The fingerprint in the cache keys prevents switching models from reusing the wrong cached resources. A stale or mismatched XAI report is not displayed as current.
- `components.py` holds shared presentation elements such as headers, metrics, charts, and prediction result widgets.
- `styles.py` applies the shared visual theme.
- `views/` implements the pages described in Section 8.

### Scripts, tests, data, and results

- `train_pipeline.py` runs the original end-to-end benchmark, tuning, final fit, evaluation, explanation examples, and artifact saving.
- `train_runs.py` archives the original model and trains optional additional model runs without replacing the original artifacts.
- `run_experiments.py` evaluates explanations, executes feature-selection/ablation and synthetic-shift experiments, regenerates explanation artifacts, and writes a versioned evaluation report. It can target the original model or a saved run.
- `notebooks/exploratory_analysis.ipynb` contains the retained exploratory notebook.
- `tests/` contains system, model-run, UI, model-switching, and browser smoke checks.
- `outputs/figures/`, `outputs/metrics/`, and `outputs/explanations/` contain generated charts, machine-readable reports, and example explanations.
- `models/` contains the original fitted estimator, fitted preprocessing pipeline, metadata, split artifacts, benchmark comparison, and optional per-run subdirectories.
- `requirements.txt` pins runtime and notebook dependencies. `requirements-dev.txt` adds Playwright for the optional browser check.

## 4. Data preparation and leakage control

The default split is stratified with random seed 42. It contains 1,253 training rows and 314 held-out test rows. Stratification preserves the rare failure fraction approximately in both partitions.

The fitted preprocessing pipeline in `src/preprocessing.py` performs the following steps in order:

1. Drop columns whose missing rate in the training data is greater than 55%.
2. Replace remaining missing values with the training median for each feature.
3. Drop columns that are constant after imputation.
4. Standardize features using training means and standard deviations.

The supervised feature-selection pipeline in `src/feature_selection.py` then removes redundant features when their absolute training correlation is greater than 0.95 and retains up to 40 features ranked by mutual information with the training labels. The model's feature names remain attached so explanations can refer back to the anonymous `Feature_n` inputs.

These steps are composed into the estimator pipeline and refit inside training cross-validation folds. This prevents test-set statistics from affecting imputation, scaling, feature selection, or resampling. The timestamp and target are not included as predictors.

## 5. Training, model selection, and metrics

`train.py` builds candidate models: a dummy baseline, Logistic Regression, Decision Tree, Random Forest, HistGradientBoosting, and XGBoost. The benchmark compares class weighting, SMOTE, and SMOTEENN to address the rare failure class. The saved original model uses SMOTEENN. The candidate pipeline places preprocessing and resampling inside each fold so synthetic/resampled data does not leak into validation folds.

The original training script evaluates candidates with stratified cross-validation, performs randomized hyperparameter searches, and ranks tuned candidates with a weighted score of recall (0.4), F1 (0.3), and average precision (0.3). It then refits the selected estimator on the full training split. The tuned comparison reuses tuning folds and is therefore a selection estimate rather than an independent nested-CV estimate.

Evaluation metrics are:

- **Accuracy:** fraction of all predictions that are correct; can hide failures under class imbalance.
- **Precision:** fraction of predicted failures that are actual failures.
- **Recall:** fraction of actual failures that the model detects.
- **F1:** harmonic mean of precision and recall at the chosen threshold.
- **ROC-AUC:** ranking performance across false-positive and true-positive rates.
- **PR-AUC (average precision):** precision-recall ranking performance, particularly relevant to rare failures.

The deployed decision threshold is selected from five-fold out-of-fold probabilities on training data. The current original XGBoost threshold is 0.47. The held-out test set has already been inspected during project development, so these results are retrospective evidence, not prospective validation. Interactive threshold sliders and test-set sweeps are descriptive tools and must not be treated as a fresh threshold-selection method.

Current original-model test metrics at the saved threshold are recorded in `outputs/metrics/final_evaluation.json` and `models/metadata.json`. The README reports accuracy 0.8248, precision 0.1600, recall 0.3810, F1 0.2254, ROC-AUC 0.7348, and average precision 0.2006. There are 21 test failures; the current model detects 8 and misses 13. These figures describe this split only.

## 6. Inference and prediction contract

`PredictionPipeline` in `src/prediction.py` loads a fitted preprocessor, classifier, and metadata. Before scoring, it validates the raw DataFrame against the saved raw feature names, reports invalid/missing input columns, transforms the input with the saved preprocessing steps, and gets the model's failure probability. It applies the requested threshold to return a normal/failure class and maps probability to configurable low/medium/high/critical risk labels.

The dashboard supports three input paths: selecting an existing held-out test observation, uploading a CSV with the expected `Feature_n` columns, or entering a small set of influential feature values while other inputs are held at their training medians. The selected raw and transformed rows are held in Streamlit session state so explanation pages can use the same observation. Prediction history is session-only and can be downloaded as CSV.

Risk bands are illustrative interface settings, not validated maintenance standards. Model probabilities are not calibrated and should not be read as verified failure frequencies.

## 7. Explainability methods and evaluation

### SHAP

`SHAPExplainer` in `src/explain_shap.py` provides global mean absolute contribution rankings and local signed feature contributions. Positive contributions push the model output toward failure; negative contributions push toward normal. The output unit depends on the estimator: the saved XGBoost explanations are in log-odds, not direct percentage-point changes in probability. SHAP describes model behavior and does not establish physical cause.

### LIME

`LIMEExplainer` in `src/explain_lime.py` samples a neighborhood around a selected row and fits an interpretable local surrogate. It reports feature conditions, signed coefficients, the model output, surrogate output, and a distance-weighted fit score. The surrogate can be a weak approximation, and its unrestricted linear output may fall outside the probability range. The project uses deterministic seeds to make repeated explanations reproducible.

### DiCE counterfactuals

`DiCEExplainer` in `src/explain_dice.py` searches for nearby examples that the active model scores across the requested decision threshold. Search ranges are bounded by training percentiles and can be restricted to selected features. Candidates are independently rescored before they are returned. Search may fail and does not guarantee a globally smallest change. The inputs are standardized sensor values with unknown physical meaning and actionability; a counterfactual is a model what-if, not an instruction to adjust equipment.

### Explanation evaluation

`run_experiments.py` calls `evaluate_explanations` in `src/xai_evaluation.py` on a seeded sample of 30 test rows by default. It records per-sample values, summary values, protocol, seed, dependency versions, and an artifact/code fingerprint in `xai_evaluation.json`. The current report measures:

- **Faithfulness:** probability change after replacing top-SHAP features with training medians, compared with changes from random-feature replacements.
- **LIME fidelity:** the local surrogate's weighted fit score on the neighborhood used to fit it, not a separate held-out neighborhood.
- **Stability and robustness:** similarity of SHAP vectors and feature rankings after adding Gaussian noise at specified fractions of training standard deviation.
- **Consistency:** agreement between SHAP and LIME feature rankings on the same instance.
- **Confidence:** summary of the model's maximum class probability in predefined bins; probabilities are uncalibrated.

The measures represent different properties and are not averaged into a single trust score. Median replacement can make unrealistic combinations of sensors, and synthetic perturbations are controlled sensitivity experiments rather than causal evidence.

### Explanation-guided experiments and decision support

The feature experiment compares an all-cleaned-features baseline, SHAP Top-40, and a model with the five highest-ranked training SHAP features removed. The SHAP ranking is created from training examples; all variants use the same split and fixed threshold. The current results show no general benefit from Top-40 and a substantial recall decrease, so explanations are used to form hypotheses rather than claim performance improvement.

The distribution-shift experiment adds 0.5 training standard deviations to five training-SHAP-ranked features while holding labels fixed. This is a synthetic stress test, not observed production drift. The dashboard's one-feature what-if form likewise changes a model input and reruns prediction; it demonstrates sensitivity and does not estimate the effect of a real maintenance intervention.

## 8. Streamlit dashboard pages

The application is launched with `streamlit run app/app.py`. It has 14 navigation pages:

1. **Overview:** project workflow, headline dataset imbalance, model metrics, and XAI summary indicators.
2. **Dataset:** class balance, missingness, feature diagnostics, and the limitations of anonymous sensor columns.
3. **Prediction:** choose a held-out sample, upload CSV, or enter selected features; display risk, probability, explanations, and session history.
4. **SHAP Explanation:** global importance and local contributions for the currently selected observation.
5. **LIME Explanation:** local surrogate conditions, output, and fidelity information.
6. **Counterfactual Analysis:** request DiCE alternatives and inspect the model-scored changed values.
7. **Model Runs:** archive the original model or train additional supported algorithms with chosen seed, feature count, and imbalance sampler.
8. **Run Comparison:** compare archived runs built on the same dataset and split cohort, including prediction disagreement.
9. **Model Benchmark:** held-out evaluation, confusion matrix, ROC/PR curves, threshold exploration, and the historical baseline comparison.
10. **XAI Evaluation:** inspect measured faithfulness, consistency, confidence, and explanation-quality summaries.
11. **Robustness:** examine how predictions and explanations change under controlled input noise.
12. **Explanation-Guided Analysis:** view feature selection/ablation and synthetic distribution-shift experiments.
13. **Decision Support:** combine a selected prediction, explanation, and model what-if for investigation.
14. **Technical Details:** review methodology, artifacts, and limitations from within the application.

The sidebar selects the active saved run and decision threshold. Activating another run switches the model, preprocessing, train/test artifacts, and explanation resources together; prediction history and selected examples are managed per Streamlit session.

## 9. Saved artifacts and report freshness

The primary `models/` artifacts are:

- `model.pkl`: fitted classifier.
- `preprocessing_pipeline.pkl`: fitted cleaning, selection, imputation, and scaling transforms used before inference.
- `metadata.json`: model identity, feature lists, training information, threshold source, metrics, sampler, and configuration.
- `X_train_transformed.pkl`, `y_train.pkl`: transformed training data and labels used by explainers and experiments.
- `X_test_raw.pkl`, `y_test.pkl`: raw held-out examples and labels.
- `model_comparison.json`: baseline, tuned, final-ranking, and imbalance-strategy results.
- `runs/<run-id>/`: optional self-contained saved model runs with their own artifacts and outputs.

`outputs/metrics/` stores machine-readable metrics and diagnostics, including `final_evaluation.json`, `threshold_oof.csv`, `threshold_analysis.csv`, `dataset_diagnostics.json`, model rankings, and `xai_evaluation.json`. `outputs/figures/` stores EDA and evaluation plots. `outputs/explanations/` stores SHAP/LIME images and text and DiCE JSON/text examples.

The experiment report is associated with a fingerprint of the active model artifacts and relevant source code. If that fingerprint no longer matches, the application treats the report as stale and asks for the experiment to be regenerated. Do not treat old output files as evidence for a changed model.

## 10. How to run the project

### Install and launch the dashboard

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app/app.py
```

Then open `http://localhost:8501`. The repository includes data and fitted artifacts, so training is not required to explore the current dashboard.

### Regenerate explanation and analysis experiments

```bash
python run_experiments.py
python run_experiments.py --samples 100
```

To rerun the six-model baseline benchmark as well:

```bash
python run_experiments.py --refresh-benchmark
```

To target a saved run:

```bash
python run_experiments.py --run <saved-run-id>
```

### Retrain or add a model run

The full tuning workflow can take several minutes or longer depending on the machine:

```bash
python train_pipeline.py
```

To save additional models without overwriting the original model artifacts:

```bash
python train_runs.py --models random_forest hist_gradient_boosting
```

Other options include `--seed`, `--features`, and `--sampler` (`none`, `smote`, or `smoteenn`).

### Tests and browser smoke check

```bash
python -m unittest discover -s tests -v
python tests/smoke_ui.py
python tests/smoke_model_switch.py
```

The optional real-browser check uses Playwright and expects the Streamlit application to be running:

```bash
python -m pip install -r requirements-dev.txt
python tests/browser_smoke.py
```

## 11. Limitations and appropriate interpretation

- Only 21 failures are in the held-out test split, so metric estimates can vary substantially.
- The original model misses some failures and produces false alarms; it is not a certified fault detector.
- The split is random and retrospective rather than temporal or prospective. Development has already inspected the test set.
- Model probabilities are not calibrated; SMOTEENN changes the class distribution seen during model fitting.
- LIME has weak average fidelity in the current measured report. SHAP stability or SHAP/LIME agreement does not prove that an explanation is correct.
- Anonymous feature identifiers do not reveal a sensor's engineering meaning or whether it can be changed safely.
- Feature replacement and noise perturbations can violate real relationships between process variables.
- Counterfactual search bounds individual features but does not establish that a joint feature combination is physically feasible.
- No demographic fairness, causal maintenance benefit, human usefulness, or operational outcome study has been performed.

The intended use is to explore predictive modeling and explainability methods, generate hypotheses, and support further engineering investigation. A qualified domain expert and prospective validation would be needed before operational use.
