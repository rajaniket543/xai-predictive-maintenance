# Predictive Maintenance XAI

Explainable AI for trustworthy manufacturing failure prediction.

**Data → Model → Prediction → Explanation → Explanation evaluation → Explanation-guided analysis → Re-evaluation → Decision support**

This project uses the included SECOM manufacturing dataset, the existing scikit-learn/imbalanced-learn training pipeline, an XGBoost classifier, optional saved model runs, SHAP, LIME, DiCE, and a 14-page Streamlit analytics application. All dashboard results come from executed models or versioned experiment reports. No fairness, human usefulness, causal benefit, or maintenance-impact study is invented.

## Run the application

Python 3.12 was used for verification. From this repository:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app/app.py
```

Open http://localhost:8501. Dataset, fitted models, and experiment results are included; training is not required to demonstrate them.

To regenerate the XAI experiments (30 randomly sampled test cases by default):

```bash
python run_experiments.py
# Also re-execute all six baseline models using five training folds:
python run_experiments.py --refresh-benchmark
# Increase the explanation-evaluation sample size:
python run_experiments.py --samples 100
```

To retrain and tune the existing model pipeline from scratch:

```bash
python train_pipeline.py
python run_experiments.py
```

To save additional comparable model runs without replacing the original fitted model:

```bash
python train_runs.py --models random_forest hist_gradient_boosting
python run_experiments.py --run <saved-run-id>
```

The full tuning run is more expensive than the experiment runner. Neither duration nor improvement is guaranteed. The runner also corrects legacy test-selected thresholds using training-only out-of-fold predictions, without replacing the fitted classifier. Saved runs use the same raw train/test IDs as the original model and store their own model, preprocessing, metrics, predictions, and optional XAI reports under `models/runs/`. Changing data, source, model artifacts, or the runner invalidates the saved XAI report; the UI asks for regeneration rather than displaying stale results.

## What the repository contains

- `src/data_loader.py`: parses raw whitespace-delimited sensors, missing values, outcome labels, timestamps, and diagnostics.
- `src/preprocessing.py`: training-fitted missingness filter (>55%), median imputation, constant removal, and standardization.
- `src/feature_selection.py`: correlation filtering and mutual-information Top-40 selection for the existing deployed model.
- `src/train.py`, `train_pipeline.py`: six-model benchmark, imbalance strategies, tuning, training-only threshold selection, evaluation, artifact generation.
- `src/prediction.py`: shared raw-input validation, transformation, probability, and threshold logic.
- `src/explain_shap.py`, `src/explain_lime.py`, `src/explain_dice.py`: existing explainers, now with deterministic LIME and independently checked counterfactuals.
- **New:** `src/model_runs.py`: immutable saved model runs, original-model snapshots, common split signatures, and per-run predictions.
- **New:** `src/xai_evaluation.py`: reproducible faithfulness, similarity, stability, consistency, confidence, and robustness evaluation.
- **New:** `run_experiments.py`: reuses existing components for feature-selection/ablation retraining and synthetic distribution shift; writes actual evidence.
- **New:** `train_runs.py`: command-line helper for saving additional model runs without overwriting the active original artifacts.
- `app/app.py`, `app/state.py`, `app/components.py`, `app/styles.py`: navigation, artifact-aware caches, shared widgets, professional light theme.
- `app/views/`: existing overview, prediction, benchmark, SHAP, LIME, counterfactual, and technical pages; **new** `dataset.py`, `evaluation.py`, and `model_runs.py` provide dataset, evaluation, robustness, guided analysis, decision support, model run training, and run comparison.
- `.streamlit/config.toml`: consistent app theme and disabled usage telemetry.
- `models/`: fitted classifier/preprocessor, training/test artifacts, metadata and benchmark records.
- `outputs/metrics/xai_evaluation.json`: sample-level evidence, summaries, protocols, seed, timestamp, package versions, and artifact/code fingerprint.
- `outputs/metrics/threshold_oof.csv`: training-only out-of-fold probabilities used for the corrected threshold.
- `outputs/figures/`, `outputs/explanations/`: real generated charts and explanation examples.
- `notebooks/exploratory_analysis.ipynb`: original exploratory notebook, retained.
- **New:** `tests/test_system.py`, `tests/test_model_runs.py`, `tests/smoke_ui.py`, `tests/smoke_model_switch.py`, `tests/browser_smoke.py`: numerical integrity, saved-run isolation, all-page rendering, model-switch rendering, and browser demo checks.

The original architecture and fitted classifier were preserved. Existing UI files, explanation wrappers, training threshold selection, model metadata, output reports, and documentation were updated. No separate prediction pipeline, authentication service, or database was introduced.

## Data and modeling

The included dataset contains 1,567 runs, 590 anonymous sensors, 104 failures, and 41,951 missing cells. Raw labels -1/+1 map to normal=0/failure=1. No protected demographic fields exist. Timestamps are not prediction features.

The original seed-42 stratified split contains 1,253 training and 314 test examples. All learned preprocessing and resampling stay inside training folds. The deployed classifier uses 40 mutual-information-selected features. Benchmark candidates are Dummy, Logistic Regression, Decision Tree, Random Forest, HistGradientBoosting, and XGBoost. The original pipeline compares class weighting, SMOTE, and SMOTEENN, then chooses between the two sampler strategies for consistent application across models. The current model uses SMOTEENN.

PR-AUC means **average precision**, not trapezoidal PR area. Baseline benchmark metrics are five-fold training-CV averages. The historical tuned-model table reuses tuning folds, so it is a selection estimate, not nested-CV evidence. Hyperparameters were selected using training data; the final test set does not select features or thresholds.

**Validity correction:** the earlier implementation chose the threshold by maximizing test F1. It now uses five-fold training-only out-of-fold predictions, with the entire preprocessing/sampling/model pipeline refit in each fold. The corrected threshold is **0.47**. Interactive test threshold sweeps are descriptive and should not be used to choose a deployment threshold. The test set has already been inspected during project development, so results are retrospective benchmark evidence, not a fresh prospective validation.

**Saved model runs:** the **Model Runs** page can archive the original fitted model and train additional untuned candidates, while **Run Comparison** compares only runs from the same dataset and split cohort. Activating a run is session-local and switches the shared prediction, SHAP, LIME, DiCE, and evaluation loaders together. XGBoost, HistGradientBoosting, and Logistic Regression SHAP values are shown in log-odds; Random Forest and Decision Tree SHAP values are shown in probability units.

## Actual executed results

Current XGBoost held-out test results, at threshold 0.47:

| Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC (AP) |
|---:|---:|---:|---:|---:|---:|
| 0.8248 | 0.1600 | 0.3810 | 0.2254 | 0.7348 | 0.2006 |

These values are recorded in `models/metadata.json` and `outputs/metrics/final_evaluation.json`; UI values are loaded, never hardcoded. Recall remains modest: the model detects 8 of 21 test failures and misses 13.

XAI experiment means across 30 random test samples:

| Dimension | Actual value | Meaning |
|---|---:|---|
| Faithfulness | 0.3004 | Excess absolute probability change over random-feature controls |
| LIME fidelity | 0.1442 | Distance-weighted neighborhood training R²; weak surrogate fit |
| Stability | 0.9848 | SHAP cosine at 1% of training SD noise |
| Consistency | 0.4779 | SHAP/LIME top-5 Jaccard similarity |
| Robustness | 0.8994 | SHAP cosine at 10% of training SD noise |

These measures have different meanings and are **not** averaged into a trust score. Strong SHAP stability does not establish model accuracy, causality, or operational reliability.

## Experimental definitions

**SHAP:** TreeSHAP for XGBoost, global training-sample importance and beeswarm, local signed contributions. XGBoost SHAP values are **log-odds**, not percentage-point probability changes. Base value plus all contributions reconstructs the model margin; applying the sigmoid reconstructs probability. The explainer also supports other existing tree and linear model families.

**LIME fidelity:** discretized local conditions and signed coefficients; model probability compared with the linear surrogate output at the selected sample. Reported R² is LIME's distance-weighted score on the neighborhood used to fit it. This is not independently held-out fidelity. Unconstrained linear surrogate outputs may exceed [0,1]. Each explanation uses the same seed so repeated UI visits do not change the neighborhood.

**Faithfulness:** for each sample, replace the top five absolute-SHAP features with their training medians. Compare the absolute probability change with the mean absolute probability change across 30 random sets of five features:

```text
faithfulness = |p(original) − p(top-5 replaced)|
               − mean_random |p(original) − p(random-5 replaced)|
```

Negative values are retained. The random-control mean probability and mean absolute change are different quantities and displayed separately. Median replacement can break sensor correlations and create off-distribution inputs; this is a model sensitivity experiment, not causal evidence.

**Stability / robustness:** add Gaussian noise scaled by training SD at 0%, 1%, 5%, and 10%. Reuse the same noise direction across magnitudes; do not clip (the 0% control is exact). Measure signed-vector cosine, absolute-value Spearman ranking correlation, top-5 overlap, probability change, and fraction of unchanged predicted classes. Model robustness and explanation robustness have separate charts. One noise draw per sample and a small sample size limit the inference.

**Consistency:** compare SHAP and LIME on the same sample using top-5 overlap and Jaccard, plus absolute ranking correlation. LIME-unselected coefficients are zero-padded and create ranking ties. The two methods use different units, neighborhoods, and approximation mechanisms; disagreement is shown rather than hidden.

**Confidence:** max(p,1−p), uncalibrated, grouped into predeclared low [0.5,0.7), medium [0.7,0.9), and high [0.9,1] bins. Current bin counts are 2, 4, and 24. Such small groups do not establish a general confidence–reliability relationship; empty bins are not fabricated.

**Feature selection and ablation:** reuse the existing preprocessing, sampler, and classifier settings. Fit an all-cleaned-feature baseline (450 sensors; no correlation/MI reduction), rank SHAP on 150 **training** cases, then retrain with fixed Top-40 or with the Top-5 removed. Evaluate each at a fixed threshold of 0.5 on the same held-out split. Test outcomes do not choose K or the removed features.

| Configuration | Features | F1 | Recall | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| All cleaned features | 450 | 0.2037 | 0.5238 | 0.6683 | 0.1232 |
| SHAP Top-40 | 40 | 0.2041 | 0.2381 | 0.6551 | 0.1242 |
| Remove SHAP Top-5 | 445 | 0.1593 | 0.4286 | 0.6390 | 0.1214 |

Top-40 does **not** provide a general performance improvement: recall drops substantially. Ablation reveals dependence, not proof that a feature is spurious. “Potentially spurious dependence” remains an investigation hypothesis. These exploratory configurations are separate from the deployed MI-selected model.

**Distribution shift:** add +0.5 training SD to five training-SHAP-ranked features. Hold outcomes fixed as an explicit stress-test assumption. Compare test metrics, mean confidence, SHAP similarity, and feature importance. This is a **controlled distribution-shift experiment**, not observed production drift, and the unchanged-label assumption is not physically verified.

**DiCE / intervention:** random-search counterfactuals use training 1st–99th percentile bounds for varying features. Every returned candidate is independently re-scored and must cross the active UI decision threshold. DiCE may internally use a stricter search threshold; unsuccessful searches are reported honestly. Search does not guarantee a globally minimal change. Values are standardized model inputs. Anonymous sensors have **unknown physical actionability**; none is identified as a validated adjustable actuator. The live what-if form modifies one influential feature, reruns the model, and shows the probability difference without claiming a causal maintenance benefit.

## Presentation sequence

1. **Overview:** introduce the workflow, class imbalance, actual model metrics, and XAI KPIs.
2. **Dataset:** show missingness and anonymous sensors; explain why accuracy and physical actionability need care.
3. **Model Runs:** inspect saved runs, activate one, or train a new candidate without overwriting the original model.
4. **Run Comparison:** compare saved runs from the same split and inspect disagreement on identical held-out samples.
5. **Model Benchmark:** open **Model Comparison** for the six baseline models, then inspect confusion matrix and PR curve.
6. **Prediction:** select sample **238** and click **Predict on this sample**. It has about 99.8% predicted failure probability in the current original model. Uploaded CSV and manual-entry modes also remain available.
7. **SHAP Explanation:** show global importance, then local contributions for the same sample. Explain positive/negative direction and the active model's SHAP units.
8. **LIME Explanation:** show local coefficients, surrogate probability, and actual R²; acknowledge weak fidelity when present.
9. **XAI Evaluation → Fidelity & consistency:** show the current sample's SHAP/LIME table and top-5 Jaccard agreement.
10. **Counterfactual Analysis:** inspect original → changed feature values → independently verified new prediction. Explicitly state physical actionability is unknown.
11. **XAI Evaluation → Faithfulness:** compare influential-feature changes with random controls, including negative sample results.
12. **Robustness:** show 0/1/5/10% noise curves and original vs perturbed SHAP vectors.
13. **Explanation-Guided Analysis:** compare 450 features, SHAP Top-40, and Top-5 ablation; discuss recall loss and controlled shift.
14. **XAI Evaluation → Evaluation framework:** summarize measured evidence and unmeasured dimensions.
15. **Decision Support:** choose `Feature_103`, keep its training-median replacement, click **Apply what-if and re-run prediction**. Current sample probability changes from about 99.8% to 96.5%; that alone does not flip the label. Expand perturbation validation and DiCE alternatives if desired.
16. **Technical Details:** explain the protocol, threshold correction, saved-run isolation, and limitations.

## Verification

```bash
python -m unittest discover -s tests -v
python tests/smoke_ui.py
python tests/smoke_model_switch.py
# Optional real browser check, with Google Chrome installed and app running:
python -m pip install -r requirements-dev.txt
python tests/browser_smoke.py
```

Verified on the included dataset and real artifacts: numerical/integration tests; all-page renders and live intervention/validation/DiCE buttons; saved-run switching across prediction, SHAP, LIME, and evaluation loaders; application startup and browser navigation through the demo. Browser screenshots are under `outputs/figures/*_browser.png`. Baseline six-model CV, threshold correction, feature retraining, SHAP, LIME, DiCE, saved model runs, and all new experiments were executed. The complete historical hyperparameter-search pipeline and notebook were inspected but were not rerun as part of this extension.

## Known limitations

- **Protected demographic fairness could not be directly evaluated because the dataset does not provide appropriate protected-group attributes.** No anonymous sensor grouping is called demographic fairness.
- Real maintenance benefit, human usefulness, causality, and physical actionability have not been evaluated.
- The failure classifier is imperfect and its probabilities are not calibrated; SMOTEENN changes the training distribution.
- The test set contains only 21 failures. Single-split estimates and 30-sample explanation summaries have no reported confidence intervals or statistical significance claim.
- Random stratified splitting is retrospective, not temporal or prospective validation. The dataset has already been inspected during development.
- LIME fidelity is weak on average and measured on its fitting neighborhood. High explanation similarity alone does not establish correctness.
- Perturbations can violate real sensor relationships. Counterfactual search may fail, and marginal bounds do not establish joint physical feasibility.
- Retaining original sensor names preserves feature identity but does not reveal physical meanings or make those sensors actionable.
- Missing/stale reports are explicitly marked; missing artifacts require training, and absent packages require installing the pinned requirements. Pickle artifacts should only be loaded from this trusted project.
