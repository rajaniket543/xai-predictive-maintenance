"""Page 7 — About / Methodology."""

from __future__ import annotations

import streamlit as st

import state
from components import page_header


def render() -> None:
    page_header("About / Methodology", "Dataset, pipeline, models, and the limits of this project.")

    pipeline = state.get_pipeline()
    meta = pipeline.metadata if pipeline else None

    st.markdown(
        """
### Dataset

The [SECOM dataset](https://archive.ics.uci.edu/dataset/179/secom) (UCI Machine
Learning Repository) contains 1,567 examples from a real semiconductor
manufacturing process. Each example has 590 anonymized sensor/process
measurements and a pass/fail label. Only 6.6% of examples are failures, and
about 4.5% of all cells are missing — both handled explicitly in the pipeline
below.

### End-to-End Pipeline
        """
    )

    st.markdown(
        """
```
SECOM raw data (590 sensors)
        │
        ▼
Drop columns >55% missing → median-impute → drop constant columns → standard-scale
        │
        ▼
Correlation filtering (|r| > 0.95) → mutual-information SelectKBest (top 40)
        │
        ▼
SMOTEENN oversampling (training folds only)
        │
        ▼
Model comparison (6 algorithms) → RandomizedSearchCV tuning (top 3 by PR-AUC)
        │
        ▼
Final model selection (recall / F1 / PR-AUC composite score)
        │
        ▼
Evaluation (confusion matrix, ROC, PR curve, threshold analysis)
        │
        ▼
SHAP · LIME · DiCE explanations → risk banding → dashboard
```
        """
    )

    st.markdown(
        """
### Preprocessing Decisions

- **High-missingness columns dropped** (>55% missing in training data): at that
  point there is too little signal to impute reliably.
- **Median imputation** for remaining missing values — robust to skew/outliers.
- **Constant columns dropped**: a sensor that never changes carries no signal
  and cannot be explained by SHAP/LIME.
- **Standard scaling** after imputation.
- All statistics above are fit **only on the training split**, inside a single
  scikit-learn `Pipeline`, and reused unchanged at prediction time — this is
  what prevents data leakage.

### Feature Selection

Correlation filtering removes near-duplicate sensors, then mutual-information
`SelectKBest` keeps the top 40 most informative features. **PCA was
deliberately not used** — its components mix all 590 original sensors
together, which would make every SHAP/LIME/DiCE explanation refer to an
uninterpretable synthetic axis instead of a named sensor.

### Class Imbalance

Class-weighting, SMOTE, and SMOTEENN were empirically compared via 5-fold
stratified cross-validation (see **Model Performance → Model Comparison**)
before picking a strategy — rather than assuming one upfront. SMOTE/SMOTEENN
are applied **only inside the training folds**, never to validation or test
data, using an `imbalanced-learn` Pipeline.

### Models Compared

Dummy (majority-class) baseline, Logistic Regression, Decision Tree, Random
Forest, HistGradientBoosting, and XGBoost. The final model was chosen with a
documented composite score — **0.4 × recall + 0.3 × F1 + 0.3 × PR-AUC** —
that deliberately weights failure recall higher than raw accuracy, since a
missed failure is typically far more costly than one extra false-alarm
inspection.

### Explainable AI

| Method | Question answered | Scope |
|---|---|---|
| **SHAP** | Which features contributed to this prediction, and by how much? | Global + local |
| **LIME** | Which features locally influenced this specific prediction? | Local only |
| **DiCE** | What would need to change for this prediction to flip? | Local, counterfactual |

All three describe **model behavior**, not verified real-world causality.
SHAP/LIME attributions and DiCE counterfactuals are hypotheses for a domain
expert to evaluate, not proof of a physical cause-and-effect relationship.

### Risk Bands

Risk thresholds (Low / Medium / High / Critical) are **illustrative, editable
project thresholds** — adjustable from the sidebar — not scientifically
validated maintenance standards.
        """
    )

    if meta:
        with st.expander("Current model configuration"):
            st.json(
                {
                    "model": meta["model_name"],
                    "hyperparameters": meta["hyperparameters"],
                    "imbalance_strategy": meta["imbalance_strategy"],
                    "n_raw_features": meta["n_raw_features"],
                    "n_selected_features": meta["n_selected_features"],
                    "chosen_threshold": meta["chosen_threshold"],
                    "trained_at": meta["trained_at"],
                }
            )

    st.markdown(
        """
### Limitations

- SECOM is an older benchmark dataset; anonymized feature names do not map to
  known physical sensors, which limits how actionable explanations can be.
- Severe class imbalance (6.6% failures) makes high precision *and* high
  recall simultaneously hard to achieve — this is visible in the test metrics.
- DiCE counterfactuals may not always be physically achievable in a real
  process, even when they are statistically plausible.
- SHAP/LIME explain the model, not necessarily the real physical cause of a
  failure.
- This system is a decision-support prototype, not a certified industrial
  control system — real deployment would require real-time sensor
  integration and domain-expert validation of every recommendation.

### Future Scope

Real-time IoT sensor integration, streaming predictions, edge deployment,
automated alerting, integration with maintenance-management systems,
time-series-aware modeling, deep learning, digital twins, and domain-specific
causal analysis validated by process engineers.
        """
    )
