"""
SHAP (SHapley Additive exPlanations) explainability.

SHAP answers: "which features contributed to this prediction, and by how
much?" It attributes a prediction's deviation from the average prediction
(the base value) to each input feature, in units of model output (here, log-
odds/probability contribution depending on explainer), based on cooperative
game theory (Shapley values). It describes what the *model* is doing — not a
verified causal, physical relationship between a sensor reading and an actual
machine failure.

The explainer type is chosen automatically based on the final model:
tree-based models (Random Forest, Decision Tree, HistGradientBoosting,
XGBoost) use the fast, exact `TreeExplainer`; linear models use
`LinearExplainer`; anything else falls back to the general-purpose,
model-agnostic `Explainer` (permutation-based).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.utils import CLASS_NAMES, RANDOM_SEED, get_logger

logger = get_logger(__name__)

_TREE_MODEL_NAMES = {
    "RandomForestClassifier",
    "DecisionTreeClassifier",
    "HistGradientBoostingClassifier",
    "GradientBoostingClassifier",
    "XGBClassifier",
    "ExtraTreesClassifier",
}
_LINEAR_MODEL_NAMES = {"LogisticRegression"}


def _positive_class_matrix(values: np.ndarray) -> np.ndarray:
    """
    Normalize SHAP output to a plain (n_samples, n_features) array of SHAP
    values for the positive (failure) class, regardless of whether the
    underlying explainer returned a 2D array (single-output) or a 3D array
    (n_samples, n_features, n_classes).
    """
    arr = np.asarray(values)
    if arr.ndim == 3:
        # Last axis indexes classes; class 1 = failure.
        class_axis = arr.shape[-1]
        return arr[..., min(1, class_axis - 1)]
    return arr


def _positive_class_base_value(base_values: Any) -> float:
    arr = np.asarray(base_values)
    if arr.ndim == 0:
        return float(arr)
    if arr.ndim == 1:
        return float(arr[min(1, arr.shape[0] - 1)]) if arr.shape[0] > 1 else float(arr[0])
    # (n_samples, n_classes) or similar — take the mean over samples of class 1.
    if arr.ndim == 2 and arr.shape[-1] > 1:
        return float(arr[:, 1].mean())
    return float(np.asarray(arr).mean())


class SHAPExplainer:
    """Wraps a fitted classifier + background data with the appropriate SHAP explainer."""

    def __init__(self, model, background_data: pd.DataFrame, feature_names: list[str] | None = None):
        self.model = model
        self.feature_names = feature_names or list(background_data.columns)
        self.background_data = background_data
        self.model_type = type(model).__name__
        self.explainer = self._build_explainer(model, background_data)

    def _build_explainer(self, model, background_data: pd.DataFrame):
        if self.model_type in _TREE_MODEL_NAMES:
            logger.info("Using shap.TreeExplainer for %s.", self.model_type)
            return shap.TreeExplainer(model)
        if self.model_type in _LINEAR_MODEL_NAMES:
            logger.info("Using shap.LinearExplainer for %s.", self.model_type)
            sample = shap.sample(background_data, min(200, len(background_data)), random_state=RANDOM_SEED)
            return shap.LinearExplainer(model, sample)
        logger.info("Using generic shap.Explainer (model-agnostic) for %s.", self.model_type)
        sample = shap.sample(background_data, min(100, len(background_data)), random_state=RANDOM_SEED)
        return shap.Explainer(model.predict_proba, sample)

    def explain(self, X: pd.DataFrame) -> shap.Explanation:
        explanation = self.explainer(X)
        return explanation

    def global_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        explanation = self.explain(X)
        values = _positive_class_matrix(explanation.values)
        mean_abs = np.abs(values).mean(axis=0)
        df = pd.DataFrame({"feature": self.feature_names, "mean_abs_shap": mean_abs})
        return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    def plot_summary(self, X: pd.DataFrame, save_path: Path, max_display: int = 20) -> Path:
        explanation = self.explain(X)
        values = _positive_class_matrix(explanation.values)
        fig = plt.figure(figsize=(9, 7))
        shap.summary_plot(values, X, feature_names=self.feature_names, max_display=max_display, show=False)
        plt.title("SHAP Summary Plot — Impact of Each Feature on Failure Prediction", fontsize=12, fontweight="bold")
        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def plot_bar(self, X: pd.DataFrame, save_path: Path, max_display: int = 20) -> Path:
        importance_df = self.global_importance(X).head(max_display)
        fig, ax = plt.subplots(figsize=(9, max(4, 0.3 * len(importance_df))))
        ax.barh(importance_df["feature"][::-1], importance_df["mean_abs_shap"][::-1], color="#1f77b4")
        ax.set_xlabel("Mean |SHAP value| (average impact on model output)")
        ax.set_title("SHAP Global Feature Importance", fontsize=12, fontweight="bold")
        fig.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        return save_path

    def local_explanation(self, x_row: pd.DataFrame, top_n: int = 5) -> dict[str, Any]:
        """Explain a single-row DataFrame prediction. Returns raw numbers — no invented values."""
        if len(x_row) != 1:
            raise ValueError("local_explanation expects a single-row DataFrame.")

        explanation = self.explain(x_row)
        values = _positive_class_matrix(explanation.values)[0]
        base_value = _positive_class_base_value(explanation.base_values)
        proba = float(self.model.predict_proba(x_row)[0, 1])

        order = np.argsort(-np.abs(values))
        contributors = []
        for idx in order[:top_n]:
            contributors.append(
                {
                    "feature": self.feature_names[idx],
                    "feature_value": float(x_row.iloc[0, idx]),
                    "shap_value": float(values[idx]),
                    "direction": "increases" if values[idx] > 0 else "decreases",
                }
            )

        return {
            "predicted_class": CLASS_NAMES[int(proba >= 0.5)],
            "failure_probability": proba,
            "base_value": base_value,
            "top_contributors": contributors,
            "all_shap_values": {self.feature_names[i]: float(values[i]) for i in range(len(values))},
        }

    def explain_prediction_text(self, x_row: pd.DataFrame, top_n: int = 5) -> str:
        result = self.local_explanation(x_row, top_n=top_n)
        lines = [
            f"Prediction: {result['predicted_class']}",
            f"Failure probability: {result['failure_probability']:.1%}",
            "",
            "Top contributing factors (SHAP):",
        ]
        for i, c in enumerate(result["top_contributors"], start=1):
            lines.append(
                f"{i}. {c['feature']} = {c['feature_value']:.3f} -> "
                f"{c['direction']} failure probability (SHAP contribution = {c['shap_value']:+.4f})"
            )
        return "\n".join(lines)
