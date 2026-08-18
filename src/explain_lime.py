"""
LIME (Local Interpretable Model-agnostic Explanations) explainability.

LIME answers: "which features locally influenced this specific prediction?"
It perturbs the input around one instance, observes how the model's
prediction changes, and fits a simple, interpretable (linear) surrogate model
to those local perturbations. The surrogate's coefficients are the reported
"contributions" — a *local approximation* of the model's behavior in the
neighborhood of that one instance, not a global rule and not a causal claim.

Continuous features are discretized into quartile bins (LIME's standard,
recommended setting for tabular data) because it produces more stable,
human-readable conditions (e.g. "Feature_12 <= 0.31") than raw linear
coefficients on unbounded, differently-scaled sensor readings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

from src.utils import CLASS_NAMES, RANDOM_SEED, get_logger

logger = get_logger(__name__)


class LIMEExplainer:
    def __init__(
        self,
        training_data: pd.DataFrame,
        feature_names: list[str] | None = None,
        class_names: tuple[str, str] = (CLASS_NAMES[0], CLASS_NAMES[1]),
        random_state: int = RANDOM_SEED,
    ):
        self.feature_names = feature_names or list(training_data.columns)
        self.class_names = list(class_names)
        self._feature_names_by_length = sorted(self.feature_names, key=len, reverse=True)
        self.explainer = LimeTabularExplainer(
            training_data=training_data.values,
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode="classification",
            discretize_continuous=True,
            random_state=random_state,
        )

    def _match_feature(self, condition: str) -> str:
        for name in self._feature_names_by_length:
            if name in condition:
                return name
        return condition

    def explain_instance(self, model, x_row: pd.DataFrame, num_features: int = 10) -> dict[str, Any]:
        if len(x_row) != 1:
            raise ValueError("explain_instance expects a single-row DataFrame.")

        lime_exp = self.explainer.explain_instance(
            x_row.iloc[0].values,
            model.predict_proba,
            num_features=num_features,
        )
        proba = float(model.predict_proba(x_row)[0, 1])
        predicted_label = int(proba >= 0.5)

        contributors = []
        for condition, weight in lime_exp.as_list():
            feature = self._match_feature(condition)
            contributors.append(
                {
                    "feature": feature,
                    "condition": condition,
                    "contribution": float(weight),
                    "direction": "increases" if weight > 0 else "decreases",
                }
            )

        return {
            "predicted_class": CLASS_NAMES[predicted_label],
            "failure_probability": proba,
            "local_model_r2": float(lime_exp.score) if lime_exp.score is not None else None,
            "contributors": contributors,
            "lime_explanation_object": lime_exp,
        }

    def explain_prediction_text(self, model, x_row: pd.DataFrame, num_features: int = 10) -> str:
        result = self.explain_instance(model, x_row, num_features=num_features)
        lines = [
            f"Prediction: {result['predicted_class']}",
            f"Failure probability: {result['failure_probability']:.1%}",
            "",
            "Important factors (LIME local approximation):",
            "",
        ]
        for c in result["contributors"]:
            lines.append(f"{c['feature']}  ({c['condition']})")
            lines.append(f"Contribution: {c['contribution']:+.4f}")
            lines.append("")
        return "\n".join(lines).rstrip()

    def plot_explanation(self, result: dict[str, Any], save_path: Path, title: str = "LIME Local Explanation") -> Path:
        contributors = sorted(result["contributors"], key=lambda c: c["contribution"])
        features = [c["condition"] for c in contributors]
        weights = [c["contribution"] for c in contributors]
        colors = ["#d62728" if w > 0 else "#1f77b4" for w in weights]

        fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(features))))
        ax.barh(features, weights, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Contribution to failure prediction")
        ax.set_title(
            f"{title}\nPredicted: {result['predicted_class']} "
            f"({result['failure_probability']:.1%} failure probability)",
            fontsize=11,
            fontweight="bold",
        )
        fig.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        return save_path
