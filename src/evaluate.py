"""
Model evaluation: point metrics, confusion matrix, ROC/PR curves, and
decision-threshold analysis.

Static (matplotlib/seaborn) figures are produced here and saved under
`outputs/figures/` for the README/report and notebook. The Streamlit
dashboard re-renders the same underlying numbers with Plotly for
interactivity — this module is the single source of truth for the numbers
either way.

Why PR-AUC and threshold analysis matter here specifically: with a ~6.6%
failure rate, a model predicting "no failure" for every sample already scores
~93% accuracy while catching zero real failures. ROC-AUC can also look
deceptively good under heavy imbalance because the false-positive rate stays
low even with many missed failures. Precision-Recall AUC and an explicit
sweep over the decision threshold are what actually reveal how well the
model finds the rare failure class.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.utils import CLASS_NAMES, get_logger

logger = get_logger(__name__)

sns.set_theme(style="whitegrid")
_FIGSIZE = (7, 5)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    """Core scalar metrics for one model at one operating threshold."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
    }


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, save_path: Path, title: str = "Confusion Matrix"
) -> Path:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=[CLASS_NAMES[0], CLASS_NAMES[1]]
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_roc_curve(
    y_true: np.ndarray, y_proba: np.ndarray, save_path: Path, model_name: str = "Model"
) -> Path:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.plot(fpr, tpr, color="#1f77b4", linewidth=2, label=f"{model_name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1, label="Random classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_precision_recall_curve(
    y_true: np.ndarray, y_proba: np.ndarray, save_path: Path, model_name: str = "Model"
) -> Path:
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    baseline = float(np.mean(y_true))

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.plot(recall, precision, color="#d62728", linewidth=2, label=f"{model_name} (PR-AUC = {ap:.3f})")
    ax.axhline(baseline, color="grey", linestyle="--", linewidth=1, label=f"Baseline (failure rate = {baseline:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def threshold_analysis(y_true: np.ndarray, y_proba: np.ndarray, n_steps: int = 91) -> pd.DataFrame:
    """
    Sweep the decision threshold from 0.05 to 0.95 and recompute
    precision/recall/F1/accuracy at each point. 0.5 is the sklearn default
    but is not guaranteed to be a good operating point under class imbalance
    — this table is what lets us choose a threshold deliberately instead.
    """
    thresholds = np.linspace(0.05, 0.95, n_steps)
    rows = []
    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        rows.append(
            {
                "threshold": t,
                "precision": precision_score(y_true, y_pred_t, zero_division=0),
                "recall": recall_score(y_true, y_pred_t, zero_division=0),
                "f1": f1_score(y_true, y_pred_t, zero_division=0),
                "accuracy": accuracy_score(y_true, y_pred_t),
            }
        )
    return pd.DataFrame(rows)


def find_best_threshold(y_true: np.ndarray, y_proba: np.ndarray, metric: str = "f1") -> float:
    df = threshold_analysis(y_true, y_proba)
    best_row = df.loc[df[metric].idxmax()]
    return float(best_row["threshold"])


def find_threshold_for_recall(
    y_true: np.ndarray, y_proba: np.ndarray, target_recall: float = 0.80
) -> float:
    """Lowest threshold (i.e. most conservative on missed failures) achieving at least `target_recall`."""
    df = threshold_analysis(y_true, y_proba, n_steps=181)
    candidates = df[df["recall"] >= target_recall]
    if candidates.empty:
        logger.warning("No threshold achieves recall >= %.2f; returning threshold with max recall.", target_recall)
        return float(df.loc[df["recall"].idxmax(), "threshold"])
    # Among thresholds hitting the recall target, prefer the one with best precision.
    best_row = candidates.loc[candidates["precision"].idxmax()]
    return float(best_row["threshold"])


def plot_threshold_analysis(df: pd.DataFrame, save_path: Path, chosen_threshold: float | None = None) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["threshold"], df["precision"], label="Precision", linewidth=2)
    ax.plot(df["threshold"], df["recall"], label="Recall", linewidth=2)
    ax.plot(df["threshold"], df["f1"], label="F1-score", linewidth=2)
    if chosen_threshold is not None:
        ax.axvline(chosen_threshold, color="black", linestyle="--", linewidth=1, label=f"Chosen threshold = {chosen_threshold:.2f}")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold Analysis: Precision / Recall / F1 vs. Decision Threshold", fontsize=12, fontweight="bold")
    ax.legend(loc="best")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_model_comparison(comparison_df: pd.DataFrame, save_path: Path) -> Path:
    """Grouped bar chart comparing models across the key metrics."""
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    melted = comparison_df.melt(id_vars="model", value_vars=metrics, var_name="metric", value_name="score")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=melted, x="metric", y="score", hue="model", ax=ax)
    ax.set_title("Model Comparison Across Evaluation Metrics", fontsize=13, fontweight="bold")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def full_evaluation_report(
    model_name: str,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    figures_dir: Path,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Run the complete Phase-9 evaluation for one model at one threshold and save all figures."""
    y_pred = (y_proba >= threshold).astype(int)
    metrics = compute_metrics(y_true, y_pred, y_proba)
    report_dict = classification_report(
        y_true, y_pred, target_names=[CLASS_NAMES[0], CLASS_NAMES[1]], output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    slug = model_name.lower().replace(" ", "_")
    plot_confusion_matrix(y_true, y_pred, figures_dir / f"confusion_matrix_{slug}.png", title=f"Confusion Matrix — {model_name}")
    plot_roc_curve(y_true, y_proba, figures_dir / f"roc_curve_{slug}.png", model_name=model_name)
    plot_precision_recall_curve(y_true, y_proba, figures_dir / f"pr_curve_{slug}.png", model_name=model_name)

    thresh_df = threshold_analysis(y_true, y_proba)
    plot_threshold_analysis(thresh_df, figures_dir / f"threshold_analysis_{slug}.png", chosen_threshold=threshold)

    return {
        "model_name": model_name,
        "threshold": threshold,
        "metrics": metrics,
        "classification_report": report_dict,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
