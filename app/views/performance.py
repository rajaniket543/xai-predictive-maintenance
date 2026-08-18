"""Page 6 — Model Performance: confusion matrix, ROC/PR curves, threshold analysis, model comparison."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

import state
from components import page_header
from src.evaluate import threshold_analysis
from src.utils import CLASS_NAMES


def _confusion_matrix_figure(y_true, y_pred) -> go.Figure:
    cm = confusion_matrix(y_true, y_pred)
    labels = [CLASS_NAMES[0], CLASS_NAMES[1]]
    fig = go.Figure(
        go.Heatmap(
            z=cm,
            x=[f"Predicted {l}" for l in labels],
            y=[f"Actual {l}" for l in labels],
            colorscale="Blues",
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 18},
            showscale=False,
        )
    )
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), yaxis=dict(autorange="reversed"))
    return fig


def render() -> None:
    pipeline = state.require_pipeline()
    meta = pipeline.metadata
    page_header("Model Performance", "Full evaluation of the final model on the held-out test set.")

    X_test_raw, y_test = state.get_test_artifacts()
    y_proba = pipeline.predict_proba(X_test_raw)

    threshold = st.slider(
        "Decision threshold", min_value=0.05, max_value=0.95,
        value=float(st.session_state.decision_threshold or meta["chosen_threshold"]), step=0.01,
        help="The probability above which a prediction is classified as FAILURE.",
    )
    y_pred = (y_proba >= threshold).astype(int)

    tab_summary, tab_curves, tab_threshold, tab_comparison = st.tabs(
        ["Confusion Matrix & Report", "ROC / PR Curves", "Threshold Analysis", "Model Comparison"]
    )

    with tab_summary:
        col1, col2 = st.columns([1, 1.2])
        with col1:
            st.subheader("Confusion Matrix")
            st.plotly_chart(_confusion_matrix_figure(y_test, y_pred), use_container_width=True, config={"displayModeBar": False})
        with col2:
            st.subheader("Classification Report")
            report = classification_report(
                y_test, y_pred, target_names=[CLASS_NAMES[0], CLASS_NAMES[1]], output_dict=True, zero_division=0
            )
            report_df = pd.DataFrame(report).T.round(3)
            st.dataframe(report_df, use_container_width=True)
        st.caption(
            "In predictive maintenance, a **false negative** (a real failure predicted as normal) is "
            "typically far costlier than a **false positive** (an unnecessary inspection) — a missed "
            "failure can mean scrapped product, unplanned downtime, or safety risk, while a false alarm "
            "only costs one avoidable inspection. This is why recall on the FAILURE class is tracked "
            "explicitly rather than relying on accuracy alone."
        )

    with tab_curves:
        col1, col2 = st.columns(2)
        with col1:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            from sklearn.metrics import roc_auc_score

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"Model (AUC={roc_auc_score(y_test, y_proba):.3f})", line=dict(color="#2563EB", width=3)))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(color="#94A3B8", dash="dash")))
            fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                               template="plotly_white", height=400, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        with col2:
            from sklearn.metrics import average_precision_score

            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            baseline = float(y_test.mean())
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=f"Model (PR-AUC={average_precision_score(y_test, y_proba):.3f})", line=dict(color="#DC2626", width=3)))
            fig.add_hline(y=baseline, line_dash="dash", line_color="#94A3B8", annotation_text=f"Baseline ({baseline:.3f})")
            fig.update_layout(title="Precision-Recall Curve", xaxis_title="Recall", yaxis_title="Precision",
                               template="plotly_white", height=400, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.caption(
            "PR-AUC is the more informative curve here: with only "
            f"{baseline:.1%} of examples being real failures, ROC-AUC can look deceptively strong even "
            "when the model finds few true failures, because the false-positive rate stays low almost "
            "by default."
        )

    with tab_threshold:
        st.subheader("Precision / Recall / F1 vs. Decision Threshold")
        thresh_df = threshold_analysis(y_test.values, y_proba)
        fig = go.Figure()
        for col, color in [("precision", "#2563EB"), ("recall", "#DC2626"), ("f1", "#16A34A")]:
            fig.add_trace(go.Scatter(x=thresh_df["threshold"], y=thresh_df[col], mode="lines", name=col.capitalize(), line=dict(color=color, width=2.5)))
        fig.add_vline(x=threshold, line_dash="dash", line_color="#0F172A", annotation_text=f"Current = {threshold:.2f}")
        fig.update_layout(template="plotly_white", xaxis_title="Decision threshold", yaxis_title="Score",
                           height=420, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.caption(
            f"The default 0.5 threshold is not assumed optimal. F1-optimal threshold on the test set: "
            f"**{meta['best_f1_threshold']:.2f}**. Threshold achieving ≥80% recall with the best "
            f"available precision: **{meta['high_recall_threshold']:.2f}** — use the slider above to "
            "explore the trade-off between missed failures and false alarms directly."
        )

    with tab_comparison:
        comparison = state.get_model_comparison()
        if not comparison:
            st.info("Model comparison data not found.")
        else:
            st.subheader("Tuned Model Comparison")
            tuned_df = pd.DataFrame(comparison.get("tuned_comparison", []))
            if not tuned_df.empty:
                st.dataframe(tuned_df.round(3), use_container_width=True, hide_index=True)
                metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
                melted = tuned_df.melt(id_vars="model", value_vars=metric_cols, var_name="metric", value_name="score")
                fig = go.Figure()
                for model_name in tuned_df["model"]:
                    sub = melted[melted["model"] == model_name]
                    fig.add_trace(go.Bar(x=sub["metric"], y=sub["score"], name=model_name))
                fig.update_layout(barmode="group", template="plotly_white", height=420,
                                   yaxis=dict(range=[0, 1]), margin=dict(l=10, r=10, t=20, b=10))
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            st.subheader("Class-Imbalance Strategy Comparison")
            imbalance_df = pd.DataFrame(comparison.get("imbalance_strategy_comparison", []))
            if not imbalance_df.empty:
                st.dataframe(imbalance_df.round(3), use_container_width=True, hide_index=True)
                st.caption(
                    f"**{meta['imbalance_strategy']}** was selected for the final pipeline based on this "
                    "comparison (evaluated via 5-fold stratified cross-validation on the training set only)."
                )
