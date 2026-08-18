"""Page 1 — Dashboard: project overview, dataset summary, model headline metrics."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

import state
from components import page_header, render_prediction_result
from styles import metric_card


def render() -> None:
    pipeline = state.require_pipeline()
    meta = pipeline.metadata

    page_header(
        "Explainable AI-Based Predictive Maintenance",
        "SECOM manufacturing dataset — failure prediction with SHAP, LIME, and DiCE explanations.",
    )

    n_total = meta["n_train"] + meta["n_test"]
    failure_rate = meta["dataset_failure_rate"]
    metrics = meta["test_metrics"]

    cols = st.columns(4)
    cards = [
        ("Total Samples", f"{n_total:,}", f"{meta['n_train']} train / {meta['n_test']} test"),
        ("Failure Rate", f"{failure_rate:.1%}", "Severely imbalanced dataset"),
        ("Model", meta["model_name"].replace("_", " ").title(), f"{meta['n_selected_features']} features used"),
        ("Test Recall (Failure)", f"{metrics['recall']:.1%}", f"F1 = {metrics['f1']:.2f}"),
    ]
    for col, (label, value, sub) in zip(cols, cards):
        with col:
            st.markdown(metric_card(label, value, sub), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([1.3, 1])
    with left:
        st.subheader("Model Performance Snapshot (held-out test set)")
        metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
        fig = go.Figure(
            go.Bar(
                x=[m.upper().replace("_", "-") for m in metric_names],
                y=[metrics[m] for m in metric_names],
                marker_color="#2563EB",
                text=[f"{metrics[m]:.2f}" for m in metric_names],
                textposition="outside",
            )
        )
        fig.update_layout(
            template="plotly_white",
            yaxis=dict(range=[0, 1.05], title="Score"),
            height=340,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.caption(
            f"Final model: **{meta['model_class']}**, selected via a composite score weighting "
            "failure recall, F1, and PR-AUC (see the Model Performance and About pages for the full comparison)."
        )

    with right:
        st.subheader("Current Prediction Summary")
        if st.session_state.get("last_prediction") is None:
            st.info("No prediction has been made yet this session. Go to **Predict Failure** to score an instance.")
        else:
            render_prediction_result(st.session_state.last_prediction)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("About This Project")
    st.markdown(
        """
This dashboard predicts whether a semiconductor manufacturing run is likely
to **fail** using the [UCI SECOM dataset](https://archive.ics.uci.edu/dataset/179/secom)
(1,567 examples, 590 anonymized sensor/process measurements), and explains
every prediction with three complementary XAI techniques:

- **SHAP** — global and local feature attributions grounded in cooperative game theory.
- **LIME** — a local, interpretable surrogate model around one prediction.
- **DiCE** — counterfactual "what-if" scenarios showing what would need to change.

Use the sidebar to navigate between predicting new instances, inspecting
explanations, reviewing model performance, and reading the full methodology.
        """
    )
