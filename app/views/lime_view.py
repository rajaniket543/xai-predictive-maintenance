"""Page 4 — LIME Explanation: local surrogate-model explanation for the current instance."""

from __future__ import annotations

import streamlit as st

import state
from components import contributor_bar_chart, disclaimer, empty_state, page_header


def render() -> None:
    pipeline = state.require_pipeline()
    page_header("LIME Explanation", "Which features locally influenced this specific prediction?")

    st.markdown(
        "LIME (Local Interpretable Model-agnostic Explanations) perturbs the input around one "
        "instance, observes how the model's prediction changes, and fits a simple linear model to "
        "approximate the model's behavior *in that local neighborhood only*. It is not a global rule."
    )

    if st.session_state.get("selected_transformed_row") is None:
        empty_state("No instance selected yet. Go to **Predict Failure** and score an instance first.")
        return

    row = st.session_state.selected_transformed_row
    num_features = st.slider("Number of features to explain", min_value=5, max_value=20, value=10)

    try:
        lime_explainer = state.get_lime_explainer(pipeline)
        with st.spinner("Computing LIME explanation..."):
            result = lime_explainer.explain_instance(pipeline.model, row, num_features=num_features)
    except Exception:
        st.error("The LIME explainer could not explain this instance. Try selecting a different sample.")
        return

    col1, col2 = st.columns(2)
    col1.metric("Predicted class", result["predicted_class"])
    col2.metric("Failure probability", f"{result['failure_probability']:.1%}")
    if result["local_model_r2"] is not None:
        st.caption(
            f"Local surrogate model fit quality (R²): {result['local_model_r2']:.3f} — how well the "
            "simple local model approximates the real model's behavior near this instance."
        )

    contributors = result["contributors"]
    fig = contributor_bar_chart(
        [c["condition"] for c in contributors],
        [c["contribution"] for c in contributors],
        title="LIME Local Feature Contributions",
        x_label="Contribution to failure prediction",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with st.expander("Plain-language explanation"):
        lines = []
        for c in contributors:
            lines.append(f"- **{c['feature']}** ({c['condition']}) → {c['direction']} failure probability by {abs(c['contribution']):.4f}")
        st.markdown("\n".join(lines))

    disclaimer(
        "LIME's explanation is a <em>local approximation</em> — valid in the immediate neighborhood of "
        "this one instance. Different instances can have different, even contradictory, top LIME "
        "features; that is expected behavior, not an error."
    )
