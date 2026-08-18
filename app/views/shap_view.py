"""Page 3 — SHAP Explanation: global feature importance + local per-instance attribution."""

from __future__ import annotations

import streamlit as st

import state
from components import contributor_bar_chart, disclaimer, empty_state, page_header
from src.utils import EXPLANATIONS_DIR


def render() -> None:
    pipeline = state.require_pipeline()
    page_header("SHAP Explanation", "Which features contributed to this prediction, and by how much?")

    st.markdown(
        "SHAP (SHapley Additive exPlanations) attributes each prediction to the input features, "
        "based on cooperative game theory. Positive SHAP values push the prediction toward "
        "**FAILURE**; negative values push it toward **NORMAL**. These values describe the "
        "*model's* behavior — not a verified physical cause of failure."
    )

    tab_global, tab_local = st.tabs(["Global Explanation", "Local Explanation (current instance)"])

    with tab_global:
        st.subheader("Global Feature Importance")
        st.caption(
            "Computed on a sample of the training data. Ranks sensors by their average impact "
            "on the model's failure-probability output, across many predictions."
        )
        X_train, _ = state.get_training_artifacts()
        shap_explainer = state.get_shap_explainer(pipeline)
        sample = X_train.sample(min(150, len(X_train)), random_state=42)
        importance_df = shap_explainer.global_importance(sample).head(20)

        fig = contributor_bar_chart(
            importance_df["feature"].tolist(),
            importance_df["mean_abs_shap"].tolist(),
            title="Top 20 Features by Mean |SHAP Value|",
            x_label="Mean |SHAP value|",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        summary_path = EXPLANATIONS_DIR / "shap_summary.png"
        if summary_path.exists():
            st.subheader("SHAP Summary Plot (beeswarm)")
            st.caption(
                "Each point is one training example. Color shows whether that feature's value was "
                "high (red) or low (blue) for that example; position shows the resulting SHAP value."
            )
            st.image(str(summary_path), use_container_width=True)

    with tab_local:
        st.subheader("Local Explanation — Current Instance")
        if st.session_state.get("selected_transformed_row") is None:
            empty_state("No instance selected yet. Go to **Predict Failure** and score an instance first.")
            return

        row = st.session_state.selected_transformed_row
        try:
            shap_explainer = state.get_shap_explainer(pipeline)
            result = shap_explainer.local_explanation(row, top_n=10)
        except Exception:
            st.error("The SHAP explainer could not explain this instance. Try selecting a different sample.")
            return

        st.markdown(
            f"**Prediction:** {result['predicted_class']}  \n"
            f"**Failure probability:** {result['failure_probability']:.1%}  \n"
            f"**Base value (average model output):** {result['base_value']:.3f}"
        )

        contributors = result["top_contributors"]
        fig = contributor_bar_chart(
            [f"{c['feature']} = {c['feature_value']:.3g}" for c in contributors],
            [c["shap_value"] for c in contributors],
            title="Top Contributing Factors (SHAP)",
            x_label="SHAP value (impact on failure probability)",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with st.expander("Plain-language explanation"):
            lines = []
            for c in contributors:
                lines.append(f"- **{c['feature']}** = {c['feature_value']:.3f} → {c['direction']} failure probability (SHAP = {c['shap_value']:+.4f})")
            st.markdown("\n".join(lines))

        disclaimer(
            "SHAP values describe how much each feature moved <em>this model's</em> prediction. "
            "They are not proof of a causal, physical relationship between a sensor reading and an "
            "actual equipment failure."
        )
