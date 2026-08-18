"""Page 5 — Counterfactual Analysis: DiCE what-if scenarios for the current instance."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import state
from components import class_badge_html, disclaimer, empty_state, page_header


def render() -> None:
    pipeline = state.require_pipeline()
    page_header("Counterfactual Analysis (DiCE)", "What would need to change for this prediction to flip?")

    st.markdown(
        "DiCE (Diverse Counterfactual Explanations) searches for the smallest realistic change to the "
        "input that would flip the model's prediction to the opposite class. These are **what-if "
        "scenarios**, not guaranteed or physically validated maintenance instructions."
    )

    if st.session_state.get("selected_transformed_row") is None:
        empty_state("No instance selected yet. Go to **Predict Failure** and score an instance first.")
        return

    row = st.session_state.selected_transformed_row

    col1, col2 = st.columns(2)
    with col1:
        total_cfs = st.slider("Number of counterfactuals to generate", 1, 5, 3)
    with col2:
        n_mutable = st.slider("Number of top-influence features allowed to vary", 5, 40, 15)

    X_train, _ = state.get_training_artifacts()
    shap_explainer = state.get_shap_explainer(pipeline)
    top_features = shap_explainer.global_importance(
        X_train.sample(min(150, len(X_train)), random_state=42)
    )["feature"].head(n_mutable).tolist()

    try:
        dice_explainer = state.get_dice_explainer(pipeline)
        with st.spinner("Searching for counterfactuals..."):
            result = dice_explainer.generate_counterfactuals(row, total_cfs=total_cfs, features_to_vary=top_features)
    except Exception:
        st.error(
            "The counterfactual engine (DiCE) is currently unavailable for this instance. "
            "Try a different sample or a different number of mutable features."
        )
        return

    st.markdown(
        f"**Current prediction:** {class_badge_html(result['original_class'])} "
        f"&nbsp;&nbsp;({result['original_probability']:.1%} failure probability)",
        unsafe_allow_html=True,
    )

    if not result["counterfactuals"]:
        st.warning(
            "No valid counterfactual was found within the permitted feature ranges and the allowed "
            "number of mutable features. Try increasing the number of features allowed to vary."
        )
        return

    st.divider()
    for i, cf in enumerate(result["counterfactuals"]):
        st.markdown(f"#### Counterfactual {i + 1} — changes {cf['n_features_changed']} feature(s)")
        st.markdown(
            f"{class_badge_html(cf['predicted_class'])} &nbsp;&nbsp;({cf['failure_probability']:.1%} failure probability)",
            unsafe_allow_html=True,
        )
        if cf["changes"]:
            changes_df = pd.DataFrame(cf["changes"])
            changes_df = changes_df.rename(
                columns={
                    "feature": "Feature",
                    "original_value": "Original Value",
                    "counterfactual_value": "Suggested Value",
                    "delta": "Change",
                }
            )
            st.dataframe(
                changes_df.style.format({"Original Value": "{:.3f}", "Suggested Value": "{:.3f}", "Change": "{:+.3f}"}),
                use_container_width=True,
                hide_index=True,
            )
        st.markdown("")

    disclaimer(
        "Counterfactuals are <em>what-if</em> scenarios produced by searching for nearby, statistically "
        "plausible inputs the model would classify differently. They are not guaranteed physical "
        "maintenance instructions — proposed changes should be reviewed by a domain expert before "
        "acting on them."
    )
