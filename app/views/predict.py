"""Page 2 — Predict Failure: score a new instance from the dataset, a CSV upload, or manual entry."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import state
from components import disclaimer, empty_state, page_header, render_prediction_result
from src.prediction import InvalidInputError
from src.utils import get_logger

logger = get_logger(__name__)


def _score_and_store(pipeline, raw_row: pd.DataFrame, source: str) -> None:
    try:
        transformed_row = pipeline.transform(raw_row)
        result = pipeline.predict(
            raw_row,
            threshold=st.session_state.decision_threshold,
            risk_thresholds=st.session_state.risk_thresholds,
        ).iloc[0].to_dict()
    except InvalidInputError as exc:
        st.error(f"Could not score this input: {exc}")
        return
    except Exception:
        logger.exception("Unexpected error while scoring an instance (source=%s).", source)
        st.error(
            "Something went wrong while scoring this input. This has been logged — try a "
            "different sample, or check that uploaded data matches the expected SECOM format."
        )
        return
    state.set_current_instance(raw_row, transformed_row, result, source)
    st.rerun()


def _dataset_tab(pipeline) -> None:
    X_test_raw, y_test = state.get_test_artifacts()
    st.caption(
        f"Pick any of the {len(X_test_raw)} held-out test examples (never used for training) to see "
        "how the full pipeline scores a real historical observation."
    )

    filter_choice = st.radio(
        "Filter by true (historical) outcome", ["All", "Known failures only", "Known normal only"],
        horizontal=True, label_visibility="collapsed",
    )
    candidate_idx = X_test_raw.index
    if filter_choice == "Known failures only":
        candidate_idx = y_test[y_test == 1].index
    elif filter_choice == "Known normal only":
        candidate_idx = y_test[y_test == 0].index

    if len(candidate_idx) == 0:
        empty_state("No examples match this filter.")
        return

    col1, col2 = st.columns([3, 1])
    with col1:
        chosen_idx = st.selectbox("Sample index", options=list(candidate_idx))
    with col2:
        st.write("")
        st.write("")
        random_click = st.button("Random sample", use_container_width=True)
    if random_click:
        chosen_idx = candidate_idx.to_series().sample(1, random_state=None).iloc[0]

    true_label = "FAILURE" if y_test.loc[chosen_idx] == 1 else "NORMAL"
    st.caption(f"True historical outcome for this example: **{true_label}** (ground truth, for reference only).")

    if st.button("Predict on this sample", type="primary"):
        raw_row = X_test_raw.loc[[chosen_idx]]
        _score_and_store(pipeline, raw_row, source=f"dataset[{chosen_idx}]")


def _upload_tab(pipeline) -> None:
    st.caption(
        "Upload a CSV with one row per instance and columns matching the raw SECOM sensor "
        f"columns (Feature_0 ... Feature_{len(pipeline.raw_feature_names) - 1})."
    )
    uploaded = st.file_uploader("CSV file", type=["csv"])
    if uploaded is None:
        return
    try:
        df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read this file as CSV: {exc}")
        return

    st.write(f"Loaded {len(df)} row(s).")
    st.dataframe(df.head(5), use_container_width=True)

    row_choice = 0
    if len(df) > 1:
        row_choice = st.number_input("Row to score", min_value=0, max_value=len(df) - 1, value=0, step=1)

    if st.button("Predict on this row", type="primary"):
        raw_row = df.iloc[[row_choice]].copy()
        raw_row.index = [f"upload_row_{row_choice}"]
        _score_and_store(pipeline, raw_row, source="csv_upload")


def _manual_tab(pipeline) -> None:
    X_train, _ = state.get_training_artifacts()
    X_test_raw, _ = state.get_test_artifacts()

    st.caption(
        "Manually adjust the model's most influential sensors (top 12 by SHAP importance). "
        "Every other sensor is held at its training-set median — entering all "
        f"{len(pipeline.raw_feature_names)} raw sensors by hand would not be usable."
    )

    shap_explainer = state.get_shap_explainer(pipeline)
    top_features = shap_explainer.global_importance(X_train.sample(min(200, len(X_train)), random_state=42))
    top_features = top_features["feature"].head(12).tolist()

    template_raw = X_test_raw.median(numeric_only=True).to_frame().T
    template_raw.index = ["manual_entry"]

    cols = st.columns(3)
    manual_values = {}
    for i, feat in enumerate(top_features):
        default = float(template_raw[feat].iloc[0])
        with cols[i % 3]:
            manual_values[feat] = st.number_input(feat, value=default, format="%.4f", key=f"manual_{feat}")

    if st.button("Predict from manual input", type="primary"):
        raw_row = template_raw.copy()
        for feat, val in manual_values.items():
            raw_row[feat] = val
        _score_and_store(pipeline, raw_row, source="manual_entry")


def render() -> None:
    pipeline = state.require_pipeline()
    page_header("Predict Failure", "Score a manufacturing run and get an instant risk assessment.")

    tab1, tab2, tab3 = st.tabs(["Select From Dataset", "Upload CSV", "Manual Entry"])
    with tab1:
        _dataset_tab(pipeline)
    with tab2:
        _upload_tab(pipeline)
    with tab3:
        _manual_tab(pipeline)

    st.divider()

    if st.session_state.get("last_prediction") is not None:
        st.subheader("Latest Prediction")
        render_prediction_result(st.session_state.last_prediction)
        disclaimer(
            "This prediction reflects patterns learned from historical SECOM data. It is a statistical "
            "estimate, not a certified engineering diagnosis — use it to prioritize inspection, not to "
            "replace one."
        )

    if st.session_state.prediction_history:
        st.divider()
        st.subheader("Prediction History (this session)")
        history_df = pd.DataFrame(st.session_state.prediction_history)
        display_cols = [
            "timestamp", "source", "instance_id", "predicted_class",
            "failure_probability", "risk_level", "threshold_used",
        ]
        st.dataframe(history_df[display_cols].iloc[::-1], use_container_width=True, hide_index=True)
        st.download_button(
            "Download history as CSV",
            data=history_df.to_csv(index=False).encode("utf-8"),
            file_name="prediction_history.csv",
            mime="text/csv",
        )
