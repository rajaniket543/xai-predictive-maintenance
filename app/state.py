"""
Cached resource/data loaders and session-state helpers shared by every page.

Streamlit reruns the whole script on each interaction, so anything expensive
(loading the model, building the SHAP/LIME/DiCE explainers) is wrapped in
`st.cache_resource`/`st.cache_data` and computed once per session rather than
once per click.
"""

from __future__ import annotations

from typing import Any

import joblib
import pandas as pd
import streamlit as st

from src.explain_dice import DiCEExplainer
from src.explain_lime import LIMEExplainer
from src.explain_shap import SHAPExplainer
from src.prediction import ArtifactNotFoundError, PredictionPipeline
from src.utils import DEFAULT_RISK_THRESHOLDS, MODEL_COMPARISON_PATH, MODELS_DIR, load_json


@st.cache_resource(show_spinner="Loading trained model...")
def get_pipeline() -> PredictionPipeline | None:
    try:
        return PredictionPipeline.load()
    except ArtifactNotFoundError:
        return None


@st.cache_resource(show_spinner="Loading training data used by the explainers...")
def get_training_artifacts() -> tuple[pd.DataFrame, pd.Series]:
    X_train = joblib.load(MODELS_DIR / "X_train_transformed.pkl")
    y_train = joblib.load(MODELS_DIR / "y_train.pkl")
    return X_train, y_train


@st.cache_data(show_spinner="Loading held-out test set...")
def get_test_artifacts() -> tuple[pd.DataFrame, pd.Series]:
    X_test_raw = joblib.load(MODELS_DIR / "X_test_raw.pkl")
    y_test = joblib.load(MODELS_DIR / "y_test.pkl")
    return X_test_raw, y_test


@st.cache_resource(show_spinner="Building SHAP explainer...")
def get_shap_explainer(_pipeline: PredictionPipeline) -> SHAPExplainer:
    X_train, _ = get_training_artifacts()
    return SHAPExplainer(_pipeline.model, X_train, feature_names=_pipeline.selected_feature_names)


@st.cache_resource(show_spinner="Building LIME explainer...")
def get_lime_explainer(_pipeline: PredictionPipeline) -> LIMEExplainer:
    X_train, _ = get_training_artifacts()
    return LIMEExplainer(X_train, feature_names=_pipeline.selected_feature_names)


@st.cache_resource(show_spinner="Building DiCE counterfactual engine...")
def get_dice_explainer(_pipeline: PredictionPipeline) -> DiCEExplainer:
    X_train, y_train = get_training_artifacts()
    return DiCEExplainer(X_train, y_train, _pipeline.model, feature_names=_pipeline.selected_feature_names)


@st.cache_data
def get_model_comparison() -> dict[str, Any]:
    if MODEL_COMPARISON_PATH.exists():
        return load_json(MODEL_COMPARISON_PATH)
    return {}


def init_session_state() -> None:
    defaults = {
        "prediction_history": [],
        "selected_raw_row": None,
        "selected_transformed_row": None,
        "selected_source": None,
        "last_prediction": None,
        "risk_thresholds": dict(DEFAULT_RISK_THRESHOLDS),
        "decision_threshold": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def set_current_instance(raw_row: pd.DataFrame, transformed_row: pd.DataFrame, result: dict[str, Any], source: str) -> None:
    st.session_state.selected_raw_row = raw_row
    st.session_state.selected_transformed_row = transformed_row
    st.session_state.selected_source = source
    st.session_state.last_prediction = result

    entry = {
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": source,
        "instance_id": str(raw_row.index[0]),
        **result,
    }
    st.session_state.prediction_history.append(entry)


def require_pipeline() -> PredictionPipeline:
    """Guard used at the top of every page that needs a trained model."""
    pipeline = get_pipeline()
    if pipeline is None:
        st.error(
            "No trained model was found in `models/`. Train the model first, then reload this page:\n\n"
            "```bash\npython train_pipeline.py\n```"
        )
        st.stop()
    return pipeline
