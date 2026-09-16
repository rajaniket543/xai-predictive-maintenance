"""
Cached resource/data loaders and session-state helpers shared by every page.

Streamlit reruns the whole script on each interaction, so anything expensive
(loading the model, building the SHAP/LIME/DiCE explainers) is wrapped in
`st.cache_resource`/`st.cache_data` and computed once per session rather than
once per click.
"""

from __future__ import annotations

from typing import Any
from src.xai_evaluation import artifact_fingerprint as compute_fingerprint
from src.model_runs import run_directory
from src.utils import METRICS_DIR

import joblib
import pandas as pd
import streamlit as st

from src.prediction import ArtifactNotFoundError, PredictionPipeline
from src.utils import DEFAULT_RISK_THRESHOLDS, MODEL_COMPARISON_PATH, MODELS_DIR, load_json


@st.cache_resource(show_spinner="Loading trained model...")
def _load_pipeline(token, directory) -> PredictionPipeline | None:
    try:
        return PredictionPipeline.load(directory/'preprocessing_pipeline.pkl', directory/'model.pkl', directory/'metadata.json')
    except Exception as exc:
        st.error(f"Model artifacts could not be loaded: {exc}. Check dependencies or run python train_pipeline.py.")
        return None


@st.cache_resource(show_spinner="Loading training data used by the explainers...")
def _load_training(token, directory) -> tuple[pd.DataFrame, pd.Series]:
    X_train = joblib.load(directory / "X_train_transformed.pkl")
    y_train = joblib.load(directory / "y_train.pkl")
    return X_train, y_train


@st.cache_data(show_spinner="Loading held-out test set...")
def _load_test(token, directory) -> tuple[pd.DataFrame, pd.Series]:
    X_test_raw = joblib.load(directory / "X_test_raw.pkl")
    y_test = joblib.load(directory / "y_test.pkl")
    return X_test_raw, y_test


@st.cache_resource(show_spinner="Building SHAP explainer...")
def _load_shap(token, _pipeline: PredictionPipeline) -> SHAPExplainer:
    X_train, _ = get_training_artifacts()
    from src.explain_shap import SHAPExplainer
    return SHAPExplainer(_pipeline.model, X_train, feature_names=_pipeline.selected_feature_names)


@st.cache_resource(show_spinner="Building LIME explainer...")
def _load_lime(token, _pipeline: PredictionPipeline) -> LIMEExplainer:
    X_train, _ = get_training_artifacts()
    from src.explain_lime import LIMEExplainer
    return LIMEExplainer(X_train, feature_names=_pipeline.selected_feature_names)


@st.cache_resource(show_spinner="Building DiCE counterfactual engine...")
def _load_dice(token, _pipeline: PredictionPipeline) -> DiCEExplainer:
    X_train, y_train = get_training_artifacts()
    from src.explain_dice import DiCEExplainer
    return DiCEExplainer(X_train, y_train, _pipeline.model, feature_names=_pipeline.selected_feature_names)


@st.cache_data
def _load_comparison(token) -> dict[str, Any]:
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
    token = artifact_fingerprint()
    if st.session_state.get("artifact_token") != token:
        for key, value in defaults.items():
            st.session_state[key] = value
        st.session_state.artifact_token = token
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def set_current_instance(raw_row: pd.DataFrame, transformed_row: pd.DataFrame, result: dict[str, Any], source: str) -> None:
    st.session_state.selected_raw_row = raw_row
    st.session_state.selected_transformed_row = transformed_row
    st.session_state.selected_source = source
    st.session_state.last_prediction = result

    entry = {
        "run_id": active_run_id(),
        "model": get_pipeline().metadata["model_name"],
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


def get_pipeline():
    return _load_pipeline(artifact_fingerprint(), active_directory())

def get_training_artifacts():
    return _load_training(artifact_fingerprint(), active_directory())

def get_test_artifacts():
    return _load_test(artifact_fingerprint(), active_directory())

def get_model_comparison():
    return _load_comparison(artifact_fingerprint())

def get_shap_explainer(pipeline):
    return _load_shap(artifact_fingerprint(), pipeline)

def get_lime_explainer(pipeline):
    return _load_lime(artifact_fingerprint(), pipeline)

def get_dice_explainer(pipeline):
    return _load_dice(artifact_fingerprint(), pipeline)

@st.cache_data(show_spinner=False)
def _read_report(token, modified, path):
    report = load_json(path)
    return report if report.get('fingerprint') == token else None

def get_evaluation():
    path = active_outputs() / 'metrics' / 'xai_evaluation.json'
    if not path.exists():
        return None
    try:
        return _read_report(artifact_fingerprint(), path.stat().st_mtime_ns, path)
    except (ValueError, OSError):
        return None

@st.cache_data(show_spinner=False)
def global_importance(token):
    pipeline = get_pipeline()
    train, _ = get_training_artifacts()
    return get_shap_explainer(pipeline).global_importance(train.sample(min(150,len(train)), random_state=42))

@st.cache_data(show_spinner=False)
def local_shap(token, row):
    return get_shap_explainer(get_pipeline()).local_explanation(row, top_n=10)

@st.cache_data(show_spinner=False)
def local_lime(token, row, num_features=10):
    # Fresh per-call explainer avoids mutable RNG shared across sessions.
    train, _ = get_training_artifacts()
    from src.explain_lime import LIMEExplainer
    result = LIMEExplainer(train).explain_instance(get_pipeline().model, row, num_features)
    result.pop('lime_explanation_object', None)
    return result

@st.cache_data(show_spinner=False)
def counterfactuals(token, row, total_cfs, features, threshold):
    train, target = get_training_artifacts()
    from src.explain_dice import DiCEExplainer
    return DiCEExplainer(train, target, get_pipeline().model).generate_counterfactuals(
        row, total_cfs=total_cfs, features_to_vary=features, threshold=threshold)

@st.cache_data(show_spinner=False)
def raw_training_medians(token):
    from src.data_loader import load_secom_dataset
    train, _ = get_training_artifacts()
    pipeline = get_pipeline()
    return load_secom_dataset().loc[train.index, pipeline.raw_feature_names].median().to_frame().T


def active_run_id():
    return st.session_state.get('active_run_id', 'original')


def active_directory():
    return run_directory(active_run_id())


def active_outputs():
    from src.utils import OUTPUTS_DIR
    return OUTPUTS_DIR if active_run_id() == 'original' else active_directory() / 'outputs'


def artifact_fingerprint():
    return compute_fingerprint(active_directory())


def activate_run(run_id):
    pipeline = _load_pipeline(compute_fingerprint(run_directory(run_id)), run_directory(run_id))
    if pipeline is None:
        return
    raw = st.session_state.get('selected_raw_row')
    st.session_state.active_run_id = run_id
    st.session_state.active_run_choice = run_id
    # Drop old feature widgets and model-specific transient results.
    for key in list(st.session_state):
        if key.startswith('manual_') or key in {'intervention_result','prediction_sample_id'}:
            del st.session_state[key]
    init_session_state()
    st.session_state.decision_threshold = pipeline.default_threshold
    if raw is not None:
        set_current_instance(raw, pipeline.transform(raw), pipeline.predict(raw).iloc[0].to_dict(), 'Retained sample after model switch')


def shap_units():
    model_type = type(require_pipeline().model).__name__
    return 'log-odds' if model_type in {'XGBClassifier','HistGradientBoostingClassifier','GradientBoostingClassifier','LogisticRegression'} else 'probability'
