"""
Streamlit dashboard entry point.

Run from the project root with:
    streamlit run app/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# `streamlit run app/app.py` puts this file's own directory on sys.path, not
# the project root — add both explicitly so `import src...` (project root)
# and plain `import state` / `import styles` (this directory) both resolve,
# regardless of the working directory the command was launched from.
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
for path in (PROJECT_ROOT, APP_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import streamlit as st

import state
import styles
from views import about, counterfactual_view, dashboard, lime_view, performance, predict, shap_view, evaluation, dataset, model_runs

st.set_page_config(
    page_title="XAI Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

styles.inject_css()
state.init_session_state()


def _render_sidebar() -> None:
    st.sidebar.markdown("### ⚙️ XAI Predictive Maintenance")
    st.sidebar.caption("SECOM manufacturing failure prediction")
    st.sidebar.divider()

    from src.model_runs import list_runs
    runs = list_runs()
    labels = {'original': 'Original working model (XGBoost)'}
    labels.update({r['run_id']: f"{r['label']} · {r['run_id']}" for r in runs})
    if state.active_run_id() not in labels:
        state.activate_run('original')
    def change_run():
        state.activate_run(st.session_state.active_run_choice)
    st.sidebar.selectbox('Active model run', options=list(labels), index=list(labels).index(state.active_run_id()),
                         format_func=labels.get, key='active_run_choice', on_change=change_run)
    pipeline = state.get_pipeline()
    if pipeline is not None:
        if st.session_state.decision_threshold is None:
            st.session_state.decision_threshold = pipeline.default_threshold

        st.sidebar.markdown("**Decision threshold**")
        st.session_state.decision_threshold = st.sidebar.slider(
            "Probability above which a prediction counts as FAILURE",
            min_value=0.05, max_value=0.95,
            value=float(st.session_state.decision_threshold), step=0.01,
            label_visibility="collapsed",
        )

        st.sidebar.markdown("**Risk-level thresholds** (demo, configurable)")
        st.sidebar.caption("Illustrative project thresholds — not scientifically validated maintenance standards.")
        low_medium = st.sidebar.slider("Low → Medium", 0.0, 1.0, st.session_state.risk_thresholds["low_medium"], 0.05)
        medium_high = st.sidebar.slider("Medium → High", 0.0, 1.0, st.session_state.risk_thresholds["medium_high"], 0.05)
        high_critical = st.sidebar.slider("High → Critical", 0.0, 1.0, st.session_state.risk_thresholds["high_critical"], 0.05)
        bounds = sorted([low_medium, medium_high, high_critical])
        st.session_state.risk_thresholds = {
            "low_medium": bounds[0], "medium_high": bounds[1], "high_critical": bounds[2],
        }

        if st.session_state.get("selected_raw_row") is not None:
            st.session_state.last_prediction = pipeline.predict(st.session_state.selected_raw_row,
                threshold=st.session_state.decision_threshold, risk_thresholds=st.session_state.risk_thresholds).iloc[0].to_dict()

        st.sidebar.divider()
        meta = pipeline.metadata
        st.sidebar.caption(
            f"**Model:** {meta['model_class']}  \n"
            f"**Trained:** {meta['trained_at']}  \n"
            f"**Features:** {meta['n_selected_features']} of {meta['n_raw_features']}"
        )
    else:
        st.sidebar.warning("No trained model found. Run `python train_pipeline.py` first.")


_render_sidebar()

pages = [
    st.Page(dashboard.render, title="Overview", icon="🏠", url_path="dashboard", default=True),
    st.Page(dataset.render, title="Dataset", icon="🗂️", url_path="dataset"),
    st.Page(predict.render, title="Prediction", icon="🎯", url_path="predict"),
    st.Page(shap_view.render, title="SHAP Explanation", icon="📊", url_path="shap"),
    st.Page(lime_view.render, title="LIME Explanation", icon="🔍", url_path="lime"),
    st.Page(counterfactual_view.render, title="Counterfactual Analysis", icon="🔄", url_path="counterfactual"),
    st.Page(model_runs.render, title="Model Runs", icon="🧠", url_path="models"),
    st.Page(model_runs.comparison, title="Run Comparison", icon="⚖️", url_path="runs"),
    st.Page(performance.render, title="Model Benchmark", icon="📈", url_path="performance"),
    st.Page(evaluation.render, title="XAI Evaluation", icon="🔬", url_path="evaluation"),
    st.Page(evaluation.robustness, title="Robustness", icon="🛡️", url_path="robustness"),
    st.Page(evaluation.analysis, title="Explanation-Guided Analysis", icon="🧪", url_path="analysis"),
    st.Page(evaluation.decision, title="Decision Support", icon="🔧", url_path="decision"),
    st.Page(about.render, title="Technical Details", icon="ℹ️", url_path="about"),
]
navigation = st.navigation(pages)
navigation.run()
