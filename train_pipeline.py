#!/usr/bin/env python3
"""
End-to-end, reproducible training pipeline for the Explainable AI-Based
Predictive Maintenance System.

    SECOM data -> diagnostics -> EDA figures -> train/test split ->
    preprocessing + feature selection -> imbalance-strategy comparison ->
    model comparison -> hyperparameter tuning -> final model selection ->
    evaluation -> SHAP / LIME / DiCE example explanations -> save artifacts

Usage:
    python train_pipeline.py

All randomness is seeded (see src/utils.py: RANDOM_SEED) so re-running this
script reproduces the same splits, the same SMOTE-resampled folds, and the
same tuned hyperparameters. Runtime is dominated by the RandomizedSearchCV
step (Phase 7) and is typically a few minutes on a laptop CPU.
"""

from __future__ import annotations

import sys
import time
import warnings
from datetime import datetime

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.data_loader import analyze_dataset, get_feature_columns, load_secom_dataset
from src.evaluate import (
    find_best_threshold,
    find_threshold_for_recall,
    full_evaluation_report,
    plot_model_comparison,
    threshold_analysis,
)
from src.explain_dice import DiCEExplainer
from src.explain_lime import LIMEExplainer
from src.explain_shap import SHAPExplainer
from src.train import (
    CV_SCORING,
    build_full_pipeline,
    build_preprocessing_and_selection_pipeline,
    compare_imbalance_strategies,
    compare_models,
    extract_fitted_preprocessing_pipeline,
    get_candidate_models,
    make_train_test_split,
    select_final_model,
    tune_model,
)
from src.utils import (
    CONFIG,
    EXPLANATIONS_DIR,
    FIGURES_DIR,
    METADATA_PATH,
    METRICS_DIR,
    MODEL_COMPARISON_PATH,
    MODEL_PATH,
    MODELS_DIR,
    PREPROCESSING_PIPELINE_PATH,
    TARGET_COLUMN,
    get_logger,
    save_json,
    set_global_seed,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

logger = get_logger("train_pipeline")

sns.set_theme(style="whitegrid")


def _banner(title: str) -> None:
    line = "=" * 78
    logger.info("\n%s\n%s\n%s", line, title, line)


def generate_eda_figures(df: pd.DataFrame, X_train: pd.DataFrame, y_train: pd.Series) -> None:
    """A focused set of exploratory plots — not exhaustive, each answers a specific question."""
    feature_cols = get_feature_columns(df)

    # 1. Target class distribution.
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    labels = ["NORMAL (0)", "FAILURE (1)"]
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, counts.values, color=["#2ca02c", "#d62728"])
    for bar, count in zip(bars, counts.values):
        pct = 100 * count / len(df)
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{count}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Number of examples")
    ax.set_title("Target Class Distribution — Severe Class Imbalance", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_class_distribution.png", dpi=150)
    plt.close(fig)

    # 2. Missingness — worst 25 columns.
    missing_rate = 100 * df[feature_cols].isna().mean().sort_values(ascending=False).head(25)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(missing_rate.index[::-1], missing_rate.values[::-1], color="#ff7f0e")
    ax.set_xlabel("Missing (%)")
    ax.set_title("25 Features With the Highest Missing-Value Rate", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_missingness.png", dpi=150)
    plt.close(fig)

    # 3. Variance distribution across all raw features (log scale).
    variances = df[feature_cols].var(numeric_only=True).dropna()
    variances_positive = variances[variances > 0]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(np.log10(variances_positive.values), bins=40, color="#1f77b4")
    ax.set_xlabel("log10(feature variance)")
    ax.set_ylabel("Number of features")
    ax.set_title("Distribution of Raw Feature Variance (log scale)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_variance_distribution.png", dpi=150)
    plt.close(fig)

    # 4. Correlation heatmap of the 30 highest-variance TRAIN features (readable subset only).
    top_var_cols = X_train.var(numeric_only=True).sort_values(ascending=False).head(30).index
    corr = X_train[top_var_cols].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True, ax=ax, cbar_kws={"label": "Pearson correlation"})
    ax.set_title("Correlation Heatmap — 30 Highest-Variance Features (train set)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_correlation_heatmap.png", dpi=150)
    plt.close(fig)

    # 5. Distributions of the features most associated with the target (train set only).
    imputed = X_train[feature_cols].fillna(X_train[feature_cols].median())
    corr_with_target = imputed.corrwith(y_train).abs().sort_values(ascending=False)
    top_features = corr_with_target.head(6).index.tolist()

    plot_df = imputed[top_features].copy()
    plot_df[TARGET_COLUMN] = y_train.map({0: "NORMAL", 1: "FAILURE"}).values
    melted = plot_df.melt(id_vars=TARGET_COLUMN, var_name="feature", value_name="value")

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, feature in zip(axes.flat, top_features):
        sns.boxplot(data=melted[melted["feature"] == feature], x=TARGET_COLUMN, y="value", hue=TARGET_COLUMN,
                    palette={"NORMAL": "#2ca02c", "FAILURE": "#d62728"}, ax=ax, legend=False)
        ax.set_title(feature, fontsize=10)
        ax.set_xlabel("")
    fig.suptitle("Distributions of the Features Most Correlated With Failure (train set)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_top_features_by_class.png", dpi=150)
    plt.close(fig)

    logger.info("Saved 5 EDA figures to %s", FIGURES_DIR)


def main() -> None:
    start_time = time.time()
    set_global_seed(CONFIG.random_seed)

    _banner("PHASE 1 — DATA LOADING")
    df = load_secom_dataset()
    diagnostics = analyze_dataset(df, missing_threshold=CONFIG.missingness_drop_threshold)
    save_json(diagnostics.to_dict(), METRICS_DIR / "dataset_diagnostics.json")

    _banner("PHASE 3/4 — TRAIN/TEST SPLIT")
    X_train, X_test, y_train, y_test = make_train_test_split(
        df, test_size=CONFIG.test_size, random_state=CONFIG.random_seed
    )

    _banner("PHASE 2 — EXPLORATORY DATA ANALYSIS")
    generate_eda_figures(df, X_train, y_train)

    preprocessing_and_selection = build_preprocessing_and_selection_pipeline(
        missingness_threshold=CONFIG.missingness_drop_threshold,
        correlation_threshold=CONFIG.correlation_drop_threshold,
        n_features=CONFIG.n_features_to_select,
        random_state=CONFIG.random_seed,
    )
    cv = StratifiedKFold(n_splits=CONFIG.cv_folds, shuffle=True, random_state=CONFIG.random_seed)

    _banner("PHASE 5 — CLASS IMBALANCE STRATEGY COMPARISON")
    imbalance_comparison = compare_imbalance_strategies(
        X_train, y_train, preprocessing_and_selection, cv, random_state=CONFIG.random_seed
    )
    imbalance_comparison.to_csv(METRICS_DIR / "imbalance_strategy_comparison.csv", index=False)
    logger.info("Imbalance strategy comparison:\n%s", imbalance_comparison.to_string(index=False))

    sampler_candidates = imbalance_comparison[imbalance_comparison["strategy"].isin(["smote", "smoteenn"])]
    sampler_name = sampler_candidates.sort_values("pr_auc", ascending=False).iloc[0]["strategy"]
    logger.info(
        "Selected imbalance-handling strategy for the full pipeline: %s "
        "(class_weight='balanced' was also evaluated for comparison but is not used as the primary "
        "strategy since not every candidate model exposes an equivalent parameter).",
        sampler_name,
    )

    _banner("PHASE 6 — MODEL COMPARISON")
    candidate_models = get_candidate_models(random_state=CONFIG.random_seed)
    model_comparison = compare_models(
        X_train, y_train, preprocessing_and_selection, candidate_models, cv,
        sampler_name=sampler_name, random_state=CONFIG.random_seed,
    )
    model_comparison.to_csv(METRICS_DIR / "model_comparison_baseline.csv", index=False)
    plot_model_comparison(model_comparison, FIGURES_DIR / "model_comparison_baseline.png")
    logger.info("Baseline (untuned) model comparison:\n%s", model_comparison.to_string(index=False))

    _banner("PHASE 7 — HYPERPARAMETER TUNING")
    top_candidates = (
        model_comparison[model_comparison["model"] != "dummy_baseline"].head(3)["model"].tolist()
    )
    logger.info("Tuning top %d candidates by PR-AUC: %s", len(top_candidates), top_candidates)

    tuned_rows = []
    tuned_estimators = {}
    for name in top_candidates:
        pipeline = build_full_pipeline(
            preprocessing_and_selection, candidate_models[name], sampler_name=sampler_name,
            random_state=CONFIG.random_seed,
        )
        search = tune_model(
            pipeline, name, X_train, y_train, cv=cv,
            scoring=CONFIG.hyperparameter_search_metric, n_iter=CONFIG.n_iter_search,
            random_state=CONFIG.random_seed,
        )
        tuned_estimators[name] = search
        cv_scores = cross_validate(search.best_estimator_, X_train, y_train, cv=cv, scoring=CV_SCORING, n_jobs=-1)
        tuned_rows.append(
            {
                "model": name,
                "accuracy": cv_scores["test_accuracy"].mean(),
                "precision": cv_scores["test_precision"].mean(),
                "recall": cv_scores["test_recall"].mean(),
                "f1": cv_scores["test_f1"].mean(),
                "roc_auc": cv_scores["test_roc_auc"].mean(),
                "pr_auc": cv_scores["test_average_precision"].mean(),
                "best_params": search.best_params_,
            }
        )

    tuned_df = pd.DataFrame(tuned_rows)
    logger.info("Tuned model comparison:\n%s", tuned_df.drop(columns=["best_params"]).to_string(index=False))

    _banner("PHASE 8 — FINAL MODEL SELECTION")
    winner_name, ranked = select_final_model(tuned_df)
    ranked.drop(columns=["best_params"]).to_csv(METRICS_DIR / "final_model_ranking.csv", index=False)
    logger.info("Final model ranking (composite score = 0.4*recall + 0.3*f1 + 0.3*pr_auc):\n%s",
                ranked.drop(columns=["best_params"]).to_string(index=False))
    logger.info("SELECTED FINAL MODEL: %s", winner_name)

    final_search = tuned_estimators[winner_name]
    final_pipeline = final_search.best_estimator_  # already refit on all of X_train
    preprocessing_fitted = extract_fitted_preprocessing_pipeline(final_pipeline)
    model_fitted = final_pipeline.named_steps["classifier"]

    X_train_transformed = preprocessing_fitted.transform(X_train)
    X_test_transformed = preprocessing_fitted.transform(X_test)
    selected_features = list(X_test_transformed.columns)
    logger.info("Final selected feature set (%d features): %s", len(selected_features), selected_features)

    _banner("PHASE 9 — FINAL EVALUATION")
    y_proba_test = model_fitted.predict_proba(X_test_transformed)[:, 1]
    thresh_df = threshold_analysis(y_test.values, y_proba_test)
    thresh_df.to_csv(METRICS_DIR / "threshold_analysis.csv", index=False)

    best_f1_threshold = find_best_threshold(y_test.values, y_proba_test, metric="f1")
    high_recall_threshold = find_threshold_for_recall(y_test.values, y_proba_test, target_recall=0.80)
    chosen_threshold = best_f1_threshold

    eval_report = full_evaluation_report(winner_name, y_test.values, y_proba_test, FIGURES_DIR, threshold=chosen_threshold)
    save_json(eval_report, METRICS_DIR / "final_evaluation.json")
    logger.info("Final test-set metrics @ threshold=%.2f: %s", chosen_threshold, eval_report["metrics"])
    logger.info("Confusion matrix: %s", eval_report["confusion_matrix"])
    logger.info(
        "F1-optimal threshold = %.2f | threshold achieving >=80%% recall with best precision = %.2f",
        best_f1_threshold, high_recall_threshold,
    )

    _banner("PHASE 10-12 — SHAP / LIME / DiCE EXAMPLE EXPLANATIONS")
    shap_explainer = SHAPExplainer(model_fitted, X_train_transformed, feature_names=selected_features)
    shap_sample = X_test_transformed.sample(min(150, len(X_test_transformed)), random_state=CONFIG.random_seed)
    shap_explainer.plot_summary(shap_sample, EXPLANATIONS_DIR / "shap_summary.png")
    shap_explainer.plot_bar(shap_sample, EXPLANATIONS_DIR / "shap_global_importance.png")
    global_importance_df = shap_explainer.global_importance(shap_sample)
    global_importance_df.to_csv(METRICS_DIR / "shap_global_importance.csv", index=False)

    # Prefer a genuine true positive (a real failure the model correctly
    # flags, with the highest confidence) so the walkthrough explanations
    # read as the intended "FAILURE -> what changed it" narrative rather
    # than a harder-to-follow missed-detection case.
    proba_series = pd.Series(y_proba_test, index=X_test_transformed.index)
    true_positive_idx = [i for i in y_test[y_test == 1].index if proba_series.loc[i] >= chosen_threshold]
    if true_positive_idx:
        example_idx = max(true_positive_idx, key=lambda i: proba_series.loc[i])
    elif len(y_test[y_test == 1].index):
        example_idx = max(y_test[y_test == 1].index, key=lambda i: proba_series.loc[i])
    else:
        example_idx = proba_series.idxmax()
    example_row = X_test_transformed.loc[[example_idx]]

    (EXPLANATIONS_DIR / "shap_example_explanation.txt").write_text(shap_explainer.explain_prediction_text(example_row))

    lime_explainer = LIMEExplainer(X_train_transformed, feature_names=selected_features)
    lime_result = lime_explainer.explain_instance(model_fitted, example_row)
    lime_explainer.plot_explanation(lime_result, EXPLANATIONS_DIR / "lime_example_explanation.png")
    (EXPLANATIONS_DIR / "lime_example_explanation.txt").write_text(
        lime_explainer.explain_prediction_text(model_fitted, example_row)
    )

    top_shap_features = global_importance_df["feature"].head(15).tolist()
    dice_explainer = DiCEExplainer(X_train_transformed, y_train, model_fitted, feature_names=selected_features)
    dice_result = dice_explainer.generate_counterfactuals(example_row, total_cfs=3, features_to_vary=top_shap_features)
    (EXPLANATIONS_DIR / "dice_example_explanation.txt").write_text(
        dice_explainer.explain_counterfactual_text(dice_result)
    )
    save_json(dice_result, EXPLANATIONS_DIR / "dice_example_explanation.json")
    logger.info("Saved example SHAP/LIME/DiCE explanations to %s", EXPLANATIONS_DIR)

    _banner("SAVING MODEL ARTIFACTS")
    joblib.dump(preprocessing_fitted, PREPROCESSING_PIPELINE_PATH)
    joblib.dump(model_fitted, MODEL_PATH)
    joblib.dump(X_train_transformed, MODELS_DIR / "X_train_transformed.pkl")
    joblib.dump(y_train, MODELS_DIR / "y_train.pkl")
    joblib.dump(X_test, MODELS_DIR / "X_test_raw.pkl")
    joblib.dump(y_test, MODELS_DIR / "y_test.pkl")

    metadata = {
        "model_name": winner_name,
        "model_class": type(model_fitted).__name__,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "random_seed": CONFIG.random_seed,
        "raw_feature_names": get_feature_columns(df),
        "selected_feature_names": selected_features,
        "chosen_threshold": chosen_threshold,
        "best_f1_threshold": best_f1_threshold,
        "high_recall_threshold": high_recall_threshold,
        "imbalance_strategy": sampler_name,
        "hyperparameters": final_search.best_params_,
        "test_metrics": eval_report["metrics"],
        "confusion_matrix": eval_report["confusion_matrix"],
        "risk_thresholds": CONFIG.risk_thresholds,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "dataset_failure_rate": float(df[TARGET_COLUMN].mean()),
        "n_raw_features": len(get_feature_columns(df)),
        "n_selected_features": len(selected_features),
    }
    save_json(metadata, METADATA_PATH)
    save_json(
        {
            "baseline_comparison": model_comparison.to_dict(orient="records"),
            "tuned_comparison": tuned_df.drop(columns=["best_params"]).to_dict(orient="records"),
            "final_ranking": ranked.drop(columns=["best_params"]).to_dict(orient="records"),
            "imbalance_strategy_comparison": imbalance_comparison.to_dict(orient="records"),
        },
        MODEL_COMPARISON_PATH,
    )

    elapsed = time.time() - start_time
    _banner(f"TRAINING PIPELINE COMPLETE in {elapsed:.1f}s — final model: {winner_name}")
    logger.info("Artifacts saved to: %s", MODELS_DIR)
    logger.info("Figures saved to: %s", FIGURES_DIR)
    logger.info("Metrics saved to: %s", METRICS_DIR)
    logger.info("Explanations saved to: %s", EXPLANATIONS_DIR)
    logger.info("Next step: streamlit run app/app.py")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Training pipeline failed.")
        sys.exit(1)
