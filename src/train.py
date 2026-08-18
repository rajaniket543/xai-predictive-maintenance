"""
Model training: pipeline assembly, imbalance-strategy comparison, multi-model
comparison, and hyperparameter tuning.

Everything that "learns" from data — imputation statistics, scaling,
correlation/mutual-information feature selection, and SMOTE oversampling —
lives inside a single `imblearn.pipeline.Pipeline`. That pipeline is what
gets cross-validated and tuned, which is what keeps every fold leakage-free:
scikit-learn/imbalanced-learn refit each pipeline step from scratch on the
training portion of every fold, and `imblearn`'s Pipeline applies its
sampler step only during `fit`, never during `transform`/`predict` — so the
test fold is never oversampled and never influences preprocessing choices.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import loguniform, randint, uniform
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.tree import DecisionTreeClassifier

from src.feature_selection import build_feature_selection_pipeline
from src.preprocessing import build_preprocessing_pipeline
from src.utils import RANDOM_SEED, TARGET_COLUMN, get_logger

logger = get_logger(__name__)

try:
    from xgboost import XGBClassifier

    _HAS_XGBOOST = True
except ImportError:  # pragma: no cover - environment dependent
    _HAS_XGBOOST = False
    logger.warning("xgboost is not installed; XGBoost will be excluded from model comparison.")

CV_SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    from src.data_loader import get_feature_columns

    X = df[get_feature_columns(df)]
    y = df[TARGET_COLUMN]
    return X, y


def make_train_test_split(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = RANDOM_SEED
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(
        "Train/test split: %d train (%.2f%% failures), %d test (%.2f%% failures).",
        len(X_train),
        100 * y_train.mean(),
        len(X_test),
        100 * y_test.mean(),
    )
    return X_train, X_test, y_train, y_test


def build_preprocessing_and_selection_pipeline(
    missingness_threshold: float = 0.55,
    correlation_threshold: float = 0.95,
    n_features: int = 40,
    random_state: int = RANDOM_SEED,
):
    """The full, leakage-safe cleaning + feature-selection pipeline (no sampler, no classifier)."""
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline(
        steps=[
            ("preprocessing", build_preprocessing_pipeline(missingness_threshold=missingness_threshold)),
            (
                "feature_selection",
                build_feature_selection_pipeline(
                    n_features=n_features,
                    correlation_threshold=correlation_threshold,
                    random_state=random_state,
                ),
            ),
        ]
    )
    pipeline.set_output(transform="pandas")
    return pipeline


def _flatten_pipeline_steps(pipeline) -> list[tuple[str, Any]]:
    """
    imblearn's Pipeline explicitly rejects nested `Pipeline` instances as
    steps ("All intermediate steps of the chain should not be Pipelines").
    `build_preprocessing_and_selection_pipeline` is a plain sklearn Pipeline
    of Pipelines (for standalone/inference reuse), so before handing it to an
    imblearn Pipeline we recursively unwrap it into one flat list of
    (name, transformer) steps. Each transformer object keeps whatever
    `set_output` configuration it was built with — that survives `clone()`.
    """
    from sklearn.pipeline import Pipeline as SkPipeline

    flat: list[tuple[str, Any]] = []
    for name, step in pipeline.steps:
        if isinstance(step, SkPipeline):
            flat.extend(_flatten_pipeline_steps(step))
        else:
            flat.append((name, step))
    return flat


def _make_probe_pipeline(preprocessing_and_selection, sampler, random_state: int) -> ImbPipeline:
    from sklearn.base import clone

    steps = _flatten_pipeline_steps(clone(preprocessing_and_selection))
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("classifier", RandomForestClassifier(random_state=random_state, n_jobs=-1)))
    return ImbPipeline(steps=steps)


def compare_imbalance_strategies(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessing_and_selection,
    cv: StratifiedKFold,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Phase 5: empirically compare imbalance-handling strategies (rather than
    assuming SMOTE is best) using a fixed Random Forest probe model, so the
    final choice used across the rest of the project is evidence-based.
    """
    strategies: dict[str, Any] = {
        "class_weight_balanced": None,
        "smote": SMOTE(random_state=random_state),
        "smoteenn": SMOTEENN(random_state=random_state),
    }

    rows = []
    for name, sampler in strategies.items():
        if name == "class_weight_balanced":
            pipe = _make_probe_pipeline(preprocessing_and_selection, None, random_state)
            pipe.named_steps["classifier"].set_params(class_weight="balanced")
        else:
            pipe = _make_probe_pipeline(preprocessing_and_selection, sampler, random_state)

        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=CV_SCORING, n_jobs=-1, error_score="raise")
        rows.append(
            {
                "strategy": name,
                "accuracy": scores["test_accuracy"].mean(),
                "precision": scores["test_precision"].mean(),
                "recall": scores["test_recall"].mean(),
                "f1": scores["test_f1"].mean(),
                "roc_auc": scores["test_roc_auc"].mean(),
                "pr_auc": scores["test_average_precision"].mean(),
            }
        )
        logger.info("Imbalance strategy '%s': %s", name, rows[-1])

    result = pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    return result


def get_candidate_models(random_state: int = RANDOM_SEED) -> dict[str, Any]:
    models: dict[str, Any] = {
        "dummy_baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=random_state),
        "decision_tree": DecisionTreeClassifier(random_state=random_state),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1),
        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=random_state),
    }
    if _HAS_XGBOOST:
        models["xgboost"] = XGBClassifier(
            random_state=random_state,
            eval_metric="logloss",
            n_jobs=-1,
        )
    return models


def build_full_pipeline(
    preprocessing_and_selection,
    classifier,
    sampler_name: str = "smote",
    random_state: int = RANDOM_SEED,
) -> ImbPipeline:
    from sklearn.base import clone

    sampler = {
        "smote": SMOTE(random_state=random_state),
        "smoteenn": SMOTEENN(random_state=random_state),
        "none": None,
    }[sampler_name]

    steps = _flatten_pipeline_steps(clone(preprocessing_and_selection))
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("classifier", classifier))
    return ImbPipeline(steps=steps)


def extract_fitted_preprocessing_pipeline(fitted_full_pipeline: ImbPipeline):
    """
    Pull the (already-fitted, in place) preprocessing + feature-selection
    steps back out of a fitted `build_full_pipeline` result and re-wrap them
    as a standalone sklearn `Pipeline` — this is what gets saved as
    `preprocessing_pipeline.pkl` and reused at inference time. No refitting
    happens here: each step object already carries the statistics it learned
    from the training fold(s).
    """
    from sklearn.pipeline import Pipeline

    preprocessing_step_names = [
        name for name, _ in fitted_full_pipeline.steps if name not in ("sampler", "classifier")
    ]
    steps = [(name, fitted_full_pipeline.named_steps[name]) for name in preprocessing_step_names]
    pipeline = Pipeline(steps=steps)
    pipeline.set_output(transform="pandas")
    return pipeline


def compare_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessing_and_selection,
    models: dict[str, Any],
    cv: StratifiedKFold,
    sampler_name: str = "smote",
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Phase 6: cross-validated comparison of the baseline + all candidate models."""
    rows = []
    for name, model in models.items():
        # The Dummy baseline should NOT be oversampled: it exists to show
        # what "always predict the majority class" looks like on the actual
        # class distribution, which is exactly the naive failure mode SMOTE
        # is meant to fix for the real models.
        this_sampler = "none" if name == "dummy_baseline" else sampler_name
        pipe = build_full_pipeline(preprocessing_and_selection, model, sampler_name=this_sampler, random_state=random_state)
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=CV_SCORING, n_jobs=-1, error_score="raise")
        row = {
            "model": name,
            "accuracy": scores["test_accuracy"].mean(),
            "precision": scores["test_precision"].mean(),
            "recall": scores["test_recall"].mean(),
            "f1": scores["test_f1"].mean(),
            "roc_auc": scores["test_roc_auc"].mean(),
            "pr_auc": scores["test_average_precision"].mean(),
        }
        rows.append(row)
        logger.info("Model '%s' CV results: %s", name, row)

    return pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)


def get_param_distributions(model_name: str) -> dict[str, Any]:
    distributions = {
        "logistic_regression": {
            "classifier__C": loguniform(1e-3, 1e2),
            "classifier__penalty": ["l2"],
            "classifier__solver": ["lbfgs"],
        },
        "decision_tree": {
            "classifier__max_depth": [3, 5, 7, 10, 15, None],
            "classifier__min_samples_split": randint(2, 20),
            "classifier__min_samples_leaf": randint(1, 10),
            "classifier__criterion": ["gini", "entropy"],
        },
        "random_forest": {
            "classifier__n_estimators": randint(150, 500),
            "classifier__max_depth": [5, 10, 15, 20, None],
            "classifier__min_samples_leaf": randint(1, 10),
            "classifier__max_features": ["sqrt", "log2", None],
        },
        "hist_gradient_boosting": {
            "classifier__max_iter": randint(100, 400),
            "classifier__max_depth": [3, 5, 7, None],
            "classifier__learning_rate": loguniform(0.01, 0.3),
            "classifier__l2_regularization": loguniform(1e-4, 1.0),
        },
        "xgboost": {
            "classifier__n_estimators": randint(100, 400),
            "classifier__max_depth": randint(3, 10),
            "classifier__learning_rate": loguniform(0.01, 0.3),
            "classifier__subsample": uniform(0.6, 0.4),
            "classifier__colsample_bytree": uniform(0.6, 0.4),
        },
    }
    return distributions.get(model_name, {})


def tune_model(
    pipeline: ImbPipeline,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
    scoring: str = "average_precision",
    n_iter: int = 25,
    random_state: int = RANDOM_SEED,
) -> RandomizedSearchCV:
    """
    Phase 7: RandomizedSearchCV over the given pipeline. `scoring` defaults to
    PR-AUC (average precision) rather than accuracy, because with a ~6.6%
    failure rate, accuracy is dominated by the majority class and barely
    moves even when failure-detection quality changes substantially.
    """
    param_distributions = get_param_distributions(model_name)
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )
    search.fit(X_train, y_train)
    logger.info(
        "Tuned '%s': best %s = %.4f, best params = %s",
        model_name,
        scoring,
        search.best_score_,
        search.best_params_,
    )
    return search


SELECTION_WEIGHTS = {"recall": 0.4, "f1": 0.3, "pr_auc": 0.3}


def select_final_model(comparison_df: pd.DataFrame, exclude: tuple[str, ...] = ("dummy_baseline",)) -> tuple[str, pd.DataFrame]:
    """
    Phase 8: rank tuned candidates with a documented composite score rather
    than defaulting to whichever has the highest accuracy (which, under this
    dataset's imbalance, would systematically favor models that under-detect
    failures). The weighting favors recall because, in predictive
    maintenance, a missed failure (false negative) is typically far more
    costly than one extra inspection triggered by a false alarm — while
    still requiring competitive F1/PR-AUC so recall isn't maximized by
    indiscriminately flagging everything as a failure.
    """
    candidates = comparison_df[~comparison_df["model"].isin(exclude)].copy()
    candidates["composite_score"] = (
        SELECTION_WEIGHTS["recall"] * candidates["recall"]
        + SELECTION_WEIGHTS["f1"] * candidates["f1"]
        + SELECTION_WEIGHTS["pr_auc"] * candidates["pr_auc"]
    )
    candidates = candidates.sort_values("composite_score", ascending=False).reset_index(drop=True)
    winner = candidates.loc[0, "model"]
    return winner, candidates
