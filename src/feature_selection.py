"""
Dimensionality reduction for the cleaned SECOM feature set.

Even after dropping high-missingness and constant columns, SECOM still has
several hundred sensor features — too many for a readable SHAP summary plot,
a legible LIME explanation, or a sparse, human-checkable DiCE counterfactual.
This module reduces that set to a compact, *interpretable* subset using two
classical, model-agnostic techniques:

  1. Correlation filtering — when two sensors move together almost perfectly
     (|r| > threshold), keeping both adds redundancy without adding
     information, and it splits any SHAP importance between near-duplicate
     features, making explanations harder to read.
  2. Mutual-information ranking (`SelectKBest` + `mutual_info_classif`) — a
     non-parametric, non-linear measure of how much each remaining feature
     tells you about the failure label, used to keep only the top-K most
     informative sensors.

PCA was deliberately **not** used: its components are linear combinations of
all 590 original sensors, which would make every SHAP/LIME/DiCE explanation
read as "component_7 increased failure risk" — meaningless to a process
engineer. Keeping original, named features is what makes the explanations in
this project actionable.

As with `preprocessing.py`, both steps are fit only on training data.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline

from src.utils import RANDOM_SEED, get_logger

logger = get_logger(__name__)


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """
    Greedily drop one feature from every pair whose absolute Pearson
    correlation (computed on training data) exceeds `threshold`.
    """

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y=None) -> "CorrelationFilter":
        self.feature_names_in_ = X.columns.tolist()
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.columns_to_drop_ = [
            col for col in upper.columns if (upper[col] > self.threshold).any()
        ]
        logger.info(
            "CorrelationFilter: dropping %d/%d columns correlated above %.2f.",
            len(self.columns_to_drop_),
            X.shape[1],
            self.threshold,
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=self.columns_to_drop_, errors="ignore")

    def get_feature_names_out(self, input_features=None) -> list[str]:
        return [c for c in self.feature_names_in_ if c not in self.columns_to_drop_]


def build_feature_selection_pipeline(
    n_features: int = 40, correlation_threshold: float = 0.95, random_state: int = RANDOM_SEED
) -> Pipeline:
    """
    Correlation filtering followed by mutual-information SelectKBest. `k` is
    automatically capped to however many features survive correlation
    filtering, so this is safe to call even on small candidate feature sets.
    """
    mi_score_func = partial(mutual_info_classif, random_state=random_state)

    pipeline = Pipeline(
        steps=[
            ("correlation_filter", CorrelationFilter(threshold=correlation_threshold)),
            ("select_k_best", SelectKBest(score_func=mi_score_func, k=n_features)),
        ]
    )
    pipeline.set_output(transform="pandas")
    return pipeline
