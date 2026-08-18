"""
Data-cleaning pipeline: excessive-missingness column removal, median
imputation, constant-column removal, and scaling.

Everything in this module is expressed as scikit-learn transformers so it can
be composed into a single `Pipeline` that is fit ONLY on training data and
then reused, unchanged, at prediction time (`predict.py`, the Streamlit app).
This is what prevents data leakage: statistics such as "which columns are
>55% missing", "the median of each column", and "mean/std for scaling" are
all learned exclusively from the training split.

Design decisions (see README for the full write-up):
  * Columns missing in more than `missingness_threshold` of training rows are
    dropped outright — at that point there is too little signal to impute
    reliably, and imputing them would mostly inject a constant (the median)
    into the model.
  * Remaining missing values are median-imputed. The median is robust to the
    outliers/skew common in raw sensor data, unlike the mean.
  * Zero-variance (constant) columns are removed — a sensor reading that
    never changes carries no predictive information and cannot help SHAP/LIME
    explain anything.
  * StandardScaling follows imputation so that distance/gradient-based models
    (Logistic Regression) and SHAP's additivity assumptions behave well; tree
    models are scale-invariant but are unaffected by this step.
"""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils import get_logger

logger = get_logger(__name__)


class HighMissingnessDropper(BaseEstimator, TransformerMixin):
    """Drop columns whose training-set missing rate exceeds `threshold`."""

    def __init__(self, threshold: float = 0.55):
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y=None) -> "HighMissingnessDropper":
        missing_rate = X.isna().mean()
        self.columns_to_drop_ = missing_rate[missing_rate > self.threshold].index.tolist()
        self.feature_names_in_ = X.columns.tolist()
        self.n_features_out_ = X.shape[1] - len(self.columns_to_drop_)
        logger.info(
            "HighMissingnessDropper: dropping %d/%d columns with >%.0f%% missing values.",
            len(self.columns_to_drop_),
            X.shape[1],
            100 * self.threshold,
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=self.columns_to_drop_, errors="ignore")

    def get_feature_names_out(self, input_features=None) -> list[str]:
        return [c for c in self.feature_names_in_ if c not in self.columns_to_drop_]


class ConstantColumnDropper(BaseEstimator, TransformerMixin):
    """Drop columns with zero variance (a single distinct value) in training data."""

    def fit(self, X: pd.DataFrame, y=None) -> "ConstantColumnDropper":
        variances = X.var(numeric_only=True)
        self.columns_to_drop_ = variances[variances == 0].index.tolist()
        self.feature_names_in_ = X.columns.tolist()
        logger.info(
            "ConstantColumnDropper: dropping %d constant column(s).",
            len(self.columns_to_drop_),
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=self.columns_to_drop_, errors="ignore")

    def get_feature_names_out(self, input_features=None) -> list[str]:
        return [c for c in self.feature_names_in_ if c not in self.columns_to_drop_]


def build_preprocessing_pipeline(missingness_threshold: float = 0.55) -> Pipeline:
    """
    Cleaning-only pipeline: drop high-missingness columns -> median-impute ->
    drop constant columns -> standard-scale. Operates purely on the raw
    Feature_* columns (the caller is responsible for excluding
    Timestamp/target beforehand). Every step is pandas-in/pandas-out so
    feature names survive the whole pipeline for SHAP/LIME/DiCE.
    """
    pipeline = Pipeline(
        steps=[
            ("drop_high_missingness", HighMissingnessDropper(threshold=missingness_threshold)),
            ("impute_median", SimpleImputer(strategy="median")),
            ("drop_constant", ConstantColumnDropper()),
            ("scale", StandardScaler()),
        ]
    )
    pipeline.set_output(transform="pandas")
    return pipeline
