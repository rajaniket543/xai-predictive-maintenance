"""
Single reusable prediction pipeline used by both the CLI/notebooks and the
Streamlit dashboard, so "how a prediction is made" is defined in exactly one
place.

    raw input -> validation -> preprocessing/feature-selection -> model
    -> failure probability -> risk level

The preprocessing pipeline loaded here is the *exact fitted object* produced
by `train_pipeline.py` — the same imputation medians, the same scaler
mean/std, the same selected features — so predictions never drift from how
the model was trained.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.utils import (
    CLASS_NAMES,
    METADATA_PATH,
    MODEL_PATH,
    PREPROCESSING_PIPELINE_PATH,
    RISK_ACTIONS,
    get_logger,
    load_json,
    risk_level,
)

logger = get_logger(__name__)


class ArtifactNotFoundError(RuntimeError):
    """Raised when trained model artifacts are missing from models/."""


class InvalidInputError(ValueError):
    """Raised when input data cannot be scored (wrong/missing columns, bad values)."""


@dataclass
class PredictionResult:
    failure_probability: float
    predicted_label: int
    predicted_class: str
    risk_level: str
    risk_action: str
    threshold_used: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure_probability": self.failure_probability,
            "predicted_label": self.predicted_label,
            "predicted_class": self.predicted_class,
            "risk_level": self.risk_level,
            "risk_action": self.risk_action,
            "threshold_used": self.threshold_used,
        }


class PredictionPipeline:
    """Loads the fitted preprocessing pipeline + model and scores new data."""

    def __init__(self, preprocessing_pipeline, model, metadata: dict[str, Any]):
        self.preprocessing_pipeline = preprocessing_pipeline
        self.model = model
        self.metadata = metadata
        self.raw_feature_names: list[str] = metadata["raw_feature_names"]
        self.selected_feature_names: list[str] = metadata["selected_feature_names"]
        self.default_threshold: float = metadata.get("chosen_threshold", 0.5)

    @classmethod
    def load(
        cls,
        preprocessing_path: Path = PREPROCESSING_PIPELINE_PATH,
        model_path: Path = MODEL_PATH,
        metadata_path: Path = METADATA_PATH,
    ) -> "PredictionPipeline":
        missing = [p for p in (preprocessing_path, model_path, metadata_path) if not Path(p).exists()]
        if missing:
            missing_str = "\n  - ".join(str(p) for p in missing)
            raise ArtifactNotFoundError(
                "Trained model artifacts are missing:\n"
                f"  - {missing_str}\n\n"
                "Run the training pipeline first:\n"
                "    python train_pipeline.py"
            )
        preprocessing_pipeline = joblib.load(preprocessing_path)
        model = joblib.load(model_path)
        metadata = load_json(metadata_path)
        logger.info("Loaded model '%s' and preprocessing pipeline.", metadata.get("model_name", "unknown"))
        return cls(preprocessing_pipeline, model, metadata)

    def validate_input(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise InvalidInputError(f"Expected a pandas DataFrame, got {type(df)}.")

        missing_cols = [c for c in self.raw_feature_names if c not in df.columns]
        if missing_cols:
            preview = ", ".join(missing_cols[:10])
            more = f" (+{len(missing_cols) - 10} more)" if len(missing_cols) > 10 else ""
            raise InvalidInputError(
                f"Input is missing {len(missing_cols)} required column(s): {preview}{more}. "
                f"Expected all {len(self.raw_feature_names)} raw SECOM sensor columns "
                f"(Feature_0 ... Feature_{len(self.raw_feature_names) - 1}, or the subset used at training time)."
            )

        extra_cols = [c for c in df.columns if c not in self.raw_feature_names]
        if extra_cols:
            logger.warning("Ignoring %d unexpected column(s) not seen during training.", len(extra_cols))

        ordered = df[self.raw_feature_names].copy()
        non_numeric = ordered.apply(lambda col: pd.to_numeric(col, errors="coerce")).isna() & ordered.notna()
        bad_cols = non_numeric.any(axis=0)
        if bad_cols.any():
            bad_names = bad_cols[bad_cols].index.tolist()
            raise InvalidInputError(
                f"Column(s) contain non-numeric values that are not missing-value markers: {bad_names[:10]}"
            )

        return ordered.apply(pd.to_numeric, errors="coerce")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Raw input -> the exact feature matrix the model consumes (post preprocessing + selection)."""
        validated = self.validate_input(df)
        transformed = self.preprocessing_pipeline.transform(validated)
        return transformed

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        transformed = self.transform(df)
        return self.model.predict_proba(transformed)[:, 1]

    def predict(
        self, df: pd.DataFrame, threshold: float | None = None, risk_thresholds: dict[str, float] | None = None
    ) -> pd.DataFrame:
        """Score every row of `df`, returning one PredictionResult-shaped row each."""
        t = threshold if threshold is not None else self.default_threshold
        probabilities = self.predict_proba(df)
        results = []
        for p in probabilities:
            label = int(p >= t)
            rl = risk_level(p, risk_thresholds)
            results.append(
                PredictionResult(
                    failure_probability=float(p),
                    predicted_label=label,
                    predicted_class=CLASS_NAMES[label],
                    risk_level=rl,
                    risk_action=RISK_ACTIONS[rl],
                    threshold_used=t,
                ).to_dict()
            )
        return pd.DataFrame(results, index=df.index)

    def predict_single(
        self, row: pd.Series | dict, threshold: float | None = None, risk_thresholds: dict[str, float] | None = None
    ) -> dict[str, Any]:
        row_df = pd.DataFrame([row]) if isinstance(row, dict) else row.to_frame().T
        return self.predict(row_df, threshold=threshold, risk_thresholds=risk_thresholds).iloc[0].to_dict()
