"""
Loading and first-pass diagnostics for the SECOM manufacturing dataset.

SECOM (UCI Machine Learning Repository) contains 1567 examples, each with 590
anonymized semiconductor-manufacturing sensor/process measurements plus a
pass/fail label. Source: https://archive.ics.uci.edu/dataset/179/secom

`secom.data`        — whitespace-separated feature matrix, one row per example,
                       missing values encoded as the literal string "NaN".
`secom_labels.data`  — two whitespace-separated columns per row: the label
                       (-1 = pass, 1 = fail) and a timestamp string.

This module only loads and *describes* the data. No fitting/transformation
happens here — that belongs in `preprocessing.py` so that every learned
transformation stays inside the train/test-aware pipeline and never leaks
information from the full dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils import (
    POSITIVE_LABEL,
    SECOM_DATA_PATH,
    SECOM_LABELS_PATH,
    TARGET_COLUMN,
    get_logger,
)

logger = get_logger(__name__)

FEATURE_PREFIX = "Feature_"

_DOWNLOAD_INSTRUCTIONS = """
SECOM dataset files were not found.

Expected files:
    {data_path}
    {labels_path}

To obtain the dataset:
    1. Download the two files from the UCI Machine Learning Repository:
       https://archive.ics.uci.edu/dataset/179/secom
       (direct links, correct as of this writing:
         https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data
         https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data)
    2. Place both files, unmodified, inside the `data/` directory of this project
       (or pass custom paths to `load_secom_dataset(data_path=..., labels_path=...)`).
"""


def _check_files_exist(data_path: Path, labels_path: Path) -> None:
    if not data_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            _DOWNLOAD_INSTRUCTIONS.format(data_path=data_path, labels_path=labels_path)
        )


def load_secom_dataset(
    data_path: Path | str = SECOM_DATA_PATH,
    labels_path: Path | str = SECOM_LABELS_PATH,
) -> pd.DataFrame:
    """
    Load the raw SECOM feature matrix and labels and combine them into one
    DataFrame.

    Returns
    -------
    DataFrame with columns:
        Feature_0 ... Feature_589  (raw sensor measurements, float, may contain NaN)
        Timestamp                  (datetime of the run)
        Failure                    (0 = pass/normal, 1 = fail) — the prediction target
    """
    data_path = Path(data_path)
    labels_path = Path(labels_path)
    _check_files_exist(data_path, labels_path)

    features = pd.read_csv(data_path, sep=r"\s+", header=None, na_values="NaN")
    features.columns = [f"{FEATURE_PREFIX}{i}" for i in range(features.shape[1])]

    labels_raw = pd.read_csv(
        labels_path, sep=r"\s+", header=None, names=["label", "date"]
    )
    # secom_labels.data timestamps are double-quoted, e.g. `-1 "19/07/2008 11:55:00"`.
    # pandas' CSV parser honors the quoting even with a regex separator, so the
    # quoted "date time" pair already arrives as a single unquoted field.
    timestamp = pd.to_datetime(labels_raw["date"], format="%d/%m/%Y %H:%M:%S")

    if len(features) != len(labels_raw):
        raise ValueError(
            f"Row count mismatch between features ({len(features)}) and "
            f"labels ({len(labels_raw)}) — the two files do not correspond."
        )

    df = features.copy()
    df["Timestamp"] = timestamp
    # Raw encoding is -1 = pass, 1 = fail. Remap to 0/1 so `1` consistently
    # means "the positive/failure class" throughout the rest of the project.
    df[TARGET_COLUMN] = (labels_raw["label"] == POSITIVE_LABEL).astype(int)

    logger.info(
        "Loaded SECOM dataset: %d rows x %d raw features, %d failures (%.2f%%).",
        df.shape[0],
        features.shape[1],
        df[TARGET_COLUMN].sum(),
        100 * df[TARGET_COLUMN].mean(),
    )
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(FEATURE_PREFIX)]


@dataclass
class DatasetDiagnostics:
    """Structured, human-readable summary of the raw dataset's quirks."""

    n_rows: int
    n_features: int
    class_counts: dict[str, int]
    failure_rate: float
    imbalance_ratio: float  # majority : minority
    n_duplicate_rows: int
    total_missing_cells: int
    missing_cell_pct: float
    n_columns_with_any_missing: int
    n_columns_over_missing_threshold: int
    missing_threshold_used: float
    worst_missing_columns: list[tuple[str, float]]
    n_constant_columns: int
    constant_columns: list[str]
    n_near_zero_variance_columns: int
    near_zero_variance_columns: list[str]
    dtypes_summary: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_features": self.n_features,
            "class_counts": self.class_counts,
            "failure_rate": self.failure_rate,
            "imbalance_ratio": self.imbalance_ratio,
            "n_duplicate_rows": self.n_duplicate_rows,
            "total_missing_cells": self.total_missing_cells,
            "missing_cell_pct": self.missing_cell_pct,
            "n_columns_with_any_missing": self.n_columns_with_any_missing,
            "n_columns_over_missing_threshold": self.n_columns_over_missing_threshold,
            "missing_threshold_used": self.missing_threshold_used,
            "worst_missing_columns": self.worst_missing_columns,
            "n_constant_columns": self.n_constant_columns,
            "constant_columns": self.constant_columns,
            "n_near_zero_variance_columns": self.n_near_zero_variance_columns,
            "near_zero_variance_columns": self.near_zero_variance_columns,
            "dtypes_summary": self.dtypes_summary,
        }

    def summary_text(self) -> str:
        lines = [
            f"Rows: {self.n_rows}",
            f"Raw features: {self.n_features}",
            f"Class counts (0=normal, 1=failure): {self.class_counts}",
            f"Failure rate: {self.failure_rate:.2%}",
            f"Imbalance ratio (normal:failure): {self.imbalance_ratio:.1f} : 1",
            f"Duplicate rows: {self.n_duplicate_rows}",
            f"Missing cells: {self.total_missing_cells} ({self.missing_cell_pct:.2f}% of all cells)",
            f"Columns with >=1 missing value: {self.n_columns_with_any_missing}",
            f"Columns with missing rate > {self.missing_threshold_used:.0%}: "
            f"{self.n_columns_over_missing_threshold}",
            f"Constant (zero-variance) columns: {self.n_constant_columns}",
            f"Near-zero-variance columns (<=1% unique values): {self.n_near_zero_variance_columns}",
        ]
        return "\n".join(lines)


def analyze_dataset(
    df: pd.DataFrame, missing_threshold: float = 0.55
) -> DatasetDiagnostics:
    """
    Compute the raw-data diagnostics described in the project brief: shape,
    target balance, missingness, duplicates, constant / near-constant
    features, and dtypes. Nothing here fits or transforms the data — it is
    read-only inspection used to justify later preprocessing decisions.
    """
    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    y = df[TARGET_COLUMN]

    class_counts = y.value_counts().sort_index()
    n_failure = int(class_counts.get(1, 0))
    n_normal = int(class_counts.get(0, 0))
    failure_rate = n_failure / len(df)
    imbalance_ratio = (n_normal / n_failure) if n_failure else float("inf")

    n_duplicates = int(df.duplicated(subset=feature_cols).sum())

    missing_mask = X.isna()
    total_missing = int(missing_mask.values.sum())
    missing_pct = 100 * total_missing / (X.shape[0] * X.shape[1])
    per_col_missing_rate = missing_mask.mean().sort_values(ascending=False)
    n_any_missing = int((per_col_missing_rate > 0).sum())
    n_over_threshold = int((per_col_missing_rate > missing_threshold).sum())
    worst_cols = [
        (col, float(rate)) for col, rate in per_col_missing_rate.head(10).items()
    ]

    # Constant columns: zero variance among non-missing values.
    variances = X.var(numeric_only=True, skipna=True)
    constant_cols = variances[variances == 0].index.tolist()

    # Near-zero-variance: very few distinct values relative to the number of
    # non-missing observations (a looser, unique-value-ratio based notion of
    # "almost constant" that is well-defined even before scaling).
    nunique = X.nunique(dropna=True)
    non_na_counts = X.notna().sum().replace(0, np.nan)
    unique_ratio = nunique / non_na_counts
    near_zero_var_cols = unique_ratio[unique_ratio <= 0.01].index.tolist()
    near_zero_var_cols = [c for c in near_zero_var_cols if c not in constant_cols]

    dtypes_summary = X.dtypes.astype(str).value_counts().to_dict()

    diagnostics = DatasetDiagnostics(
        n_rows=len(df),
        n_features=len(feature_cols),
        class_counts={"normal_0": n_normal, "failure_1": n_failure},
        failure_rate=failure_rate,
        imbalance_ratio=imbalance_ratio,
        n_duplicate_rows=n_duplicates,
        total_missing_cells=total_missing,
        missing_cell_pct=missing_pct,
        n_columns_with_any_missing=n_any_missing,
        n_columns_over_missing_threshold=n_over_threshold,
        missing_threshold_used=missing_threshold,
        worst_missing_columns=worst_cols,
        n_constant_columns=len(constant_cols),
        constant_columns=constant_cols,
        n_near_zero_variance_columns=len(near_zero_var_cols),
        near_zero_variance_columns=near_zero_var_cols,
        dtypes_summary=dtypes_summary,
    )

    logger.info("Dataset diagnostics:\n%s", diagnostics.summary_text())
    return diagnostics


if __name__ == "__main__":
    dataset = load_secom_dataset()
    report = analyze_dataset(dataset)
    print(report.summary_text())
