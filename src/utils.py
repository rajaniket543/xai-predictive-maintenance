"""
Shared configuration, paths, logging, and small utilities used across the project.

Centralizing paths and constants here means every module (training script,
Streamlit app, notebooks) agrees on where data/models/outputs live, and there
are no hard-coded absolute paths scattered through the codebase.
"""

from __future__ import annotations

import json
import logging
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Project paths (all relative to the repository root, derived from this file's
# location so the project runs correctly regardless of the current working
# directory it is launched from).
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
SECOM_DATA_PATH = DATA_DIR / "secom.data"
SECOM_LABELS_PATH = DATA_DIR / "secom_labels.data"

MODELS_DIR = PROJECT_ROOT / "models"
PREPROCESSING_PIPELINE_PATH = MODELS_DIR / "preprocessing_pipeline.pkl"
MODEL_PATH = MODELS_DIR / "model.pkl"
METADATA_PATH = MODELS_DIR / "metadata.json"
MODEL_COMPARISON_PATH = MODELS_DIR / "model_comparison.json"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
EXPLANATIONS_DIR = OUTPUTS_DIR / "explanations"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

for _dir in (DATA_DIR, MODELS_DIR, FIGURES_DIR, METRICS_DIR, EXPLANATIONS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42


def set_global_seed(seed: int = RANDOM_SEED) -> None:
    """Seed every RNG this project touches so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured module-level logger (idempotent — safe to call repeatedly)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------
TARGET_COLUMN = "Failure"
POSITIVE_LABEL = 1  # 1 = failure / fail, 0 = pass / normal
CLASS_NAMES = {0: "NORMAL", 1: "FAILURE"}

# Predictive-maintenance risk bands. These are configurable, illustrative
# thresholds for the project demo — NOT scientifically validated maintenance
# standards. They exist to translate a raw failure probability into a
# human-actionable recommendation, and can be changed from the dashboard.
DEFAULT_RISK_THRESHOLDS: dict[str, float] = {
    "low_medium": 0.30,
    "medium_high": 0.60,
    "high_critical": 0.80,
}

RISK_ACTIONS: dict[str, str] = {
    "LOW": "Continue normal operation.",
    "MEDIUM": "Monitor the machine/process more closely.",
    "HIGH": "Schedule an inspection.",
    "CRITICAL": "Immediate inspection recommended.",
}


def risk_level(probability: float, thresholds: dict[str, float] | None = None) -> str:
    """Map a failure probability to a risk band using configurable thresholds."""
    t = thresholds or DEFAULT_RISK_THRESHOLDS
    if probability < t["low_medium"]:
        return "LOW"
    if probability < t["medium_high"]:
        return "MEDIUM"
    if probability < t["high_critical"]:
        return "HIGH"
    return "CRITICAL"


# ---------------------------------------------------------------------------
# JSON helpers (numpy-safe)
# ---------------------------------------------------------------------------
class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that understands numpy scalar/array types produced by sklearn."""

    def default(self, o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
        return super().default(o)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, cls=NumpyJSONEncoder)


def load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


@dataclass
class ProjectConfig:
    """A single place to tweak the main pipeline knobs used across scripts."""

    random_seed: int = RANDOM_SEED
    test_size: float = 0.2
    cv_folds: int = 5

    missingness_drop_threshold: float = 0.55  # drop columns missing in >55% of rows
    correlation_drop_threshold: float = 0.95  # drop one of any pair correlated above this
    n_features_to_select: int = 40  # final interpretable feature-set size for the model

    hyperparameter_search_metric: str = "average_precision"  # PR-AUC — robust under imbalance
    n_iter_search: int = 25

    risk_thresholds: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_RISK_THRESHOLDS))


CONFIG = ProjectConfig()
