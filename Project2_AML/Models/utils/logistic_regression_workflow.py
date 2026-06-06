"""Project 2 Logistic Regression workflow.

The pipeline follows the teammate-friendly order used in the notebook:

    variance filter -> correlation filter -> standardization -> SelectFromModel -> LogisticRegression

These steps mirror `utils/data_processing.py`, but are implemented as sklearn
transformers so the entire workflow stays inside one Pipeline and does not leak
information from validation folds during cross-validation or grid search.

The filter transformers are also reused by `utils/svm_workflow.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SelectorKind = Literal["l1", "extra_trees"]


def make_selector_estimator(
    kind: SelectorKind,
    *,
    selector_c: float = 1.0,
    random_state: int = 42,
    max_iter: int = 2000,
) -> LogisticRegression | ExtraTreesClassifier:
    """Build the SelectFromModel estimator used by LR and SVM pipelines."""

    if kind == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1,
        )

    return LogisticRegression(
        penalty="l1",
        solver="liblinear",
        C=selector_c,
        random_state=random_state,
        max_iter=max_iter,
    )


def _to_frame(X: Any) -> pd.DataFrame:
    """Convert array-like inputs to a DataFrame so the filters can track columns."""

    if isinstance(X, pd.DataFrame):
        return X
    if isinstance(X, pd.Series):
        return X.to_frame()

    return pd.DataFrame(np.asarray(X))


@dataclass
class VarianceFilterTransformer(BaseEstimator, TransformerMixin):
    """Drop columns whose training-set variance is below a threshold."""

    threshold: float = 0.01

    def fit(self, X: Any, y: Any = None):
        frame = _to_frame(X)
        self.feature_names_in_ = list(frame.columns)
        variances = frame.var(axis=0)
        self.support_ = (variances > self.threshold).to_numpy()

        if not np.any(self.support_):
            raise ValueError(
                "Variance filter removed every feature. Lower the threshold before running the pipeline."
            )

        return self

    def transform(self, X: Any):
        frame = _to_frame(X)
        return frame.loc[:, self.get_support()].copy()

    def get_support(self) -> np.ndarray:
        return np.asarray(self.support_, dtype=bool)


@dataclass
class CorrelationFilterTransformer(BaseEstimator, TransformerMixin):
    """Drop highly correlated features using the training fold only."""

    threshold: float = 0.9

    def fit(self, X: Any, y: Any = None):
        frame = _to_frame(X)
        self.feature_names_in_ = list(frame.columns)

        corr_matrix = frame.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [
            column for column in upper.columns if any(upper[column] > self.threshold)
        ]
        self.support_ = ~frame.columns.isin(to_drop)

        if not np.any(self.support_):
            raise ValueError(
                "Correlation filter removed every feature. Lower the threshold before running the pipeline."
            )

        return self

    def transform(self, X: Any):
        frame = _to_frame(X)
        return frame.loc[:, self.get_support()].copy()

    def get_support(self) -> np.ndarray:
        return np.asarray(self.support_, dtype=bool)


def build_logistic_regression_pipeline(
    *,
    variance_threshold: float = 0.01,
    correlation_threshold: float = 0.9,
    selector_kind: SelectorKind = "l1",
    selector_c: float = 1.0,
    model_c: float = 1.0,
    random_state: int = 42,
    max_iter: int = 2000,
    selector_threshold: str | float | None = None,
) -> Pipeline:
    """Build the leakage-safe Project 2 Logistic Regression pipeline.

    Notes for teammates:
    - `selector_kind="l1"`: SelectFromModel with L1 LogisticRegression (liblinear).
    - `selector_kind="extra_trees"`: SelectFromModel with ExtraTreesClassifier.
    - Use stricter `selector_threshold` values (e.g. "median", "1.25*mean") to keep
      NoVariables low for the business score.
    """

    selector_estimator = make_selector_estimator(
        selector_kind,
        selector_c=selector_c,
        random_state=random_state,
        max_iter=max_iter,
    )

    final_model = LogisticRegression(
        penalty="l2",
        solver="liblinear",
        C=model_c,
        random_state=random_state,
        max_iter=max_iter,
    )

    return Pipeline(
        steps=[
            (
                "variance_filter",
                VarianceFilterTransformer(threshold=variance_threshold),
            ),
            (
                "correlation_filter",
                CorrelationFilterTransformer(threshold=correlation_threshold),
            ),
            # Standardization is placed inside the pipeline so each CV fold fits it only on training data.
            ("scaler", StandardScaler()),
            # L1 logistic gives a sparse selector that is easy to interpret and aligns with the project codebase.
            (
                "selector",
                SelectFromModel(
                    estimator=selector_estimator,
                    threshold=selector_threshold,
                ),
            ),
            ("model", final_model),
        ]
    )


__all__ = [
    "CorrelationFilterTransformer",
    "SelectorKind",
    "VarianceFilterTransformer",
    "build_logistic_regression_pipeline",
    "make_selector_estimator",
]
