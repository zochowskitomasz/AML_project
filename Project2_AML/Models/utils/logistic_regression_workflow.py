"""Project 2 Logistic Regression workflow.

The pipeline follows the teammate-friendly order used in the notebook:

    variance filter -> correlation filter -> standardization -> feature selection -> LogisticRegression

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
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import (
    SelectFromModel,
    SelectKBest,
    SequentialFeatureSelector,
    f_classif,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SelectorKind = Literal["l1", "extra_trees", "kbest", "sfs"]
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
        solver="liblinear",
        penalty="l1",
        C=selector_c,
        random_state=random_state,
        max_iter=max_iter,
    )


class AdaptiveLogisticRegression(BaseEstimator, ClassifierMixin):
    """Logistic regression that picks a fast solver for L2 and saga for elastic-net."""

    def __init__(
        self,
        *,
        C: float = 1.0,
        l1_ratio: float = 0.0,
        class_weight: str | dict | None = None,
        random_state: int = 42,
        max_iter: int = 5000,
    ) -> None:
        self.C = C
        self.l1_ratio = l1_ratio
        self.class_weight = class_weight
        self.random_state = random_state
        self.max_iter = max_iter

    def _build_model(self) -> LogisticRegression:
        if self.l1_ratio and self.l1_ratio > 0:
            return LogisticRegression(
                solver="saga",
                C=self.C,
                l1_ratio=self.l1_ratio,
                class_weight=self.class_weight,
                random_state=self.random_state,
                max_iter=self.max_iter,
            )
        return LogisticRegression(
            solver="liblinear",
            C=self.C,
            l1_ratio=0.0,
            class_weight=self.class_weight,
            random_state=self.random_state,
            max_iter=self.max_iter,
        )

    def fit(self, X: Any, y: Any):
        self.model_ = self._build_model()
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        self.n_features_in_ = self.model_.n_features_in_
        self.coef_ = self.model_.coef_
        self.intercept_ = self.model_.intercept_
        return self

    def predict(self, X: Any):
        return self.model_.predict(X)

    def predict_proba(self, X: Any):
        return self.model_.predict_proba(X)

    def decision_function(self, X: Any):
        return self.model_.decision_function(X)


def make_logistic_model(
    *,
    model_c: float = 1.0,
    l1_ratio: float = 0.0,
    class_weight: str | dict | None = None,
    random_state: int = 42,
    max_iter: int = 5000,
) -> AdaptiveLogisticRegression:
    """Build the final adaptive logistic regression classifier."""

    return AdaptiveLogisticRegression(
        C=model_c,
        l1_ratio=l1_ratio,
        class_weight=class_weight,
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


@dataclass
class KBestFilterTransformer(BaseEstimator, TransformerMixin):
    """Univariate pre-filter to shrink the search space before SFS or the final model."""

    k: int = 50

    def fit(self, X: Any, y: Any = None):
        frame = _to_frame(X)
        self.feature_names_in_ = list(frame.columns)
        effective_k = min(int(self.k), frame.shape[1])
        selector = SelectKBest(score_func=f_classif, k=effective_k)
        selector.fit(frame, np.asarray(y).reshape(-1))
        self.support_ = selector.get_support()
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
    model_l1_ratio: float = 0.0,
    class_weight: str | dict | None = None,
    random_state: int = 42,
    max_iter: int = 5000,
    selector_threshold: str | float | None = None,
    kbest_k: int = 50,
    sfs_n_features: int = 10,
    sfs_cv: int = 3,
) -> Pipeline:
    """Build the leakage-safe Project 2 Logistic Regression pipeline.

    Selector options:
    - `l1`: SelectFromModel with L1 LogisticRegression (liblinear).
    - `extra_trees`: SelectFromModel with ExtraTreesClassifier.
    - `kbest`: SelectKBest with ANOVA F-test.
    - `sfs`: KBest pre-filter followed by forward SequentialFeatureSelector.
    """

    final_model = make_logistic_model(
        model_c=model_c,
        l1_ratio=model_l1_ratio,
        class_weight=class_weight,
        random_state=random_state,
        max_iter=max_iter,
    )

    steps: list[tuple[str, Any]] = [
        ("variance_filter", VarianceFilterTransformer(threshold=variance_threshold)),
        (
            "correlation_filter",
            CorrelationFilterTransformer(threshold=correlation_threshold),
        ),
        ("scaler", StandardScaler()),
    ]

    if selector_kind == "kbest":
        steps.append(("selector", KBestFilterTransformer(k=kbest_k)))
    elif selector_kind == "sfs":
        sfs_estimator = make_logistic_model(
            model_c=1.0,
            l1_ratio=0.0,
            random_state=random_state,
            max_iter=max_iter,
        )
        steps.extend(
            [
                ("kbest_prefilter", KBestFilterTransformer(k=kbest_k)),
                (
                    "selector",
                    SequentialFeatureSelector(
                        estimator=sfs_estimator,
                        n_features_to_select=sfs_n_features,
                        direction="forward",
                        scoring="roc_auc",
                        cv=sfs_cv,
                        n_jobs=1,
                    ),
                ),
            ]
        )
    else:
        selector_estimator = make_selector_estimator(
            selector_kind,
            selector_c=selector_c,
            random_state=random_state,
            max_iter=max_iter,
        )
        steps.append(
            (
                "selector",
                SelectFromModel(
                    estimator=selector_estimator,
                    threshold=selector_threshold,
                ),
            )
        )

    steps.append(("model", final_model))
    return Pipeline(steps=steps)


def get_model_coefficients(pipeline: Pipeline) -> np.ndarray:
    """Return absolute coefficients from the fitted final logistic model."""

    if not isinstance(pipeline, Pipeline):
        raise ValueError("Expected a fitted sklearn Pipeline.")

    model = pipeline.named_steps["model"]
    if not hasattr(model, "coef_"):
        raise ValueError("Final model does not expose coef_.")

    coefficients = np.asarray(model.coef_, dtype=float).reshape(-1)
    return np.abs(coefficients)


def feature_dominance_ratio(pipeline: Pipeline) -> float:
    """Share of total coefficient mass attributed to the single strongest feature."""

    coefficients = get_model_coefficients(pipeline)
    total = float(coefficients.sum())
    if total <= 0:
        return 1.0
    return float(coefficients.max() / total)


def passes_feature_diversity_check(
    pipeline: Pipeline,
    *,
    min_features: int = 3,
    max_dominance_ratio: float = 0.65,
    X: Any | None = None,
) -> bool:
    """Reject models that rely on too few features or one dominant coefficient."""

    from utils.business_scoring import infer_n_variables

    n_features = infer_n_variables(pipeline, X=X)
    if n_features < min_features:
        return False
    return feature_dominance_ratio(pipeline) <= max_dominance_ratio


__all__ = [
    "CorrelationFilterTransformer",
    "KBestFilterTransformer",
    "AdaptiveLogisticRegression",
    "SelectorKind",
    "VarianceFilterTransformer",
    "build_logistic_regression_pipeline",
    "feature_dominance_ratio",
    "get_model_coefficients",
    "make_logistic_model",
    "make_selector_estimator",
    "passes_feature_diversity_check",
]
