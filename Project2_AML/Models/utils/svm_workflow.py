"""Project 2 SVM workflow.

Teammate convention:

    variance filter -> correlation filter -> standardization
    -> SelectFromModel(ExtraTreesClassifier) -> SVC
"""

from __future__ import annotations

from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

from .logistic_regression_workflow import (
    CorrelationFilterTransformer,
    VarianceFilterTransformer,
    make_selector_estimator,
)


def build_svm_pipeline(
    *,
    variance_threshold: float = 0.01,
    correlation_threshold: float = 0.9,
    model_c: float = 1.0,
    kernel: str = "rbf",
    gamma: str | float = "scale",
    random_state: int = 42,
    max_iter: int = 5000,
    selector_threshold: str | float | None = "median",
) -> Pipeline:
    """Build the leakage-safe Project 2 SVM pipeline."""

    selector_estimator = make_selector_estimator(
        "extra_trees",
        random_state=random_state,
        max_iter=max_iter,
    )

    if kernel == "linear":
        final_model = LinearSVC(
            C=model_c,
            dual=True,
            random_state=random_state,
            max_iter=max_iter,
        )
    else:
        final_model = SVC(
            kernel=kernel,
            C=model_c,
            gamma=gamma,
            probability=True,
            random_state=random_state,
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
            ("scaler", StandardScaler()),
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


__all__ = ["build_svm_pipeline"]
