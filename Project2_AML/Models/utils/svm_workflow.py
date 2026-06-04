"""Project 2 SVM workflow.

The preprocessing order matches Logistic Regression and the teammate notebook:

    variance filter -> correlation filter -> standardization -> SelectFromModel -> SVM

The first three filters are reused from `logistic_regression_workflow.py` so both
linear models follow the same Project 2 preprocessing convention.
"""

from __future__ import annotations

from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

from .logistic_regression_workflow import (
    CorrelationFilterTransformer,
    VarianceFilterTransformer,
)


def build_svm_pipeline(
    *,
    variance_threshold: float = 0.01,
    correlation_threshold: float = 0.9,
    selector_c: float = 1.0,
    model_c: float = 1.0,
    kernel: str = "rbf",
    gamma: str | float = "scale",
    random_state: int = 42,
    max_iter: int = 5000,
    selector_threshold: str | float | None = None,
) -> Pipeline:
    """Build the leakage-safe Project 2 SVM pipeline.

    Notes for teammates:
    - L1 `LinearSVC` (`dual=False`) is used inside SelectFromModel for sparse selection.
    - The final model defaults to RBF SVC with `probability=True` so ranking works like LR.
    - Set `kernel="linear"` for a faster linear SVM baseline.
    - Keep all preprocessing inside this pipeline during CV to avoid leakage.
    """

    selector_estimator = LinearSVC(
        penalty="l1",
        dual=False,
        C=selector_c,
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
