"""Business scoring utilities for Project 2.

This module implements the project-specific score:

    Score = (TP * 10) - (FP * 5) - (NoVariables * 200)

Used by the Logistic Regression and SVM experiments to pick k and tune hyperparameters
during GridSearchCV. Linear models use `predict_proba`; SVM may use `decision_function`.

The design keeps the pieces separate:
- model fitting happens outside this module,
- confidence scores are extracted from a fitted estimator,
- a top-k targeting policy turns confidence into customer contacts,
- the business score is computed from the chosen contacts and feature count.

Why this differs from standard metrics:
- Accuracy only checks whether each prediction matches the label.
- F1 balances precision and recall, but still ignores the cost of contacting customers.
- ROC-AUC measures ranking quality across all thresholds, not the final business payoff.
- Default scikit-learn scoring usually targets predictive quality, while this project
  must optimize a net business objective that penalizes false positives and feature count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline


def _as_numpy_1d(values: Any) -> np.ndarray:
    """Convert supported array-like inputs to a flat NumPy array."""

    if isinstance(values, (pd.Series, pd.Index)):
        array = values.to_numpy()
    elif isinstance(values, pd.DataFrame):
        array = values.to_numpy()
    else:
        array = np.asarray(values)
    return np.asarray(array).reshape(-1)


def _row_subset(data: Any, row_indices: np.ndarray) -> Any:
    """Subset rows while preserving pandas objects when possible."""

    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data.iloc[row_indices]
    return np.asarray(data)[row_indices]


def _selector_support_count(step: Any) -> int | None:
    """Return the number of retained variables for selector-like fitted steps."""

    if hasattr(step, "get_support"):
        support = np.asarray(step.get_support(), dtype=bool)
        return int(support.sum())
    return None


def infer_n_variables(estimator: Any, X: Any | None = None) -> int:
    """Infer the number of features actually used by a fitted estimator.

    Priority order:
    1. explicit selector inside a fitted pipeline,
    2. explicit selector-like estimator object,
    3. `n_features_in_` from the fitted estimator,
    4. the width of `X` if it was provided.

    If there is no explicit selector, we treat all post-filter features as used.
    """

    if hasattr(estimator, "best_estimator_"):
        return infer_n_variables(estimator.best_estimator_, X=X)

    if isinstance(estimator, Pipeline):
        for _, step in reversed(estimator.steps):
            retained = _selector_support_count(step)
            if retained is not None:
                return retained

        if hasattr(estimator, "n_features_in_"):
            return int(estimator.n_features_in_)

    retained = _selector_support_count(estimator)
    if retained is not None:
        return retained

    if hasattr(estimator, "n_features_in_"):
        return int(estimator.n_features_in_)

    if X is not None:
        X_array = np.asarray(X)
        if X_array.ndim == 1:
            return 1
        return int(X_array.shape[1])

    raise ValueError(
        "Unable to infer the number of variables. Provide a fitted estimator or X."
    )


def get_selected_feature_names(estimator: Any, X: Any) -> list[str]:
    """Return original column names that survive the fitted pipeline's selection steps.

    Project 2 requires reporting which variables were used. This walks the pipeline
    steps before the final classifier and applies each step's support mask.
    """

    if hasattr(estimator, "best_estimator_"):
        return get_selected_feature_names(estimator.best_estimator_, X)

    if not isinstance(estimator, Pipeline):
        raise ValueError("Expected a fitted sklearn Pipeline.")

    if isinstance(X, pd.DataFrame):
        names = [str(column) for column in X.columns]
    else:
        names = [str(index) for index in range(np.asarray(X).shape[1])]

    selection_steps = {
        "variance_filter",
        "correlation_filter",
        "kbest_prefilter",
        "selector",
    }
    for step_name, step in estimator.steps:
        if step_name == "model":
            break
        if step_name in selection_steps and hasattr(step, "get_support"):
            support = np.asarray(step.get_support(), dtype=bool)
            if support.shape[0] != len(names):
                raise ValueError(
                    f"Support mask length ({support.shape[0]}) does not match "
                    f"feature name count ({len(names)}) at step '{step_name}'."
                )
            names = [name for name, keep in zip(names, support, strict=True) if keep]

    return names


def compute_business_score(
    y_true: Any,
    *,
    selected_mask: Any | None = None,
    selected_indices: Sequence[int] | np.ndarray | None = None,
    n_variables: int,
    positive_label: Any = 1,
) -> float:
    """Compute the project business score from the chosen contacts.

    The selected customers are those we would contact.
    `selected_mask` and `selected_indices` are mutually exclusive inputs.
    """

    if selected_mask is not None and selected_indices is not None:
        raise ValueError("Pass either selected_mask or selected_indices, not both.")
    if n_variables < 0:
        raise ValueError("n_variables must be non-negative.")

    y_array = _as_numpy_1d(y_true)

    if selected_mask is not None:
        contact_mask = np.asarray(selected_mask, dtype=bool).reshape(-1)
        if contact_mask.shape[0] != y_array.shape[0]:
            raise ValueError("selected_mask must have the same length as y_true.")
    elif selected_indices is not None:
        contact_mask = np.zeros(y_array.shape[0], dtype=bool)
        indices = np.asarray(selected_indices, dtype=int).reshape(-1)
        if indices.size:
            if np.any(indices < 0) or np.any(indices >= y_array.shape[0]):
                raise IndexError("selected_indices contains an out-of-range value.")
            contact_mask[np.unique(indices)] = True
    else:
        raise ValueError("Provide selected_mask or selected_indices.")

    selected_y = y_array[contact_mask]
    tp = int(np.sum(selected_y == positive_label))
    fp = int(np.sum(selected_y != positive_label))

    return float((tp * 10) - (fp * 5) - (n_variables * 200))


def select_top_k_targets(
    confidence_scores: Any,
    k: int,
    *,
    max_contacts: int = 1000,
    return_mask: bool = True,
) -> np.ndarray:
    """Convert confidence scores into the top-k customers to contact.

    The caller provides scores where larger values mean "more likely positive".
    We rank by confidence and select the highest scores, capped by `max_contacts`.
    """

    if k < 0:
        raise ValueError("k must be non-negative.")
    if max_contacts < 1:
        raise ValueError("max_contacts must be at least 1.")

    scores = _as_numpy_1d(confidence_scores)
    if scores.size == 0 or k == 0:
        selected = np.zeros(scores.shape[0], dtype=bool)
        return selected if return_mask else np.array([], dtype=int)

    effective_k = min(int(k), int(max_contacts), int(scores.shape[0]))
    if effective_k == 0:
        selected = np.zeros(scores.shape[0], dtype=bool)
        return selected if return_mask else np.array([], dtype=int)

    # Stable descending ranking keeps the result deterministic when scores tie.
    ranked_indices = np.argsort(-scores, kind="mergesort")
    chosen_indices = ranked_indices[:effective_k]

    if return_mask:
        mask = np.zeros(scores.shape[0], dtype=bool)
        mask[chosen_indices] = True
        return mask
    return chosen_indices


def extract_confidence_scores(
    estimator: Any, X: Any, *, positive_label: Any = 1
) -> np.ndarray:
    """Extract confidence scores from a fitted estimator.

    The function prefers `predict_proba`. If that is unavailable, it falls back
    to `decision_function`. This keeps the targeting policy separate from the
    fitted model itself.
    """

    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(X)
        probabilities = np.asarray(probabilities)

        if probabilities.ndim == 1:
            return probabilities.reshape(-1)

        classes = getattr(estimator, "classes_", None)
        if classes is not None and positive_label in classes:
            positive_index = int(np.where(np.asarray(classes) == positive_label)[0][0])
        else:
            positive_index = probabilities.shape[1] - 1

        return probabilities[:, positive_index].reshape(-1)

    if hasattr(estimator, "decision_function"):
        decision_scores = np.asarray(estimator.decision_function(X))

        if decision_scores.ndim == 1:
            return decision_scores.reshape(-1)

        classes = getattr(estimator, "classes_", None)
        if classes is not None and decision_scores.shape[1] == len(classes):
            if positive_label in classes:
                positive_index = int(
                    np.where(np.asarray(classes) == positive_label)[0][0]
                )
            else:
                positive_index = decision_scores.shape[1] - 1
            return decision_scores[:, positive_index].reshape(-1)

        return decision_scores[:, -1].reshape(-1)

    raise ValueError("The estimator must expose predict_proba or decision_function.")


@dataclass(slots=True)
class BusinessScorer:
    """Scikit-learn-compatible scorer for the Project 2 business objective.

    Use this directly in GridSearchCV or cross-validation as `scoring=BusinessScorer(...)`.
    The scorer evaluates the current estimator by ranking customers on confidence,
    contacting the top-k cases, and then applying the business penalty formula.
    """

    k: int = 1000
    max_contacts: int = 1000
    positive_label: Any = 1

    def __call__(self, estimator: Any, X: Any, y: Any) -> float:
        confidence_scores = extract_confidence_scores(
            estimator, X, positive_label=self.positive_label
        )
        selected_mask = select_top_k_targets(
            confidence_scores, self.k, max_contacts=self.max_contacts, return_mask=True
        )
        n_variables = infer_n_variables(estimator, X=X)
        return compute_business_score(
            y,
            selected_mask=selected_mask,
            n_variables=n_variables,
            positive_label=self.positive_label,
        )


def evaluate_business_score_over_k(
    estimator: Any,
    X: Any,
    y: Any,
    k_values: Sequence[int],
    *,
    cv: int = 5,
    max_contacts: int = 1000,
    positive_label: Any = 1,
    random_state: int = 42,
) -> pd.DataFrame:
    """Evaluate multiple k values with cross-validation.

    This function is useful when you want to tune the number of contacted customers
    before running a final grid search with the best business-aware targeting policy.
    """

    y_array = _as_numpy_1d(y)
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    rows: list[dict[str, float | int]] = []
    for k in k_values:
        fold_scores: list[float] = []
        for train_indices, valid_indices in splitter.split(np.asarray(X), y_array):
            X_train = _row_subset(X, train_indices)
            X_valid = _row_subset(X, valid_indices)
            y_train = _row_subset(y_array, train_indices)
            y_valid = _row_subset(y_array, valid_indices)

            fitted_estimator = clone(estimator)
            fitted_estimator.fit(X_train, y_train)

            confidence_scores = extract_confidence_scores(
                fitted_estimator,
                X_valid,
                positive_label=positive_label,
            )
            selected_mask = select_top_k_targets(
                confidence_scores,
                k,
                max_contacts=max_contacts,
                return_mask=True,
            )
            n_variables = infer_n_variables(fitted_estimator, X=X_train)
            fold_score = compute_business_score(
                y_valid,
                selected_mask=selected_mask,
                n_variables=n_variables,
                positive_label=positive_label,
            )
            fold_scores.append(fold_score)

        rows.append(
            {
                "k": int(k),
                "mean_score": float(np.mean(fold_scores)),
                "std_score": float(np.std(fold_scores, ddof=0)),
                "min_score": float(np.min(fold_scores)),
                "max_score": float(np.max(fold_scores)),
            }
        )

    return pd.DataFrame(rows)


def make_business_scorer(
    *,
    k: int = 1000,
    max_contacts: int = 1000,
    positive_label: Any = 1,
) -> BusinessScorer:
    """Convenience constructor for GridSearchCV and other sklearn APIs."""

    return BusinessScorer(k=k, max_contacts=max_contacts, positive_label=positive_label)


__all__ = [
    "BusinessScorer",
    "compute_business_score",
    "evaluate_business_score_over_k",
    "extract_confidence_scores",
    "get_selected_feature_names",
    "infer_n_variables",
    "make_business_scorer",
    "select_top_k_targets",
]
