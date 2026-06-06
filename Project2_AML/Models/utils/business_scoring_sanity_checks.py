"""Small sanity checks for the Project 2 business scoring helpers.

Run this file directly to verify the core behaviors before wiring the scorer
into a notebook or a GridSearchCV run.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

# Keep this import flexible so the file works both from the notebook context
# (`from utils...`) and when run directly as a local smoke test.
try:
    from utils.business_scoring import (
        BusinessScorer,
        compute_business_score,
        evaluate_business_score_over_k,
        select_top_k_targets,
    )
except ImportError:
    from business_scoring import (
        BusinessScorer,
        compute_business_score,
        evaluate_business_score_over_k,
        select_top_k_targets,
    )


def test_score_from_mask_and_indices_agree() -> None:
    y_true = np.array([1, 0, 1, 0, 1])
    selected_mask = np.array([True, False, True, True, False])
    selected_indices = np.array([0, 2, 3])

    score_from_mask = compute_business_score(
        y_true, selected_mask=selected_mask, n_variables=3
    )
    score_from_indices = compute_business_score(
        y_true, selected_indices=selected_indices, n_variables=3
    )

    assert score_from_mask == score_from_indices


def test_top_k_policy_picks_highest_scores() -> None:
    scores = np.array([0.1, 0.9, 0.4, 0.8])
    selected_indices = select_top_k_targets(scores, 2, return_mask=False)

    assert list(selected_indices) == [1, 3]


def test_scorer_works_with_logistic_regression() -> None:
    """Smoke test with LogisticRegression, the Project 2 classifier in scope."""
    X, y = make_classification(
        n_samples=120,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        random_state=42,
    )

    log_reg = LogisticRegression(max_iter=500, solver="liblinear")
    log_reg.fit(X, y)
    scorer = BusinessScorer(k=20)
    score_prob = scorer(log_reg, X, y)

    cv_table = evaluate_business_score_over_k(log_reg, X, y, [10, 20], cv=3)

    assert isinstance(score_prob, float)
    assert list(cv_table["k"]) == [10, 20]
    assert {"mean_score", "std_score", "min_score", "max_score"}.issubset(
        cv_table.columns
    )


if __name__ == "__main__":
    test_score_from_mask_and_indices_agree()
    test_top_k_policy_picks_highest_scores()
    test_scorer_works_with_logistic_regression()
    print("Business scoring sanity checks passed.")
