"""Run the Project 2 SVM business experiment.

This mirrors `logistic_regression_business_experiment.py` so teammates can compare
both assigned linear models with the same business-score workflow.

How to run (from `Project2_AML/Models`):
    python svm_experiment.py
    python svm_business_experiment.py

Outputs are written to `outputs/svm/`:
    - k_search_results.csv
    - svm_pipeline.joblib
    - test_rankings.csv
    - selected_variables.txt
    - summary.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from utils.business_scoring import (  # noqa: E402
    BusinessScorer,
    evaluate_business_score_over_k,
    extract_confidence_scores,
    get_selected_feature_names,
    infer_n_variables,
    select_top_k_targets,
)
from utils.data_processing import load_data  # noqa: E402
from utils.svm_workflow import build_svm_pipeline  # noqa: E402


def _prepare_output_dir() -> Path:
    output_dir = SCRIPT_DIR / "outputs" / "svm"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _k_values(*, quick: bool = False) -> list[int]:
    # Project rule: contact at most 1000 customers; search k in steps up to that cap.
    if quick:
        return [250, 500, 1000]
    return list(range(50, 1001, 50))


def run_experiment(*, quick: bool = False) -> dict[str, object]:
    """Fit and evaluate the SVM workflow with the Project 2 business score.

    Set `quick=True` in notebooks for a faster smoke run (fewer k values and a
    smaller grid). Use the default full search for final model selection.
    """

    output_dir = _prepare_output_dir()

    data_dir = SCRIPT_DIR.parent / "data"
    X_train, y_train, X_test = load_data(path=f"{data_dir}/")
    y_train = y_train.squeeze()

    base_pipeline = build_svm_pipeline(
        variance_threshold=0.01,
        correlation_threshold=0.9,
        selector_c=1.0,
        model_c=1.0,
        kernel="rbf",
        gamma="scale",
        random_state=42,
        max_iter=5000,
    )

    k_results = evaluate_business_score_over_k(
        base_pipeline,
        X_train,
        y_train,
        _k_values(quick=quick),
        cv=3 if quick else 5,
        max_contacts=1000,
        positive_label=1,
        random_state=42,
    )
    k_results.to_csv(output_dir / "k_search_results.csv", index=False)

    best_k_row = k_results.loc[k_results["mean_score"].idxmax()]
    best_k = int(best_k_row["k"])

    # Same two-stage tuning as Logistic Regression: pick k first, then tune C.
    param_grid = {
        "selector__estimator__C": [0.1, 1.0] if quick else [0.1, 0.3, 1.0, 3.0],
        "model__C": [0.1, 1.0] if quick else [0.1, 0.3, 1.0, 3.0],
        "model__gamma": ["scale"] if quick else ["scale", 0.01, 0.1],
    }
    grid = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        scoring=BusinessScorer(k=best_k, max_contacts=1000, positive_label=1),
        cv=3 if quick else 5,
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )
    grid.fit(X_train, y_train)

    best_pipeline = grid.best_estimator_
    joblib.dump(best_pipeline, output_dir / "svm_pipeline.joblib")

    selected_variables = get_selected_feature_names(best_pipeline, X_train)
    (output_dir / "selected_variables.txt").write_text(
        "\n".join(selected_variables) + "\n", encoding="utf-8"
    )

    test_scores = extract_confidence_scores(best_pipeline, X_test, positive_label=1)
    selected_mask = select_top_k_targets(
        test_scores, best_k, max_contacts=1000, return_mask=True
    )
    selected_indices = np.flatnonzero(selected_mask)

    rankings = pd.DataFrame(
        {
            "row_index": np.arange(len(X_test)),
            "confidence_score": test_scores,
            "targeted": selected_mask.astype(int),
        }
    ).sort_values("confidence_score", ascending=False)
    rankings.to_csv(output_dir / "test_rankings.csv", index=False)

    summary = {
        "best_k": best_k,
        "best_k_cv_score": float(best_k_row["mean_score"]),
        "best_model_params": grid.best_params_,
        "best_cv_business_score": float(grid.best_score_),
        "n_variables": infer_n_variables(best_pipeline, X=X_train),
        "selected_variables": selected_variables,
        "selected_customers": int(selected_mask.sum()),
        "selected_customer_indices": selected_indices.tolist(),
        "output_dir": str(output_dir),
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("Best k:", best_k)
    print("Best model params:", grid.best_params_)
    print("Best CV score:", grid.best_score_)
    print("Selected features:", summary["n_variables"])
    print("Outputs written to:", output_dir)

    return summary


run_svm_experiment = run_experiment


if __name__ == "__main__":
    run_experiment()
