"""Run the Project 2 Logistic Regression business experiment.

How to run (from `Project2_AML/Models`):
    python logistic_regression_experiment.py
    python logistic_regression_business_experiment.py
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
from utils.logistic_regression_workflow import (  # noqa: E402
    build_logistic_regression_pipeline,
)


def _prepare_output_dir() -> Path:
    output_dir = SCRIPT_DIR / "outputs" / "logistic_regression"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _k_values(*, quick: bool = False) -> list[int]:
    if quick:
        return [200, 500, 1000]
    return [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]


def _preprocess_param_grid(*, quick: bool) -> dict[str, list]:
    if quick:
        return {
            "variance_filter__threshold": [0.01, 0.05],
            "correlation_filter__threshold": [0.9],
        }
    return {
        "variance_filter__threshold": [0.01, 0.05],
        "correlation_filter__threshold": [0.9, 0.8],
    }


def _l1_param_grid(*, quick: bool) -> dict[str, list]:
    grid = {
        **_preprocess_param_grid(quick=quick),
        "selector__threshold": ["median", "1.25*mean"]
        if quick
        else ["median", "1.25*mean", "2*mean"],
        "model__C": [0.1, 1.0] if quick else [0.1, 1.0, 3.0],
    }
    grid["selector__estimator__C"] = [0.01, 0.1] if quick else [0.001, 0.01, 0.1]
    return grid


def _extra_trees_param_grid(*, quick: bool) -> dict[str, list]:
    return {
        **_preprocess_param_grid(quick=quick),
        "selector__threshold": ["median", "1.25*mean"]
        if quick
        else ["mean", "median", "1.25*mean", "2*mean"],
        "model__C": [0.1, 1.0] if quick else [0.1, 1.0, 3.0],
    }


def _fit_selector_grid(
    X_train,
    y_train,
    *,
    best_k: int,
    selector_kind: str,
    quick: bool,
    cv: int,
) -> GridSearchCV:
    pipeline = build_logistic_regression_pipeline(
        selector_kind=selector_kind,  # type: ignore[arg-type]
        selector_threshold="median",
    )
    param_grid = (
        _l1_param_grid(quick=quick)
        if selector_kind == "l1"
        else _extra_trees_param_grid(quick=quick)
    )
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=BusinessScorer(k=best_k, max_contacts=1000, positive_label=1),
        cv=cv,
        n_jobs=-1,
        refit=True,
        error_score=np.nan,
    )
    grid.fit(X_train, y_train)
    return grid


def run_experiment(*, quick: bool = False) -> dict[str, object]:
    """Tune k, then grid-search L1 and ExtraTrees selectors; keep the best."""

    output_dir = _prepare_output_dir()
    cv = 3 if quick else 5

    data_dir = SCRIPT_DIR.parent / "data"
    X_train, y_train, X_test = load_data(path=f"{data_dir}/")
    y_train = y_train.squeeze()

    # k-search on a sparse default pipeline (ExtraTrees + median threshold).
    k_probe = build_logistic_regression_pipeline(
        selector_kind="extra_trees",
        selector_threshold="median",
    )
    k_results = evaluate_business_score_over_k(
        k_probe,
        X_train,
        y_train,
        _k_values(quick=quick),
        cv=cv,
        max_contacts=1000,
        positive_label=1,
        random_state=42,
    )
    k_results.to_csv(output_dir / "k_search_results.csv", index=False)

    best_k_row = k_results.loc[k_results["mean_score"].idxmax()]
    best_k = int(best_k_row["k"])

    grids: list[tuple[str, GridSearchCV]] = []
    for selector_kind in ("l1", "extra_trees"):
        print(f"Grid search: Logistic Regression with selector={selector_kind}")
        grids.append(
            (
                selector_kind,
                _fit_selector_grid(
                    X_train,
                    y_train,
                    best_k=best_k,
                    selector_kind=selector_kind,
                    quick=quick,
                    cv=cv,
                ),
            )
        )

    selector_kind, grid = max(
        grids,
        key=lambda item: item[1].best_score_ if item[1].best_score_ == item[1].best_score_ else float("-inf"),
    )
    if grid.best_score_ != grid.best_score_:
        raise RuntimeError("All grid-search candidates failed. Relax the parameter grid.")
    best_pipeline = grid.best_estimator_
    joblib.dump(best_pipeline, output_dir / "logistic_regression_pipeline.joblib")

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
        "best_selector_kind": selector_kind,
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
    print("Best selector:", selector_kind)
    print("Best model params:", grid.best_params_)
    print("Best CV score:", grid.best_score_)
    print("Selected features:", summary["n_variables"])
    print("Outputs written to:", output_dir)

    return summary


run_logistic_regression_experiment = run_experiment


if __name__ == "__main__":
    run_experiment()
