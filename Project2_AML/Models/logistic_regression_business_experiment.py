"""Run the Project 2 Logistic Regression business experiment.

How to run (from `Project2_AML/Models`):
    python logistic_regression_experiment.py
    python logistic_regression_business_experiment.py

Tuning strategy:
1. Cross-validated k search for the contact budget.
2. RandomizedSearchCV over preprocessing, selector, and model hyperparameters.
3. GridSearchCV refinement around the best randomized candidates.
4. Dedicated sequential forward feature selection phase (kept separate because it is slow).
5. Prefer models that pass the feature-diversity guardrails.
"""

from __future__ import annotations

import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, uniform
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

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
    feature_dominance_ratio,
    passes_feature_diversity_check,
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)


@dataclass(frozen=True)
class SearchResult:
    name: str
    search: GridSearchCV | RandomizedSearchCV
    diversity_ok: bool

    @property
    def score(self) -> float:
        value = self.search.best_score_
        return float(value) if value == value else float("-inf")

    @property
    def pipeline(self):
        return self.search.best_estimator_


def _prepare_output_dir() -> Path:
    output_dir = SCRIPT_DIR / "outputs" / "logistic_regression"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _k_values(*, quick: bool = False, fast: bool = False) -> list[int]:
    if fast:
        return [500, 1000]
    if quick:
        return [200, 500, 1000]
    return [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]


def _business_scorer(best_k: int) -> BusinessScorer:
    return BusinessScorer(k=best_k, max_contacts=1000, positive_label=1)


def _preprocess_grid(*, quick: bool) -> dict[str, list[Any]]:
    if quick:
        return {
            "variance_filter__threshold": [0.01, 0.05],
            "correlation_filter__threshold": [0.9],
        }
    return {
        "variance_filter__threshold": [0.005, 0.01, 0.02, 0.05],
        "correlation_filter__threshold": [0.75, 0.8, 0.85, 0.9],
    }


def _model_grid(*, quick: bool) -> dict[str, list[Any]]:
    if quick:
        return {
            "model__C": [0.1, 1.0, 3.0],
            "model__l1_ratio": [0.0, 0.5],
            "model__class_weight": [None, "balanced"],
        }
    return {
        "model__C": [0.01, 0.05, 0.1, 0.3, 1.0, 3.0, 10.0],
        "model__l1_ratio": [0.0, 0.15, 0.35, 0.5, 0.75],
        "model__class_weight": [None, "balanced"],
    }


def _l1_param_grid(*, quick: bool) -> dict[str, list[Any]]:
    return {
        **_preprocess_grid(quick=quick),
        **_model_grid(quick=quick),
        "selector__threshold": ["median", "1.25*mean", "2*mean"]
        if not quick
        else ["median", "1.25*mean"],
        "selector__estimator__C": [0.001, 0.01, 0.1, 1.0]
        if not quick
        else [0.01, 0.1],
    }


def _extra_trees_param_grid(*, quick: bool) -> dict[str, list[Any]]:
    return {
        **_preprocess_grid(quick=quick),
        **_model_grid(quick=quick),
        "selector__threshold": ["mean", "median", "1.25*mean", "2*mean"]
        if not quick
        else ["median", "1.25*mean"],
    }


def _kbest_param_grid(*, quick: bool) -> dict[str, list[Any]]:
    return {
        **_preprocess_grid(quick=quick),
        **_model_grid(quick=quick),
        "selector__k": [15, 25, 40, 60] if not quick else [20, 40],
    }


def _sfs_param_grid(*, quick: bool) -> dict[str, list[Any]]:
    return {
        "kbest_prefilter__k": [30, 40, 50] if not quick else [30, 40],
        "selector__n_features_to_select": [8, 10, 12, 15] if not quick else [8, 10],
    }


def _randomized_distributions(selector_kind: str) -> dict[str, Any]:
    distributions: dict[str, Any] = {
        "variance_filter__threshold": uniform(0.005, 0.045),
        "correlation_filter__threshold": uniform(0.72, 0.18),
        "model__C": loguniform(1e-3, 1e2),
        "model__l1_ratio": uniform(0.0, 0.85),
        "model__class_weight": [None, "balanced"],
    }
    if selector_kind == "l1":
        distributions["selector__threshold"] = [
            "median",
            "mean",
            "1.25*mean",
            "2*mean",
            "0.75*mean",
        ]
        distributions["selector__estimator__C"] = loguniform(1e-4, 1.0)
    elif selector_kind == "extra_trees":
        distributions["selector__threshold"] = [
            "mean",
            "median",
            "1.25*mean",
            "2*mean",
            "0.75*mean",
        ]
    else:
        distributions["selector__k"] = [10, 15, 20, 25, 30, 40, 50, 60, 80]
    return distributions


def _fit_search(
    X_train,
    y_train,
    *,
    best_k: int,
    selector_kind: str,
    param_grid: dict[str, list[Any]],
    cv: int,
    search_kind: str,
    n_iter: int = 40,
    random_state: int = 42,
    n_jobs: int = -1,
) -> GridSearchCV | RandomizedSearchCV:
    pipeline = build_logistic_regression_pipeline(
        selector_kind=selector_kind,  # type: ignore[arg-type]
        selector_threshold="median",
        sfs_cv=2,
    )
    scoring = _business_scorer(best_k)
    if search_kind == "randomized":
        search: GridSearchCV | RandomizedSearchCV = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=_randomized_distributions(selector_kind),
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            refit=True,
            random_state=random_state,
            error_score=np.nan,
        )
    else:
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            refit=True,
            error_score=np.nan,
        )
    search.fit(X_train, y_train)
    return search


def _refined_grid_from_random(
    selector_kind: str,
    best_params: dict[str, Any],
    *,
    quick: bool,
) -> dict[str, list[Any]]:
    """Build a narrow grid around the best randomized-search configuration."""

    def _around(value: float, factors: tuple[float, ...]) -> list[float]:
        values = sorted({max(value * factor, 1e-5) for factor in factors})
        return [float(v) for v in values]

    variance = float(best_params.get("variance_filter__threshold", 0.01))
    correlation = float(best_params.get("correlation_filter__threshold", 0.9))
    model_c = float(best_params.get("model__C", 1.0))
    l1_ratio = float(best_params.get("model__l1_ratio", 0.0))

    grid: dict[str, list[Any]] = {
        "variance_filter__threshold": sorted(
            {variance, max(variance * 0.5, 0.001), min(variance * 1.5, 0.1)}
        ),
        "correlation_filter__threshold": sorted(
            {correlation, max(correlation - 0.05, 0.7), min(correlation + 0.05, 0.95)}
        ),
        "model__C": _around(model_c, (0.3, 1.0, 3.0)),
        "model__l1_ratio": sorted(
            {
                0.0 if l1_ratio < 0.05 else l1_ratio,
                max(l1_ratio - 0.15, 0.0),
                min(l1_ratio + 0.15, 0.95),
            }
        ),
        "model__class_weight": [best_params.get("model__class_weight")],
    }

    if selector_kind == "l1":
        grid["selector__threshold"] = [best_params.get("selector__threshold", "median")]
        selector_c = float(best_params.get("selector__estimator__C", 0.01))
        grid["selector__estimator__C"] = _around(selector_c, (0.3, 1.0, 3.0))
    elif selector_kind == "extra_trees":
        grid["selector__threshold"] = [best_params.get("selector__threshold", "median")]
    else:
        k_value = int(best_params.get("selector__k", 30))
        grid["selector__k"] = sorted({max(k_value - 10, 5), k_value, k_value + 10})

    if quick:
        grid["model__C"] = grid["model__C"][:2]
    return grid


def _result_from_search(
    name: str,
    search: GridSearchCV | RandomizedSearchCV,
    X_train,
) -> SearchResult:
    diversity_ok = passes_feature_diversity_check(
        search.best_estimator_,
        min_features=3,
        max_dominance_ratio=0.65,
        X=X_train,
    )
    return SearchResult(name=name, search=search, diversity_ok=diversity_ok)


def _pick_best_result(results: list[SearchResult]) -> SearchResult:
    valid = [result for result in results if result.score == result.score]
    if not valid:
        raise RuntimeError("All tuning candidates failed. Relax the parameter grids.")

    diverse = [result for result in valid if result.diversity_ok]
    pool = diverse if diverse else valid

    return max(
        pool,
        key=lambda result: (
            result.score,
            -feature_dominance_ratio(result.pipeline),
            result.diversity_ok,
        ),
    )


def run_experiment(*, quick: bool = False, fast: bool = False) -> dict[str, object]:
    """Tune k, then run randomized search, grid search, and sequential selection.

    Use `fast=True` for a submission-ready run (~10-15 min): l1 + kbest only,
    skips SFS and broad grid search.
    """

    output_dir = _prepare_output_dir()
    cv = 3 if quick or fast else 5
    random_iter = 8 if fast else (12 if quick else 20)

    data_dir = SCRIPT_DIR.parent / "data"
    X_train, y_train, X_test = load_data(path=f"{data_dir}/")
    y_train = y_train.squeeze()

    k_probe = build_logistic_regression_pipeline(
        selector_kind="extra_trees",
        selector_threshold="median",
    )
    k_results = evaluate_business_score_over_k(
        k_probe,
        X_train,
        y_train,
        _k_values(quick=quick, fast=fast),
        cv=cv,
        max_contacts=1000,
        positive_label=1,
        random_state=42,
    )
    k_results.to_csv(output_dir / "k_search_results.csv", index=False)
    best_k = int(k_results.loc[k_results["mean_score"].idxmax(), "k"])

    results: list[SearchResult] = []
    tuning_records: list[dict[str, object]] = []

    fast_selector_configs = {
        "l1": _l1_param_grid,
        "extra_trees": _extra_trees_param_grid,
        "kbest": _kbest_param_grid,
    }
    if fast:
        fast_selector_configs = {
            "l1": _l1_param_grid,
            "kbest": _kbest_param_grid,
        }

    for selector_kind, grid_builder in fast_selector_configs.items():
        print(f"Randomized search: selector={selector_kind}")
        random_search = _fit_search(
            X_train,
            y_train,
            best_k=best_k,
            selector_kind=selector_kind,
            param_grid={},
            cv=cv,
            search_kind="randomized",
            n_iter=random_iter,
        )
        random_result = _result_from_search(
            f"randomized_{selector_kind}",
            random_search,
            X_train,
        )
        results.append(random_result)
        tuning_records.append(
            {
                "stage": "randomized",
                "selector_kind": selector_kind,
                "best_score": random_result.score,
                "best_params": random_search.best_params_,
                "diversity_ok": random_result.diversity_ok,
            }
        )

        print(f"Grid search (refined): selector={selector_kind}")
        refined_grid = _refined_grid_from_random(
            selector_kind,
            random_search.best_params_,
            quick=quick,
        )
        grid_search = _fit_search(
            X_train,
            y_train,
            best_k=best_k,
            selector_kind=selector_kind,
            param_grid=refined_grid,
            cv=cv,
            search_kind="grid",
        )
        grid_result = _result_from_search(
            f"grid_refined_{selector_kind}",
            grid_search,
            X_train,
        )
        results.append(grid_result)
        tuning_records.append(
            {
                "stage": "grid_refined",
                "selector_kind": selector_kind,
                "best_score": grid_result.score,
                "best_params": grid_search.best_params_,
                "diversity_ok": grid_result.diversity_ok,
            }
        )

    if not fast:
        print("Sequential feature selection: dedicated grid search")
        sfs_seed = _pick_best_result(
            [result for result in results if not result.name.endswith("_sfs")]
        )
        sfs_base = {
            key: [value]
            for key, value in sfs_seed.search.best_params_.items()
            if key.startswith(("variance_filter__", "correlation_filter__", "model__"))
        }
        sfs_param_grid = {**sfs_base, **_sfs_param_grid(quick=quick)}
        sfs_search = _fit_search(
            X_train,
            y_train,
            best_k=best_k,
            selector_kind="sfs",
            param_grid=sfs_param_grid,
            cv=cv,
            search_kind="grid",
            n_jobs=1,
        )
        sfs_result = _result_from_search("grid_sfs", sfs_search, X_train)
        results.append(sfs_result)
        tuning_records.append(
            {
                "stage": "sequential_feature_selection",
                "selector_kind": "sfs",
                "best_score": sfs_result.score,
                "best_params": sfs_search.best_params_,
                "diversity_ok": sfs_result.diversity_ok,
            }
        )

    best_result = _pick_best_result(results)
    best_search = best_result.search
    best_pipeline = best_result.pipeline

    winning_selector = str(best_result.name).rsplit("_", maxsplit=1)[-1]
    if winning_selector in {"l1", "extra_trees", "kbest"} and not quick and not fast:
        print(f"Grid search (broad winner): selector={winning_selector}")
        broad_search = _fit_search(
            X_train,
            y_train,
            best_k=best_k,
            selector_kind=winning_selector,
            param_grid=fast_selector_configs[winning_selector](quick=False),
            cv=cv,
            search_kind="grid",
            n_jobs=2,
        )
        broad_result = _result_from_search(
            f"grid_broad_{winning_selector}",
            broad_search,
            X_train,
        )
        results.append(broad_result)
        tuning_records.append(
            {
                "stage": "grid_broad",
                "selector_kind": winning_selector,
                "best_score": broad_result.score,
                "best_params": broad_search.best_params_,
                "diversity_ok": broad_result.diversity_ok,
            }
        )
        best_result = _pick_best_result(results)
        best_search = best_result.search
        best_pipeline = best_result.pipeline

    if not fast:
        print("Re-tuning k on the best pipeline")
        k_results = evaluate_business_score_over_k(
            best_pipeline,
            X_train,
            y_train,
            _k_values(quick=quick, fast=fast),
            cv=cv,
            max_contacts=1000,
            positive_label=1,
            random_state=42,
        )
        k_results.to_csv(output_dir / "k_search_results.csv", index=False)
        best_k = int(k_results.loc[k_results["mean_score"].idxmax(), "k"])
    best_k_cv_score = float(k_results.loc[k_results["k"] == best_k, "mean_score"].iloc[0])
    joblib.dump(best_pipeline, output_dir / "logistic_regression_pipeline.joblib")

    tuning_df = pd.DataFrame(tuning_records).sort_values("best_score", ascending=False)
    tuning_df.to_csv(output_dir / "tuning_results.csv", index=False)

    selected_variables = get_selected_feature_names(best_pipeline, X_train)
    (output_dir / "selected_variables.txt").write_text(
        "\n".join(selected_variables) + "\n",
        encoding="utf-8",
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

    dominance = feature_dominance_ratio(best_pipeline)
    summary = {
        "best_k": best_k,
        "best_k_cv_score": best_k_cv_score,
        "best_search_name": best_result.name,
        "best_model_params": best_search.best_params_,
        "best_cv_business_score": float(best_search.best_score_),
        "feature_dominance_ratio": dominance,
        "passes_feature_diversity_check": best_result.diversity_ok,
        "n_variables": infer_n_variables(best_pipeline, X=X_train),
        "selected_variables": selected_variables,
        "selected_customers": int(selected_mask.sum()),
        "selected_customer_indices": selected_indices.tolist(),
        "tuning_candidates": len(results),
        "output_dir": str(output_dir),
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("Best k:", best_k)
    print("Best search:", best_result.name)
    print("Best model params:", best_search.best_params_)
    print("Best CV score:", best_search.best_score_)
    print("Feature dominance ratio:", dominance)
    print("Diversity check passed:", best_result.diversity_ok)
    print("Selected features:", summary["n_variables"])
    print("Outputs written to:", output_dir)

    return summary


run_logistic_regression_experiment = run_experiment


if __name__ == "__main__":
    run_experiment(fast=True)
