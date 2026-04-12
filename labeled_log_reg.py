from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import FISTA


@dataclass
class _LambdaResult:
    lambda_value: float
    coef: np.ndarray
    intercept: float


class LabeledLogReg:
    """Logistic regression wrapper that selects lambda by validation.

    This version keeps the implementation intentionally small: coefficients are
    obtained from the shared `FISTA.FISTA` function, while this class handles
    data validation, lambda search, scoring, prediction, and plotting.
    """

    _METRIC_ALIASES = {
        "recall": "recall",
        "precision": "precision",
        "f1": "f1",
        "f_measure": "f1",
        "f-measure": "f1",
        "balanced_accuracy": "balanced_accuracy",
        "balanced accuracy": "balanced_accuracy",
        "roc_auc": "roc_auc",
        "roc auc": "roc_auc",
        "area_under_roc_curve": "roc_auc",
        "area under roc curve": "roc_auc",
        "average_precision": "average_precision",
        "pr_auc": "average_precision",
        "area_under_the_sensitivity_precision_curve": "average_precision",
        "area_under_sensitivity_precision_curve": "average_precision",
        "area under the sensitivity precision curve": "average_precision",
        "area under sensitivity precision curve": "average_precision",
    }

    def __init__(
        self,
        implementation: str = "fista",
        lambda_grid: np.ndarray | list[float] | None = None,
        n_lambdas: int = 30,
        lambda_min_ratio: float = 1e-3,
        max_iter: int = 500,
        standardize: bool = True,
    ) -> None:
        implementation = implementation.lower().strip()
        if implementation != "fista":
            raise ValueError("Only implementation='fista' is supported here. Use sklearn directly in the notebook for comparison.")
        if lambda_grid is not None:
            lambda_grid = np.asarray(lambda_grid, dtype=float).ravel()
            if lambda_grid.size == 0:
                raise ValueError("lambda_grid must not be empty.")
            if np.any(lambda_grid <= 0):
                raise ValueError("lambda values must be strictly positive.")
        if n_lambdas < 1:
            raise ValueError("n_lambdas must be at least 1.")
        if not (0.0 < lambda_min_ratio <= 1.0):
            raise ValueError("lambda_min_ratio must be in (0, 1].")

        self.implementation = implementation
        self.lambda_grid = lambda_grid
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.max_iter = max_iter
        self.standardize = standardize

        self.is_fitted_ = False
        self.is_validated_ = False
        self.best_index_: int | None = None
        self.best_lambda_: float | None = None
        self.best_validation_score_: float | None = None
        self.validation_history_: dict[str, np.ndarray] = {}
        self.models_: list[_LambdaResult] = []

    @staticmethod
    def _to_numpy(array: Any) -> np.ndarray:
        if isinstance(array, (pd.Series, pd.DataFrame)):
            return array.to_numpy()
        return np.asarray(array)

    def _validate_X(self, X: Any, name: str) -> np.ndarray:
        if isinstance(X, pd.DataFrame) and hasattr(self, "feature_names_in_"):
            X = X.reindex(columns=self.feature_names_in_, fill_value=0)
        X = self._to_numpy(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if pd.isna(X).any():
            raise ValueError(f"{name} contains missing values. This implementation requires complete data.")
        try:
            return X.astype(float, copy=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be numeric.") from exc

    def _validate_y(self, y: Any, name: str) -> np.ndarray:
        y = self._to_numpy(y).reshape(-1)
        if pd.isna(y).any():
            raise ValueError(f"{name} contains missing values. This implementation requires complete data.")
        try:
            return y.astype(float, copy=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be numeric binary labels.") from exc

    def _prepare_X(self, X: Any, fit: bool = False) -> np.ndarray:
        if fit and isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        elif fit:
            self.feature_names_in_ = None

        X = self._validate_X(X, "X")
        if fit:
            self.n_features_in_ = X.shape[1]
            if self.standardize:
                self.feature_mean_ = X.mean(axis=0)
                self.feature_scale_ = X.std(axis=0, ddof=0)
                self.feature_scale_[self.feature_scale_ == 0.0] = 1.0
            else:
                self.feature_mean_ = np.zeros(X.shape[1], dtype=float)
                self.feature_scale_ = np.ones(X.shape[1], dtype=float)

        if not hasattr(self, "feature_mean_") or not hasattr(self, "feature_scale_"):
            raise RuntimeError("Call fit before transforming new data.")

        return (X - self.feature_mean_) / self.feature_scale_

    def _check_binary_labels(self, y: np.ndarray, name: str) -> np.ndarray:
        unique_values = np.unique(y)
        if unique_values.size != 2:
            raise ValueError(f"{name} must contain exactly two classes.")
        self.classes_ = unique_values
        self.negative_class_ = unique_values[0]
        self.positive_class_ = unique_values[1]
        return (y == self.positive_class_).astype(float)

    def _resolve_lambda_grid(self, X: np.ndarray, y_binary: np.ndarray) -> np.ndarray:
        if self.lambda_grid is not None:
            return np.unique(np.sort(self.lambda_grid)[::-1])

        positive_rate = float(np.clip(y_binary.mean(), 1e-8, 1.0 - 1e-8))
        intercept_only = np.full((X.shape[0], 1), positive_rate, dtype=float)
        centered_response = intercept_only - y_binary.reshape(-1, 1)
        lambda_max = np.max(np.abs(X.T @ centered_response)) / max(X.shape[0], 1)
        if not np.isfinite(lambda_max) or lambda_max <= 0.0:
            lambda_max = 1.0

        lambda_min = lambda_max * self.lambda_min_ratio
        grid = np.logspace(np.log10(lambda_max), np.log10(lambda_min), num=self.n_lambdas)
        grid = np.unique(np.asarray(grid, dtype=float))
        return grid[::-1] if grid[0] < grid[-1] else grid

    def fit(self, X_train: Any, y_train: Any) -> "LabeledLogReg":
        X = self._prepare_X(X_train, fit=True)
        y = self._validate_y(y_train, "y_train")
        y_binary = self._check_binary_labels(y, "y_train")

        self.lambdas_ = self._resolve_lambda_grid(X, y_binary)

        self.models_ = []
        warm_start = np.zeros((X.shape[1] + 1, 1), dtype=float)
        for lambda_value in self.lambdas_:
            coef, intercept = FISTA.FISTA(
                X,
                y_binary,
                lam=float(lambda_value),
                bet=warm_start,
                iterations=self.max_iter,
            )
            coef = np.asarray(coef, dtype=float).reshape(-1)
            intercept_value = float(np.asarray(intercept).reshape(-1)[0])
            self.models_.append(_LambdaResult(lambda_value=float(lambda_value), coef=coef, intercept=intercept_value))
            warm_start = np.concatenate([coef.reshape(-1, 1), np.array([[intercept_value]])], axis=0)

        self.is_fitted_ = True
        self.is_validated_ = False
        self.best_index_ = None
        self.best_lambda_ = None
        self.best_validation_score_ = None
        self.validation_history_ = {}
        return self

    def _normalize_metric_name(self, measure: str) -> str:
        normalized = measure.lower().strip().replace("-", " ").replace("/", " ")
        normalized = " ".join(normalized.split())
        normalized = normalized.replace(" ", "_")
        return self._METRIC_ALIASES.get(normalized, normalized)

    def _predict_positive_proba_from_model(self, X: np.ndarray, model: _LambdaResult) -> np.ndarray:
        logits = X @ model.coef + model.intercept
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))

    def _score_measure(self, y_true: np.ndarray, proba: np.ndarray, measure: str) -> float:
        measure = self._normalize_metric_name(measure)
        y_pred = (proba >= 0.5).astype(int)

        if measure == "recall":
            return float(recall_score(y_true, y_pred, zero_division=0))
        if measure == "precision":
            return float(precision_score(y_true, y_pred, zero_division=0))
        if measure == "f1":
            return float(f1_score(y_true, y_pred, zero_division=0))
        if measure == "balanced_accuracy":
            return float(balanced_accuracy_score(y_true, y_pred))
        if measure == "roc_auc":
            if np.unique(y_true).size < 2:
                raise ValueError("roc_auc requires both classes to be present in the validation data.")
            return float(roc_auc_score(y_true, proba))
        if measure == "average_precision":
            if np.unique(y_true).size < 2:
                raise ValueError("area under the sensitivity-precision curve requires both classes to be present in the validation data.")
            return float(average_precision_score(y_true, proba))

        raise ValueError(
            "Unknown measure. Use one of: recall, precision, f_measure, balanced accuracy, roc_auc, or area under the sensitivity-precision curve."
        )

    def validate(self, X_valid: Any, y_valid: Any, measure: str) -> dict[str, Any]:
        if not self.is_fitted_:
            raise RuntimeError("Call fit before validate.")

        X = self._prepare_X(X_valid, fit=False)
        y = self._validate_y(y_valid, "y_valid")
        y_binary = (y == self.positive_class_).astype(int)

        valid_classes = np.unique(y)
        if valid_classes.size != 2 or set(valid_classes.tolist()) != set(self.classes_.tolist()):
            raise ValueError("y_valid must use the same two classes as y_train.")

        normalized_measure = self._normalize_metric_name(measure)
        scores = np.array(
            [self._score_measure(y_binary, self._predict_positive_proba_from_model(X, model), normalized_measure) for model in self.models_],
            dtype=float,
        )

        best_index = int(np.nanargmax(scores))
        self.best_index_ = best_index
        self.best_lambda_ = float(self.models_[best_index].lambda_value)
        self.best_validation_score_ = float(scores[best_index])
        self.is_validated_ = True
        self.validation_history_[normalized_measure] = scores

        return {
            "measure": normalized_measure,
            "lambdas": self.lambdas_.copy(),
            "scores": scores,
            "best_index": best_index,
            "best_lambda": self.best_lambda_,
            "best_score": self.best_validation_score_,
        }

    def predict_proba(self, X_test: Any) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("Call fit before predict_proba.")
        if not self.is_validated_ or self.best_index_ is None:
            raise RuntimeError("Call validate before predict_proba so the best lambda can be selected.")

        X = self._prepare_X(X_test, fit=False)
        model = self.models_[self.best_index_]
        proba_pos = self._predict_positive_proba_from_model(X, model)
        return np.column_stack([1.0 - proba_pos, proba_pos])

    def predict(self, X_test: Any) -> np.ndarray:
        proba = self.predict_proba(X_test)[:, 1]
        return np.where(proba >= 0.5, self.positive_class_, self.negative_class_)

    def plot(
        self,
        measure: str,
        ax: plt.Axes | None = None,
        figsize: tuple[float, float] = (8.0, 5.0),
        show: bool = True,
        filename: str | None = None,
    ) -> plt.Axes:
        if not self.validation_history_:
            raise RuntimeError("Call validate before plot so the measure values are available.")

        normalized_measure = self._normalize_metric_name(measure)
        if normalized_measure not in self.validation_history_:
            raise ValueError(f"Measure '{measure}' has not been validated yet.")

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        order = np.argsort(self.lambdas_)
        lambdas = self.lambdas_[order]
        scores = self.validation_history_[normalized_measure][order]

        ax.plot(lambdas, scores, marker="o")
        ax.set_xscale("log")
        ax.set_xlabel("lambda")
        ax.set_ylabel(normalized_measure.replace("_", " "))
        ax.set_title(f"Validation {normalized_measure.replace('_', ' ')} vs lambda")
        ax.grid(True, alpha=0.3)

        if self.best_lambda_ is not None:
            ax.axvline(self.best_lambda_, color="tab:red", linestyle="--", alpha=0.7)

        if filename is not None:
            ax.figure.savefig(filename, bbox_inches="tight")
        if show:
            plt.show()

        return ax

    def plot_coefficients(
        self,
        ax: plt.Axes | None = None,
        feature_names: list[str] | np.ndarray | None = None,
        figsize: tuple[float, float] = (10.0, 6.0),
        show: bool = True,
        filename: str | None = None,
    ) -> plt.Axes:
        if not self.is_fitted_:
            raise RuntimeError("Call fit before plot_coefficients.")

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        order = np.argsort(self.lambdas_)
        lambdas = self.lambdas_[order]
        coefficients = np.vstack([model.coef for model in self.models_])[order]

        if feature_names is None:
            feature_names = [f"x{i}" for i in range(coefficients.shape[1])]
        else:
            feature_names = list(feature_names)
            if len(feature_names) != coefficients.shape[1]:
                raise ValueError("feature_names must have the same length as the number of input features.")

        for feature_index, feature_name in enumerate(feature_names):
            ax.plot(lambdas, coefficients[:, feature_index], label=feature_name)

        ax.set_xscale("log")
        ax.set_xlabel("lambda")
        ax.set_ylabel("coefficient value")
        ax.set_title("Coefficient path")
        ax.grid(True, alpha=0.3)

        if len(feature_names) <= 12:
            ax.legend(loc="best", fontsize="small")

        if filename is not None:
            ax.figure.savefig(filename, bbox_inches="tight")
        if show:
            plt.show()

        return ax


LabeledLogisticRegression = LabeledLogReg
