from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class _LambdaModel:
    lambda_value: float
    coef: np.ndarray
    intercept: float
    estimator: LogisticRegression | None = None


class LabeledLogReg:
    """Logistic Lasso regression with a FISTA path or sklearn baseline.

    The class fits a sequence of models over a lambda grid on the training set,
    then selects the best lambda on the validation set using the requested
    metric.
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
        max_iter: int = 1000,
        tol: float = 1e-6,
        standardize: bool = True,
        random_state: int | None = 42,
    ) -> None:
        implementation = implementation.lower().strip()
        if implementation not in {"fista", "sklearn"}:
            raise ValueError("implementation must be either 'fista' or 'sklearn'.")

        if lambda_grid is not None and n_lambdas < 1:
            raise ValueError("n_lambdas must be at least 1.")

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
        self.tol = tol
        self.standardize = standardize
        self.random_state = random_state

        self.is_fitted_ = False
        self.is_validated_ = False
        self.best_index_: int | None = None
        self.best_lambda_: float | None = None
        self.best_validation_score_: float | None = None
        self.validation_history_: dict[str, np.ndarray] = {}
        self.models_: list[_LambdaModel] = []

    @staticmethod
    def _to_numpy(array: Any) -> np.ndarray:
        if isinstance(array, (pd.Series, pd.DataFrame)):
            return array.to_numpy()
        return np.asarray(array)

    @staticmethod
    def _soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
        return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)

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
        is_dataframe = isinstance(X, pd.DataFrame)
        if fit:
            if is_dataframe:
                self.feature_names_in_ = np.asarray(X.columns, dtype=object)
            else:
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

        positive_rate = float(y_binary.mean())
        if positive_rate <= 0.0 or positive_rate >= 1.0:
            raise ValueError("Both classes must be present in the training data.")

        residual = positive_rate - y_binary
        lambda_max = np.max(np.abs(X.T @ residual)) / X.shape[0]
        if not np.isfinite(lambda_max) or lambda_max <= 0.0:
            lambda_max = 1.0

        lambda_min = lambda_max * self.lambda_min_ratio
        grid = np.logspace(np.log10(lambda_max), np.log10(lambda_min), num=self.n_lambdas)
        grid = np.unique(np.asarray(grid, dtype=float))
        return grid[::-1] if grid[0] < grid[-1] else grid

    def _fit_fista_single(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lambda_value: float,
        step_size: float,
        warm_start: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float]:
        n_samples, n_features = X.shape
        X_aug = np.column_stack([X, np.ones(n_samples)])

        if warm_start is None:
            theta = np.zeros(n_features + 1, dtype=float)
        else:
            theta = warm_start.astype(float, copy=True)

        z = theta.copy()
        t = 1.0

        for _ in range(self.max_iter):
            logits = X_aug @ z
            probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
            gradient = (X_aug.T @ (probabilities - y)) / n_samples

            candidate = z - step_size * gradient
            next_theta = candidate.copy()
            next_theta[:-1] = self._soft_threshold(candidate[:-1], lambda_value * step_size)

            difference = next_theta - theta
            if np.linalg.norm(difference) <= self.tol * (1.0 + np.linalg.norm(theta)):
                theta = next_theta
                break

            next_t = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
            z = next_theta + ((t - 1.0) / next_t) * (next_theta - theta)
            theta = next_theta
            t = next_t

        return theta[:-1], float(theta[-1])

    def _fit_fista_path(self, X: np.ndarray, y: np.ndarray) -> list[_LambdaModel]:
        X_aug = np.column_stack([X, np.ones(X.shape[0])])
        spectral_norm = np.linalg.norm(X_aug, ord=2)
        lipschitz = (spectral_norm * spectral_norm) / (4.0 * X.shape[0])
        step_size = 1.0 / max(lipschitz, 1e-12)

        models: list[_LambdaModel] = []
        warm_start: np.ndarray | None = None

        for lambda_value in self.lambdas_:
            coef, intercept = self._fit_fista_single(
                X,
                y,
                lambda_value=lambda_value,
                step_size=step_size,
                warm_start=warm_start,
            )
            warm_start = np.concatenate([coef, [intercept]])
            models.append(_LambdaModel(lambda_value=lambda_value, coef=coef, intercept=intercept))

        return models

    def _fit_sklearn_path(self, X: np.ndarray, y: np.ndarray) -> list[_LambdaModel]:
        models: list[_LambdaModel] = []

        for lambda_value in self.lambdas_:
            C = 1.0 / lambda_value
            estimator = LogisticRegression(
                penalty="l1",
                solver="liblinear",
                C=C,
                fit_intercept=True,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                warnings.simplefilter("ignore", category=UserWarning)
                estimator.fit(X, y)
            coef = estimator.coef_.reshape(-1).astype(float, copy=True)
            intercept = float(estimator.intercept_.reshape(-1)[0])
            models.append(
                _LambdaModel(
                    lambda_value=lambda_value,
                    coef=coef,
                    intercept=intercept,
                    estimator=estimator,
                )
            )

        return models

    def fit(self, X_train: Any, y_train: Any) -> "LabeledLogReg":
        X = self._prepare_X(X_train, fit=True)
        y = self._validate_y(y_train, "y_train")
        y_binary = self._check_binary_labels(y, "y_train")

        self.lambdas_ = self._resolve_lambda_grid(X, y_binary)

        if self.implementation == "fista":
            self.models_ = self._fit_fista_path(X, y_binary)
        else:
            self.models_ = self._fit_sklearn_path(X, y_binary)

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

    def _predict_positive_proba_from_model(self, X: np.ndarray, model: _LambdaModel) -> np.ndarray:
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
                raise ValueError(
                    "area under the sensitivity-precision curve requires both classes to be present in the validation data."
                )
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

        allowed_classes = np.unique(np.concatenate([self.classes_, np.unique(y)]))
        if np.unique(allowed_classes).size > 2:
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
