import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

class UnlabeledLogReg:
    def __init__(self, logreg_impl):
        if logreg_impl == "sklearn":
            self.logreg = LogisticRegression(penalty="l1")
        else:
            self.logreg = None # TODO: replace with own logreg

    def fit(self, X: np.ndarray, y: np.ndarray, method: str = "naive") -> None:
        """
        Fits the logistic regression model to the data X and labels y using the specified method.

        The "naive" method removes all unlabeled observations and fits the model based on labeled data only.

        The "oracle" method assumes that no labels are missing. Functionally, it works in the same way as the "naive" method, but it checks if all labels are present.

        The "knn" method uses `sklearn`'s K Nearest Neighbors classifier to fill in missing labels. Then, it fits the logistic regression model.

        The "lda" method uses Linear Discriminant Analysis to estimate distributions of features for each class and fills missing labels based on the highest probability. Then, it fits the logistic regression model.

        Parameters:
        X (np.ndarray): The input data, shape (n_samples, n_features).
        y (np.ndarray): The labels for the input data, shape (n_samples,).
        method (str): The method to use for fitting the model. Options are "naive", "oracle", "knn" or "lda". Default is "naive".
        """
        match method:
            case "naive":
                self.logreg.fit(X[y != -1], y[y != -1])
            case "oracle":
                if (y == -1).sum() > 0:
                    raise ValueError("Missing labels found in y while using 'oracle' method.")
                self.logreg.fit(X, y)
            case "knn":
                knn = KNeighborsClassifier()
                knn.fit(X[y != -1], y[y != -1])
                y_filled = knn.predict(X)
                self.logreg.fit(X, y_filled)
            case "lda":
                lda = LinearDiscriminantAnalysis()
                lda.fit(X[y != -1], y[y != -1])
                y_filled = lda.predict(X)
                self.logreg.fit(X, y_filled)
            case _:
                raise ValueError(f"Unknown method: {method}. Use one of following: 'naive', 'oracle', 'knn', 'lda'.")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the probabilities for the input data X using the fitted logistic regression model.

        Parameters:
        X (np.ndarray): Input data.

        Returns:
        np.ndarray: Predicted labels.
        """

        return self.logreg.predict_proba(X)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the labels for the input data X using the fitted logistic regression model.

        Parameters:
        X (np.ndarray): Input data.

        Returns:
        np.ndarray: Predicted labels.
        """

        return self.logreg.predict(X)
    
    def score(self, X: np.ndarray, y: np.ndarray, metric: str = "accuracy") -> float:
        """
        Computes `metric` of the fitted logistic regression model on the input data `X` and true labels `y`.

        Parameters:
        X (np.ndarray): Input data.
        y (np.ndarray): True labels.
        metric (str): The metric to compute. Options are "accuracy", "balanced_accuracy", "f1" and "roc_auc". Default is "accuracy".

        Returns:
        float: The computed metric.
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]
        match metric:
            case "accuracy":
                return accuracy_score(y, y_pred)
            case "balanced_accuracy":
                return balanced_accuracy_score(y, y_pred)
            case "f1":
                return f1_score(y, y_pred)
            case "roc_auc":
                return roc_auc_score(y, y_proba)
            case _:
                raise ValueError(f"Unknown metric: {metric}. Use one of following: 'accuracy', 'balanced_accuracy', 'f1', 'roc_auc'.")