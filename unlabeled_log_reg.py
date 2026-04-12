import numpy as np
from labeled_log_reg import LabeledLogReg
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

class UnlabeledLogReg:
    def fit(self, model: LogisticRegression | LabeledLogReg, X: np.ndarray, y: np.ndarray, method: str = "naive") -> LogisticRegression | LabeledLogReg:
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
                return model.fit(X[y != -1], y[y != -1])
            case "oracle":
                if (y == -1).sum() > 0:
                    raise ValueError("Missing labels found in y while using 'oracle' method.")
                return model.fit(X, y)
            case "knn":
                knn = KNeighborsClassifier()
                knn.fit(X[y != -1], y[y != -1])
                y_imputed = knn.predict(X[y == -1])
                y_all = np.array(y).copy()
                y_all[y_all == -1] = y_imputed
                return model.fit(X, y_all)
            case "lda":
                lda = LinearDiscriminantAnalysis()
                lda.fit(X[y != -1], y[y != -1])
                y_imputed = lda.predict(X[y == -1]) 
                y_all = np.array(y).copy()
                y_all[y_all == -1] = y_imputed
                return model.fit(X, y_all)
            case _:
                raise ValueError(f"Unknown method: {method}. Use one of following: 'naive', 'oracle', 'knn', 'lda'.")