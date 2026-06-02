import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler

from statsmodels.stats.outliers_influence import variance_inflation_factor



########### LOADING AND PREPROCESSING ###########

def load_data(path: str = "../data/") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the dataset from the chosen directory.

    Note that the passed directory should contain the following files: `x_train.txt`, `y_train.txt`, `x_test.txt` containing space-separated numpy arrays. Otherwise, the function will fail.
    
    Parameters:
        path (str): A relative path to the data file.
    
    Returns:
        out (tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]): The contents of the `x_train.txt` file, the `y_train.txt` file, and the `x_test.txt` file, respectively.
    """

    X_train = pd.read_csv(path + "x_train.txt", sep=" ")
    y_train = pd.read_csv(path + "y_train.txt")
    X_test = pd.read_csv(path + "x_test.txt", sep=" ")

    return X_train, y_train, X_test



def standarize(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Standarize the dataset, i.e. transform the columns such that their mean is equal to 0 and their standard deviation is equal to 1.

    This function uses `sklearn.preprocessing.StandardScaler`. It is trained on the `X_train` DataFrame and then transforms `X_train` and `X_test`.

    Parameters:
        X_train (pd.DataFrame): Features from the training sample.
        X_test (pd.DataFrame): Features from the test sample.
    
    Returns:
        (X_train_new, X_test_new) (tuple[pd.DataFrame, pd.DataFrame]): The transformed DataFrames with original columns and index.
    """

    ss = StandardScaler().fit(X_train)
    X_train_new = ss.transform(X_train)
    X_test_new = ss.transform(X_test)
    return (pd.DataFrame(X_train_new, columns=X_train.columns, index=X_train.index), pd.DataFrame(X_test_new, columns=X_test.columns, index=X_test.index))



########### FEATURE SELECTION ###########

# very slow and actaully ill-advised for this high amount of features
def variance_inflation_factor(X_train: pd.DataFrame, X_test: pd.DataFrame, threshold: float = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform Variance Inflation Factor analysis and return a list of columns after filtering by threshold.

    At each iteration, the column with the highest VIF value is removed from the set.

    Parameters:
        X (pd.DataFrame): The DataFrame on which the analysis will be performed.

    Returns:
        index (pd.Index): An index containing columns that were not removed during the procedure.
    """
    
    chosen_columns = X_train.columns

    while True:
        vif_data = pd.DataFrame()
        vif_data["feature"] = chosen_columns
        vif_data["VIF"] = [variance_inflation_factor(X_train[chosen_columns].values, i) for i in range(len(chosen_columns))]
        candidate = vif_data.sort_values("VIF", ascending=False).iloc[0]
        if candidate["VIF"] > threshold:
            chosen_columns = chosen_columns.drop(candidate["feature"])
            print(f"Removed column {candidate["feature"]} with VIF={candidate["VIF"]:.2f}")
        else:
            break

    return X_train[chosen_columns], X_test[chosen_columns]



# good for early selection - run it with a relatively low threshold like 0.01
def variance_threshold(X_train: pd.DataFrame, X_test: pd.DataFrame, threshold: float = 0.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove features whose variance in `X_train` is below the given threshold.

    Parameters:
        X_train (pd.DataFrame): Features from the training sample.
        X_test (pd.DataFrame): Features from the test sample.
        threshold (float, default=0.0): Minimal variance required to keep a feature.

    Returns:
        (X_train_new, X_test_new) (tuple[pd.DataFrame, pd.DataFrame]): DataFrames containing only selected features.
    """

    variances = X_train.var(axis=0)
    selected_columns = variances[variances > threshold].index
    return X_train[selected_columns], X_test[selected_columns]



# good to run with a high threshold (default is good) before select_from_model to stabilize it
def correlation_filter(X_train: pd.DataFrame, X_test: pd.DataFrame, threshold: float = 0.9) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove highly correlated features from `X_train` and `X_test`.

    Features are dropped when their absolute correlation with another feature in `X_train` exceeds the threshold.

    Parameters:
        X_train (pd.DataFrame): Features from the training sample.
        X_test (pd.DataFrame): Features from the test sample.
        threshold (float, default=0.9): Maximum allowed absolute pairwise correlation.

    Returns:
        (X_train_new, X_test_new) (tuple[pd.DataFrame, pd.DataFrame]): DataFrames containing only selected features.
    """

    corr_matrix = X_train.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    selected_columns = X_train.columns.difference(to_drop)

    return X_train[selected_columns], X_test[selected_columns]


# use ExtraTreesClassifier for any model or LogisticRegression(penalty="l1") for logistic regression
# remember to set a seed for the classifier passed
# leave default threshold for lasso; do a gridsearch for forest (suggested values: ['mean', 'median', '1.25*mean', '0.75*mean'])
def select_from_model(X_train: pd.DataFrame, y_train: pd.Series | pd.DataFrame, X_test: pd.DataFrame, estimator = ExtraTreesClassifier(random_state=42), threshold: str | float | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform feature selection based on the chosen model's feature importances.

    Either ExtraTreesClassifier or LogisticRegression(penalty="l1") should be passed. It is possible to adjust hyperparameters () in order to find the optimal model.
    
    Parameters:
        X_train (pd.DataFrame): Features from the training sample.
        y_train (pd.Series | pd.DataFrame): Target values for the training sample.
        X_test (pd.DataFrame): Features from the test sample.
        estimator (default=ExtraTreesClassifier): Estimator to be used.
        threshold (str | float): Threshold for feature selection. If ommited, 1e-5 for LogisticRegression and "mean" for ExtraTreesClassifier is used. 
    Returns:
        (X_train_new, X_test_new) (tuple[pd.DataFrame, pd.DataFrame]): DataFrames containing only the selected features.
    """

    selector = SelectFromModel(estimator, threshold=threshold)
    selector.fit(X_train, y_train)
    mask = selector.get_support()

    return X_train[mask], X_test[mask]



# use as an alternative to select_from_model
# default f_classif performs the ANOVA test (linear relationships)
# use mutual_info_classif for non-linear relationships (good for trees)
def select_k_best(X_train: pd.DataFrame, y_train: pd.Series | pd.DataFrame, X_test: pd.DataFrame, k: int = 20, score_func=f_classif) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Select the top `k` features according to a univariate statistical test.

    Parameters:
        X_train (pd.DataFrame): Features from the training sample.
        y_train (pd.Series | pd.DataFrame): Target values for the training sample.
        X_test (pd.DataFrame): Features from the test sample.
        k (int, default=20): Number of top features to select.
        score_func (callable, default=f_classif): Scoring function from sklearn.feature_selection.

    Returns:
        (X_train_new, X_test_new) (tuple[pd.DataFrame, pd.DataFrame]): DataFrames containing only the selected features.
    """

    y = y_train.squeeze()
    selector = SelectKBest(score_func=score_func, k=min(k, X_train.shape[1])).fit(X_train, y)
    selected_columns = X_train.columns[selector.get_support()]
    return X_train[selected_columns], X_test[selected_columns]
