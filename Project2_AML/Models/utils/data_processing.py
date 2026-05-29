import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from statsmodels.stats.outliers_influence import variance_inflation_factor



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



def pca(X_train: pd.DataFrame, X_test: pd.DataFrame, criterion: str | None = "cumulative", threshold: float = 0.9) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform the PCA transformation on the train and test DataFrames.

    The function uses `sklearn.decomposition.PCA` underneath.

    The `criterion` along with the `threshold` can be set to automatically perform variable selection after calculating the principal components. By default, 90% of cumulative variance is kept.

    Parameters:
        X_train (pd.DataFrame): Features from the training sample.
        X_test (pd.DataFrame): Features from the test sample.
        criterion ({"cumulative", "eigenvalue", None}, default="cumulative"): The variable selection criterion.
        threshold (float, default=0.9): The threshold for the criterion - a fraction of retained cumulative variance if `criterion` is set to `"cumulative"` or a minimal eigenvalue when `criterion` is set to `"eigenvalue"`. No effect if `criterion` is set to `None`.
    
    Returns:
        (X_train_new, X_test_new) (tuple[pd.DataFrame, pd.DataFrame]): The transformed DataFrames.
    """

    pca = PCA().fit(X_train)
    X_train_new = pd.DataFrame(pca.transform(X_train), columns=X_train.columns)
    X_test_new = pd.DataFrame(pca.transform(X_test), columns=X_test.columns)

    if criterion == "cumulative":
        cumulative_variance_ratio = pca.explained_variance_ratio_.cumsum()
        X_train_new = X_train_new.loc[:, cumulative_variance_ratio < threshold]
        X_test_new = X_test_new.loc[:, cumulative_variance_ratio < threshold]
    elif criterion == "eigenvalue":
        X_train_new = X_train_new.loc[:, pca.explained_variance_ > threshold]
        X_test_new = X_test_new.loc[:, pca.explained_variance_ > threshold]

    return X_train_new, X_test_new



def vif_filter(X: pd.DataFrame, threshold: float = 10) -> pd.Index:
    """
    Perform Variance Inflation Factor analysis and return a list of columns after filtering by threshold.

    At each iteration, the column with the highest VIF value is removed from the set.

    Warning: the procedure is time intensive and unstable for all predictors.

    Parameters:
        X (pd.DataFrame): The DataFrame on which the analysis will be performed.

    Returns:
        index (pd.Index): An index containing columns that were not removed during the procedure.
    """
    
    chosen_columns = X.columns

    while True:
        vif_data = pd.DataFrame()
        vif_data["feature"] = chosen_columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
        candidate = vif_data.sort_values("VIF", ascending=False).iloc[0]
        if candidate["VIF"] > threshold:
            chosen_columns = chosen_columns.drop(candidate["feature"])
            print(f"Removed column {candidate["feature"]} with VIF={candidate["VIF"]:.2f}")
        else:
            break

    return chosen_columns
