import kagglehub
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42

def remove_labels(df: pd.DataFrame, target: str, scheme: str, p_mcar: float = 0.15, p_mar: list[float] = [0.9, 0.05], p_mnar: list[float] = [0.9, 0.1, 0.5, 0.05]) -> pd.DataFrame:
    """
    Remove labels in the dataset based on the chosen scheme. A new column named `target`_`scheme` is added to the original DataFrame.

    Schemes:
        MCAR: Missing Completely At Random. P(S=1|X,Y)=P(S=1)=c.
              Labels are removed uniformly at random with probability c.

        MAR1: Missing At Random 1. P(S=1|X,Y)=P(S=1|X).
              Missingness depends only on a single explanatory variable.

        MAR2: Missing At Random 2. P(S=1|X,Y)=P(S=1|X).
              Firstly, all feature columns are normalized to interval [0, 1]. Then, for each row, the mean of values is calculated.
              The resulting series is categorized into 10 bins, of which an arbitrarily chosen bin (in this implementation, bin 3)
              is used to make labels more likely to be missing.

        MNAR: Missing Not At Random. P(S=1|X,Y).
              Analogous to MAR2 with one exception: if an observation belongs to class 1, the mean corresponding to it is transformed
              using formula x -> 1 - x.

    Parameters:
        df (pandas.DataFrame): A Dataframe containing a column in which the labels are to be removed.
        target (str): The name of the column containing labels.
        scheme (str): Scheme for generating missing labels: `"mcar"`, `"mar1"`, `"mar2"` or `"mnar"`.
        p_mcar (float): The probability of label missing for the MCAR scheme. Used only if scheme is set to `"mcar"`.
        p_mar (list[float]): A list of 2 probabilities (the first one corresponding to a chosen decile, the second one corresponding to the remaining deciles) for the MAR schemes. Used only if scheme is set to `"mar1"` or `"mar2"`.
        p_mnar (list[float]): A list of 4 probabilities for the MNAR scheme. They correspond to: chosen decile in class 1, remaining deciles in class 1, chosen decile in class 0, and remaining deciles in class 0, respectively. Used only if scheme is set to `"mnar"`.

    Returns:
        df (pandas.DataFrame): The original DataFrame with a new column containing removed labels (missing labels are set to -1).
    """

    generator = np.random.default_rng(seed=RANDOM_SEED)

    if scheme == "mcar":
        is_missing = generator.binomial(1, p_mcar, size=len(df))

    elif scheme == "mar1":
        feature_cols = df.columns.drop(list(df.filter(regex=target)))
        if len(feature_cols) == 0:
            raise ValueError("MAR1 requires at least one explanatory variable column.")

        mar_feature = df[feature_cols].nunique().idxmax()
        x = pd.to_numeric(df[mar_feature], errors="coerce")
        x = x.fillna(x.mean())

        denom = x.max() - x.min()
        if denom == 0 or np.isnan(denom):
            normalized_x = pd.Series(0.5, index=df.index)
        else:
            normalized_x = (x.max() - x) / denom

        deciles = pd.qcut(normalized_x, 10, labels=False, duplicates="drop")
        chosen_bin = min(3, int(deciles.max()))
        missing_prob = deciles.map(lambda d: p_mar[0] if d == chosen_bin else p_mar[1])
        is_missing = generator.binomial(1, missing_prob)

    elif scheme == "mar2":
        normalized_value_means = df[df.columns.drop(list(df.filter(regex=target)))].apply(
            lambda x: (x.max() - x) / (x.max() - x.min()), axis=0
        ).mean(axis=1)
        deciles = pd.qcut(normalized_value_means, 10, labels=False)
        is_missing = generator.binomial(1, deciles.map(lambda x: p_mar[0] if x == 3 else p_mar[1]))

    elif scheme == "mnar":
        normalized_value_means = df[df.columns.drop(list(df.filter(regex=target)))].apply(
            lambda x: (x.max() - x) / (x.max() - x.min()), axis=0
        ).mean(axis=1)
        deciles = pd.qcut(normalized_value_means, 10, labels=False)
        is_missing = generator.binomial(1, [(p_mnar[0] if d == 3 else p_mnar[1]) if c == 1 else (p_mnar[2] if d == 3 else p_mnar[3]) for d, c in zip(deciles, df[target])])

    else:
        raise ValueError("Argument scheme accepts only one of following values: 'mcar', 'mar1', 'mar2' or 'mnar'.")

    is_missing = np.asarray(is_missing, dtype=bool)
    print(f"Percentage of labels removed: {is_missing.mean()*100:.2f}%")
    print(f"percentages: {p_mcar if scheme == 'mcar' else (p_mar if scheme in ['mar1', 'mar2'] else p_mnar)}")

    df[target + "_" + scheme] = df[target]
    df.loc[is_missing, target + "_" + scheme] = -1
    return df


def get_dataset(name: str, p_mcar: float = 0.15, p_mar: list[float] = [0.9, 0.05], p_mnar: list[float] = [0.9, 0.1, 0.5, 0.05]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Download, preprocess, split, and remove labels from the chosen dataset.

    Parameters:
        name (str): The name of the dataset to be returned. Accepted values are: `"shopping"`, `"smartphone"`, `"software"` and `"wine"`.
        p_mcar (float): The probability of label missing for the MCAR scheme.
        p_mar (list[float]): A list of 2 probabilities (the first one corresponding to a chosen decile, the second one corresponding to the remaining deciles) for schemes MAR1 and MAR2.
        p_mnar (list[float]): A list of 4 probabilities for the MNAR scheme. They correspond to: chosen decile in class 1, remaining deciles in class 1, chosen decile in class 0, and remaining deciles in class 0, respectively.
        
    Returns:
        tuple[pandas.DataFrame, pandas.DataFrame]: A tuple of (train DataFrame, val DataFrame, test DataFrame). The train DataFrame contains columns with removed labels, while the val and test DataFrames are left unchanged.
    """

    # Dataset downloading, reading, preprocessing
    match name:
        case "shopping":
            path = kagglehub.dataset_download("shree0910/online-vs-in-store-shopping-behaviour-dataset")
            data = pd.read_csv(f"{path}/{os.listdir(path)[0]}")
            
            data["shopping_preference"] = data["shopping_preference"].map({"Store": 0, "Hybrid": 1, "Online": 1})
            data["city_tier"] = data["city_tier"].map({"Tier 1": 0, "Tier 2": 1, "Tier 3": 2})
            data.drop("gender", axis=1, inplace=True)
        case "smartphone":
            path = kagglehub.dataset_download("vishardmehta/smartphone-battery-health-prediction-dataset")
            target = pd.read_csv(f"{path}/{os.listdir(path)[0]}")
            features = pd.read_csv(f"{path}/{os.listdir(path)[1]}")
            data = features.merge(target, on="Device_ID")

            data["target_action"] = data["recommended_action"].map({"Change Phone": 1, "Replace Battery": 1, "Keep Using": 0})
            data["background_app_usage_level"] = data["background_app_usage_level"].map({"Low": 0, "Medium": 1, "High": 2})
            data["signal_strength_avg"] = data["signal_strength_avg"].map({"Poor": 0, "Moderate": 1, "Good": 2})
            data.drop(["recommended_action", "Device_ID", "current_battery_health_percent"], axis=1, inplace=True)
        case "software":
            path = kagglehub.dataset_download("ziya07/software-defect-prediction-dataset")
            data = pd.read_csv(f"{path}/{os.listdir(path)[0]}")
        case "wine":
            path = kagglehub.dataset_download("yasserh/wine-quality-dataset")
            data = pd.read_csv(f"{path}/{os.listdir(path)[0]}")
            data.drop('Id', axis=1, inplace=True)
            data["quality"] = data["quality"].map(lambda x: 1 if x > 5 else 0)
        case _:
            raise ValueError(f"Unknown dataset name: {name}. Accepted values are: 'shopping', 'smartphone', 'software', and 'wine'.")
    
    # Standarization
    data = data.astype(float)
    data.iloc[:, :-1] = (data.iloc[:, :-1] - data.iloc[:, :-1].mean())/data.iloc[:, :-1].std()

    # Split
    data_train, data_val_test = train_test_split(data, test_size=0.5, random_state=RANDOM_SEED)
    data_val, data_test = train_test_split(data_val_test, test_size=0.5, random_state=RANDOM_SEED)

    # Label removal
    label_name = {
        "shopping": "shopping_preference",
        "smartphone": "target_action",
        "software": "DEFECT_LABEL",
        "wine": "quality"
    }[name]

    data_train = remove_labels(data_train, label_name, "mcar", p_mcar=p_mcar)
    data_train = remove_labels(data_train, label_name, "mar1", p_mar=p_mar)
    data_train = remove_labels(data_train, label_name, "mar2", p_mar=p_mar)
    data_train = remove_labels(data_train, label_name, "mnar", p_mnar=p_mnar)

    return data_train, data_val, data_test