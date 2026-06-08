import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, f_classif
from sklearn.preprocessing import MinMaxScaler

from utils.data_processing import load_data, standarize, variance_threshold, correlation_filter, select_from_model, select_k_best

from sklearn.feature_selection import VarianceThreshold, SelectFromModel, RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# def drop_collinear_features(df, target, threshold=0.80):
#     corr_matrix = df.corr().abs()
#     upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
#     to_drop = set()
#     for column in upper.columns:
#         highly_correlated_with = upper.index[upper[column] > threshold].tolist()
#         for correlated_col in highly_correlated_with:
#             if abs(df[column].corr(target)) > abs(df[correlated_col].corr(target)):
#                 to_drop.add(correlated_col)
#             else:
#                 to_drop.add(column)
#     return df.drop(columns=list(to_drop))


def give_data():
    X, y, XfinalTest = load_data()

    print(f"Original dataset shape: {X.shape}")

    X_var_filtered = X.loc[:, VarianceThreshold(threshold=0.01).fit(X).get_support()]


    # X_filtered = drop_collinear_features(X_var_filtered, y, threshold=0.80)
    # print(f"Features surviving basic filters: {X_filtered.shape[1]}")

    # --- 2. Collinearity (Correlation) Filter ---
    def drop_collinear_features(df, target, threshold=0.85):
        """
        Finds highly correlated pairs and drops the one with lower 
        absolute correlation to the target.
        """
        # Calculate absolute correlation matrix of features
        corr_matrix = df.corr().abs()
        
        # Upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features with correlation greater than threshold
        to_drop = set()
        for column in upper.columns:
            highly_correlated_with = upper.index[upper[column] > threshold].tolist()
            for correlated_col in highly_correlated_with:
                # Check which one has lower correlation with the target
                corr_col1 = abs(df[column].corr(target["y"]))
                corr_col2 = abs(df[correlated_col].corr(target["y"]))
                
                if corr_col1 > corr_col2:
                    to_drop.add(correlated_col)
                else:
                    to_drop.add(column)
                    
        return df.drop(columns=list(to_drop))

    X_filtered = drop_collinear_features(X_var_filtered, y, threshold=0.85)
    print(f"Features after dropping highly collinear ones: {X_filtered.shape[1]}")

    # Scale data for L1 Regularization

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_filtered), columns=X_filtered.columns)

    # --- STRICT TEST 1: L1 Regularization (Lasso) ---
    # C=0.05 is highly penalized; it will force most coefficients to 0
    l1_model = LogisticRegression(penalty='l1', solver='liblinear', C=0.05, random_state=42, max_iter=1000)
    l1_selector = SelectFromModel(l1_model)
    l1_selector.fit(X_scaled, y)
    l1_survivors = X_scaled.columns[l1_selector.get_support()].tolist()
    print(f"Features passing L1 strict penalty: {len(l1_survivors)}")

    # --- STRICT TEST 2: RFECV with XGBoost ---
    # Iteratively drops the bottom 5 features and validates with 5-fold CV
    xgb_estimator = xgb.XGBClassifier(n_estimators=50, max_depth=3, eval_metric='logloss', random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rfecv = RFECV(
        estimator=xgb_estimator, 
        step=5, 
        cv=cv, 
        scoring='roc_auc', 
        min_features_to_select=5,
        n_jobs=-1
    )
    rfecv.fit(X_filtered, y)
    rfecv_survivors = X_filtered.columns[rfecv.support_].tolist()
    print(f"Features passing RFECV: {len(rfecv_survivors)}")

    # --- THE CONSENSUS INTERSECTION ---
    # A feature MUST survive BOTH strict tests to move to the final round
    consensus_features = list(set(l1_survivors).intersection(set(rfecv_survivors)))
    print(f"\nFeatures surviving BOTH strict tests: {len(consensus_features)}")

    # If the strict filters were extremely aggressive, ensure we have features to test
    if len(consensus_features) == 0:
        print("Filters were too strict! No features survived. Relax the L1 'C' parameter or RFECV step.")
    else:
        # Train a final validation model on just the consensus features
        X_consensus = X_filtered[consensus_features]
        final_xgb = xgb.XGBClassifier(n_estimators=100, max_depth=4, eval_metric='logloss', random_state=42)
        final_xgb.fit(X_consensus, y)

        # Calculate Permutation Importance (repeating the shuffle 10 times per feature for stability)
        perm_importance = permutation_importance(
            final_xgb, X_consensus, y, n_repeats=10, random_state=42, scoring='roc_auc'
        )

        # Create a clean dataframe of the results
        perm_df = pd.DataFrame({
            'Feature': X_consensus.columns,
            'Importance_Mean': perm_importance["importances_mean"],
            'Importance_Std': perm_importance.importances_std
        })

        # Sort strictly by the mean performance drop
        perm_df = perm_df.sort_values(by='Importance_Mean', ascending=False)
        
        # Drop any feature that actually hurt the model (Importance <= 0)
        perm_df = perm_df[perm_df['Importance_Mean'] > 0]

        # Strictest cut: Take up to the top 20
        final_20 = perm_df.head(20)

        print("\n--- FINAL STRICT TOP 20 FEATURES ---")
        display(final_20)
        
        # Your final dataset
        X_final_strict = X_filtered[final_20['Feature'].tolist()]
        print(f"\nFinal strict dataset shape: {X_final_strict.shape}")

        return X_final_strict, final_20['Feature'].tolist()