import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_selection import VarianceThreshold, SelectFromModel, RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import warnings

from utils.data_processing import load_data

# Suppress sklearn/xgboost warnings for clean output
# warnings.filterwarnings("ignore")

def drop_collinear_features(df, target, threshold=0.80):
    """Drops highly correlated features, keeping the one most correlated with the target."""
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = set()
    
    for column in upper.columns:
        highly_correlated_with = upper.index[upper[column] > threshold].tolist()
        for correlated_col in highly_correlated_with:
            if abs(df[column].corr(target)) > abs(df[correlated_col].corr(target)):
                to_drop.add(correlated_col)
            else:
                to_drop.add(column)
                
    return df.drop(columns=list(to_drop))

def get_data(max_features=20, random_state=42):
    
    X, y, xsmall = load_data()
    print(f"--- STARTING STRICT PIPELINE ---")
    print(f"Initial features: {X.shape[1]}")

    # Ensure y is a series for index matching
    if isinstance(y, np.ndarray):
        y = pd.Series(y, index=X.index)

    # ---------------------------------------------------------
    # STAGE 1: Variance and Strict Collinearity
    # ---------------------------------------------------------
    var_thresholder = VarianceThreshold(threshold=0.01)
    X_var = X.loc[:, var_thresholder.fit(X).get_support()]
    
    X_stage1 = drop_collinear_features(X_var, y["y"], threshold=0.80)
    print(f"Stage 1 (Variance & Collinearity) survivors: {X_stage1.shape[1]}")
    
    if X_stage1.shape[1] == 0: return X_stage1, []

    # ---------------------------------------------------------
    # STAGE 2: L1 Regularization (Lasso)
    # ---------------------------------------------------------
    # Scale locally just for L1 so we don't alter the original X data
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_stage1), columns=X_stage1.columns, index=X_stage1.index)
    
    # C=0.05 is highly aggressive
    l1_model = LogisticRegression(solver='liblinear', l1_ratio=1, C=0.05, random_state=random_state)
    l1_selector = SelectFromModel(l1_model)
    l1_selector.fit(X_scaled, y.to_numpy().ravel())
    
    stage2_cols = X_stage1.columns[l1_selector.get_support()].tolist()
    X_stage2 = X_stage1[stage2_cols]
    print(f"Stage 2 (L1 Penalty) survivors: {X_stage2.shape[1]}")
    print("check")
    if X_stage2.shape[1] == 0: return X_stage2, []

    # ---------------------------------------------------------
    # STAGE 3: RFECV (Recursive Feature Elimination)
    # ---------------------------------------------------------
    xgb_base = xgb.XGBClassifier(n_estimators=50, max_depth=3, eval_metric='logloss', random_state=random_state, n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    
    rfecv = RFECV(estimator=xgb_base, step=3, cv=cv, scoring='roc_auc', min_features_to_select=5, n_jobs=-1)
    rfecv.fit(X_stage2, y.to_numpy().ravel())
    
    stage3_cols = X_stage2.columns[rfecv.support_].tolist()
    X_stage3 = X_stage2[stage3_cols]
    print(f"Stage 3 (RFECV) survivors: {X_stage3.shape[1]}")
    
    if X_stage3.shape[1] == 0: return X_stage3, []

    # ---------------------------------------------------------
    # STAGE 4: Shadow Feature Elimination (Mini-Boruta)
    # ---------------------------------------------------------
    # Feature must beat the maximum importance of completely randomized data
    surviving_cols = list(X_stage3.columns)
    trials = 3 # Hard consensus across 3 random shuffles
    
    for trial in range(trials):
        X_shadow = X_stage3[surviving_cols].copy()
        # Shuffle each column to create noise
        for col in X_shadow.columns:
            X_shadow[col] = np.random.permutation(X_shadow[col])
        X_shadow.columns = [f"{col}_shadow" for col in surviving_cols]
        
        X_combined = pd.concat([X_stage3[surviving_cols], X_shadow], axis=1)
        
        xgb_shadow = xgb.XGBClassifier(n_estimators=50, max_depth=3, eval_metric='logloss', random_state=random_state+trial)
        xgb_shadow.fit(X_combined, y)
        
        importances = xgb_shadow.feature_importances_
        original_imp = importances[:len(surviving_cols)]
        shadow_imp = importances[len(surviving_cols):]
        
        max_shadow_imp = shadow_imp.max()
        
        # Keep only features that beat the best shadow feature
        surviving_cols = [surviving_cols[i] for i in range(len(surviving_cols)) if original_imp[i] > max_shadow_imp]
        
        if not surviving_cols:
            break
            
    X_stage4 = X_stage3[surviving_cols]
    print(f"Stage 4 (Shadow Feature Protocol) survivors: {X_stage4.shape[1]}")

    if X_stage4.shape[1] == 0: return X_stage4, []

    # ---------------------------------------------------------
    # STAGE 5: Out-of-Fold Permutation Importance
    # ---------------------------------------------------------
    # Evaluate features on hold-out data to prevent overfitting
    oof_importances = np.zeros(X_stage4.shape[1])
    
    for train_idx, val_idx in cv.split(X_stage4, y):
        X_train, X_val = X_stage4.iloc[train_idx], X_stage4.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        xgb_oof = xgb.XGBClassifier(n_estimators=100, max_depth=4, eval_metric='logloss', random_state=random_state)
        xgb_oof.fit(X_train, y_train)
        
        perm_res = permutation_importance(xgb_oof, X_val, y_val, n_repeats=5, random_state=random_state, scoring='roc_auc')
        oof_importances += perm_res["importances_mean"] / cv.get_n_splits()
        
    perm_df = pd.DataFrame({
        'Feature': X_stage4.columns,
        'OOF_Importance': oof_importances
    }).sort_values(by='OOF_Importance', ascending=False)
    
    # Must actually help the model (> 0) and take up to max_features
    final_features = perm_df[perm_df['OOF_Importance'] > 0].head(max_features)['Feature'].tolist()
    
    X_new = X[final_features]
    print(f"Stage 5 (OOF Permutation) Final Selection: {len(final_features)} features.")
    print(f"--- PIPELINE COMPLETE ---\n")
    
    return X_new, final_features
