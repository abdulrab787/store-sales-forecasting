import sys, os
import numpy as np
import pandas as pd
import joblib
import re
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from config import PROCESSED_PATH, MODEL_DIR

import xgboost as xgb
import lightgbm as lgb

def clean_columns(df):
    df = df.copy()
    df.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', c) for c in df.columns]
    return df

def load_data():
    path = os.path.join(PROCESSED_PATH, "train_final.csv")
    print("Loading:", path)
    df = pd.read_csv(path)
    return df

def time_split(df, val_fraction=0.2):
    df = df.sort_values("date")
    split_idx = int(len(df) * (1 - val_fraction))
    return df.iloc[:split_idx], df.iloc[split_idx:]

def prepare_xy(df):
    X = df.drop(columns=["sales", "date"])
    y = df["sales"]
    return X, y

#Add OOF generation for XGBoost + LightGBM
def generate_oof_predictions(X_train, y_train, n_splits=5, random_state=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    oof_xgb = np.zeros(len(X_train))
    oof_lgb = np.zeros(len(X_train))

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train), 1):
        print(f"\nFold {fold}/{n_splits}")

        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        # XGBoost
        xgb_model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=42
        )
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        oof_xgb[val_idx] = xgb_model.predict(X_val)

        # LightGBM
        lgb_train = lgb.Dataset(X_tr, label=y_tr)
        lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

        params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.05,
            "num_leaves": 64,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "seed": 42
        }

        lgb_model = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_val],
            num_boost_round=500
        )
        oof_lgb[val_idx] = lgb_model.predict(X_val)

        fold_rmse_xgb = np.sqrt(mean_squared_error(y_val, oof_xgb[val_idx]))
        fold_rmse_lgb = np.sqrt(mean_squared_error(y_val, oof_lgb[val_idx]))
        print(f"  XGB fold RMSE: {fold_rmse_xgb:.4f}")
        print(f"  LGB fold RMSE: {fold_rmse_lgb:.4f}")

    return oof_xgb, oof_lgb

#main function to run OOF generation and save stacking dataset
def run_oof():
    df = load_data()

    train_df, val_df = time_split(df)
    X_train, y_train = prepare_xy(train_df)

    # 🔥 Clean feature names once for both XGB & LGB
    X_train = clean_columns(X_train)

    print("Generating OOF predictions...")
    oof_xgb, oof_lgb = generate_oof_predictions(X_train, y_train)
    
    # Build OOF stacking dataset
    oof_df = pd.DataFrame({
        "xgb_oof": oof_xgb,
        "lgb_oof": oof_lgb,
        "target": y_train.values
    })

    oof_path = os.path.join(PROJECT_ROOT, "experiments", "oof_stacking_train.csv")
    os.makedirs(os.path.dirname(oof_path), exist_ok=True)
    oof_df.to_csv(oof_path, index=False)
    print("Saved OOF stacking data to:", oof_path)

if __name__ == "__main__":
    run_oof()