import sys, os
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import mean_squared_error

import xgboost as xgb
import lightgbm as lgb

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# Import correct paths
from config import PROCESSED_PATH, MODEL_DIR

# Load processed data
def load_data():
    path = os.path.join(PROCESSED_PATH, "train_final.csv")
    print("Loading:", path)
    df = pd.read_csv(path)
    return df

# Time-based split
def time_split(df, val_fraction=0.2):
    df = df.sort_values("date")

    split_idx = int(len(df) * (1 - val_fraction))
    train = df.iloc[:split_idx]
    val = df.iloc[split_idx:]

    print(f"Train size: {train.shape}")
    print(f"Validation size: {val.shape}")

    return train, val

# Prepare features
def prepare_xy(df):
    X = df.drop(columns=["sales", "date"])
    y = df["sales"]
    return X, y

# Train XGBoost
def train_xgboost(X_train, y_train, X_val, y_val):
    print("\n🚀 Training XGBoost...")

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

    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    preds = xgb_model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))

    print(f"XGBoost RMSE: {rmse:.6f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "xgb_model.pkl")
    joblib.dump(xgb_model, model_path)
    print("Saved:", model_path)

    return rmse

# Train LightGBM
import re

def clean_columns(df):
    df = df.copy()
    df.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', c) for c in df.columns]
    return df

def train_lightgbm(X_train, y_train, X_val, y_val):
    print("\n🚀 Training LightGBM...")

    X_train = clean_columns(X_train)
    X_val = clean_columns(X_val)

    lgb_train = lgb.Dataset(X_train, label=y_train)
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

    preds = lgb_model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))

    print(f"LightGBM RMSE: {rmse:.6f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "lgb_model.pkl")
    joblib.dump(lgb_model, model_path)
    print("Saved:", model_path)

    return rmse

# Main pipeline
def run_advanced_models():
    df = load_data()

    train_df, val_df = time_split(df)

    X_train, y_train = prepare_xy(train_df)
    X_val, y_val = prepare_xy(val_df)

    xgb_rmse = train_xgboost(X_train, y_train, X_val, y_val)
    lgb_rmse = train_lightgbm(X_train, y_train, X_val, y_val)

    print("\n✅ FINAL RESULTS")
    print(f"XGBoost RMSE: {xgb_rmse:.6f}")
    print(f"LightGBM RMSE: {lgb_rmse:.6f}")

if __name__ == "__main__":
    run_advanced_models()