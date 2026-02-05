import sys, os
import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# Import correct paths
from config import PROCESSED_PATH, MODEL_DIR

# Load data & base models
def load_resources():
    df_path = os.path.join(PROCESSED_PATH, "train_final.csv")
    xgb_path = os.path.join(MODEL_DIR, "xgb_model.pkl")
    lgb_path = os.path.join(MODEL_DIR, "lgb_model.pkl")

    print("Loading:", df_path)
    df = pd.read_csv(df_path)

    xgb_model = joblib.load(xgb_path)
    lgb_model = joblib.load(lgb_path)

    return df, xgb_model, lgb_model

# Time-based split
def time_split(df, val_fraction=0.2):
    df = df.sort_values("date")
    split_idx = int(len(df) * (1 - val_fraction))
    return df.iloc[:split_idx], df.iloc[split_idx:]

# Prepare features
def prepare_xy(df):
    X = df.drop(columns=["sales", "date"])
    y = df["sales"]
    return X, y

# Build stacking features
def create_stacking_features(X_train, X_val, xgb_model, lgb_model):
    """
    Generate base model predictions as new features
    """

    # Base model predictions
    xgb_train_pred = xgb_model.predict(X_train)
    xgb_val_pred = xgb_model.predict(X_val)

    lgb_train_pred = lgb_model.predict(X_train)
    lgb_val_pred = lgb_model.predict(X_val)

    # Create new feature matrices
    X_train_stack = pd.DataFrame({
        "xgb_pred": xgb_train_pred,
        "lgb_pred": lgb_train_pred
    })

    X_val_stack = pd.DataFrame({
        "xgb_pred": xgb_val_pred,
        "lgb_pred": lgb_val_pred
    })

    return X_train_stack, X_val_stack

# Train meta-model
def train_stacking_model():
    df, xgb_model, lgb_model = load_resources()

    # Time-based split
    train_df, val_df = time_split(df)

    X_train, y_train = prepare_xy(train_df)
    X_val, y_val = prepare_xy(val_df)

    # Create stacking features
    X_train_stack, X_val_stack = create_stacking_features(
        X_train, X_val, xgb_model, lgb_model
    )

    # Meta-learner
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(X_train_stack, y_train)

    # Evaluate
    preds = meta_model.predict(X_val_stack)
    rmse = np.sqrt(mean_squared_error(y_val, preds))

    print(f"\n🔥 STACKING ENSEMBLE RMSE: {rmse:.6f}")

    # Save meta-model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "stacking_meta_model.pkl")
    joblib.dump(meta_model, model_path)
    print("Saved:", model_path)

if __name__ == "__main__":
    train_stacking_model()