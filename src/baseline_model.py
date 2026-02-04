import sys, os
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import joblib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from config import PROCESSED_PATH, MODEL_DIR
os.makedirs(MODEL_DIR, exist_ok=True)

# Load processed data
def load_processed_data():
    path = os.path.join(PROCESSED_PATH, "train_final.csv")
    df = pd.read_csv(path)
    return df

# Time-based train/val split
def time_based_split(df, val_fraction=0.2):
    df = df.sort_values("date")

    split_idx = int(len(df) * (1 - val_fraction))
    train = df.iloc[:split_idx]
    val = df.iloc[split_idx:]

    print(f"Train size: {train.shape}")
    print(f"Validation size: {val.shape}")

    return train, val

# Prepare features & target
def prepare_features(df):
    X = df.drop(columns=["sales", "date"])
    y = df["sales"]
    return X, y

# Train baseline model
model = Ridge(alpha=1.0)

def train_baseline_model():
    df = load_processed_data()

    # Split by time
    train_df, val_df = time_based_split(df)

    # Prepare features
    X_train, y_train = prepare_features(train_df)
    X_val, y_val = prepare_features(val_df)

    # Baseline model: Ridge Regression
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    # Predictions
    preds = model.predict(X_val)

    # Metric
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    print(f"Baseline Ridge RMSE: {rmse:.6f}")

    # Ensure model directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save model
    model_path = os.path.join(MODEL_DIR, "ridge_baseline.pkl")
    joblib.dump(model, model_path)
    return model_path

if __name__ == "__main__":
    saved_path = train_baseline_model()
    print("Saved model at:", saved_path)
