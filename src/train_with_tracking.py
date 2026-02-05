import mlflow
import mlflow.sklearn
import pandas as pd
import joblib
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import re
import os

# project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "train_final.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

print("Loading training data from:", TRAIN_PATH)

# Load data 
train = pd.read_csv(TRAIN_PATH)
if "date" in train.columns:
    train = train.drop(columns=["date"])

# Clean columns 
train = train.copy()
train.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', c) for c in train.columns]

X = train.drop(columns=["sales"])
y = train["sales"]

# Save training feature list
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(X.columns.tolist(), os.path.join(MODEL_DIR, "train_features.pkl"))
print("Saved training feature list.")

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# MLflow experiment
mlflow.set_experiment("store_sales_forecasting")

with mlflow.start_run():

    # Train XGBoost
    xgb = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6
    )

    xgb.fit(X_train, y_train)
    preds = xgb.predict(X_val)
    rmse = mean_squared_error(y_val, preds) ** 0.5

    # Log parameters & metrics
    mlflow.log_param("model", "xgboost")
    mlflow.log_param("n_estimators", 500)
    mlflow.log_metric("val_rmse", rmse)

    # Save model
    joblib.dump(xgb, os.path.join(MODEL_DIR, "xgb_model.pkl"))

    # Log model in MLflow
    mlflow.sklearn.log_model(xgb, "xgb_model")

print("Training + tracking completed!")