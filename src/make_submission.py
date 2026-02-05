import pandas as pd
import joblib
import os
from datetime import datetime
import sys

# Project paths
# ============================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from config import PROCESSED_PATH, MODEL_DIR

RAW_PATH = os.path.join(PROJECT_ROOT, "data", "raw")
SUBMISSION_PATH = os.path.join(PROJECT_ROOT, "submissions")
os.makedirs(SUBMISSION_PATH, exist_ok=True)


# ==========================
# Load test data
# ==========================
def load_test_data():
    test = pd.read_csv(f"{RAW_PATH}/test.csv")
    test_proc = pd.read_csv(f"{PROCESSED_PATH}/test_final.csv")
    return test, test_proc

# Load models + feature list
def load_models():
    xgb_model = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
    lgb_model = joblib.load(os.path.join(MODEL_DIR, "lgb_model.pkl"))
    stack_model = joblib.load(os.path.join(MODEL_DIR, "stacking_meta_model.pkl"))
    feature_list = joblib.load(os.path.join(MODEL_DIR, "train_features.pkl"))
    return xgb_model, lgb_model, stack_model, feature_list

# ==========================
# Create stacking features
# ==========================
def create_stack_features(X, xgb_model, lgb_model):
    xgb_pred = xgb_model.predict(X)
    lgb_pred = lgb_model.predict(X)

    X_stack = pd.DataFrame({
        "xgb_pred": xgb_pred,
        "lgb_pred": lgb_pred
    })

    return X_stack

# ==========================
# Make submission
# ==========================
import re

def clean_columns(df):
    df = df.copy()
    df.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', c) for c in df.columns]
    return df

def make_submission():
    test_raw, test_proc = load_test_data()
    xgb_model, lgb_model, stack_model, feature_list = load_models()

    # Prepare features (must match training)
    X_test = test_proc.drop(columns=["date"], errors="ignore")
    X_test = clean_columns(X_test)
    X_test = X_test.reindex(columns=feature_list, fill_value=0)


    # Create stacking features
    X_test_stack = create_stack_features(X_test, xgb_model, lgb_model)

    # Final predictions
    final_preds = lgb_model.predict(X_test)


    # Create Kaggle submission format
    submission = pd.DataFrame({
        "id": test_raw["id"],
        "sales": final_preds
    })

    # Versioned filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{SUBMISSION_PATH}/submission_{timestamp}.csv"

    submission.to_csv(filename, index=False)

    print(f"Submission saved: {filename}")

if __name__ == "__main__":
    make_submission()
