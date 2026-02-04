import sys, os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# Import correct paths
from config import PROCESSED_PATH, MODEL_DIR

# Experiments directory
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

# Load data and models
def load_data_and_models():
    df_path = os.path.join(PROCESSED_PATH, "train_final.csv")
    xgb_path = os.path.join(MODEL_DIR, "xgb_model.pkl")
    lgb_path = os.path.join(MODEL_DIR, "lgb_model.pkl")

    print("Loading:", df_path)
    df = pd.read_csv(df_path)

    xgb_model = joblib.load(xgb_path)
    lgb_model = joblib.load(lgb_path)

    return df, xgb_model, lgb_model

# Time-based validation split
def time_split(df, val_fraction=0.2):
    df = df.sort_values("date")
    split_idx = int(len(df) * (1 - val_fraction))
    return df.iloc[:split_idx], df.iloc[split_idx:]

# Prepare features
def prepare_xy(df):
    X = df.drop(columns=["sales", "date"])
    y = df["sales"]
    return X, y

# Feature Importance - XGBoost
def plot_xgb_feature_importance(model, feature_names):
    importance = model.feature_importances_
    fi_df = pd.DataFrame({"feature": feature_names, "importance": importance})
    fi_df = fi_df.sort_values("importance", ascending=False)

    fi_df.head(20).plot(kind="barh", x="feature", y="importance", figsize=(8, 6))
    plt.title("Top 20 XGBoost Feature Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(EXPERIMENTS_DIR, "xgb_feature_importance.png"))
    plt.close()

    return fi_df

# Feature Importance - LightGBM
def plot_lgb_feature_importance(model, feature_names):
    importance = model.feature_importance(importance_type="gain")
    fi_df = pd.DataFrame({"feature": feature_names, "importance": importance})
    fi_df = fi_df.sort_values("importance", ascending=False)

    fi_df.head(20).plot(kind="barh", x="feature", y="importance", figsize=(8, 6))
    plt.title("Top 20 LightGBM Feature Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(EXPERIMENTS_DIR, "lgb_feature_importance.png"))
    plt.close()

    return fi_df

# Error Analysis
def error_analysis(val_df, y_true, preds):
    # Always define results first
    results = val_df.copy()
    results["actual_sales"] = y_true.values
    results["predicted_sales"] = preds
    results["error"] = results["actual_sales"] - results["predicted_sales"]
    results["abs_error"] = np.abs(results["error"])

    # Save worst predictions
    worst_path = os.path.join(EXPERIMENTS_DIR, "top_100_worst_errors.csv")
    results.sort_values("abs_error", ascending=False).head(100).to_csv(worst_path, index=False)

    # Error by store
    if "store_nbr" in results.columns:
        store_error = (
            results.groupby("store_nbr")["abs_error"]
            .mean()
            .sort_values(ascending=False)
        )

        store_error.head(20).plot(kind="bar", figsize=(8,5))
        plt.title("Top 20 Stores with Highest Prediction Error")
        plt.ylabel("Mean Absolute Error")
        plt.tight_layout()
        plt.savefig(os.path.join(EXPERIMENTS_DIR, "error_by_store.png"))
        plt.close()
    else:
        print("⚠️ No 'store_nbr' column found — skipping store-level error analysis.")

    # Error by family (auto-detect)
    family_cols = [c for c in results.columns if "family" in c.lower()]

    if len(family_cols) == 0:
        print("⚠️ No family column found — skipping family-level error analysis.")
    else:
        fam_col = family_cols[0]
        family_error = (
            results.groupby(fam_col)["abs_error"]
            .mean()
            .sort_values(ascending=False)
        )

        family_error.head(20).plot(kind="bar", figsize=(8,5))
        plt.title("Top 20 Families with Highest Prediction Error")
        plt.ylabel("Mean Absolute Error")
        plt.tight_layout()
        plt.savefig(os.path.join(EXPERIMENTS_DIR, "error_by_family.png"))
        plt.close()

    print("Saved error analysis outputs in /experiments")

# Main Analysis Pipeline
def run_analysis():
    df, xgb_model, lgb_model = load_data_and_models()

    train_df, val_df = time_split(df)
    X_val, y_val = prepare_xy(val_df)

    # XGBoost predictions
    xgb_preds = xgb_model.predict(X_val)
    xgb_rmse = np.sqrt(mean_squared_error(y_val, xgb_preds))
    print(f"XGBoost Validation RMSE: {xgb_rmse:.6f}")

    # Feature importance
    feature_names = X_val.columns.tolist()
    plot_xgb_feature_importance(xgb_model, feature_names)
    plot_lgb_feature_importance(lgb_model, feature_names)

    # Error analysis
    error_analysis(val_df, y_val, xgb_preds)

if __name__ == "__main__":
    run_analysis()