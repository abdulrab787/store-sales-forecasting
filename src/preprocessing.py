import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np

from config import RAW_PATH
import pandas as pd
import os

def load_data(raw_path=RAW_PATH):
    train = pd.read_csv(os.path.join(raw_path, "train.csv"))
    test = pd.read_csv(os.path.join(raw_path, "test.csv"))
    stores = pd.read_csv(os.path.join(raw_path, "stores.csv"))
    oil = pd.read_csv(os.path.join(raw_path, "oil.csv"))
    transactions = pd.read_csv(os.path.join(raw_path, "transactions.csv"))
    return train, test, stores, oil, transactions

def preprocess_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert date to datetime and extract useful features
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5,6]).astype(int)

    return df


def merge_external_data(train, test, stores, oil, transactions):
    # Convert all date columns to datetime
    train["date"] = pd.to_datetime(train["date"])
    test["date"] = pd.to_datetime(test["date"])
    oil["date"] = pd.to_datetime(oil["date"])
    transactions["date"] = pd.to_datetime(transactions["date"])

    # Merge oil prices
    train = train.merge(oil, on="date", how="left")
    test = test.merge(oil, on="date", how="left")

    # Merge store info
    train = train.merge(stores, on="store_nbr", how="left")
    test = test.merge(stores, on="store_nbr", how="left")

    # Merge transactions
    train = train.merge(transactions, on=["store_nbr", "date"], how="left")
    test = test.merge(transactions, on=["store_nbr", "date"], how="left")

    return train, test

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Smart missing value handling
    """
    df = df.copy()

    # Forward-fill oil prices (time series method)
    df["dcoilwtico"] = df["dcoilwtico"].ffill()

    # Fill numeric NaNs with median
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    # Fill categorical NaNs with "Unknown"
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        df[col] = df[col].fillna("Unknown")

    return df


def create_lag_features(df: pd.DataFrame, lags=[7, 14, 30]) -> pd.DataFrame:
    """
    Create lag features for time series forecasting
    """
    df = df.copy()

    df = df.sort_values(["store_nbr", "family", "date"])

    for lag in lags:
        df[f"sales_lag_{lag}"] = (
            df.groupby(["store_nbr", "family"])["sales"]
            .shift(lag)
        )

    return df


def create_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create rolling window statistics
    """
    df = df.copy()

    df = df.sort_values(["store_nbr", "family", "date"])

    df["sales_roll_mean_7"] = (
        df.groupby(["store_nbr", "family"])["sales"]
        .rolling(7)
        .mean()
        .reset_index(level=[0,1], drop=True)
    )

    df["sales_roll_std_7"] = (
        df.groupby(["store_nbr", "family"])["sales"]
        .rolling(7)
        .std()
        .reset_index(level=[0,1], drop=True)
    )

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode categorical variables
    """
    cat_cols = ["family", "city", "state", "type"]

    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    return df


def full_preprocessing_pipeline(train, test, stores, oil, transactions):
    """
    Run full preprocessing pipeline
    """

    # Step 1: Date features
    train = preprocess_dates(train)
    test = preprocess_dates(test)

    # Step 2: Merge external data
    train, test = merge_external_data(train, test, stores, oil, transactions)

    # Step 3: Handle missing values
    train = handle_missing_values(train)
    test = handle_missing_values(test)

    # Step 4: Lag features (only for train)
    train = create_lag_features(train)

    # Step 5: Rolling features (only for train)
    train = create_rolling_features(train)

    # Drop NaNs created by lag/rolling
    train = train.dropna()

    # Step 6: Encode categoricals
    train = encode_categoricals(train)
    test = encode_categoricals(test)

    return train, test
