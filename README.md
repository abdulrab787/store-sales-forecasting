![Python](https://img.shields.io/badge/python-3.9-blue)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-green)
![CI](https://img.shields.io/badge/CI-GitHub%20Actions-success)


# Store Sales Forecasting (Kaggle)

## 🎯 Objective
Build a production-ready machine learning pipeline to forecast daily store sales.

## 📊 Dataset
- Time series store sales data from Kaggle
- Includes store, item, date, and sales

## 🚀 Approach
- Exploratory Data Analysis (EDA)
- Feature Engineering (lag features, rolling stats, trends)
- Models:
  - XGBoost
  - LightGBM
  - Stacking Ensemble

## 🛠️ Tech Stack
- Python, Pandas, Scikit-learn, XGBoost, LightGBM
- MLflow for experiment tracking
- GitHub Actions for CI
- VS Code for development

## 📁 Project Structure
- `src/` — reusable ML pipeline
- `notebooks/` — EDA & experiments
- `models/` — trained models
- `submissions/` — Kaggle submissions

## 🔁 Reproducibility
Run:
```bash
python -m src.train_with_tracking
python -m src.make_submission
