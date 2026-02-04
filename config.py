import os

# Project root (directory where config.py lives)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Data directories
RAW_PATH = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_PATH = os.path.join(PROJECT_ROOT, "data", "processed")

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")