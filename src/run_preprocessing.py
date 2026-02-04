import sys, os

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# Import config paths
from config import RAW_PATH, PROCESSED_PATH

# Import preprocessing functions
from preprocessing import load_data, full_preprocessing_pipeline

# Load raw data
train, test, stores, oil, transactions = load_data(RAW_PATH)

# Run preprocessing
train_proc, test_proc = full_preprocessing_pipeline(
    train, test, stores, oil, transactions
)

# Ensure processed directory exists
os.makedirs(PROCESSED_PATH, exist_ok=True)

# Save outputs
train_proc.to_csv(os.path.join(PROCESSED_PATH, "train_final.csv"), index=False)
test_proc.to_csv(os.path.join(PROCESSED_PATH, "test_final.csv"), index=False)

print("Preprocessing completed successfully!")