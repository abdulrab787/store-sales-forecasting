import pandas as pd
from src.preprocessing import preprocess_data

def test_preprocessing_shape():
    df = pd.DataFrame({
        "sales": [100, 200, 300],
        "store": ["A", "B", "C"]
    })

    processed = preprocess_data(df)

    assert processed.shape[0] == 3  # same number of rows
