# phase3/data/data_loader.py
import os
import numpy as np
import pandas as pd

class DataLoader:
    """
    Loads prepared data from CSV located in the project's `data/` folder.
    Returns numeric feature matrix (float32) and target (float64).
    """
    def __init__(self, path: str = None):
        if path:
            self.path = path
        else:
            base = os.path.abspath(
                os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data')
            )
            self.path = os.path.join(base, 'prepared_data.csv')

    def load(self, target_column: str = 'Price', bin_column: str = 'bin'):
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Data file not found at {self.path}")
        df = pd.read_csv(self.path)
        # Drop target and bin columns if present
        X = df.drop(columns=[target_column, bin_column], errors='ignore')
        # Convert all columns to numeric, fill missing
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        features = X.values.astype(np.float32)
        targets = pd.to_numeric(df[target_column], errors='coerce').fillna(0).values.astype(np.float64)
        return features, targets
