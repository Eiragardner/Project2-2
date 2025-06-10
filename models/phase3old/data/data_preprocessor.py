# phase3/data/data_preprocessor.py
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import Dict, Tuple, Any
import logging

class DataPreprocessor:
    """Handles all data preprocessing tasks"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scalers = {}
        self.logger = logging.getLogger(__name__)
    
    def load_and_preprocess_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess data with advanced techniques"""
        self.logger.info("Loading and preprocessing data...")
        
        # Try to load actual data first
        try:
            if os.path.exists('../data/prepared_data.csv'):
                df = pd.read_csv('../data/prepared_data.csv')
                X = df.iloc[:, :-1].values
                y = df.iloc[:, -1].values
            elif os.path.exists('data/prepared_data.csv'):
                df = pd.read_csv('data/prepared_data.csv')
                X = df.iloc[:, :-1].values
                y = df.iloc[:, -1].values
            else:
                loader = DataLoader()
                X, y = loader.load()
        except Exception as e:
            self.logger.warning(f"Could not load data: {e}. Using sample data.")
            np.random.seed(42)
            X = np.random.randn(5500, 50)
            y = np.random.randn(5500) * 10 + 50
        
        self.logger.info(f"Loaded dataset: {X.shape[0]} samples, {X.shape[1]} features")
        self.logger.info(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
        
        # Handle outliers
        X, y = self._remove_outliers(X, y)
        
        # Feature scaling
        X = self._scale_features(X)
        
        return X, y
   