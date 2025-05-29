# phase3/data/data_manager.py
import os
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from ..core.config import MoEConfig
from ..core.logger import MoELogger


class DataManager:
    """Handles all data loading, preprocessing, and splitting operations"""
    
    def __init__(self, config: MoEConfig, logger: MoELogger):
        self.config = config
        self.logger = logger
        self.scalers = {}
        self.original_shape = None
        self.feature_names = None
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load data from various sources"""
        self.logger.info("Loading data...")
        
        X, y = None, None
        
        # Try to load from specified path
        if self.config.data_path and os.path.exists(self.config.data_path):
            X, y = self._load_from_file(self.config.data_path)
        
        # Try common data locations
        if X is None:
            data_paths = [
                '../data/prepared_data.csv',
                'data/prepared_data.csv',
                'prepared_data.csv'
            ]
            
            for path in data_paths:
                if os.path.exists(path):
                    X, y = self._load_from_file(path)
                    break
        
        # Generate sample data if no real data found
        if X is None:
            self.logger.warning("No data file found. Generating sample data.")
            X, y = self._generate_sample_data()
        
        self.original_shape = X.shape
        self.logger.info(f"Loaded dataset: {X.shape[0]} samples, {X.shape[1]} features")
        self.logger.info(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
        
        return X, y
    
    def _load_from_file(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load data from file"""
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
                self.feature_names = df.columns[:-1].tolist()
                X = df.iloc[:, :-1].values
                y = df.iloc[:, -1].values
                return X, y
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to load data from {file_path}: {str(e)}")
            return None, None
    
    def _generate_sample_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate sample data for testing"""
        np.random.seed(self.config.random_state)
        n_samples, n_features = 5500, 50
        
        X = np.random.randn(n_samples, n_features)
        # Create realistic target with some correlation to features
        y = (X[:, :5].sum(axis=1) * 2 + 
             np.random.randn(n_samples) * 5 + 50)
        
        self.feature_names = [f'feature_{i}' for i in range(n_features)]
        return X, y
    
    def preprocess_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess data with outlier removal and scaling"""
        self.logger.info("Preprocessing data...")
        
        # Remove outliers from target
        X, y = self._remove_outliers(X, y)
        
        # Scale features
        X = self._scale_features(X)
        
        return X, y
    
    def _remove_outliers(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Remove outliers using IQR method"""
        Q1, Q3 = np.percentile(y, [25, 75])
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_mask = (y >= lower_bound) & (y <= upper_bound)
        n_outliers = (~outlier_mask).sum()
        
        if n_outliers > 0:
            self.logger.info(f"Removed {n_outliers} outliers ({n_outliers/len(y)*100:.1f}%)")
            return X[outlier_mask], y[outlier_mask]
        
        return X, y
    
    def _scale_features(self, X: np.ndarray) -> np.ndarray:
        """Scale features based on configuration"""
        if self.config.scaling_method == 'none':
            return X
        
        if self.config.scaling_method == 'standard':
            self.scalers['features'] = StandardScaler()
        elif self.config.scaling_method == 'robust':
            self.scalers['features'] = RobustScaler()
        else:
            self.logger.warning(f"Unknown scaling method: {self.config.scaling_method}")
            return X
        
        X_scaled = self.scalers['features'].fit_transform(X)
        self.logger.info(f"Applied {self.config.scaling_method} scaling")
        return X_scaled
    
    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Split data into train/validation/test sets"""
        self.logger.info("Splitting data...")
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=self.config.test_size,
            random_state=self.config.random_state
        )
        
        # Second split: separate train and validation
        val_size_adjusted = self.config.val_size / (1 - self.config.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=self.config.random_state
        )
        
        self.logger.info(
            f"Data split - Train: {len(X_train)}, "
            f"Val: {len(X_val)}, Test: {len(X_test)}"
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test

