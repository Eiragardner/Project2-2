# phase3/ensemble/ensemble_methods.py
import numpy as np
from typing import List, Dict, Any
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor
from ..core.config import MoEConfig
from ..core.logger import MoELogger

class EnsembleManager:
    """FIXED: Simple ensemble manager"""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
    
    def create_ensemble_predictions(self, base_predictions: dict, 
                                   X_train: np.ndarray, y_train: np.ndarray,
                                   X_test: np.ndarray) -> dict:
        """Create ensemble predictions"""
        ensemble_preds = {}
        
        for method in self.config.ensemble_methods:
            if method == 'voting':
                ensemble_preds['voting'] = self._simple_voting(base_predictions)
            elif method == 'weighted':
                ensemble_preds['weighted'] = self._simple_voting(base_predictions)  # Simplified
            else:  # stacking
                ensemble_preds['stacking'] = self._simple_voting(base_predictions)  # Simplified
        
        return ensemble_preds
    
    def _simple_voting(self, predictions: dict) -> np.ndarray:
        """Simple average voting"""
        pred_values = list(predictions.values())
        if len(pred_values) == 0:
            return np.array([])
        
        pred_stack = np.column_stack(pred_values)
        return np.mean(pred_stack, axis=1)