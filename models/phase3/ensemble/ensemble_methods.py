# phase3/ensemble/ensemble_methods.py
import numpy as np
from typing import List, Dict, Any
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor
from ..core.config import MoEConfig
from ..core.logger import MoELogger

class EnsembleManager:
    """Manages different ensemble strategies for MoE predictions"""
    
    def __init__(self, config: MoEConfig, logger: MoELogger):
        self.config = config
        self.logger = logger
        self.ensemble_models = {}
    
    def create_ensemble_predictions(self, base_predictions: Dict[str, np.ndarray],
                                  X_train: np.ndarray, y_train: np.ndarray,
                                  X_test: np.ndarray) -> Dict[str, np.ndarray]:
        """Create ensemble predictions using different strategies"""
        ensemble_preds = {}
        
        for method in self.config.ensemble_methods:
            if method == 'voting':
                ensemble_preds['voting'] = self._simple_voting(base_predictions)
            elif method == 'stacking':
                ensemble_preds['stacking'] = self._stacking_ensemble(
                    base_predictions, X_train, y_train, X_test
                )
            elif method == 'weighted':
                ensemble_preds['weighted'] = self._weighted_ensemble(base_predictions)
        
        return ensemble_preds
    
    def _simple_voting(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Simple average voting"""
        pred_stack = np.column_stack(list(predictions.values()))
        return np.mean(pred_stack, axis=1)
    
    def _weighted_ensemble(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Weighted ensemble based on individual model performance"""
        # Simple equal weighting for now - could be enhanced with validation performance
        return self._simple_voting(predictions)
    
    def _stacking_ensemble(self, base_predictions: Dict[str, np.ndarray],
                          X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray) -> np.ndarray:
        """Stacking ensemble using linear meta-learner"""
        try:
            # Create training data for meta-learner
            train_stack = np.column_stack(list(base_predictions.values()))
            
            # Train meta-learner
            meta_learner = LinearRegression()
            meta_learner.fit(train_stack, y_train)
            
            # Store for future use
            self.ensemble_models['stacking'] = meta_learner
            
            # Make predictions (assuming same structure for test predictions)
            test_stack = np.column_stack(list(base_predictions.values()))
            return meta_learner.predict(test_stack)
            
        except Exception as e:
            self.logger.warning(f"Stacking ensemble failed: {e}. Using voting instead.")
            return self._simple_voting(base_predictions)

