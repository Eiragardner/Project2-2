# phase3/optimization/hyperparameter_optimizer.py
import numpy as np
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
import lightgbm as lgb
import xgboost as xgb
from typing import List, Tuple, Any, Dict
import logging

class HyperparameterOptimizer:
    """
    Handles hyperparameter optimization for expert models.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def optimize_experts(self, X_train: np.ndarray, y_train: np.ndarray, 
                        bin_assignments: np.ndarray, selected_experts: List[Tuple[str, Any]]) -> List[Tuple[str, Any]]:
        """Optimize hyperparameters for selected experts"""
        self.logger.info("Optimizing expert hyperparameters...")
        
        optimized_experts = []
        
        for bin_idx, (name, expert) in enumerate(selected_experts):
            mask = bin_assignments == bin_idx
            if mask.sum() < 20:  # Skip optimization for small bins
                optimized_experts.append((name, expert))
                continue
            
            X_bin, y_bin = X_train[mask], y_train[mask]
            
            try:
                optimized_expert = self._optimize_single_expert(name, expert, X_bin, y_bin)
                optimized_experts.append((f"{name}_opt", optimized_expert))
                self.logger.info(f"Optimized expert for bin {bin_idx}")
            except Exception as e:
                self.logger.warning(f"Optimization failed for bin {bin_idx}: {str(e)}")
                optimized_experts.append((name, expert))
        
        return optimized_experts
    
    def _optimize_single_expert(self, name: str, expert: Any, X: np.ndarray, y: np.ndarray) -> Any:
        """Optimize a single expert model"""
        def objective(trial):
            try:
                # Define hyperparameter search space based on model type
                if 'lgb' in name:
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                        'max_depth': trial.suggest_int('max_depth', 3, 12),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                        'num_leaves': trial.suggest_int('num_leaves', 10, 200),
                        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    }
                    model = lgb.LGBMRegressor(**params, random_state=42, verbose=-1)
                elif 'xgb' in name:
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                        'max_depth': trial.suggest_int('max_depth', 3, 12),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                    }
                    model = xgb.XGBRegressor(**params, random_state=42, verbosity=0)
                elif 'ridge' in name:
                    alpha = trial.suggest_float('alpha', 0.1, 100, log=True)
                    model = Ridge(alpha=alpha)
                else:  # Default case
                    return 0
                
                # Cross-validation
                kf = KFold(n_splits=3, shuffle=True, random_state=42)
                scores = []
                for train_idx, val_idx in kf.split(X):
                    model.fit(X[train_idx], y[train_idx])
                    y_pred = model.predict(X[val_idx])
                    scores.append(mean_squared_error(y[val_idx], y_pred))
                
                return np.mean(scores)
                
            except Exception as e:
                return float('inf')
        
        # Optimize with limited trials for efficiency
        study = optuna.create_study(direction='minimize')
        n_trials = min(30, self.config['optimization_trials'] // 10)
        study.optimize(objective, n_trials=n_trials)
        
        # Create optimized expert
        best_params = study.best_params
        if 'lgb' in name:
            optimized_expert = lgb.LGBMRegressor(**best_params, random_state=42, verbose=-1)
        elif 'xgb' in name:
            optimized_expert = xgb.XGBRegressor(**best_params, random_state=42, verbosity=0)
        elif 'ridge' in name:
            optimized_expert = Ridge(**best_params)
        else:
            optimized_expert = expert
        
        return optimized_expert