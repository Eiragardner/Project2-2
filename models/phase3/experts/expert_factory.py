# phase3/experts/expert_factory.py
import numpy as np
from typing import List, Tuple, Any
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
import lightgbm as lgb
import xgboost as xgb
from ..core.config import MoEConfig
from ..core.logger import MoELogger

class ExpertFactory:
    """Factory for creating and selecting expert models"""
    
    def __init__(self, config: MoEConfig, logger: MoELogger):
        self.config = config
        self.logger = logger
    
    def create_expert_pool(self) -> List[Tuple[str, Any]]:
        """Create a diverse pool of expert regressors"""
        experts = []
        
        # LightGBM variants
        experts.extend([
            ('lgb_fast', lgb.LGBMRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1, 
                random_state=self.config.random_state, verbose=-1
            )),
            ('lgb_deep', lgb.LGBMRegressor(
                n_estimators=200, max_depth=10, learning_rate=0.05, 
                random_state=self.config.random_state, verbose=-1
            )),
            ('lgb_wide', lgb.LGBMRegressor(
                n_estimators=150, max_depth=4, learning_rate=0.08, 
                num_leaves=100, random_state=self.config.random_state, verbose=-1
            )),
        ])
        
        # XGBoost variants
        experts.extend([
            ('xgb_standard', xgb.XGBRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1, 
                random_state=self.config.random_state, verbosity=0
            )),
            ('xgb_conservative', xgb.XGBRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.05, 
                reg_alpha=0.1, random_state=self.config.random_state, verbosity=0
            )),
            ('xgb_aggressive', xgb.XGBRegressor(
                n_estimators=150, max_depth=8, learning_rate=0.12, 
                subsample=0.8, random_state=self.config.random_state, verbosity=0
            )),
        ])
        
        # Linear models
        experts.extend([
            ('ridge_l1', Ridge(alpha=1.0)),
            ('ridge_l2', Ridge(alpha=10.0)),
            ('elastic_net', ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=self.config.random_state)),
        ])
        
        # Tree-based
        experts.extend([
            ('rf_standard', RandomForestRegressor(
                n_estimators=100, max_depth=8, random_state=self.config.random_state
            )),
            ('rf_deep', RandomForestRegressor(
                n_estimators=150, max_depth=12, min_samples_split=5, 
                random_state=self.config.random_state
            )),
        ])
        
        return experts
    
    def select_experts_for_bins(self, X: np.ndarray, y: np.ndarray, 
                              bin_assignments: np.ndarray, n_bins: int) -> List[Tuple[str, Any]]:
        """Select best experts for each bin"""
        self.logger.info("Selecting optimal experts for each bin...")
        
        expert_pool = self.create_expert_pool()
        selected_experts = []
        
        for bin_idx in range(n_bins):
            mask = bin_assignments == bin_idx
            if mask.sum() < 5:
                # Use simple model for small bins
                selected_experts.append(('ridge_simple', Ridge(alpha=1.0)))
                continue
            
            X_bin, y_bin = X[mask], y[mask]
            
            if self.config.expert_selection_strategy == 'performance_based':
                best_expert = self._select_best_expert_cv(X_bin, y_bin, expert_pool)
            elif self.config.expert_selection_strategy == 'specialized':
                best_expert = self._select_specialized_expert(bin_idx, n_bins, expert_pool)
            else:  # random
                best_expert = expert_pool[bin_idx % len(expert_pool)]
            
            selected_experts.append(best_expert)
            self.logger.info(f"Bin {bin_idx}: Selected {best_expert[0]} ({mask.sum()} samples)")
        
        return selected_experts
    
    def _select_best_expert_cv(self, X: np.ndarray, y: np.ndarray, 
                              expert_pool: List[Tuple[str, Any]]) -> Tuple[str, Any]:
        """Select best expert using cross-validation"""
        if len(y) < 10:
            return expert_pool[0]
        
        best_score = np.inf
        best_expert = expert_pool[0]
        
        n_folds = min(self.config.cv_folds, len(y) // 3)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.config.random_state)
        
        # Test only top experts to save time
        for name, expert in expert_pool[:6]:
            scores = []
            try:
                for train_idx, val_idx in kf.split(X):
                    X_train_cv, X_val_cv = X[train_idx], X[val_idx]
                    y_train_cv, y_val_cv = y[train_idx], y[val_idx]
                    
                    expert_copy = clone(expert)
                    expert_copy.fit(X_train_cv, y_train_cv)
                    y_pred = expert_copy.predict(X_val_cv)
                    
                    score = mean_squared_error(y_val_cv, y_pred)
                    scores.append(score)
                
                avg_score = np.mean(scores)
                if avg_score < best_score:
                    best_score = avg_score
                    best_expert = (name, expert)
                    
            except Exception as e:
                self.logger.warning(f"Expert {name} failed: {str(e)}")
                continue
        
        return best_expert
    
    def _select_specialized_expert(self, bin_idx: int, n_bins: int, 
                                 expert_pool: List[Tuple[str, Any]]) -> Tuple[str, Any]:
        """Select expert based on bin characteristics"""
        # Low bins: Linear models
        if bin_idx < n_bins // 3:
            return ('ridge_l1', Ridge(alpha=1.0))
        # High bins: Complex models
        elif bin_idx >= 2 * n_bins // 3:
            return ('lgb_deep', lgb.LGBMRegressor(
                n_estimators=200, max_depth=10, learning_rate=0.05, 
                random_state=self.config.random_state, verbose=-1
            ))
        # Middle bins: Balanced models
        else:
            return ('xgb_standard', xgb.XGBRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1, 
                random_state=self.config.random_state, verbosity=0
            ))
    
    def train_experts(self, X_train: np.ndarray, y_train: np.ndarray,
                     bin_assignments: np.ndarray, selected_experts: List[Tuple[str, Any]]) -> List[Any]:
        """Train all expert models"""
        self.logger.info("Training expert models...")
        
        trained_experts = []
        
        for bin_idx, (name, expert) in enumerate(selected_experts):
            mask = bin_assignments == bin_idx
            if mask.sum() == 0:
                self.logger.warning(f"Bin {bin_idx} has no training samples")
                trained_experts.append(expert)
                continue
            
            X_bin, y_bin = X_train[mask], y_train[mask]
            
            try:
                expert.fit(X_bin, y_bin)
                trained_experts.append(expert)
                self.logger.info(f"Trained expert for bin {bin_idx}: {mask.sum()} samples")
            except Exception as e:
                self.logger.error(f"Failed to train expert for bin {bin_idx}: {str(e)}")
                # Fallback to simple model
                fallback = Ridge(alpha=1.0)
                fallback.fit(X_bin, y_bin)
                trained_experts.append(fallback)
        
        return trained_experts
