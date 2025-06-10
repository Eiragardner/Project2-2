# models/phase3/experts/enhanced_expert_factory_fixed.py
"""
Enhanced Expert Factory with Rule-Based Selection for Small Bins

When bins are too small for reliable CV (< 800 samples), use rule-based selection:
1. Force proven good models (LightGBM, XGBoost)
2. Use simple train/val split validation
3. Fallback hierarchy based on bin characteristics
"""
import numpy as np
import warnings
from typing import List, Tuple, Any
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


class EnhancedExpertFactory:
    """Enhanced expert factory with intelligent selection based on bin size"""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.feature_names = None
        
        # Key thresholds for selection strategy
        self.cv_threshold = 800  # Use CV only for bins with 800+ samples
        self.min_val_size = 100  # Minimum validation size for reliable evaluation
        
    def create_expert_pool(self) -> List[Tuple[str, Any]]:
        """Create comprehensive expert pool"""
        experts = []
        
        # High-quality models (prefer these for small bins)
        if LGB_AVAILABLE:
            experts.extend([
                ('lgb_conservative', lgb.LGBMRegressor(
                    n_estimators=50, max_depth=4, learning_rate=0.05,
                    min_child_samples=20, num_leaves=15,
                    random_state=self.config.random_state, verbose=-1,
                    objective='regression', force_col_wise=True
                )),
                ('lgb_balanced', lgb.LGBMRegressor(
                    n_estimators=100, max_depth=6, learning_rate=0.1,
                    min_child_samples=10, num_leaves=31,
                    random_state=self.config.random_state, verbose=-1,
                    objective='regression', force_col_wise=True
                )),
                ('lgb_aggressive', lgb.LGBMRegressor(
                    n_estimators=150, max_depth=8, learning_rate=0.15,
                    min_child_samples=5, num_leaves=63,
                    random_state=self.config.random_state, verbose=-1,
                    objective='regression', force_col_wise=True
                )),
            ])
        
        if XGB_AVAILABLE:
            experts.extend([
                ('xgb_conservative', xgb.XGBRegressor(
                    n_estimators=50, max_depth=4, learning_rate=0.05,
                    min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                    random_state=self.config.random_state, n_jobs=1, verbosity=0
                )),
                ('xgb_balanced', xgb.XGBRegressor(
                    n_estimators=100, max_depth=6, learning_rate=0.1,
                    min_child_weight=3, subsample=0.9, colsample_bytree=0.9,
                    random_state=self.config.random_state, n_jobs=1, verbosity=0
                )),
            ])
        
        # Tree-based models (good middle ground)
        experts.extend([
            ('rf_shallow', RandomForestRegressor(
                n_estimators=50, max_depth=5, min_samples_split=10,
                min_samples_leaf=5, random_state=self.config.random_state, n_jobs=1
            )),
            ('rf_medium', RandomForestRegressor(
                n_estimators=100, max_depth=10, min_samples_split=5,
                min_samples_leaf=2, random_state=self.config.random_state, n_jobs=1
            )),
        ])
        
        # Linear models (use for very small bins or as fallback)
        experts.extend([
            ('ridge_light', Ridge(alpha=1.0, random_state=self.config.random_state)),
            ('ridge_medium', Ridge(alpha=10.0, random_state=self.config.random_state)),
            ('ridge_heavy', Ridge(alpha=100.0, random_state=self.config.random_state)),
            ('elastic_light', ElasticNet(
                alpha=1.0, l1_ratio=0.3, random_state=self.config.random_state, 
                max_iter=2000, tol=1e-4
            )),
            ('elastic_balanced', ElasticNet(
                alpha=10.0, l1_ratio=0.5, random_state=self.config.random_state, 
                max_iter=2000, tol=1e-4
            )),
        ])
        
        self.logger.info(f"Created expert pool with {len(experts)} models")
        if LGB_AVAILABLE:
            self.logger.info("LightGBM models available for high-quality selection")
        if XGB_AVAILABLE:
            self.logger.info("XGBoost models available for high-quality selection")
        
        return experts
    
    def select_experts_for_bins(self, X: np.ndarray, y: np.ndarray, 
                              bin_assignments: np.ndarray, n_bins: int) -> List[Tuple[str, Any]]:
        """Intelligent expert selection based on bin size"""
        self.logger.info("Selecting optimal experts with size-aware strategy...")
        
        expert_pool = self.create_expert_pool()
        selected_experts = []
        
        for bin_idx in range(n_bins):
            mask = bin_assignments == bin_idx
            bin_size = mask.sum()
            
            if bin_size < 10:  # Extremely small bin
                selected_experts.append(self._get_fallback_expert(bin_size))
                self.logger.warning(f"Bin {bin_idx}: Tiny bin ({bin_size}), using fallback")
                continue
            
            X_bin, y_bin = X[mask], y[mask]
            
            # Choose strategy based on bin size
            if bin_size >= self.cv_threshold:
                # Large bin: Use CV selection
                self.logger.info(f"Bin {bin_idx}: Large bin ({bin_size}), using CV selection")
                best_expert = self._select_expert_cv_validation(X_bin, y_bin, expert_pool, bin_idx)
            else:
                # Small bin: Use rule-based selection
                self.logger.info(f"Bin {bin_idx}: Small bin ({bin_size}), using rule-based selection")
                best_expert = self._select_expert_rule_based(X_bin, y_bin, expert_pool, bin_idx, bin_size)
            
            selected_experts.append(best_expert)
        
        return selected_experts
    
    def _select_expert_rule_based(self, X: np.ndarray, y: np.ndarray, 
                                expert_pool: List[Tuple[str, Any]], bin_idx: int, bin_size: int) -> Tuple[str, Any]:
        """Rule-based expert selection for small bins"""
        
        # Rule 1: Always try high-quality models first
        priority_models = []
        
        if LGB_AVAILABLE:
            priority_models.extend(['lgb_conservative', 'lgb_balanced'])
        if XGB_AVAILABLE:
            priority_models.extend(['xgb_conservative', 'xgb_balanced'])
        
        priority_models.extend(['rf_shallow', 'rf_medium'])
        
        # Rule 2: Determine target complexity based on bin size
        if bin_size >= 400:
            # Medium-small bins: Can handle moderate complexity
            allowed_models = priority_models + ['ridge_medium', 'elastic_light']
        elif bin_size >= 200:
            # Small bins: Conservative models only
            allowed_models = [m for m in priority_models if 'conservative' in m or 'shallow' in m] + ['ridge_medium']
        else:
            # Very small bins: Only most conservative models
            allowed_models = [m for m in priority_models if 'conservative' in m or 'shallow' in m] + ['ridge_heavy']
        
        # Rule 3: Filter expert pool to allowed models
        candidate_experts = [(name, model) for name, model in expert_pool if name in allowed_models]
        
        if not candidate_experts:
            # Fallback if no allowed models
            return self._get_fallback_expert(bin_size)
        
        # Rule 4: Simple validation on best candidates
        if bin_size >= self.min_val_size:
            return self._evaluate_candidates_simple(X, y, candidate_experts, bin_idx)
        else:
            # Too small for validation, use heuristic selection
            return self._select_by_heuristics(candidate_experts, bin_size, y)
    
    def _evaluate_candidates_simple(self, X: np.ndarray, y: np.ndarray,
                                  candidate_experts: List[Tuple[str, Any]], bin_idx: int) -> Tuple[str, Any]:
        """Simple train/val evaluation for small bins"""
        
        # Use 70/30 split for more stable validation
        val_size = max(20, min(int(0.3 * len(y)), len(y) - 30))
        
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=val_size, random_state=42
            )
        except ValueError:
            # If split fails, use heuristic selection
            return self._select_by_heuristics(candidate_experts, len(y), y)
        
        best_score = np.inf
        best_expert = candidate_experts[0]
        results = []
        
        for name, expert in candidate_experts:
            try:
                expert_copy = clone(expert)
                expert_copy.fit(X_train, y_train)
                
                y_pred = expert_copy.predict(X_val)
                
                # Validate predictions
                if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                    continue
                
                # Use composite score: MAE + RMSE (both important)
                mae = mean_absolute_error(y_val, y_pred)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                r2 = r2_score(y_val, y_pred)
                
                # Composite score: normalize MAE and RMSE, penalize negative R²
                mae_norm = mae / np.mean(y_val)  # Relative MAE
                rmse_norm = rmse / np.std(y_val)  # Relative RMSE
                r2_penalty = max(0, -r2) * 2  # Penalty for negative R²
                
                composite_score = mae_norm + rmse_norm + r2_penalty
                
                results.append((name, composite_score, mae, rmse, r2))
                
                if composite_score < best_score:
                    best_score = composite_score
                    best_expert = (name, expert)
                    
            except Exception as e:
                self.logger.debug(f"Expert {name} failed on bin {bin_idx}: {e}")
                continue
        
        # Log results
        if results:
            results.sort(key=lambda x: x[1])  # Sort by composite score
            self.logger.info(f"Bin {bin_idx}: Rule-based selection results:")
            for i, (name, score, mae, rmse, r2) in enumerate(results[:3]):
                marker = " [SELECTED]" if name == best_expert[0] else ""
                self.logger.info(f"  {i+1}. {name}: Score={score:.3f}, MAE=${mae:.0f}, RMSE=${rmse:.0f}, R²={r2:.3f}{marker}")
        else:
            self.logger.warning(f"Bin {bin_idx}: No valid results, using heuristic selection")
            return self._select_by_heuristics(candidate_experts, len(y), y)
        
        return best_expert
    
    def _select_by_heuristics(self, candidate_experts: List[Tuple[str, Any]], 
                            bin_size: int, y: np.ndarray) -> Tuple[str, Any]:
        """Heuristic expert selection when validation is not possible"""
        
        # Heuristic 1: Prefer order based on reliability and bin size
        preference_order = []
        
        if bin_size >= 300:
            # Can handle moderate complexity
            preference_order = ['lgb_conservative', 'xgb_conservative', 'rf_shallow', 'lgb_balanced', 'ridge_medium']
        elif bin_size >= 150:
            # Conservative models only
            preference_order = ['lgb_conservative', 'xgb_conservative', 'rf_shallow', 'ridge_medium', 'ridge_heavy']
        else:
            # Very small: simplest models
            preference_order = ['ridge_medium', 'ridge_heavy', 'lgb_conservative', 'elastic_light']
        
        # Heuristic 2: Consider target variance
        target_cv = np.std(y) / np.mean(y)  # Coefficient of variation
        
        if target_cv > 0.5:  # High variance
            # Prefer robust models
            preference_order = ['lgb_conservative', 'xgb_conservative', 'ridge_heavy'] + preference_order
        
        # Select first available model from preference order
        available_names = [name for name, _ in candidate_experts]
        
        for preferred_name in preference_order:
            if preferred_name in available_names:
                selected_expert = next((name, model) for name, model in candidate_experts if name == preferred_name)
                self.logger.info(f"Heuristic selection: {preferred_name} (bin_size={bin_size}, target_cv={target_cv:.2f})")
                return selected_expert
        
        # Fallback to first available
        self.logger.info(f"Heuristic fallback: {candidate_experts[0][0]}")
        return candidate_experts[0]
    
    def _select_expert_cv_validation(self, X: np.ndarray, y: np.ndarray, 
                                   expert_pool: List[Tuple[str, Any]], bin_idx: int) -> Tuple[str, Any]:
        """Cross-validation for large bins (unchanged from original)"""
        
        best_score = np.inf
        best_expert = expert_pool[0]
        results = []
        
        # Use 3-fold CV for efficiency
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        
        for name, expert in expert_pool:
            try:
                cv_scores = []
                
                for train_idx, val_idx in kf.split(X):
                    X_train_cv, X_val_cv = X[train_idx], X[val_idx]
                    y_train_cv, y_val_cv = y[train_idx], y[val_idx]
                    
                    if len(y_val_cv) < 3:
                        continue
                    
                    expert_copy = clone(expert)
                    expert_copy.fit(X_train_cv, y_train_cv)
                    
                    y_pred = expert_copy.predict(X_val_cv)
                    
                    if not (np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred))):
                        mse = mean_squared_error(y_val_cv, y_pred)
                        if np.isfinite(mse):
                            cv_scores.append(mse)
                
                if len(cv_scores) >= 2:
                    avg_mse = np.mean(cv_scores)
                    std_mse = np.std(cv_scores)
                    avg_r2 = 1 - (avg_mse / np.var(y))  # Approximate R²
                    
                    results.append((name, avg_mse, avg_r2, std_mse))
                    
                    if avg_mse < best_score:
                        best_score = avg_mse
                        best_expert = (name, expert)
                        
            except Exception as e:
                self.logger.debug(f"Expert {name} failed on bin {bin_idx}: {e}")
                continue
        
        # Log results
        if results:
            results.sort(key=lambda x: x[1])
            self.logger.info(f"Bin {bin_idx}: CV selection results:")
            for i, (name, avg_mse, r2, std_mse) in enumerate(results[:3]):
                marker = " [SELECTED]" if name == best_expert[0] else ""
                self.logger.info(f"  {i+1}. {name}: MSE={avg_mse:.0f}±{std_mse:.0f}, R²={r2:.3f}{marker}")
        else:
            self.logger.warning(f"Bin {bin_idx}: No valid CV results, using rule-based fallback")
            return self._select_expert_rule_based(X, y, expert_pool, bin_idx, len(y))
        
        return best_expert
    
    def _get_fallback_expert(self, bin_size: int) -> Tuple[str, Any]:
        """Get a safe fallback expert for tiny bins"""
        if bin_size < 5:
            # Extremely small: Use simple Ridge
            return ('ridge_fallback', Ridge(alpha=100.0, random_state=self.config.random_state))
        else:
            # Small but manageable: Use moderate Ridge
            return ('ridge_fallback', Ridge(alpha=10.0, random_state=self.config.random_state))
    
    def train_experts(self, X: np.ndarray, y: np.ndarray, 
                     bin_assignments: np.ndarray, selected_experts: List[Tuple[str, Any]]) -> List[Any]:
        """Train the selected expert models (unchanged from original)"""
        self.logger.info("Training selected expert models...")
        
        trained_experts = []
        successful_trainings = 0
        
        for bin_idx, (expert_name, expert_model) in enumerate(selected_experts):
            try:
                mask = bin_assignments == bin_idx
                bin_size = mask.sum()
                
                if bin_size < 3:
                    fallback = Ridge(alpha=100.0, random_state=self.config.random_state)
                    if bin_size > 0:
                        fallback.fit(X[mask], y[mask])
                    else:
                        fallback.coef_ = np.zeros(X.shape[1])
                        fallback.intercept_ = np.mean(y)
                    trained_experts.append(fallback)
                    self.logger.warning(f"Bin {bin_idx}: Tiny bin ({bin_size}), using fallback")
                    continue
                
                X_bin, y_bin = X[mask], y[mask]
                expert_copy = clone(expert_model)
                expert_copy.fit(X_bin, y_bin)
                
                trained_experts.append(expert_copy)
                successful_trainings += 1
                
                self.logger.info(f"Successfully trained {expert_name} for bin {bin_idx}: {bin_size} samples")
                
            except Exception as e:
                self.logger.error(f"Training failed for bin {bin_idx} ({expert_name}): {e}")
                fallback = Ridge(alpha=100.0, random_state=self.config.random_state)
                try:
                    mask = bin_assignments == bin_idx
                    if mask.sum() > 0:
                        fallback.fit(X[mask], y[mask])
                    else:
                        fallback.coef_ = np.zeros(X.shape[1])
                        fallback.intercept_ = np.mean(y)
                    trained_experts.append(fallback)
                    self.logger.info(f"Using Ridge fallback for bin {bin_idx}")
                except Exception as e2:
                    self.logger.error(f"Even fallback failed for bin {bin_idx}: {e2}")
                    class DummyModel:
                        def __init__(self, mean_val):
                            self.mean_val = mean_val
                        def predict(self, X):
                            return np.full(X.shape[0], self.mean_val)
                    trained_experts.append(DummyModel(np.mean(y)))
        
        self.logger.info(f"Training completed: {successful_trainings} experts trained")
        return trained_experts