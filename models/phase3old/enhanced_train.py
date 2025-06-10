# phase3/enhanced_train.py
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from scipy import stats
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
import optuna
import logging
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Fix imports - make them more flexible
try:
    from .data.data_loader import DataLoader
except ImportError:
    # Fallback data loader
    class DataLoader:
        def load(self):
            # Load your actual data here
            # For now, create sample data
            np.random.seed(42)
            X = np.random.randn(5500, 50)  # 5500 samples, 50 features
            y = np.random.randn(5500) * 10 + 50  # Target values
            return X, y

try:
    from .binning.learnable_binning import LearnableBinning
except ImportError:
    # Simple fallback binning
    class LearnableBinning:
        def __init__(self, num_bins, init_edges):
            self.num_bins = num_bins
            self.edges = init_edges

try:
    from .classifier.moe_classifier import MoEClassifier
except ImportError:
    # We'll define the gate directly in the trainer
    pass


class EnhancedMoETrainer:
    """
    Advanced Mixture of Experts trainer with dynamic binning,
    intelligent expert assignment, and comprehensive optimization.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = self._get_default_config()
        if config:
            self.config.update(config)
        
        self.setup_logging()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self.scalers = {}
        self.bin_layer = None
        self.experts = []
        self.gate = None
        self.bin_assignments = None
        self.feature_importance = None
        
    def _get_default_config(self) -> Dict[str, Any]:
        return {
            'min_bin_size': 50,  # Minimum samples per bin
            'max_bins': 7,       # Maximum number of bins
            'min_bins': 3,       # Minimum number of bins
            'test_size': 0.2,
            'val_size': 0.15,
            'random_state': 42,
            'cv_folds': 5,
            'gate_epochs': 200,
            'gate_lr': 1e-3,
            'early_stopping_patience': 20,
            'optimization_trials': 100,
            'ensemble_methods': ['voting', 'stacking'],
            'scaling_method': 'robust',  # 'standard', 'robust', 'none'
            'binning_strategy': 'adaptive',  # 'quantile', 'kmeans', 'adaptive'
            'expert_selection_strategy': 'performance_based',  # 'random', 'specialized', 'performance_based'
        }
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_and_preprocess_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess data with advanced techniques"""
        self.logger.info("Loading and preprocessing data...")
        
        # Try to load actual data first
        try:
            # Try to load from CSV files in data directory
            if os.path.exists('../data/prepared_data.csv'):
                df = pd.read_csv('../data/prepared_data.csv')
                X = df.iloc[:, :-1].values  # All columns except last
                y = df.iloc[:, -1].values   # Last column
            elif os.path.exists('data/prepared_data.csv'):
                df = pd.read_csv('data/prepared_data.csv')
                X = df.iloc[:, :-1].values
                y = df.iloc[:, -1].values
            else:
                # Use DataLoader
                loader = DataLoader()
                X, y = loader.load()
        except Exception as e:
            self.logger.warning(f"Could not load data: {e}. Using sample data.")
            # Create sample data
            np.random.seed(42)
            X = np.random.randn(5500, 50)
            y = np.random.randn(5500) * 10 + 50
        
        self.logger.info(f"Loaded dataset: {X.shape[0]} samples, {X.shape[1]} features")
        self.logger.info(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
        
        # Handle outliers in target
        Q1, Q3 = np.percentile(y, [25, 75])
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_mask = (y >= lower_bound) & (y <= upper_bound)
        self.logger.info(f"Removed {(~outlier_mask).sum()} outliers")
        
        X, y = X[outlier_mask], y[outlier_mask]
        
        # Feature scaling
        if self.config['scaling_method'] == 'standard':
            self.scalers['features'] = StandardScaler()
        elif self.config['scaling_method'] == 'robust':
            self.scalers['features'] = RobustScaler()
        
        if self.config['scaling_method'] != 'none':
            X = self.scalers['features'].fit_transform(X)
        
        return X, y
    
    def determine_optimal_bins(self, y: np.ndarray) -> Tuple[int, np.ndarray]:
        """Dynamically determine optimal number of bins and boundaries"""
        self.logger.info("Determining optimal binning strategy...")
        
        n_samples = len(y)
        max_bins = min(self.config['max_bins'], n_samples // self.config['min_bin_size'])
        max_bins = max(max_bins, self.config['min_bins'])
        
        best_score = -np.inf
        best_bins = self.config['min_bins']
        best_boundaries = None
        
        for n_bins in range(self.config['min_bins'], max_bins + 1):
            if self.config['binning_strategy'] == 'quantile':
                boundaries = np.percentile(y, np.linspace(0, 100, n_bins + 1))[1:-1]
            elif self.config['binning_strategy'] == 'kmeans':
                kmeans = KMeans(n_clusters=n_bins, random_state=self.config['random_state'], n_init=10)
                clusters = kmeans.fit_predict(y.reshape(-1, 1))
                boundaries = []
                for i in range(n_bins - 1):
                    mask_curr = clusters == i
                    mask_next = clusters == i + 1
                    if mask_curr.any() and mask_next.any():
                        boundary = (y[mask_curr].max() + y[mask_next].min()) / 2
                        boundaries.append(boundary)
                boundaries = np.array(sorted(boundaries))
            else:  # adaptive
                # Combine quantile and density-based approaches
                q_boundaries = np.percentile(y, np.linspace(0, 100, n_bins + 1))[1:-1]
                
                # Adjust boundaries based on density
                hist, edges = np.histogram(y, bins=n_bins * 3)
                peak_indices = self._find_density_peaks(hist)
                
                if len(peak_indices) >= n_bins - 1:
                    density_boundaries = edges[peak_indices[:n_bins-1]]
                    # Weighted combination
                    boundaries = 0.7 * q_boundaries + 0.3 * density_boundaries
                else:
                    boundaries = q_boundaries
            
            # Evaluate binning quality
            bin_assignments = np.digitize(y, boundaries)
            score = self._evaluate_binning_quality(y, bin_assignments, n_bins)
            
            if score > best_score:
                best_score = score
                best_bins = n_bins
                best_boundaries = boundaries
        
        self.logger.info(f"Optimal bins: {best_bins}, Score: {best_score:.4f}")
        return best_bins, best_boundaries
    
    def _find_density_peaks(self, hist: np.ndarray) -> List[int]:
        """Find peaks in density histogram"""
        peaks = []
        for i in range(1, len(hist) - 1):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                peaks.append(i)
        return peaks
    
    def _evaluate_binning_quality(self, y: np.ndarray, bin_assignments: np.ndarray, n_bins: int) -> float:
        """Evaluate the quality of binning using multiple criteria"""
        scores = []
        
        # 1. Balance score (prefer balanced bins)
        bin_counts = np.bincount(bin_assignments, minlength=n_bins)
        balance_score = 1.0 - np.std(bin_counts) / np.mean(bin_counts)
        scores.append(balance_score * 0.3)
        
        # 2. Separation score (prefer well-separated bins)
        bin_means = [y[bin_assignments == i].mean() for i in range(n_bins) if (bin_assignments == i).sum() > 0]
        if len(bin_means) > 1:
            separation_score = np.std(bin_means) / np.std(y)
            scores.append(separation_score * 0.4)
        
        # 3. Homogeneity score (prefer homogeneous bins)
        homogeneity_scores = []
        for i in range(n_bins):
            if (bin_assignments == i).sum() > 1:
                bin_std = y[bin_assignments == i].std()
                homogeneity_scores.append(1.0 - bin_std / np.std(y))
        if homogeneity_scores:
            homogeneity_score = np.mean(homogeneity_scores)
            scores.append(homogeneity_score * 0.3)
        
        return sum(scores)
    
    def create_expert_pool(self) -> List[Any]:
        """Create a pool of diverse expert regressors"""
        experts = []
        
        # LightGBM variants
        experts.extend([
            ('lgb_fast', lgb.LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, verbose=-1)),
            ('lgb_deep', lgb.LGBMRegressor(n_estimators=200, max_depth=10, learning_rate=0.05, random_state=42, verbose=-1)),
            ('lgb_wide', lgb.LGBMRegressor(n_estimators=150, max_depth=4, learning_rate=0.08, num_leaves=100, random_state=42, verbose=-1)),
        ])
        
        # XGBoost variants
        experts.extend([
            ('xgb_standard', xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, verbosity=0)),
            ('xgb_conservative', xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, reg_alpha=0.1, random_state=42, verbosity=0)),
            ('xgb_aggressive', xgb.XGBRegressor(n_estimators=150, max_depth=8, learning_rate=0.12, subsample=0.8, random_state=42, verbosity=0)),
        ])
        
        # Linear models
        experts.extend([
            ('ridge_l1', Ridge(alpha=1.0)),
            ('ridge_l2', Ridge(alpha=10.0)),
            ('elastic_net', ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)),
        ])
        
        # Tree-based
        experts.extend([
            ('rf_standard', RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)),
            ('rf_deep', RandomForestRegressor(n_estimators=150, max_depth=12, min_samples_split=5, random_state=42)),
        ])
        
        return experts
    
    def select_experts_for_bins(self, X: np.ndarray, y: np.ndarray, bin_assignments: np.ndarray, 
                              n_bins: int) -> List[Any]:
        """Intelligently select best experts for each bin"""
        self.logger.info("Selecting optimal experts for each bin...")
        
        expert_pool = self.create_expert_pool()
        selected_experts = []
        
        for bin_idx in range(n_bins):
            mask = bin_assignments == bin_idx
            if mask.sum() < 5:  # Too few samples
                # Use a simple model
                selected_experts.append(('ridge_simple', Ridge(alpha=1.0)))
                continue
            
            X_bin, y_bin = X[mask], y[mask]
            
            if self.config['expert_selection_strategy'] == 'performance_based':
                best_expert = self._select_best_expert_cv(X_bin, y_bin, expert_pool)
            elif self.config['expert_selection_strategy'] == 'specialized':
                best_expert = self._select_specialized_expert(bin_idx, n_bins, expert_pool)
            else:  # random
                best_expert = expert_pool[bin_idx % len(expert_pool)]
            
            selected_experts.append(best_expert)
            self.logger.info(f"Bin {bin_idx}: Selected {best_expert[0]} ({mask.sum()} samples)")
        
        return selected_experts
    
    def _select_best_expert_cv(self, X: np.ndarray, y: np.ndarray, expert_pool: List[Any]) -> Any:
        """Select best expert using cross-validation"""
        if len(y) < 10:  # Too few samples for CV
            return expert_pool[0]
        
        best_score = np.inf
        best_expert = expert_pool[0]
        
        n_folds = min(5, len(y) // 3)  # Adaptive CV folds
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.config['random_state'])
        
        for name, expert in expert_pool[:6]:  # Test top 6 experts to save time
            scores = []
            try:
                for train_idx, val_idx in kf.split(X):
                    X_train_cv, X_val_cv = X[train_idx], X[val_idx]
                    y_train_cv, y_val_cv = y[train_idx], y[val_idx]
                    
                    expert_copy = self._clone_expert(expert)
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
    
    def _select_specialized_expert(self, bin_idx: int, n_bins: int, expert_pool: List[Any]) -> Any:
        """Select expert based on bin characteristics"""
        # Low bins: Linear models for budget segment
        if bin_idx < n_bins // 3:
            return ('ridge_l1', Ridge(alpha=1.0))
        # High bins: Complex models for luxury segment
        elif bin_idx >= 2 * n_bins // 3:
            return ('lgb_deep', lgb.LGBMRegressor(n_estimators=200, max_depth=10, learning_rate=0.05, verbose=-1))
        # Middle bins: Balanced models
        else:
            return ('xgb_standard', xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, verbosity=0))
    
    def _clone_expert(self, expert: Any) -> Any:
        """Create a copy of an expert model"""
        from sklearn.base import clone
        return clone(expert)
    
    def optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray, 
                                bin_assignments: np.ndarray, selected_experts: List[Any]) -> List[Any]:
        """Optimize hyperparameters for selected experts"""
        self.logger.info("Optimizing expert hyperparameters...")
        
        optimized_experts = []
        
        for bin_idx, (name, expert) in enumerate(selected_experts):
            mask = bin_assignments == bin_idx
            if mask.sum() < 20:  # Skip optimization for small bins
                optimized_experts.append((name, expert))
                continue
            
            X_bin, y_bin = X_train[mask], y_train[mask]
            
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
                        optimized_experts.append((name, expert))
                        return 0
                    
                    # Cross-validation
                    kf = KFold(n_splits=3, shuffle=True, random_state=42)
                    scores = []
                    for train_idx, val_idx in kf.split(X_bin):
                        model.fit(X_bin[train_idx], y_bin[train_idx])
                        y_pred = model.predict(X_bin[val_idx])
                        scores.append(mean_squared_error(y_bin[val_idx], y_pred))
                    
                    return np.mean(scores)
                    
                except Exception as e:
                    return float('inf')
            
            # Optimize with limited trials for efficiency
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=min(30, self.config['optimization_trials'] // len(selected_experts)))
            
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
            
            optimized_experts.append((f"{name}_opt", optimized_expert))
            self.logger.info(f"Bin {bin_idx} optimized: {study.best_value:.4f}")
        
        return optimized_experts
    
    def train_experts(self, X_train: np.ndarray, y_train: np.ndarray, 
                     bin_assignments: np.ndarray, selected_experts: List[Any]) -> List[Any]:
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
    
    def train_enhanced_gate(self, X_train: np.ndarray, X_val: np.ndarray, 
                           y_train: np.ndarray, y_val: np.ndarray,
                           train_preds: np.ndarray, val_preds: np.ndarray) -> nn.Module:
        """Train enhanced gating network with validation and early stopping"""
        self.logger.info("Training enhanced gating network...")
        
        n_features = X_train.shape[1]
        n_bins = train_preds.shape[1]
        
        # Enhanced MoE gate with additional layers
        class EnhancedMoEGate(nn.Module):
            def __init__(self, feature_dim: int, num_bins: int, hidden_dim: int = 64):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(feature_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim // 2, num_bins)
                )
                
            def forward(self, features: torch.Tensor, bin_outputs: torch.Tensor) -> torch.Tensor:
                logits = self.layers(features)
                weights = torch.softmax(logits, dim=1)
                return (weights * bin_outputs).sum(dim=1)
        
        gate = EnhancedMoEGate(n_features, n_bins).to(self.device)
        optimizer = torch.optim.AdamW(gate.parameters(), lr=self.config['gate_lr'], weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        criterion = nn.MSELoss()
        
        # Convert to tensors
        X_train_t = torch.from_numpy(X_train).float().to(self.device)
        X_val_t = torch.from_numpy(X_val).float().to(self.device)
        y_train_t = torch.from_numpy(y_train).float().to(self.device)
        y_val_t = torch.from_numpy(y_val).float().to(self.device)
        train_preds_t = torch.from_numpy(train_preds).float().to(self.device)
        val_preds_t = torch.from_numpy(val_preds).float().to(self.device)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        for epoch in range(self.config['gate_epochs']):
            # Training
            gate.train()
            optimizer.zero_grad()
            train_out = gate(X_train_t, train_preds_t)
            train_loss = criterion(train_out, y_train_t)
            train_loss.backward()
            optimizer.step()
            
            # Validation
            gate.eval()
            with torch.no_grad():
                val_out = gate(X_val_t, val_preds_t)
                val_loss = criterion(val_out, y_val_t)
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = gate.state_dict().copy()
            else:
                patience_counter += 1
            
            if epoch % 20 == 0:
                self.logger.info(f"Epoch {epoch}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")
            
            if patience_counter >= self.config['early_stopping_patience']:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Load best state
        if best_state:
            gate.load_state_dict(best_state)
        
        return gate
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      bin_assignments: np.ndarray = None) -> Dict[str, Any]:
        """Comprehensive model evaluation"""
        metrics = {}
        
        # Overall metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        
        metrics['overall'] = {
            'MAE': float(mae),
            'RMSE': float(rmse),
            'R2': float(r2),
            'MAPE': float(mape),
            'samples': len(y_true)
        }
        
        # Per-bin metrics
        if bin_assignments is not None:
            metrics['per_bin'] = {}
            n_bins = len(np.unique(bin_assignments))
            
            for bin_idx in range(n_bins):
                mask = bin_assignments == bin_idx
                if mask.sum() > 0:
                    y_t, y_p = y_true[mask], y_pred[mask]
                    bin_mae = mean_absolute_error(y_t, y_p)
                    bin_rmse = np.sqrt(mean_squared_error(y_t, y_p))
                    bin_r2 = r2_score(y_t, y_p) if len(y_t) > 1 else 0
                    bin_mape = np.mean(np.abs((y_t - y_p) / (y_t + 1e-8))) * 100
                    
                    metrics['per_bin'][bin_idx] = {
                        'MAE': float(bin_mae),
                        'RMSE': float(bin_rmse),
                        'R2': float(bin_r2),
                        'MAPE': float(bin_mape),
                        'samples': int(mask.sum()),
                        'target_range': [float(y_t.min()), float(y_t.max())]
                    }
        
        return metrics
    
    def save_model(self, model_dir: str = 'models_enhanced'):
        """Save all model components"""
        os.makedirs(model_dir, exist_ok=True)
        
        # Save scalers
        if self.scalers:
            import joblib
            joblib.dump(self.scalers, os.path.join(model_dir, 'scalers.pkl'))
        
        # Save bin layer
        if self.bin_layer:
            torch.save(self.bin_layer.state_dict(), os.path.join(model_dir, 'bin_layer.pt'))
        
        # Save experts
        import joblib
        for i, expert in enumerate(self.experts):
            joblib.dump(expert, os.path.join(model_dir, f'expert_{i}.pkl'))
        
        # Save gate
        if self.gate:
            torch.save(self.gate.state_dict(), os.path.join(model_dir, 'gate.pt'))
        
        # Save configuration
        with open(os.path.join(model_dir, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=2)
        
        self.logger.info(f"Model saved to {model_dir}")
    
    def train(self) -> Dict[str, Any]:
        """Main training pipeline"""
        self.logger.info("Starting enhanced MoE training pipeline...")
        
        # 1. Load and preprocess data
        X, y = self.load_and_preprocess_data()
        
        # 2. Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=self.config['test_size'], 
            random_state=self.config['random_state'], stratify=None
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=self.config['val_size'] / (1 - self.config['test_size']),
            random_state=self.config['random_state']
        )
        
        self.logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # 3. Determine optimal binning
        n_bins, boundaries = self.determine_optimal_bins(y_train)
        
        # 4. Create bin assignments
        train_bins = np.digitize(y_train, boundaries)
        val_bins = np.digitize(y_val, boundaries)
        test_bins = np.digitize(y_test, boundaries)
        
        # 5. Create learnable binning layer
        edges_tensor = torch.from_numpy(boundaries).float()
        self.bin_layer = LearnableBinning(num_bins=n_bins, init_edges=edges_tensor)
        
        # 6. Select and optimize experts
        selected_experts = self.select_experts_for_bins(X_train, y_train, train_bins, n_bins)
        optimized_experts = self.optimize_hyperparameters(X_train, y_train, train_bins, selected_experts)
        
        # 7. Train experts
        self.experts = self.train_experts(X_train, y_train, train_bins, optimized_experts)
        
        # 8. Generate expert predictions
        train_preds = np.column_stack([expert.predict(X_train) for expert in self.experts])
        val_preds = np.column_stack([expert.predict(X_val) for expert in self.experts])
        test_preds = np.column_stack([expert.predict(X_test) for expert in self.experts])
        
        # 9. Train enhanced gating network
        self.gate = self.train_enhanced_gate(X_train, X_val, y_train, y_val, train_preds, val_preds)
        
        # 10. Generate final predictions
        self.gate.eval()
        with torch.no_grad():
            final_test_preds = self.gate(
                torch.from_numpy(X_test).float().to(self.device),
                torch.from_numpy(test_preds).float().to(self.device)
            ).cpu().numpy()
        
        # 11. Evaluate model
        metrics = self.evaluate_model(y_test, final_test_preds, test_bins)
        
        # 12. Feature importance analysis
        self.analyze_feature_importance(X_train, y_train)
        
        # 13. Save model and results
        self.save_model()
        self.save_metrics(metrics)
        
        # 14. Generate detailed report
        self.generate_report(metrics, n_bins, boundaries)
        
        return metrics
    
    def analyze_feature_importance(self, X: np.ndarray, y: np.ndarray):
        """Analyze feature importance across experts"""
        self.logger.info("Analyzing feature importance...")
        
        importance_dict = {}
        
        for i, expert in enumerate(self.experts):
            try:
                if hasattr(expert, 'feature_importances_'):
                    importance_dict[f'expert_{i}'] = expert.feature_importances_
                elif hasattr(expert, 'coef_'):
                    importance_dict[f'expert_{i}'] = np.abs(expert.coef_)
            except:
                continue
        
        if importance_dict:
            # Average importance across experts
            all_importances = list(importance_dict.values())
            avg_importance = np.mean(all_importances, axis=0)
            
            # Store feature importance
            self.feature_importance = {
                'average': avg_importance.tolist(),
                'per_expert': {k: v.tolist() for k, v in importance_dict.items()}
            }
            
            # Log top features
            top_indices = np.argsort(avg_importance)[-10:][::-1]
            self.logger.info("Top 10 most important features:")
            for i, idx in enumerate(top_indices):
                self.logger.info(f"  {i+1}. Feature {idx}: {avg_importance[idx]:.4f}")
    
    def save_metrics(self, metrics: Dict[str, Any]):
        """Save detailed metrics"""
        os.makedirs('models_enhanced', exist_ok=True)
        
        # Add feature importance to metrics
        if self.feature_importance:
            metrics['feature_importance'] = self.feature_importance
        
        with open('models_enhanced/detailed_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        self.logger.info("Metrics saved to models_enhanced/detailed_metrics.json")
    
    def generate_report(self, metrics: Dict[str, Any], n_bins: int, boundaries: np.ndarray):
        """Generate comprehensive training report"""
        self.logger.info("=" * 60)
        self.logger.info("ENHANCED MOE TRAINING REPORT")
        self.logger.info("=" * 60)
        
        # Overall performance
        overall = metrics['overall']
        self.logger.info(f"\nOVERALL PERFORMANCE:")
        self.logger.info(f"  MAE:     {overall['MAE']:.4f}")
        self.logger.info(f"  RMSE:    {overall['RMSE']:.4f}")
        self.logger.info(f"  R²:      {overall['R2']:.4f}")
        self.logger.info(f"  MAPE:    {overall['MAPE']:.2f}%")
        self.logger.info(f"  Samples: {overall['samples']}")
        
        # Binning information
        self.logger.info(f"\nBINNING STRATEGY:")
        self.logger.info(f"  Number of bins: {n_bins}")
        self.logger.info(f"  Boundaries: {boundaries}")
        self.logger.info(f"  Strategy: {self.config['binning_strategy']}")
        
        # Per-bin performance
        if 'per_bin' in metrics:
            self.logger.info(f"\nPER-BIN PERFORMANCE:")
            for bin_idx, bin_metrics in metrics['per_bin'].items():
                self.logger.info(f"  Bin {bin_idx}:")
                self.logger.info(f"    Samples: {bin_metrics['samples']}")
                self.logger.info(f"    Range: [{bin_metrics['target_range'][0]:.2f}, {bin_metrics['target_range'][1]:.2f}]")
                self.logger.info(f"    MAE: {bin_metrics['MAE']:.4f}")
                self.logger.info(f"    RMSE: {bin_metrics['RMSE']:.4f}")
                self.logger.info(f"    R²: {bin_metrics['R2']:.4f}")
                self.logger.info(f"    MAPE: {bin_metrics['MAPE']:.2f}%")
        
        # Configuration summary
        self.logger.info(f"\nCONFIGURATION SUMMARY:")
        key_configs = ['min_bin_size', 'max_bins', 'binning_strategy', 'expert_selection_strategy', 'scaling_method']
        for key in key_configs:
            if key in self.config:
                self.logger.info(f"  {key}: {self.config[key]}")
        
        self.logger.info("=" * 60)


def main():
    """Main execution function"""
    # Custom configuration for your dataset
    config = {
        'min_bin_size': 80,  # Adjusted for 5.5k dataset
        'max_bins': 6,
        'min_bins': 4,
        'test_size': 0.2,
        'val_size': 0.15,
        'gate_epochs': 300,
        'gate_lr': 1e-3,
        'early_stopping_patience': 25,
        'optimization_trials': 150,
        'scaling_method': 'robust',
        'binning_strategy': 'adaptive',
        'expert_selection_strategy': 'performance_based',
    }
    
    # Initialize and train
    trainer = EnhancedMoETrainer(config)
    results = trainer.train()
    
    return results


if __name__ == '__main__':
    main()