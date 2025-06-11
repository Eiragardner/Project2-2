# models/phase3/neural_mixture_model.py
import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import torch
import lightgbm as lgb
import xgboost as xgb

sys.path.append(str(Path(__file__).parent / 'gates'))
try:
    from neural_gate_fixed_proper import FixedNeuralGateTrainer
except ImportError:
    print("Warning: FixedNeuralGateTrainer not found. Using dummy implementation.")
    class FixedNeuralGateTrainer:
        def __init__(self, *args, **kwargs):
            pass
        def train(self, *args, **kwargs):
            return 0.0
        def predict(self, X):
            return np.zeros(len(X))
        def analyze_weights(self, X):
            return {}

# Try to import evaluation modules
try:
    sys.path.append(str(Path(__file__).resolve().parent))
    from evaluation.evaluator import ModelEvaluator
except ImportError:
    ModelEvaluator = None
    
try:
    from core.logger import MoELogger
except ImportError:
    MoELogger = None


class SimpleLogger:
    """Simple logger implementation if MoELogger is not available"""
    def info(self, message):
        print(message)
    
    def warning(self, message):
        print(f"WARNING: {message}")
    
    def error(self, message):
        print(f"ERROR: {message}")


class FallbackEvaluator:
    """Fallback evaluator if ModelEvaluator is not available"""
    def evaluate(self, y_true, y_pred):
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        # Calculate additional metrics
        errors = np.abs(y_true - y_pred)
        percentage_errors = (errors / np.abs(y_true)) * 100
        mape = np.mean(percentage_errors)
        median_ape = np.median(percentage_errors)
        
        # Calculate accuracy brackets
        within_5_percent = np.mean(percentage_errors <= 5) * 100
        within_10_percent = np.mean(percentage_errors <= 10) * 100
        within_15_percent = np.mean(percentage_errors <= 15) * 100
        
        # Calculate bias (MPE)
        signed_errors = ((y_pred - y_true) / np.abs(y_true)) * 100
        mpe = np.mean(signed_errors)
        
        # Calculate other metrics
        max_error = np.max(errors)
        error_95th_percentile = np.percentile(errors, 95)
        
        overall_metrics = {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'median_ape': median_ape,
            'within_5_percent': within_5_percent,
            'within_10_percent': within_10_percent,
            'within_15_percent': within_15_percent,
            'mpe': mpe,
            'max_error': max_error,
            'error_95th_percentile': error_95th_percentile
        }
        
        return {
            'overall': overall_metrics
        }


def get_evaluator():
    """Get evaluator with proper logger"""
    if ModelEvaluator is not None:
        try:
            if MoELogger is not None:
                logger = MoELogger()
            else:
                logger = SimpleLogger()
            return ModelEvaluator(logger)
        except:
            pass
    
    return FallbackEvaluator()


class CaliforniaProcessor:
    def __init__(self):
        self.is_california = False
        self.scale_factor = 1.0
    
    def process(self, X, y, data_path):
        self.is_california = self._is_california(X, y, data_path)
        
        if self.is_california:
            print("California dataset detected: Converting to actual prices (*100,000)")
            self.scale_factor = 100000.0
            y_scaled = y * self.scale_factor
            print(f"Scaled range: ${y_scaled.min():,.0f} - ${y_scaled.max():,.0f}")
            return X, y_scaled
        
        return X, y
    
    def _is_california(self, X, y, path):
        return (
            'california' in path.lower() or
            (19000 <= len(X) <= 22000 and X.shape[1] >= 8 and 0.1 <= y.min() <= 0.2 and 4.5 <= y.max() <= 5.5)
        )


class ExpertPool:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scalers = {}
    
    def create_experts(self, X_train, y_train, X_val, y_val):
        print("Creating expert pool...")
        evaluator = get_evaluator()
        
        experts = []
        expert_names = []
        expert_performances = []
        
        expert_configs = [
            ('LowPrice_XGB', self._create_range_expert, 'low'),
            ('MidPrice_LGB', self._create_range_expert, 'mid'),
            ('HighPrice_RF', self._create_range_expert, 'high'),
            ('Conservative_Ridge', self._create_conservative_expert, None),
            ('Aggressive_XGB', self._create_aggressive_expert, None),
            ('Balanced_LGB', self._create_balanced_expert, None),
            ('Deep_RF', self._create_deep_rf_expert, None),
            ('Linear_Expert', self._create_linear_expert, None),
            ('Robust_Expert', self._create_robust_expert, None),
        ]
        
        print(f"Training {len(expert_configs)} experts...")
        
        for name, creator_func, param in expert_configs:
            try:
                print(f"\nTraining {name}...")
                expert = creator_func(X_train, y_train, param)
                
                val_pred = expert.predict(X_val)
                val_metrics = evaluator.evaluate(y_val, val_pred)
                
                experts.append(expert)
                expert_names.append(name)
                expert_performances.append(max(val_metrics['overall']['r2'], 0.01))
                
                # Quick summary for expert selection
                overall = val_metrics['overall']
                print(f"{name}: R²={overall['r2']:.4f}, RMSE=${overall['rmse']:,.0f}, "
                      f"MAPE={overall['mape']:.1f}%, Within 10%={overall.get('within_10_percent', 0):.1f}%")
                
            except Exception as e:
                print(f"{name}: Failed ({e}), using Ridge fallback")
                expert = Ridge(alpha=10.0, random_state=self.random_state)
                expert.fit(X_train, y_train)
                val_pred = expert.predict(X_val)
                val_metrics = evaluator.evaluate(y_val, val_pred)
                
                experts.append(expert)
                expert_names.append(f"{name}_Fallback")
                expert_performances.append(max(val_metrics['overall']['r2'], 0.01))
                
                overall = val_metrics['overall']
                print(f"{name}_Fallback: R²={overall['r2']:.4f}, RMSE=${overall['rmse']:,.0f}")
        
        performance_indices = np.argsort(expert_performances)[::-1]
        top_k = min(6, len(experts))
        
        final_experts = [experts[i] for i in performance_indices[:top_k]]
        final_names = [expert_names[i] for i in performance_indices[:top_k]]
        final_performances = [expert_performances[i] for i in performance_indices[:top_k]]
        
        print(f"\nSelected top {top_k} experts:")
        for i, (name, perf) in enumerate(zip(final_names, final_performances)):
            print(f"{i+1}: {name} (R² = {perf:.4f})")
        
        return final_experts, final_names, final_performances
    
    def _create_range_expert(self, X_train, y_train, price_range):
        percentiles = {
            'low': (0, 33),
            'mid': (25, 75), 
            'high': (67, 100)
        }
        
        low_p, high_p = percentiles[price_range]
        low_val = np.percentile(y_train, low_p)
        high_val = np.percentile(y_train, high_p)
        
        mask = (y_train >= low_val) & (y_train <= high_val)
        
        if price_range == 'low':
            model = xgb.XGBRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.08,
                subsample=0.8, colsample_bytree=0.8,
                random_state=self.random_state, verbosity=0
            )
        elif price_range == 'mid':
            model = lgb.LGBMRegressor(
                n_estimators=250, max_depth=8, learning_rate=0.06,
                subsample=0.85, colsample_bytree=0.9,
                random_state=self.random_state, verbose=-1
            )
        else:
            model = RandomForestRegressor(
                n_estimators=300, max_depth=12, min_samples_split=3,
                min_samples_leaf=2, random_state=self.random_state
            )
        
        model.fit(X_train, y_train)
        return model
    
    def _create_conservative_expert(self, X_train, y_train, _):
        return Ridge(alpha=50.0, random_state=self.random_state)
    
    def _create_aggressive_expert(self, X_train, y_train, _):
        return xgb.XGBRegressor(
            n_estimators=400, max_depth=10, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.8,
            gamma=0.1, min_child_weight=1,
            random_state=self.random_state, verbosity=0
        )
    
    def _create_balanced_expert(self, X_train, y_train, _):
        return lgb.LGBMRegressor(
            n_estimators=200, max_depth=8, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8,
            min_child_samples=20, random_state=self.random_state, verbose=-1
        )
    
    def _create_deep_rf_expert(self, X_train, y_train, _):
        return RandomForestRegressor(
            n_estimators=250, max_depth=20, min_samples_split=5,
            min_samples_leaf=2, max_features='sqrt',
            random_state=self.random_state
        )
    
    def _create_linear_expert(self, X_train, y_train, _):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        self.scalers['linear'] = scaler
        
        model = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=self.random_state)
        model.fit(X_scaled, y_train)
        
        class ScaledLinearExpert:
            def __init__(self, model, scaler):
                self.model = model
                self.scaler = scaler
            
            def predict(self, X):
                X_scaled = self.scaler.transform(X)
                return self.model.predict(X_scaled)
        
        return ScaledLinearExpert(model, scaler)
    
    def _create_robust_expert(self, X_train, y_train, _):
        return GradientBoostingRegressor(
            n_estimators=150, max_depth=6, learning_rate=0.05,
            subsample=0.8, random_state=self.random_state
        )


class NeuralMixtureModel:
    def __init__(self):
        self.processor = CaliforniaProcessor()
        self.expert_pool = ExpertPool()
        self.gate_trainer = None
        self.experts = None
        self.expert_names = None
        self.expert_performances = None
        
    def train(self, data_path):
        print("Neural Mixture Model Training")
        print(f"Dataset: {data_path}")
        print("="*60)
        
        X, y = self._load_data(data_path)
        X, y = self.processor.process(X, y, data_path)
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=self._create_strata(y)
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.2, random_state=42, stratify=self._create_strata(y_temp)
        )
        
        print(f"Data: {len(X):,} samples, {X.shape[1]} features")
        print(f"Split: {len(X_train):,} train, {len(X_val):,} val, {len(X_test):,} test")
        
        self.experts, self.expert_names, self.expert_performances = self.expert_pool.create_experts(
            X_train, y_train, X_val, y_val
        )
        
        # Calculate ensemble baseline
        ensemble_pred = self._predict_ensemble(X_test)
        evaluator = get_evaluator()
        
        print("\n" + "="*60)
        print("BASELINE PERFORMANCE (Weighted Ensemble):")
        ensemble_metrics = evaluator.evaluate(y_test, ensemble_pred)
        self._print_metrics(ensemble_metrics['overall'], "Weighted Ensemble")
        
        # Train neural gate
        print("\nTraining neural gate...")
        self.gate_trainer = FixedNeuralGateTrainer(
            n_features=X.shape[1],
            experts=self.experts,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        val_loss = self.gate_trainer.train(
            X_train, y_train, X_val, y_val, 
            epochs=300, lr=1e-3
        )
        
        # Calculate neural mixture performance
        neural_pred = self.gate_trainer.predict(X_test)
        
        print("\n" + "="*60)
        print("NEURAL MIXTURE PERFORMANCE:")
        neural_metrics = evaluator.evaluate(y_test, neural_pred)
        self._print_metrics(neural_metrics['overall'], "Neural Mixture")
        
        # Calculate improvements
        improvements = self._calculate_improvements_comprehensive(
            ensemble_metrics['overall'], neural_metrics['overall']
        )
        
        print("\n" + "="*60)
        print("PERFORMANCE COMPARISON & IMPROVEMENTS:")
        self._print_comprehensive_comparison(
            ensemble_metrics['overall'], 
            neural_metrics['overall'], 
            improvements
        )
        
        # Expected improvement analysis
        print("\n" + "="*60)
        print("EXPECTED IMPROVEMENT ANALYSIS:")
        self._analyze_expected_improvement_comprehensive(improvements, neural_metrics['overall'])
        
        # Weight analysis
        try:
            weight_analysis = self.gate_trainer.analyze_weights(X_test[:1000])
        except:
            weight_analysis = {}
        
        # Print chosen models summary
        print("\n" + "="*60)
        print("CHOSEN EXPERT MODELS:")
        for i, (name, expert, perf) in enumerate(zip(self.expert_names, self.experts, self.expert_performances)):
            model_type = type(expert).__name__
            if hasattr(expert, 'model'):
                model_type = type(expert.model).__name__
            print(f"{i+1}. {name}")
            print(f"   Type: {model_type}")
            print(f"   Validation R²: {perf:.4f}")
            if hasattr(expert, 'n_estimators'):
                print(f"   Trees: {expert.n_estimators}")
            elif hasattr(expert, 'model') and hasattr(expert.model, 'n_estimators'):
                print(f"   Trees: {expert.model.n_estimators}")
            print()
        
        return {
            'ensemble_metrics': ensemble_metrics,
            'neural_metrics': neural_metrics,
            'improvements': improvements,
            'weight_analysis': weight_analysis,
            'expert_names': self.expert_names,
            'expert_performances': self.expert_performances
        }
    
    def _print_metrics(self, metrics, model_name):
        """Print formatted metrics"""
        print(f"{model_name} Results:")
        print(f"  R² = {metrics['r2']:.4f}")
        print(f"  RMSE = ${metrics['rmse']:,.0f}")
        print(f"  MAE = ${metrics['mae']:,.0f}")
        print(f"  MAPE = {metrics['mape']:.1f}%")
        print(f"  Within 5% = {metrics.get('within_5_percent', 0):.1f}%")
        print(f"  Within 10% = {metrics.get('within_10_percent', 0):.1f}%")
        print(f"  Within 15% = {metrics.get('within_15_percent', 0):.1f}%")
        if 'mpe' in metrics:
            bias_direction = 'over' if metrics['mpe'] > 0 else 'under'
            print(f"  MPE (Bias) = {metrics['mpe']:+.1f}% ({bias_direction}prediction)")
    
    def _calculate_improvements_comprehensive(self, baseline_metrics, neural_metrics):
        """Calculate comprehensive improvement metrics using evaluator results"""
        improvements = {}
        
        # Absolute improvements
        improvements['r2_improvement'] = neural_metrics['r2'] - baseline_metrics['r2']
        improvements['rmse_improvement'] = baseline_metrics['rmse'] - neural_metrics['rmse']
        improvements['mae_improvement'] = baseline_metrics['mae'] - neural_metrics['mae']
        improvements['mape_improvement'] = baseline_metrics['mape'] - neural_metrics['mape']
        improvements['mpe_improvement'] = baseline_metrics.get('mpe', 0) - neural_metrics.get('mpe', 0)
        
        # Percentage improvements
        improvements['rmse_pct_improvement'] = (improvements['rmse_improvement'] / baseline_metrics['rmse']) * 100
        improvements['mae_pct_improvement'] = (improvements['mae_improvement'] / baseline_metrics['mae']) * 100
        improvements['mape_pct_improvement'] = (improvements['mape_improvement'] / baseline_metrics['mape']) * 100
        
        # Accuracy bracket improvements
        for bracket in ['within_5_percent', 'within_10_percent', 'within_15_percent']:
            baseline_val = baseline_metrics.get(bracket, 0)
            neural_val = neural_metrics.get(bracket, 0)
            improvements[f'{bracket}_improvement'] = neural_val - baseline_val
        
        # Bias improvements
        improvements['bias_improvement'] = abs(baseline_metrics.get('mpe', 0)) - abs(neural_metrics.get('mpe', 0))
        
        return improvements
    
    def _print_comprehensive_comparison(self, baseline_metrics, neural_metrics, improvements):
        """Print comprehensive comparison using evaluator metrics"""
        print("WEIGHTED ENSEMBLE vs NEURAL MIXTURE:")
        print()
        
        # Core metrics comparison
        print("Core Metrics:")
        print(f"                    Ensemble    Neural      Improvement")
        print(f"  R²              {baseline_metrics['r2']:8.4f}  {neural_metrics['r2']:8.4f}  {improvements['r2_improvement']:+8.4f}")
        print(f"  RMSE           ${baseline_metrics['rmse']:8,.0f} ${neural_metrics['rmse']:8,.0f} ${improvements['rmse_improvement']:+8,.0f} ({improvements['rmse_pct_improvement']:+.1f}%)")
        print(f"  MAE            ${baseline_metrics['mae']:8,.0f} ${neural_metrics['mae']:8,.0f} ${improvements['mae_improvement']:+8,.0f} ({improvements['mae_pct_improvement']:+.1f}%)")
        print(f"  MAPE           {baseline_metrics['mape']:8.1f}% {neural_metrics['mape']:8.1f}% {improvements['mape_improvement']:+8.1f}% ({improvements['mape_pct_improvement']:+.1f}%)")
        
        # Bias analysis
        baseline_mpe = baseline_metrics.get('mpe', 0)
        neural_mpe = neural_metrics.get('mpe', 0)
        print(f"  MPE (Bias)     {baseline_mpe:+8.1f}% {neural_mpe:+8.1f}% {improvements['mpe_improvement']:+8.1f}%")
        
        # Accuracy brackets
        print()
        print("Accuracy Brackets:")
        for bracket, label in [('within_5_percent', 'Within 5%'), ('within_10_percent', 'Within 10%'), ('within_15_percent', 'Within 15%')]:
            baseline_val = baseline_metrics.get(bracket, 0)
            neural_val = neural_metrics.get(bracket, 0)
            improvement = improvements.get(f'{bracket}_improvement', 0)
            print(f"  {label:12} {baseline_val:8.1f}% {neural_val:8.1f}% {improvement:+8.1f}%")
        
        # Additional insights
        print()
        print("Additional Metrics:")
        print(f"  Median APE     {baseline_metrics.get('median_ape', 0):8.1f}% {neural_metrics.get('median_ape', 0):8.1f}%")
        print(f"  Max Error     ${baseline_metrics.get('max_error', 0):8,.0f} ${neural_metrics.get('max_error', 0):8,.0f}")
        print(f"  95th Pct Err  ${baseline_metrics.get('error_95th_percentile', 0):8,.0f} ${neural_metrics.get('error_95th_percentile', 0):8,.0f}")
    
    def _analyze_expected_improvement_comprehensive(self, improvements, neural_metrics):
        """Comprehensive expected improvement analysis"""
        r2_improvement = improvements['r2_improvement']
        rmse_pct_improvement = improvements['rmse_pct_improvement']
        mae_pct_improvement = improvements['mae_pct_improvement']
        within_10_improvement = improvements.get('within_10_percent_improvement', 0)
        bias_improvement = improvements.get('bias_improvement', 0)
        
        print("Neural Gate vs Weighted Ensemble Analysis:")
        print()
        
        # R² improvement analysis
        if r2_improvement >= 0.05:
            r2_status = "EXCELLENT"
        elif r2_improvement >= 0.02:
            r2_status = "GOOD"
        elif r2_improvement >= 0.01:
            r2_status = "MODERATE"
        elif r2_improvement > 0:
            r2_status = "MINIMAL"
        else:
            r2_status = "NO IMPROVEMENT"
        
        print(f"R² Improvement: {r2_status} ({r2_improvement:+.4f})")
        
        # Error reduction analysis
        avg_error_reduction = (rmse_pct_improvement + mae_pct_improvement) / 2
        
        if avg_error_reduction >= 5.0:
            error_status = "SIGNIFICANT"
        elif avg_error_reduction >= 2.0:
            error_status = "MODERATE"
        elif avg_error_reduction >= 1.0:
            error_status = "SMALL"
        elif avg_error_reduction > 0:
            error_status = "MINIMAL"
        else:
            error_status = "NO REDUCTION"
        
        print(f"Error Reduction: {error_status} ({avg_error_reduction:+.2f}% average)")
        
        # Accuracy improvement
        if within_10_improvement >= 5.0:
            accuracy_status = "SIGNIFICANT"
        elif within_10_improvement >= 2.0:
            accuracy_status = "MODERATE"
        elif within_10_improvement >= 1.0:
            accuracy_status = "SMALL"
        elif within_10_improvement > 0:
            accuracy_status = "MINIMAL"
        else:
            accuracy_status = "NO IMPROVEMENT"
        
        print(f"Accuracy Improvement (10% bracket): {accuracy_status} ({within_10_improvement:+.1f}%)")
        
        # Overall assessment
        print()
        print("EXPECTED IMPROVEMENT OVER WEIGHTED ENSEMBLE:")
        
        # Multi-dimensional assessment
        improvement_score = 0
        if r2_improvement >= 0.02: improvement_score += 3
        elif r2_improvement >= 0.01: improvement_score += 2
        elif r2_improvement > 0: improvement_score += 1
        
        if avg_error_reduction >= 2.0: improvement_score += 2
        elif avg_error_reduction >= 1.0: improvement_score += 1
        
        if within_10_improvement >= 2.0: improvement_score += 2
        elif within_10_improvement >= 1.0: improvement_score += 1
        
        if bias_improvement >= 1.0: improvement_score += 1
        
        if improvement_score >= 6:
            assessment = "SUBSTANTIAL - Neural gate provides significant multi-dimensional improvement"
        elif improvement_score >= 4:
            assessment = "MODERATE - Neural gate provides meaningful improvement across multiple metrics"
        elif improvement_score >= 2:
            assessment = "MARGINAL - Neural gate provides some improvement in key areas"
        else:
            assessment = "MINIMAL - Weighted ensemble remains competitive"
        
        print(f"{assessment}")
        print(f"Improvement Score: {improvement_score}/8")
        
        # Performance tier
        if neural_metrics['r2'] >= 0.90:
            tier = "EXCELLENT (R² ≥ 0.90)"
        elif neural_metrics['r2'] >= 0.85:
            tier = "VERY GOOD (R² ≥ 0.85)"
        elif neural_metrics['r2'] >= 0.80:
            tier = "GOOD (R² ≥ 0.80)"
        elif neural_metrics['r2'] >= 0.75:
            tier = "ACCEPTABLE (R² ≥ 0.75)"
        else:
            tier = "NEEDS IMPROVEMENT (R² < 0.75)"
        
        print(f"Overall Performance Tier: {tier}")
        
        # Key insights
        within_10_pct = neural_metrics.get('within_10_percent', 0)
        if within_10_pct >= 80:
            print(f"✓ High Accuracy: {within_10_pct:.1f}% of predictions within 10%")
        elif within_10_pct >= 60:
            print(f"• Moderate Accuracy: {within_10_pct:.1f}% of predictions within 10%")
        else:
            print(f"⚠ Low Accuracy: {within_10_pct:.1f}% of predictions within 10%")
    
    def _create_strata(self, y, n_bins=10):
        return pd.qcut(y, q=n_bins, labels=False, duplicates='drop')
    
    def _predict_ensemble(self, X_test):
        predictions = np.zeros((len(X_test), len(self.experts)))
        
        for i, expert in enumerate(self.experts):
            predictions[:, i] = expert.predict(X_test)
        
        weights = np.array(self.expert_performances)
        weights = weights / weights.sum()
        
        return np.average(predictions, axis=1, weights=weights)
    
    def _load_data(self, data_path):
        print(f"Loading: {data_path}")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"File not found: {data_path}")
        
        df = pd.read_csv(data_path)
        
        numeric_cols = []
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                numeric_cols.append(col)
            except:
                continue
        
        df = df[numeric_cols]
        
        if 'median_house_value' in df.columns:
            target_col = 'median_house_value'
        elif 'price' in df.columns.str.lower():
            target_col = [c for c in df.columns if 'price' in c.lower()][0]
        elif 'value' in df.columns.str.lower():
            target_col = [c for c in df.columns if 'value' in c.lower()][0]
        else:
            variances = df.var()
            target_col = variances.idxmax()
        
        feature_cols = [c for c in df.columns if c != target_col]
        
        X = df[feature_cols].values
        y = df[target_col].values
        
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X, y = X[mask], y[mask]
        
        print(f"Features: {len(feature_cols)}, Target: {target_col}")
        print(f"Shape: {X.shape}, Target range: ${y.min():.2f} - ${y.max():.2f}")
        print(f"Cleaned {(~mask).sum():,} rows with missing values")
        
        return X, y


def main():
    if len(sys.argv) != 2:
        print("Usage: python -m models.phase3.neural_mixture_model 'California Dataset.csv'")
        return
    
    data_file = sys.argv[1]
    
    try:
        model = NeuralMixtureModel()
        results = model.train(data_file)
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        
        neural_metrics = results['neural_metrics']['overall']
        improvements = results['improvements']
        
        print(f"Final Neural Mixture Performance:")
        print(f"  R² = {neural_metrics['r2']:.4f}")
        print(f"  RMSE = ${neural_metrics['rmse']:,.0f}")
        print(f"  MAE = ${neural_metrics['mae']:,.0f}")
        print(f"  MAPE = {neural_metrics['mape']:.1f}%")
        
        print(f"\nKey Improvements over Weighted Ensemble:")
        print(f"  R² gained: {improvements['r2_improvement']:+.4f} points")
        print(f"  RMSE reduced: ${improvements['rmse_improvement']:+,.0f} ({improvements['rmse_pct_improvement']:+.2f}%)")
        print(f"  MAE reduced: ${improvements['mae_improvement']:+,.0f} ({improvements['mae_pct_improvement']:+.2f}%)")
        
        print(f"\nFinal Expert Selection:")
        for i, name in enumerate(results['expert_names']):
            print(f"  {i+1}. {name} (R² = {results['expert_performances'][i]:.4f})")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()