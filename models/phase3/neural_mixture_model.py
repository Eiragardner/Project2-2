# models/phase3/neural_mixture_model.py
import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import torch
import lightgbm as lgb
import xgboost as xgb

sys.path.append(str(Path(__file__).parent / 'gates'))
from neural_gate_fixed_proper import FixedNeuralGateTrainer


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
                expert = creator_func(X_train, y_train, param)
                
                val_pred = expert.predict(X_val)
                val_r2 = r2_score(y_val, val_pred)
                val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
                
                experts.append(expert)
                expert_names.append(name)
                expert_performances.append(max(val_r2, 0.01))
                
                print(f"{name}: R² = {val_r2:.4f}, RMSE = ${val_rmse:,.0f}")
                
            except Exception as e:
                print(f"{name}: Failed ({e}), using Ridge fallback")
                expert = Ridge(alpha=10.0, random_state=self.random_state)
                expert.fit(X_train, y_train)
                val_pred = expert.predict(X_val)
                val_r2 = r2_score(y_val, val_pred)
                
                experts.append(expert)
                expert_names.append(f"{name}_Fallback")
                expert_performances.append(max(val_r2, 0.01))
                
                print(f"{name}_Fallback: R² = {val_r2:.4f}")
        
        performance_indices = np.argsort(expert_performances)[::-1]
        top_k = min(6, len(experts))
        
        final_experts = [experts[i] for i in performance_indices[:top_k]]
        final_names = [expert_names[i] for i in performance_indices[:top_k]]
        final_performances = [expert_performances[i] for i in performance_indices[:top_k]]
        
        print(f"\nSelected top {top_k} experts:")
        for i, (name, perf) in enumerate(zip(final_names, final_performances)):
            print(f"{i}: {name} (R² = {perf:.4f})")
        
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
        
        ensemble_pred = self._predict_ensemble(X_test)
        ensemble_r2 = r2_score(y_test, ensemble_pred)
        ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
        
        print(f"\nWeighted Ensemble: R² = {ensemble_r2:.4f}, RMSE = ${ensemble_rmse:,.0f}")
        
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
        
        neural_pred = self.gate_trainer.predict(X_test)
        neural_r2 = r2_score(y_test, neural_pred)
        neural_rmse = np.sqrt(mean_squared_error(y_test, neural_pred))
        
        print("\nFINAL RESULTS:")
        print(f"Weighted Ensemble: R² = {ensemble_r2:.4f}, RMSE = ${ensemble_rmse:,.0f}")
        print(f"Neural Mixture:    R² = {neural_r2:.4f}, RMSE = ${neural_rmse:,.0f}")
        
        improvement = neural_r2 - ensemble_r2
        print(f"Improvement: {improvement:+.4f} R² points")
        
        if improvement > 0:
            print("Neural gate outperforms weighted ensemble")
        
        weight_analysis = self.gate_trainer.analyze_weights(X_test[:1000])
        
        return {
            'ensemble_r2': ensemble_r2,
            'neural_r2': neural_r2,
            'ensemble_rmse': ensemble_rmse,
            'neural_rmse': neural_rmse,
            'improvement': improvement
        }
    
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
        
        print("\nTraining Complete!")
        
        improvement = results['improvement']
        neural_r2 = results['neural_r2']
        
        print(f"\nPerformance Summary:")
        print(f"Neural R²: {neural_r2:.4f}")
        print(f"Improvement: {improvement:+.4f} R² points")
        
        if neural_r2 >= 0.85:
            print("Performance: Good")
        elif neural_r2 >= 0.75:
            print("Performance: Acceptable")
        else:
            print("Performance: Needs improvement")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()