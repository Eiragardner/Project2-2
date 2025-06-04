# models/phase3/fixed_neural_gate_comparison.py
import os
import sys
import numpy as np
from typing import Dict, Any
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.phase3.core.config import MoEConfig
from models.phase3.core.logger import MoELogger
from models.phase3.evaluation.evaluator import ModelEvaluator
from models.phase3.data.hybrid_data_manager import HybridDataManager
from models.phase3.experts.enhanced_expert_factory import EnhancedExpertFactory
from models.phase3.gates.fixed_neural_gate import FixedNeuralGate, FixedGateTrainer, FeatureBasedBinning


class FixedNeuralGateComparison:
    """Compare Enhanced MoE vs Fixed Neural Gate vs Feature-based Binning"""
    
    def __init__(self, config: MoEConfig):
        self.config = config
        self.logger = MoELogger(config.log_file)
        self.evaluator = ModelEvaluator(self.logger)
        
    def run_comprehensive_comparison(self) -> Dict[str, Any]:
        """Run comprehensive comparison of all approaches"""
        self.logger.info("=== FIXED NEURAL GATE COMPARISON STARTING ===")
        
        results = {}
        
        try:
            # Load and prepare data
            self.logger.info("Loading data with Hybrid DataManager...")
            data_manager = HybridDataManager(self.config, self.logger)
            
            X, y = data_manager.load_data()
            X, y = data_manager.preprocess_data(X, y)
            X_train, X_val, X_test, y_train, y_val, y_test = data_manager.split_data(X, y)
            
            # Get feature names for interpretability
            feature_names = getattr(data_manager, 'feature_names', [f'feature_{i}' for i in range(X.shape[1])])
            
            # Test 1: Feature-based binning (no target needed for prediction)
            self.logger.info("\n=== TESTING FEATURE-BASED BINNING ===")
            feature_result = self._test_feature_based_approach(
                X_train, y_train, X_val, y_val, X_test, y_test, feature_names
            )
            results['feature_based'] = feature_result
            
            # Test 2: Fixed Neural Gate
            self.logger.info("\n=== TESTING FIXED NEURAL GATE ===")
            neural_gate_result = self._test_fixed_neural_gate(
                X_train, y_train, X_val, y_val, X_test, y_test
            )
            results['fixed_neural_gate'] = neural_gate_result
            
            # Test 3: Simple baseline for comparison
            self.logger.info("\n=== TESTING SIMPLE BASELINE ===")
            baseline_result = self._test_simple_baseline(X_train, y_train, X_test, y_test)
            results['baseline'] = baseline_result
            
            # Compare all results
            comparison = self._compare_all_results(results)
            results['comparison'] = comparison
            
        except Exception as e:
            self.logger.error(f"Comparison failed: {e}")
            raise
            
        return results
    
    def _test_feature_based_approach(self, X_train, y_train, X_val, y_val, X_test, y_test, feature_names):
        """Test feature-based binning approach"""
        self.logger.info("Testing feature-based expert assignment...")
        
        # Create feature-based binning
        binner = FeatureBasedBinning(self.logger)
        binner.create_feature_bins(X_train, y_train, feature_names, n_bins=3)
        
        # Assign bins based on features (not target!)
        train_bins = binner.assign_bins(X_train)
        test_bins = binner.assign_bins(X_test)
        
        # Train experts using enhanced expert factory
        expert_factory = EnhancedExpertFactory(self.config, self.logger)
        selected_experts = expert_factory.select_experts_for_bins(
            X_train, y_train, train_bins, n_bins=3
        )
        trained_experts = expert_factory.train_experts(
            X_train, y_train, train_bins, selected_experts
        )
        
        # Make predictions
        predictions = np.zeros(len(y_test))
        for bin_idx in range(3):
            test_mask = test_bins == bin_idx
            if test_mask.sum() > 0 and bin_idx < len(trained_experts):
                bin_pred = trained_experts[bin_idx].predict(X_test[test_mask])
                predictions[test_mask] = bin_pred
        
        # Calculate metrics using our enhanced evaluator
        metrics = self.evaluator.evaluate(y_test, predictions, test_bins)
        
        return {
            'predictions': predictions,
            'bin_assignments': test_bins,
            'metrics': metrics,
            'approach': 'feature_based_binning',
            'experts': [expert[0] for expert in selected_experts]
        }
    
    def _test_fixed_neural_gate(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """Test fixed neural gate approach"""
        self.logger.info("Testing fixed neural gate...")
        
        # Train multiple experts on overlapping data (no hard binning needed)
        # We'll train each expert on different subsets to create diversity
        n_experts = 4
        experts = []
        expert_names = []
        
        # Create diverse expert pool
        expert_configs = [
            ('lgb_conservative', {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.05}),
            ('xgb_balanced', {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1}),
            ('rf_medium', {'n_estimators': 100, 'max_depth': 8, 'min_samples_split': 5}),
            ('ridge_ensemble', {'alpha': 10.0})
        ]
        
        # Train experts on random subsets for diversity
        for i, (name, params) in enumerate(expert_configs):
            np.random.seed(42 + i)  # Different seed for each expert
            subset_indices = np.random.choice(len(X_train), size=int(0.8 * len(X_train)), replace=False)
            X_subset = X_train[subset_indices]
            y_subset = y_train[subset_indices]
            
            if 'lgb' in name:
                import lightgbm as lgb
                expert = lgb.LGBMRegressor(**params, random_state=42, verbose=-1)
            elif 'xgb' in name:
                import xgboost as xgb
                expert = xgb.XGBRegressor(**params, random_state=42, verbosity=0)
            elif 'rf' in name:
                from sklearn.ensemble import RandomForestRegressor
                expert = RandomForestRegressor(**params, random_state=42)
            else:  # ridge
                from sklearn.linear_model import Ridge
                expert = Ridge(**params, random_state=42)
            
            expert.fit(X_subset, y_subset)
            experts.append(expert)
            expert_names.append(name)
        
        # Get expert predictions for gate training
        train_expert_preds = np.column_stack([expert.predict(X_train) for expert in experts])
        val_expert_preds = np.column_stack([expert.predict(X_val) for expert in experts])
        test_expert_preds = np.column_stack([expert.predict(X_test) for expert in experts])
        
        # Train neural gate
        gate_trainer = FixedGateTrainer(self.config, self.logger)
        gate = gate_trainer.train_gate(
            X_train, X_val, y_train, y_val,
            train_expert_preds, val_expert_preds
        )
        
        # Make final predictions
        predictions, gate_weights = gate_trainer.predict_with_gate(gate, X_test, test_expert_preds)
        
        # Calculate metrics
        metrics = self.evaluator.evaluate(y_test, predictions)
        
        # Log gate weight statistics
        self.logger.info("Gate weight statistics:")
        for i, name in enumerate(expert_names):
            mean_weight = np.mean(gate_weights[:, i])
            self.logger.info(f"  {name}: {mean_weight:.3f} average weight")
        
        return {
            'predictions': predictions,
            'gate_weights': gate_weights,
            'metrics': metrics,
            'approach': 'fixed_neural_gate',
            'experts': expert_names,
            'expert_predictions': test_expert_preds
        }
    
    def _test_simple_baseline(self, X_train, y_train, X_test, y_test):
        """Test simple baseline for comparison"""
        self.logger.info("Testing simple baseline...")
        
        # Simple Random Forest baseline
        from sklearn.ensemble import RandomForestRegressor
        baseline = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        baseline.fit(X_train, y_train)
        predictions = baseline.predict(X_test)
        
        # Calculate metrics
        metrics = self.evaluator.evaluate(y_test, predictions)
        
        return {
            'predictions': predictions,
            'metrics': metrics,
            'approach': 'random_forest_baseline'
        }
    
    def _compare_all_results(self, results):
        """Compare all approaches"""
        comparison = {}
        
        for name, result in results.items():
            if 'metrics' in result:
                overall = result['metrics']['overall']
                comparison[name] = {
                    'r2': overall.get('r2', 0),
                    'rmse': overall.get('rmse', 0),
                    'mae': overall.get('mae', 0),
                    'mae_percentage': overall.get('mae_percentage', overall.get('mape', 0)),  # Fallback to MAPE
                    'within_10_percent': overall.get('within_10_percent', 0),
                    'approach': result.get('approach', name)
                }
        
        # Find best approach
        if comparison:
            best_r2 = max(comparison.values(), key=lambda x: x['r2'])
            best_mae = min(comparison.values(), key=lambda x: x['mae_percentage'])
            
            comparison['best_r2'] = best_r2
            comparison['best_mae_percentage'] = best_mae
        
        return comparison


def main():
    """Main comparison function"""
    
    config = MoEConfig(
        min_bin_size=30,
        max_bins=6,
        min_bins=3,
        scaling_method='robust',
        random_state=42,
        gate_epochs=100,  # More epochs for better training
        gate_learning_rate=1e-4,  # Conservative learning rate
        gate_patience=15,  # More patience
        use_neural_gate=True
    )
    
    comparison = FixedNeuralGateComparison(config)
    results = comparison.run_comprehensive_comparison()
    
    print("\n" + "="*80)
    print("COMPREHENSIVE APPROACH COMPARISON")
    print("="*80)
    
    if 'comparison' in results:
        comp = results['comparison']
        
        print("\nResults Summary:")
        print("-" * 50)
        
        for approach, metrics in comp.items():
            if approach not in ['best_r2', 'best_mae_percentage']:
                print(f"\n{approach.replace('_', ' ').title()}:")
                print(f"  R²: {metrics['r2']:.4f}")
                print(f"  RMSE: ${metrics['rmse']:,.0f}")
                print(f"  MAE: ${metrics['mae']:,.0f} ({metrics['mae_percentage']:.1f}% of mean)")
                print(f"  Within 10%: {metrics['within_10_percent']:.1f}% of predictions")
        
        print(f"\n🏆 WINNERS:")
        print(f"  Best R²: {comp['best_r2']['approach'].replace('_', ' ').title()} (R² = {comp['best_r2']['r2']:.4f})")
        print(f"  Best MAE%: {comp['best_mae_percentage']['approach'].replace('_', ' ').title()} (MAE = {comp['best_mae_percentage']['mae_percentage']:.1f}%)")
        
        # Detailed analysis
        print(f"\n📊 DETAILED ANALYSIS:")
        
        # Check if neural gate is working
        if 'fixed_neural_gate' in results and 'gate_weights' in results['fixed_neural_gate']:
            weights = results['fixed_neural_gate']['gate_weights']
            print(f"\nNeural Gate Analysis:")
            print(f"  Gate weight diversity: {np.std(weights.mean(axis=0)):.3f}")
            print(f"  Most used expert: {np.argmax(weights.mean(axis=0))}")
            print(f"  Weight distribution: {weights.mean(axis=0)}")
            
            # Check if gate is actually blending or just picking one expert
            weight_entropy = -np.sum(weights.mean(axis=0) * np.log(weights.mean(axis=0) + 1e-8))
            max_entropy = np.log(weights.shape[1])
            normalized_entropy = weight_entropy / max_entropy
            print(f"  Blending effectiveness: {normalized_entropy:.3f} (1.0 = perfect blending)")
        
        # Feature-based binning analysis
        if 'feature_based' in results and 'bin_assignments' in results['feature_based']:
            bins = results['feature_based']['bin_assignments']
            print(f"\nFeature-based Binning Analysis:")
            unique_bins, counts = np.unique(bins, return_counts=True)
            for bin_idx, count in zip(unique_bins, counts):
                print(f"  Bin {bin_idx}: {count} samples ({count/len(bins)*100:.1f}%)")
    
    print("\n" + "="*80)
    
    return results


if __name__ == "__main__":
    results = main()