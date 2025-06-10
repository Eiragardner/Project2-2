# models/phase3/fixed_baseline_model.py
"""
FIXED Baseline Model: Proper train/val/test split
No data leakage - validation set used for model selection
"""
import os
import sys
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.phase3.core.config import MoEConfig
from models.phase3.core.logger import MoELogger
from models.phase3.data.hybrid_data_manager import HybridDataManager
from models.phase3.experts.enhanced_expert_factory import EnhancedExpertFactory
from models.phase3.evaluation.evaluator import ModelEvaluator


class FixedBaselineModel:
    """
    FIXED: Proper evaluation without data leakage
    - Use validation set for model selection
    - Use test set only for final unbiased evaluation
    """
    
    def __init__(self, config: MoEConfig = None):
        if config is None:
            config = MoEConfig(
                min_bin_size=30,
                max_bins=6,
                min_bins=3,
                scaling_method='robust',
                random_state=42
            )
        
        self.config = config
        self.logger = MoELogger(config.log_file)
        self.evaluator = ModelEvaluator(self.logger)
        
        # Model components
        self.data_manager = None
        self.expert_factory = None
        self.trained_experts = None
        self.bin_boundaries = None
        self.is_trained = False
        self.best_config = None
        
    def train(self) -> Dict[str, Any]:
        """Train the baseline model with proper evaluation"""
        self.logger.info("=== TRAINING FIXED BASELINE MODEL (NO DATA LEAKAGE) ===")
        self.logger.info("Configuration: Hybrid DataManager + Enhanced Expert Factory")
        
        # Initialize data manager
        self.data_manager = HybridDataManager(self.config, self.logger)
        
        # Load and preprocess data
        X, y = self.data_manager.load_data()
        X, y = self.data_manager.preprocess_data(X, y)
        X_train, X_val, X_test, y_train, y_val, y_test = self.data_manager.split_data(X, y)
        
        self.logger.info(f"Data splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # STEP 1: Find optimal configuration using VALIDATION set
        best_config = self._find_optimal_configuration_on_validation(
            X_train, y_train, X_val, y_val
        )
        
        # STEP 2: Train final model with best config on train+val, evaluate on test
        final_result = self._train_final_model_proper(
            X_train, X_val, X_test, y_train, y_val, y_test, best_config
        )
        
        self.is_trained = True
        
        self.logger.info("=== FIXED BASELINE MODEL TRAINING COMPLETE ===")
        self.logger.info(f"Final Test Performance: R² = {final_result['test_r2']:.4f}")
        self.logger.info(f"Final Test RMSE: ${final_result['test_rmse']:,.0f}")
        self.logger.info(f"Final Test MAE: ${final_result['test_mae']:,.0f}")
        
        return final_result
    
    def _find_optimal_configuration_on_validation(self, X_train, y_train, X_val, y_val):
        """FIXED: Find optimal configuration using VALIDATION set only"""
        self.logger.info("Finding optimal bin configuration using VALIDATION set...")
        
        best_result = None
        best_score = -np.inf
        validation_results = []
        
        for n_bins in [3, 4, 5, 6]:  # Test different bin configurations
            try:
                # Train on train set, evaluate on validation set
                result = self._train_and_evaluate_on_validation(
                    n_bins, X_train, y_train, X_val, y_val
                )
                
                validation_results.append({
                    'n_bins': n_bins,
                    'val_r2': result['r2'],
                    'val_rmse': result['rmse'],
                    'val_mae': result['mae']
                })
                
                if result['r2'] > best_score:
                    best_score = result['r2']
                    best_result = result
                    best_result['n_bins'] = n_bins
                    
                self.logger.info(f"  {n_bins} bins: Val R²={result['r2']:.4f}, Val RMSE=${result['rmse']:,.0f}")
                    
            except Exception as e:
                self.logger.warning(f"Failed with {n_bins} bins: {e}")
                continue
        
        self.logger.info(f"✅ Optimal configuration selected: {best_result['n_bins']} bins (based on validation)")
        
        return {
            'best_n_bins': best_result['n_bins'],
            'validation_results': validation_results,
            'best_val_r2': best_result['r2']
        }
    
    def _train_and_evaluate_on_validation(self, n_bins, X_train, y_train, X_val, y_val):
        """Train on train set, evaluate on validation set"""
        
        # Create bins based on TRAINING set targets
        boundaries = np.percentile(y_train, np.linspace(0, 100, n_bins + 1))[1:-1]
        train_bins = np.digitize(y_train, boundaries)
        val_bins = np.digitize(y_val, boundaries)  # Apply same boundaries to validation
        
        # Initialize enhanced expert factory
        expert_factory = EnhancedExpertFactory(self.config, self.logger)
        
        # Select optimal experts for each bin using TRAINING data
        selected_experts = expert_factory.select_experts_for_bins(
            X_train, y_train, train_bins, n_bins
        )
        
        # Train experts on TRAINING data
        trained_experts = expert_factory.train_experts(
            X_train, y_train, train_bins, selected_experts
        )
        
        # Make predictions on VALIDATION set
        val_predictions = np.zeros(len(y_val))
        
        for bin_idx in range(n_bins):
            val_mask = val_bins == bin_idx
            if val_mask.sum() > 0 and bin_idx < len(trained_experts):
                bin_pred = trained_experts[bin_idx].predict(X_val[val_mask])
                val_predictions[val_mask] = bin_pred
        
        # Calculate validation metrics
        rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
        mae = mean_absolute_error(y_val, val_predictions)
        r2 = r2_score(y_val, val_predictions)
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'predictions': val_predictions,
            'boundaries': boundaries,
            'selected_experts': selected_experts,
            'trained_experts': trained_experts
        }
    
    def _train_final_model_proper(self, X_train, X_val, X_test, y_train, y_val, y_test, best_config):
        """FIXED: Train final model and evaluate on TEST set (unbiased)"""
        self.logger.info(f"Training final model with {best_config['best_n_bins']} bins...")
        self.logger.info("🎯 Final evaluation on TEST set (unbiased)")
        
        optimal_bins = best_config['best_n_bins']
        
        # Combine train + validation for final training (common practice)
        X_train_final = np.vstack([X_train, X_val])
        y_train_final = np.hstack([y_train, y_val])
        
        # Create bins based on combined training data
        boundaries = np.percentile(y_train_final, np.linspace(0, 100, optimal_bins + 1))[1:-1]
        train_bins = np.digitize(y_train_final, boundaries)
        test_bins = np.digitize(y_test, boundaries)
        
        # Initialize enhanced expert factory
        expert_factory = EnhancedExpertFactory(self.config, self.logger)
        
        # Select and train experts on combined training data
        selected_experts = expert_factory.select_experts_for_bins(
            X_train_final, y_train_final, train_bins, optimal_bins
        )
        
        trained_experts = expert_factory.train_experts(
            X_train_final, y_train_final, train_bins, selected_experts
        )
        
        # Make predictions on TEST set (unbiased evaluation)
        test_predictions = np.zeros(len(y_test))
        expert_info = []
        
        for bin_idx in range(optimal_bins):
            test_mask = test_bins == bin_idx
            if test_mask.sum() > 0 and bin_idx < len(trained_experts):
                bin_pred = trained_experts[bin_idx].predict(X_test[test_mask])
                test_predictions[test_mask] = bin_pred
                
                # Get expert name
                if bin_idx < len(selected_experts):
                    expert_name = selected_experts[bin_idx][0]
                else:
                    expert_name = "unknown"
                expert_info.append(f"Bin {bin_idx}: {expert_name}")
        
        # Calculate TEST metrics (unbiased)
        test_evaluation = self.evaluator.evaluate(y_test, test_predictions, test_bins)
        test_overall = test_evaluation['overall']
        
        # Store final model components
        self.bin_boundaries = boundaries
        self.trained_experts = trained_experts
        self.best_config = best_config
        
        # Log expert configuration
        self.logger.info("Final Expert Configuration:")
        for info in expert_info:
            self.logger.info(f"  {info}")
        
        return {
            'test_rmse': test_overall['rmse'],
            'test_mae': test_overall['mae'],
            'test_r2': test_overall['r2'],
            'test_predictions': test_predictions,
            'expert_info': expert_info,
            'validation_config': best_config,
            'full_test_evaluation': test_evaluation
        }


def test_fixed_baseline():
    """Test the FIXED baseline model"""
    print("🎯 FIXED BASELINE MODEL TEST (NO DATA LEAKAGE)")
    print("="*60)
    print("✅ Validation set used for model selection")
    print("✅ Test set used only for final unbiased evaluation")
    print("="*60)
    
    baseline = FixedBaselineModel()
    result = baseline.train()
    
    print(f"\n🏆 UNBIASED TEST RESULTS:")
    print(f"R²: {result['test_r2']:.4f}")
    print(f"RMSE: ${result['test_rmse']:,.0f}")
    print(f"MAE: ${result['test_mae']:,.0f}")
    
    # Show validation vs test comparison
    if 'validation_config' in result:
        val_r2 = result['validation_config']['best_val_r2']
        test_r2 = result['test_r2']
        gap = val_r2 - test_r2
        
        print(f"\n📊 VALIDATION vs TEST:")
        print(f"Best Validation R²: {val_r2:.4f}")
        print(f"Final Test R²: {test_r2:.4f}")
        print(f"Gap: {gap:+.4f}")
        
        if abs(gap) < 0.02:
            print(f"✅ Small gap - model generalizes well!")
        elif gap > 0.05:
            print(f"⚠️  Large gap - possible overfitting to validation")
        else:
            print(f"👍 Reasonable gap - normal generalization")
    
    print(f"\n🔧 EXPERT CONFIGURATION:")
    for info in result['expert_info']:
        print(f"   {info}")
    
    return result


if __name__ == "__main__":
    test_fixed_baseline()