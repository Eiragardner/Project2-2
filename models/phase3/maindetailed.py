# models/phase3/baseline_expert_diagnostic.py
"""
Enhanced Baseline Diagnostic: Deep Expert Analysis

This diagnostic version tests:
1. Expert selection consistency across seeds
2. Individual expert performance on their assigned bins
3. Expert performance outside their bins (generalization)
4. Comparison of expert selection methods
"""
import os
import sys
import numpy as np
from typing import Dict, Any, Tuple, List
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import clone

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.phase3.core.config import MoEConfig
from models.phase3.core.logger import MoELogger
from models.phase3.data.hybrid_data_manager import HybridDataManager
from models.phase3.experts.enhanced_expert_factory import EnhancedExpertFactory
from models.phase3.evaluation.evaluator import ModelEvaluator


class BaselineExpertDiagnostic:
    """
    Deep diagnostic of expert selection and performance
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
        
    def run_comprehensive_diagnostic(self) -> Dict[str, Any]:
        """Run comprehensive expert diagnostic"""
        print("🔬 COMPREHENSIVE BASELINE EXPERT DIAGNOSTIC")
        print("="*60)
        
        results = {}
        
        # Test 1: Seed consistency
        print("\n📊 TEST 1: Expert Selection Consistency Across Seeds")
        print("-" * 50)
        seed_results = self._test_seed_consistency()
        results['seed_consistency'] = seed_results
        
        # Test 2: Expert performance on their bins
        print("\n📊 TEST 2: Expert Performance on Assigned Bins")
        print("-" * 50)
        bin_performance = self._test_expert_bin_performance()
        results['bin_performance'] = bin_performance
        
        # Test 3: Expert generalization
        print("\n📊 TEST 3: Expert Generalization (Performance Outside Bins)")
        print("-" * 50)
        generalization = self._test_expert_generalization()
        results['generalization'] = generalization
        
        # Test 4: Alternative expert selection
        print("\n📊 TEST 4: Alternative Expert Selection Methods")
        print("-" * 50)
        alternatives = self._test_alternative_selection()
        results['alternatives'] = alternatives
        
        # Final summary
        print("\n🎯 DIAGNOSTIC SUMMARY")
        print("="*60)
        self._print_diagnostic_summary(results)
        
        return results
    
    def _test_seed_consistency(self) -> Dict[str, Any]:
        """Test if expert selection is consistent across different random seeds"""
        seeds = [42, 123, 456, 789, 999]
        expert_selections = {}
        performance_scores = []
        
        print(f"Testing {len(seeds)} different random seeds...")
        
        for i, seed in enumerate(seeds):
            try:
                # Create config with different seed
                config = MoEConfig(
                    min_bin_size=30,
                    max_bins=6,
                    min_bins=3,
                    scaling_method='robust',
                    random_state=seed
                )
                
                # Load data with this seed
                data_manager = HybridDataManager(config, self.logger)
                X, y = data_manager.load_data()
                X, y = data_manager.preprocess_data(X, y)
                X_train, X_val, X_test, y_train, y_val, y_test = data_manager.split_data(X, y)
                
                # Train with 6 bins (optimal)
                boundaries = np.percentile(y_train, np.linspace(0, 100, 7))[1:-1]
                train_bins = np.digitize(y_train, boundaries)
                test_bins = np.digitize(y_test, boundaries)
                
                # Select experts
                expert_factory = EnhancedExpertFactory(config, self.logger)
                selected_experts = expert_factory.select_experts_for_bins(
                    X_train, y_train, train_bins, 6
                )
                
                # Train and evaluate
                trained_experts = expert_factory.train_experts(
                    X_train, y_train, train_bins, selected_experts
                )
                
                # Make predictions
                predictions = np.zeros(len(y_test))
                for bin_idx in range(6):
                    test_mask = test_bins == bin_idx
                    if test_mask.sum() > 0 and bin_idx < len(trained_experts):
                        bin_pred = trained_experts[bin_idx].predict(X_test[test_mask])
                        predictions[test_mask] = bin_pred
                
                r2 = r2_score(y_test, predictions)
                performance_scores.append(r2)
                
                # Store expert selection
                expert_names = [expert[0] for expert in selected_experts]
                expert_selections[seed] = expert_names
                
                print(f"  Seed {seed}: R² = {r2:.4f}, Experts = {expert_names}")
                
            except Exception as e:
                print(f"  Seed {seed}: Failed - {e}")
                continue
        
        # Analyze consistency
        if len(expert_selections) > 1:
            # Check how often the same expert is selected for each bin
            bin_consistency = []
            for bin_idx in range(6):
                bin_experts = [experts[bin_idx] if bin_idx < len(experts) else 'none' 
                              for experts in expert_selections.values()]
                most_common = max(set(bin_experts), key=bin_experts.count)
                consistency = bin_experts.count(most_common) / len(bin_experts)
                bin_consistency.append(consistency)
                print(f"  Bin {bin_idx} consistency: {consistency:.1%} (most common: {most_common})")
        
        avg_performance = np.mean(performance_scores) if performance_scores else 0
        std_performance = np.std(performance_scores) if performance_scores else 0
        
        print(f"\nPerformance stability: {avg_performance:.4f} ± {std_performance:.4f}")
        
        return {
            'expert_selections': expert_selections,
            'performance_scores': performance_scores,
            'avg_performance': avg_performance,
            'std_performance': std_performance,
            'bin_consistency': bin_consistency if 'bin_consistency' in locals() else []
        }
    
    def _test_expert_bin_performance(self) -> Dict[str, Any]:
        """Test how well experts perform on their assigned bins"""
        
        # Load data with default seed
        data_manager = HybridDataManager(self.config, self.logger)
        X, y = data_manager.load_data()
        X, y = data_manager.preprocess_data(X, y)
        X_train, X_val, X_test, y_train, y_val, y_test = data_manager.split_data(X, y)
        
        # Create bins
        boundaries = np.percentile(y_train, np.linspace(0, 100, 7))[1:-1]
        train_bins = np.digitize(y_train, boundaries)
        test_bins = np.digitize(y_test, boundaries)
        
        # Select and train experts
        expert_factory = EnhancedExpertFactory(self.config, self.logger)
        selected_experts = expert_factory.select_experts_for_bins(
            X_train, y_train, train_bins, 6
        )
        trained_experts = expert_factory.train_experts(
            X_train, y_train, train_bins, selected_experts
        )
        
        # Test each expert on its assigned bin
        bin_performances = []
        
        print("Individual expert performance on their assigned bins:")
        for bin_idx in range(6):
            test_mask = test_bins == bin_idx
            train_mask = train_bins == bin_idx
            
            if test_mask.sum() > 0 and bin_idx < len(trained_experts):
                # Test on assigned bin
                bin_pred = trained_experts[bin_idx].predict(X_test[test_mask])
                bin_r2 = r2_score(y_test[test_mask], bin_pred)
                bin_rmse = np.sqrt(mean_squared_error(y_test[test_mask], bin_pred))
                
                # Get bin statistics
                bin_mean = y_test[test_mask].mean()
                bin_std = y_test[test_mask].std()
                
                expert_name = selected_experts[bin_idx][0] if bin_idx < len(selected_experts) else "unknown"
                
                print(f"  Bin {bin_idx} ({expert_name}):")
                print(f"    Performance: R² = {bin_r2:.4f}, RMSE = ${bin_rmse:,.0f}")
                print(f"    Bin stats: {test_mask.sum()} samples, mean=${bin_mean:,.0f}, std=${bin_std:,.0f}")
                print(f"    Training size: {train_mask.sum()} samples")
                
                bin_performances.append({
                    'bin_idx': bin_idx,
                    'expert_name': expert_name,
                    'r2': bin_r2,
                    'rmse': bin_rmse,
                    'test_samples': test_mask.sum(),
                    'train_samples': train_mask.sum(),
                    'bin_mean': bin_mean,
                    'bin_std': bin_std
                })
            else:
                print(f"  Bin {bin_idx}: No test samples or expert")
        
        avg_bin_r2 = np.mean([bp['r2'] for bp in bin_performances])
        print(f"\nAverage bin R²: {avg_bin_r2:.4f}")
        
        return {
            'bin_performances': bin_performances,
            'avg_bin_r2': avg_bin_r2,
            'boundaries': boundaries.tolist()
        }
    
    def _test_expert_generalization(self) -> Dict[str, Any]:
        """Test how experts perform outside their assigned bins"""
        
        # Load data
        data_manager = HybridDataManager(self.config, self.logger)
        X, y = data_manager.load_data()
        X, y = data_manager.preprocess_data(X, y)
        X_train, X_val, X_test, y_train, y_val, y_test = data_manager.split_data(X, y)
        
        # Create bins and train experts
        boundaries = np.percentile(y_train, np.linspace(0, 100, 7))[1:-1]
        train_bins = np.digitize(y_train, boundaries)
        test_bins = np.digitize(y_test, boundaries)
        
        expert_factory = EnhancedExpertFactory(self.config, self.logger)
        selected_experts = expert_factory.select_experts_for_bins(
            X_train, y_train, train_bins, 6
        )
        trained_experts = expert_factory.train_experts(
            X_train, y_train, train_bins, selected_experts
        )
        
        # Test each expert on ALL test data
        print("Expert performance on ALL test data (generalization):")
        
        generalization_matrix = np.zeros((6, 6))  # [expert_idx, bin_idx] -> R²
        expert_all_data_performance = []
        
        for expert_idx in range(len(trained_experts)):
            expert_name = selected_experts[expert_idx][0] if expert_idx < len(selected_experts) else f"expert_{expert_idx}"
            
            # Test on all data
            all_pred = trained_experts[expert_idx].predict(X_test)
            all_r2 = r2_score(y_test, all_pred)
            all_rmse = np.sqrt(mean_squared_error(y_test, all_pred))
            
            print(f"  Expert {expert_idx} ({expert_name}) on all data: R² = {all_r2:.4f}, RMSE = ${all_rmse:,.0f}")
            
            expert_all_data_performance.append({
                'expert_idx': expert_idx,
                'expert_name': expert_name,
                'all_data_r2': all_r2,
                'all_data_rmse': all_rmse
            })
            
            # Test on each bin individually
            for bin_idx in range(6):
                test_mask = test_bins == bin_idx
                if test_mask.sum() > 0:
                    bin_pred = trained_experts[expert_idx].predict(X_test[test_mask])
                    bin_r2 = r2_score(y_test[test_mask], bin_pred)
                    generalization_matrix[expert_idx, bin_idx] = bin_r2
        
        # Print generalization matrix
        print(f"\nGeneralization Matrix (Expert vs Bin R²):")
        print("Expert\\Bin    ", end="")
        for bin_idx in range(6):
            print(f"  Bin{bin_idx:1d}  ", end="")
        print()
        
        for expert_idx in range(len(trained_experts)):
            expert_name = selected_experts[expert_idx][0][:8] if expert_idx < len(selected_experts) else f"exp{expert_idx}"
            print(f"{expert_name:10s}  ", end="")
            for bin_idx in range(6):
                r2_val = generalization_matrix[expert_idx, bin_idx]
                print(f"{r2_val:6.2f}  ", end="")
            print()
        
        return {
            'expert_all_data_performance': expert_all_data_performance,
            'generalization_matrix': generalization_matrix.tolist(),
            'best_generalist_r2': max([ep['all_data_r2'] for ep in expert_all_data_performance])
        }
    
    def _test_alternative_selection(self) -> Dict[str, Any]:
        """Test alternative expert selection methods"""
        
        # Load data
        data_manager = HybridDataManager(self.config, self.logger)
        X, y = data_manager.load_data()
        X, y = data_manager.preprocess_data(X, y)
        X_train, X_val, X_test, y_train, y_val, y_test = data_manager.split_data(X, y)
        
        boundaries = np.percentile(y_train, np.linspace(0, 100, 7))[1:-1]
        train_bins = np.digitize(y_train, boundaries)
        test_bins = np.digitize(y_test, boundaries)
        
        alternatives = {}
        
        # Method 1: Current enhanced selection
        print("1. Current EnhancedExpertFactory:")
        expert_factory = EnhancedExpertFactory(self.config, self.logger)
        selected_experts = expert_factory.select_experts_for_bins(X_train, y_train, train_bins, 6)
        trained_experts = expert_factory.train_experts(X_train, y_train, train_bins, selected_experts)
        
        predictions = np.zeros(len(y_test))
        for bin_idx in range(6):
            test_mask = test_bins == bin_idx
            if test_mask.sum() > 0 and bin_idx < len(trained_experts):
                bin_pred = trained_experts[bin_idx].predict(X_test[test_mask])
                predictions[test_mask] = bin_pred
        
        current_r2 = r2_score(y_test, predictions)
        print(f"   R² = {current_r2:.4f}")
        alternatives['current'] = current_r2
        
        # Method 2: Force LightGBM for all bins
        print("2. Force LightGBM for all bins:")
        try:
            import lightgbm as lgb
            lgb_experts = []
            for bin_idx in range(6):
                mask = train_bins == bin_idx
                if mask.sum() > 10:
                    expert = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
                    expert.fit(X_train[mask], y_train[mask])
                    lgb_experts.append(expert)
                else:
                    from sklearn.linear_model import Ridge
                    expert = Ridge(alpha=10.0)
                    expert.fit(X_train, y_train)
                    lgb_experts.append(expert)
            
            lgb_predictions = np.zeros(len(y_test))
            for bin_idx in range(6):
                test_mask = test_bins == bin_idx
                if test_mask.sum() > 0:
                    bin_pred = lgb_experts[bin_idx].predict(X_test[test_mask])
                    lgb_predictions[test_mask] = bin_pred
            
            lgb_r2 = r2_score(y_test, lgb_predictions)
            print(f"   R² = {lgb_r2:.4f}")
            alternatives['force_lgb'] = lgb_r2
            
        except ImportError:
            print("   LightGBM not available")
            alternatives['force_lgb'] = None
        
        # Method 3: Force XGBoost for all bins
        print("3. Force XGBoost for all bins:")
        try:
            import xgboost as xgb
            xgb_experts = []
            for bin_idx in range(6):
                mask = train_bins == bin_idx
                if mask.sum() > 10:
                    expert = xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
                    expert.fit(X_train[mask], y_train[mask])
                    xgb_experts.append(expert)
                else:
                    from sklearn.linear_model import Ridge
                    expert = Ridge(alpha=10.0)
                    expert.fit(X_train, y_train)
                    xgb_experts.append(expert)
            
            xgb_predictions = np.zeros(len(y_test))
            for bin_idx in range(6):
                test_mask = test_bins == bin_idx
                if test_mask.sum() > 0:
                    bin_pred = xgb_experts[bin_idx].predict(X_test[test_mask])
                    xgb_predictions[test_mask] = bin_pred
            
            xgb_r2 = r2_score(y_test, xgb_predictions)
            print(f"   R² = {xgb_r2:.4f}")
            alternatives['force_xgb'] = xgb_r2
            
        except ImportError:
            print("   XGBoost not available")
            alternatives['force_xgb'] = None
        
        # Method 4: Random Forest for all bins
        print("4. Random Forest for all bins:")
        from sklearn.ensemble import RandomForestRegressor
        rf_experts = []
        for bin_idx in range(6):
            mask = train_bins == bin_idx
            expert = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
            if mask.sum() > 10:
                expert.fit(X_train[mask], y_train[mask])
            else:
                expert.fit(X_train, y_train)
            rf_experts.append(expert)
        
        rf_predictions = np.zeros(len(y_test))
        for bin_idx in range(6):
            test_mask = test_bins == bin_idx
            if test_mask.sum() > 0:
                bin_pred = rf_experts[bin_idx].predict(X_test[test_mask])
                rf_predictions[test_mask] = bin_pred
        
        rf_r2 = r2_score(y_test, rf_predictions)
        print(f"   R² = {rf_r2:.4f}")
        alternatives['force_rf'] = rf_r2
        
        return alternatives
    
    def _print_diagnostic_summary(self, results: Dict[str, Any]):
        """Print comprehensive diagnostic summary"""
        
        print("🎯 EXPERT SELECTION DIAGNOSIS:")
        
        # Seed consistency
        if 'seed_consistency' in results:
            std_perf = results['seed_consistency']['std_performance']
            avg_perf = results['seed_consistency']['avg_performance']
            if std_perf < 0.01:
                print("  ✅ Expert selection is STABLE across seeds")
            else:
                print(f"  ⚠️  Expert selection varies across seeds (std={std_perf:.4f})")
            print(f"     Average performance: {avg_perf:.4f}")
        
        # Bin performance
        if 'bin_performance' in results:
            avg_bin_r2 = results['bin_performance']['avg_bin_r2']
            if avg_bin_r2 > 0.3:
                print(f"  ✅ Experts perform WELL on their bins (avg R²={avg_bin_r2:.4f})")
            elif avg_bin_r2 > 0.1:
                print(f"  ⚠️  Experts perform OKAY on their bins (avg R²={avg_bin_r2:.4f})")
            else:
                print(f"  ❌ Experts perform POORLY on their bins (avg R²={avg_bin_r2:.4f})")
        
        # Generalization
        if 'generalization' in results:
            best_gen = results['generalization']['best_generalist_r2']
            if best_gen > 0.7:
                print(f"  ✅ Best expert generalizes WELL (R²={best_gen:.4f})")
            elif best_gen > 0.4:
                print(f"  ⚠️  Best expert generalizes OKAY (R²={best_gen:.4f})")
            else:
                print(f"  ❌ Experts are POOR generalists (best R²={best_gen:.4f})")
        
        # Alternative methods
        if 'alternatives' in results:
            current = results['alternatives']['current']
            print(f"\n📊 ALTERNATIVE EXPERT SELECTION COMPARISON:")
            print(f"   Current method: R² = {current:.4f}")
            
            for method, r2 in results['alternatives'].items():
                if method != 'current' and r2 is not None:
                    improvement = ((r2 - current) / current) * 100 if current > 0 else 0
                    if r2 > current:
                        print(f"   {method:12s}: R² = {r2:.4f} (🎯 +{improvement:.1f}% better)")
                    else:
                        print(f"   {method:12s}: R² = {r2:.4f} ({improvement:.1f}%)")
        
        print(f"\n💡 RECOMMENDATIONS:")
        
        # Specific recommendations based on results
        if 'bin_performance' in results and results['bin_performance']['avg_bin_r2'] < 0.2:
            print("   🔧 Expert selection is broken - experts can't even handle their own bins")
            print("   💡 Fix: Use forced model selection (LightGBM/XGBoost) instead of CV selection")
        
        if 'generalization' in results and results['generalization']['best_generalist_r2'] > 0.7:
            print("   ✅ Experts can generalize - neural gate should work with these experts")
            print("   💡 Try: Train experts on ALL data, let neural gate handle routing")
        
        if 'alternatives' in results:
            best_alt = max([(k, v) for k, v in results['alternatives'].items() if v is not None], 
                          key=lambda x: x[1])
            if best_alt[1] > results['alternatives']['current']:
                print(f"   🎯 Switch to {best_alt[0]} method (R²={best_alt[1]:.4f})")


def main():
    """Run the comprehensive diagnostic"""
    diagnostic = BaselineExpertDiagnostic()
    results = diagnostic.run_comprehensive_diagnostic()
    return results


if __name__ == "__main__":
    results = main()