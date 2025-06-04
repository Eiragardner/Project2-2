# models/phase3/expert_diagnostic.py
"""
Expert Quality Diagnostic

Analyze why the experts are performing so poorly compared to baseline.
The neural gate can't fix fundamentally broken experts.
"""
import os
import sys
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.phase3.core.config import MoEConfig
from models.phase3.core.logger import MoELogger
from models.phase3.data.hybrid_data_manager import HybridDataManager
from models.phase3.experts.enhanced_expert_factory import EnhancedExpertFactory


def diagnose_expert_quality():
    """Diagnose why experts are performing poorly"""
    print("🔍 EXPERT QUALITY DIAGNOSTIC")
    print("="*60)
    print("Goal: Find out why experts are terrible")
    print("="*60)
    
    config = MoEConfig(
        min_bin_size=30,
        max_bins=6,
        min_bins=3,
        scaling_method='robust',
        random_state=42
    )
    
    logger = MoELogger(config.log_file)
    
    # Load data
    data_manager = HybridDataManager(config, logger)
    X, y = data_manager.load_data()
    X, y = data_manager.preprocess_data(X, y)
    X_train, X_val, X_test, y_train, y_val, y_test = data_manager.split_data(X, y)
    
    print(f"Data: {len(y_train)} train, {len(y_val)} val, {len(y_test)} test")
    print(f"Target range: ${y_test.min():,.0f} - ${y_test.max():,.0f}")
    
    # Test different bin configurations
    for n_bins in [3, 6]:
        print(f"\n📊 TESTING {n_bins} BINS:")
        print("-" * 40)
        
        # Create bins
        boundaries = np.percentile(y_train, np.linspace(0, 100, n_bins + 1))[1:-1]
        train_bins = np.digitize(y_train, boundaries)
        test_bins = np.digitize(y_test, boundaries)
        
        print(f"Bin boundaries: {boundaries}")
        
        # Analyze bin distribution
        for bin_idx in range(n_bins):
            train_count = (train_bins == bin_idx).sum()
            test_count = (test_bins == bin_idx).sum()
            if test_count > 0:
                bin_mean = y_test[test_bins == bin_idx].mean()
                bin_std = y_test[test_bins == bin_idx].std()
                print(f"  Bin {bin_idx}: {train_count} train, {test_count} test, "
                      f"mean=${bin_mean:,.0f}, std=${bin_std:,.0f}")
        
        # Train experts
        expert_factory = EnhancedExpertFactory(config, logger)
        selected_experts = expert_factory.select_experts_for_bins(
            X_train, y_train, train_bins, n_bins
        )
        trained_experts = expert_factory.train_experts(
            X_train, y_train, train_bins, selected_experts
        )
        
        print(f"\nEXPERT ANALYSIS:")
        
        # Test 1: Baseline hard assignment (what your baseline does)
        baseline_predictions = np.zeros(len(y_test))
        baseline_r2_per_bin = []
        
        for bin_idx in range(n_bins):
            test_mask = test_bins == bin_idx
            expert_name = selected_experts[bin_idx][0] if bin_idx < len(selected_experts) else "unknown"
            
            if test_mask.sum() > 0 and bin_idx < len(trained_experts):
                bin_pred = trained_experts[bin_idx].predict(X_test[test_mask])
                baseline_predictions[test_mask] = bin_pred
                
                # Check this bin's performance
                bin_r2 = r2_score(y_test[test_mask], bin_pred)
                bin_rmse = np.sqrt(mean_squared_error(y_test[test_mask], bin_pred))
                baseline_r2_per_bin.append(bin_r2)
                
                print(f"  Bin {bin_idx} ({expert_name}): R²={bin_r2:.4f}, RMSE=${bin_rmse:,.0f}, {test_mask.sum()} samples")
            else:
                print(f"  Bin {bin_idx}: No samples or expert")
                baseline_r2_per_bin.append(0)
        
        baseline_r2 = r2_score(y_test, baseline_predictions)
        baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_predictions))
        
        print(f"\n  BASELINE HARD ASSIGNMENT: R² = {baseline_r2:.4f}, RMSE = ${baseline_rmse:,.0f}")
        print(f"  Average bin R²: {np.mean(baseline_r2_per_bin):.4f}")
        
        # Test 2: What if we use each expert on ALL test data?
        print(f"\n  EXPERT PERFORMANCE ON ALL TEST DATA:")
        expert_r2s = []
        expert_predictions_all = []
        
        for bin_idx in range(len(trained_experts)):
            expert_name = selected_experts[bin_idx][0] if bin_idx < len(selected_experts) else f"expert_{bin_idx}"
            
            # Test expert on ALL test data (not just its bin)
            all_pred = trained_experts[bin_idx].predict(X_test)
            all_r2 = r2_score(y_test, all_pred)
            all_rmse = np.sqrt(mean_squared_error(y_test, all_pred))
            expert_r2s.append(all_r2)
            expert_predictions_all.append(all_pred)
            
            print(f"    {expert_name}: R²={all_r2:.4f}, RMSE=${all_rmse:,.0f}")
        
        # Test 3: What's the best possible blending?
        if len(expert_predictions_all) > 1:
            expert_matrix = np.column_stack(expert_predictions_all)
            
            # Equal weighting
            equal_weights = np.ones(len(expert_predictions_all)) / len(expert_predictions_all)
            equal_blend = np.dot(expert_matrix, equal_weights)
            equal_r2 = r2_score(y_test, equal_blend)
            
            # Best individual expert
            best_expert_r2 = max(expert_r2s)
            
            print(f"\n  BLENDING ANALYSIS:")
            print(f"    Best individual expert: R² = {best_expert_r2:.4f}")
            print(f"    Equal weighting blend: R² = {equal_r2:.4f}")
            print(f"    Baseline hard assignment: R² = {baseline_r2:.4f}")
            
            # The key insight
            if baseline_r2 > max(best_expert_r2, equal_r2):
                print(f"    🎯 INSIGHT: Hard assignment BEATS individual experts!")
                print(f"              Specialization is working!")
            else:
                print(f"    ⚠️  WARNING: Experts are fundamentally bad")
        
        print(f"\n  DIAGNOSIS:")
        if baseline_r2 > 0.8:
            print(f"    ✅ Experts are good when used correctly (hard assignment)")
            print(f"    🎯 Neural gate should learn to approximate hard assignment")
        elif np.mean(baseline_r2_per_bin) < 0.3:
            print(f"    ❌ Individual bin experts are terrible")
            print(f"    🔧 Issue: Expert selection or training is broken")
        else:
            print(f"    ⚠️  Mixed results - some experts work, others don't")


def compare_expert_selection_methods():
    """Compare different expert selection approaches"""
    print(f"\n🔬 EXPERT SELECTION COMPARISON")
    print("="*60)
    
    config = MoEConfig(random_state=42)
    logger = MoELogger(config.log_file)
    
    # Load data
    data_manager = HybridDataManager(config, logger)
    X, y = data_manager.load_data()
    X, y = data_manager.preprocess_data(X, y)
    X_train, X_val, X_test, y_train, y_val, y_test = data_manager.split_data(X, y)
    
    # Create 6 bins (optimal from baseline)
    boundaries = np.percentile(y_train, np.linspace(0, 100, 7))[1:-1]
    train_bins = np.digitize(y_train, boundaries)
    test_bins = np.digitize(y_test, boundaries)
    
    # Method 1: Your current EnhancedExpertFactory
    print(f"\nMethod 1: EnhancedExpertFactory (current)")
    expert_factory = EnhancedExpertFactory(config, logger)
    selected_experts = expert_factory.select_experts_for_bins(
        X_train, y_train, train_bins, 6
    )
    trained_experts = expert_factory.train_experts(
        X_train, y_train, train_bins, selected_experts
    )
    
    baseline_predictions = np.zeros(len(y_test))
    for bin_idx in range(6):
        test_mask = test_bins == bin_idx
        if test_mask.sum() > 0 and bin_idx < len(trained_experts):
            bin_pred = trained_experts[bin_idx].predict(X_test[test_mask])
            baseline_predictions[test_mask] = bin_pred
    
    method1_r2 = r2_score(y_test, baseline_predictions)
    print(f"  R² = {method1_r2:.4f}")
    
    # Method 2: Force LightGBM for all bins
    print(f"\nMethod 2: Force LightGBM for all bins")
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
                # Fallback
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
        
        method2_r2 = r2_score(y_test, lgb_predictions)
        print(f"  R² = {method2_r2:.4f}")
        
        if method2_r2 > method1_r2:
            print(f"  🎯 LightGBM approach is better!")
        
    except ImportError:
        print(f"  ❌ LightGBM not available")
    
    print(f"\n💡 RECOMMENDATION:")
    if method1_r2 > 0.8:
        print(f"   Current expert selection is good")
        print(f"   Neural gate should work with proper implementation")
    else:
        print(f"   Expert selection needs improvement")
        print(f"   Fix experts first, then try neural gate")


if __name__ == "__main__":
    diagnose_expert_quality()
    compare_expert_selection_methods()