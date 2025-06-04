# test_baseline.py - Quick verification of our baseline
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.phase3.baseline_model import BaselineModel

def test_baseline():
    """Test our established baseline"""
    print("🎯 BASELINE MODEL TEST")
    print("="*60)
    print("Testing: Hybrid DataManager + Enhanced Expert Factory")
    print("Expected: R² ≈ 0.8380, RMSE ≈ $111,248")
    print("="*60)
    
    baseline = BaselineModel()
    result = baseline.train()
    
    print(f"\n🏆 FINAL RESULTS:")
    print(f"R²: {result['r2']:.4f} (Target: 0.8380)")
    print(f"RMSE: ${result['rmse']:,.0f} (Target: $111,248)")
    print(f"MAE: ${result['mae']:,.0f}")
    
    # EXTRACT AND SHOW ENHANCED METRICS
    if 'full_evaluation' in result:
        print(f"\n📊 ENHANCED METRICS:")
        overall = result['full_evaluation']['overall']
        
        print(f"   RMSE: ${overall['rmse']:,.0f}")
        print(f"   MAE: ${overall['mae']:,.0f} ({overall.get('mae_percentage', 0):.1f}% of mean)")
        print(f"   R²: {overall['r2']:.4f}")
        print(f"   MAPE: {overall['mape']:.1f}%")
        print(f"   MPE: {overall.get('mpe', 0):+.1f}% (bias: {'over' if overall.get('mpe', 0) > 0 else 'under'}prediction)")
        print(f"   Median APE: {overall.get('median_ape', 0):.1f}%")
        print(f"   Within 5%: {overall.get('within_5_percent', 0):.1f}% of predictions")
        print(f"   Within 10%: {overall.get('within_10_percent', 0):.1f}% of predictions")
        print(f"   Within 15%: {overall.get('within_15_percent', 0):.1f}% of predictions")
        
        # BIAS ANALYSIS
        mpe = overall.get('mpe', 0)
        print(f"\n🔍 BIAS ANALYSIS:")
        if abs(mpe) < 1:
            print(f"   ✅ Excellent! Model is unbiased (MPE = {mpe:+.1f}%)")
        elif abs(mpe) < 3:
            print(f"   👍 Good! Model has minimal bias (MPE = {mpe:+.1f}%)")
        else:
            print(f"   ⚠️  Model has noticeable bias (MPE = {mpe:+.1f}%)")
            if mpe > 0:
                print(f"      Model tends to OVERESTIMATE prices")
            else:
                print(f"      Model tends to UNDERESTIMATE prices")
    else:
        print(f"\n⚠️  Enhanced metrics not available (missing full_evaluation)")
    
    # Check if we hit our target
    r2_diff = abs(result['r2'] - 0.8380)
    rmse_diff = abs(result['rmse'] - 111248)
    
    print(f"\n📊 PERFORMANCE CHECK:")
    if r2_diff < 0.01:
        print(f"✅ R² within 1% of target ({r2_diff:.4f} difference)")
    else:
        print(f"⚠️  R² off by {r2_diff:.4f} from target")
    
    if rmse_diff < 5000:
        print(f"✅ RMSE within $5K of target (${rmse_diff:,.0f} difference)")
    else:
        print(f"⚠️  RMSE off by ${rmse_diff:,.0f} from target")
    
    print(f"\n🔧 EXPERT CONFIGURATION:")
    for info in result['expert_info']:
        print(f"   {info}")
    
    if result['r2'] > 0.83:
        print(f"\n🎉 BASELINE CONFIRMED - Ready for production!")
        print(f"   This model achieves excellent performance")
        print(f"   Next step: Add neural gate for inference-time expert selection")
    else:
        print(f"\n⚠️  BASELINE NEEDS INVESTIGATION")
        print(f"   Performance is below our established benchmark")
    
    return result

if __name__ == "__main__":
    test_baseline()