# leakage_detector.py - Detect data leakage issues
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

def detect_data_leakage():
    """Detect potential data leakage issues"""
    print("🔍 DATA LEAKAGE DETECTION")
    print("="*50)
    
    # figure out where this file lives
    here = os.path.dirname(__file__)              # …/models/phase3
    project_root = os.path.abspath(os.path.join(here, '..', '..'))  # …/Project2-2

    data_paths = [
        os.path.join(project_root, 'data', 'prepared_data.csv'),
        # you can still fall back to these if you ever run from root or elsewhere
        os.path.join(project_root, 'data', 'without30.csv'),
    ]
    
    for path in data_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"📁 Loaded: {path} - Shape: {df.shape}")
            break
    
    if df is None:
        print("❌ No data found")
        return
    
    # Find target
    price_cols = [col for col in df.columns if 'price' in col.lower()]
    target_col = price_cols[0] if price_cols else df.columns[-1]
    
    y = df[target_col].values
    feature_cols = [col for col in df.columns if col != target_col]
    
    print(f"🎯 Target: {target_col}")
    print(f"📊 Features: {len(feature_cols)}")
    print(f"💰 Price range: ${y.min():,.0f} - ${y.max():,.0f}")
    
    # Test 1: Check for perfect correlations (data leakage)
    print(f"\n🧪 TEST 1: Perfect Correlations")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlations = df[numeric_cols].corr()[target_col].abs().sort_values(ascending=False)
    
    print("Top 10 correlations with target:")
    for i, (col, corr) in enumerate(correlations.head(10).items()):
        if col != target_col:
            warning = " ⚠️ SUSPICIOUS!" if corr > 0.95 else " 🚨 LEAKAGE!" if corr > 0.99 else ""
            print(f"  {i+1:2d}. {col[:30]:30s}: {corr:.4f}{warning}")
    
    # Test 2: Check if features contain target information
    print(f"\n🧪 TEST 2: Feature Names Analysis")
    suspicious_features = []
    for col in feature_cols[:20]:  # Check first 20
        col_lower = col.lower()
        if any(word in col_lower for word in ['price', 'cost', 'value', 'worth', 'dollar', '$']):
            suspicious_features.append(col)
    
    if suspicious_features:
        print("🚨 SUSPICIOUS FEATURE NAMES (might contain target info):")
        for feat in suspicious_features:
            print(f"  - {feat}")
    else:
        print("✅ No obviously suspicious feature names found")
    
    # Test 3: Simple train/test split validation
    print(f"\n🧪 TEST 3: Train/Test Split Validation")
    
    # Use only numeric features, limit to 10
    numeric_features = df.select_dtypes(include=[np.number]).columns
    feature_cols_numeric = [col for col in numeric_features if col != target_col][:10]
    
    X = df[feature_cols_numeric].fillna(df[feature_cols_numeric].median()).values
    
    # Multiple random splits to check consistency
    r2_scores = []
    
    for seed in [42, 123, 456, 789, 999]:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )
        
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = Ridge(alpha=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        r2_scores.append(r2)
    
    mean_r2 = np.mean(r2_scores)
    std_r2 = np.std(r2_scores)
    
    print(f"R² across 5 different splits:")
    for i, r2 in enumerate(r2_scores):
        print(f"  Split {i+1}: {r2:.4f}")
    
    print(f"Mean R²: {mean_r2:.4f} ± {std_r2:.4f}")
    
    # Analysis
    if mean_r2 > 0.9:
        print("🚨 HIGHLY SUSPICIOUS: R² > 0.9 suggests data leakage!")
    elif mean_r2 > 0.8:
        print("⚠️  SUSPICIOUS: R² > 0.8 is unusually high for real-world data")
    elif std_r2 > 0.1:
        print("⚠️  UNSTABLE: High variance across splits suggests overfitting")
    elif mean_r2 < 0.1:
        print("❌ POOR: Very low R² suggests data quality issues")
    else:
        print("✅ REASONABLE: Results seem plausible")
    
    # Test 4: Feature importance analysis
    print(f"\n🧪 TEST 4: Feature Importance Analysis")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = Ridge(alpha=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Get feature importance (absolute coefficients)
    importance = np.abs(model.coef_)
    feature_importance = list(zip(feature_cols_numeric, importance))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    print("Top 5 most important features:")
    for i, (feat, imp) in enumerate(feature_importance[:5]):
        print(f"  {i+1}. {feat[:30]:30s}: {imp:.4f}")
    
    # Test 5: Temporal leakage check (if dates present)
    print(f"\n🧪 TEST 5: Temporal Analysis")
    date_cols = [col for col in df.columns if any(word in col.lower() 
                 for word in ['date', 'time', 'year', 'month', 'day'])]
    
    if date_cols:
        print(f"📅 Found date columns: {date_cols}")
        print("⚠️  CHECK: Ensure no future information is used!")
    else:
        print("✅ No obvious date columns found")
    
    # Final verdict
    print(f"\n" + "="*50)
    print("🏁 FINAL VERDICT")
    print("="*50)
    
    issues = []
    
    if any(corr > 0.95 for corr in correlations.values if not np.isnan(corr)):
        issues.append("Very high correlations detected")
    
    if suspicious_features:
        issues.append("Suspicious feature names found")
    
    if mean_r2 > 0.8:
        issues.append("Unusually high R² scores")
    
    if std_r2 > 0.1:
        issues.append("High variance across splits")
    
    if issues:
        print("🚨 POTENTIAL ISSUES DETECTED:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nRECOMMENDATIONS:")
        print("  1. Manually inspect highly correlated features")
        print("  2. Remove features that might contain target information")
        print("  3. Use more conservative train/test splits")
        print("  4. Consider using less complex models")
    else:
        print("✅ No major issues detected")
        print("Data appears to be clean for modeling")
    
    return {
        'mean_r2': mean_r2,
        'std_r2': std_r2,
        'suspicious_features': suspicious_features,
        'top_correlations': correlations.head(5).to_dict(),
        'issues': issues
    }

if __name__ == "__main__":
    results = detect_data_leakage()