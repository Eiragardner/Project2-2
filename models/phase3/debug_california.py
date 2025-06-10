# debug_california.py - Find the data leakage issue in California Dataset
import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.phase3.core.config import MoEConfig
from models.phase3.core.logger import MoELogger
from models.phase3.data.hybrid_data_manager import HybridDataManager

def debug_california_dataset():
    """Debug what's wrong with California Dataset.csv"""
    print("🔍 DEBUGGING CALIFORNIA DATASET")
    print("="*50)
    
    # Create config pointing to California dataset
    config = MoEConfig()
    config.data_path = "California Dataset.csv"
    
    logger = MoELogger(config.log_file)
    
    # Step 1: Check raw data loading
    print("📁 STEP 1: Raw Data Loading")
    try:
        df = pd.read_csv("California Dataset.csv")
        print(f"✅ Raw CSV shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Check target detection logic
        price_cols = [col for col in df.columns if 'price' in col.lower()]
        if price_cols:
            target_col = price_cols[0]
        elif 'Price' in df.columns:
            target_col = 'Price'
        else:
            target_col = df.columns[-1]
        
        print(f"Target column: {target_col}")
        
        y_raw = df[target_col].values
        print(f"Target range: {y_raw.min():.3f} to {y_raw.max():.3f}")
        print(f"Target sample: {y_raw[:5]}")
        
        feature_cols = [col for col in df.columns if col != target_col]
        print(f"Feature columns ({len(feature_cols)}): {feature_cols}")
        
    except Exception as e:
        print(f"❌ Raw data loading failed: {e}")
        return
    
    # Step 2: Check HybridDataManager processing
    print(f"\n📊 STEP 2: HybridDataManager Processing")
    try:
        data_manager = HybridDataManager(config, logger)
        X, y = data_manager.load_data()
        
        print(f"✅ Processed shape: X={X.shape}, y={y.shape}")
        print(f"Processed target range: {y.min():.3f} to {y.max():.3f}")
        print(f"Processed target sample: {y[:5]}")
        
        # Check if target changed during processing
        if not np.allclose(y_raw, y, rtol=1e-5):
            print(f"⚠️  Target was modified during processing!")
            print(f"   Raw: {y_raw[:3]}")
            print(f"   Processed: {y[:3]}")
        else:
            print(f"✅ Target unchanged during processing")
        
        # Check feature names if available
        if hasattr(data_manager, 'feature_names') and data_manager.feature_names:
            print(f"Final features: {data_manager.feature_names}")
        
    except Exception as e:
        print(f"❌ HybridDataManager failed: {e}")
        return
    
    # Step 3: Check for data leakage (perfect correlations)
    print(f"\n🚨 STEP 3: Data Leakage Detection")
    try:
        # Reconstruct the DataFrame after processing
        if hasattr(data_manager, 'feature_names') and data_manager.feature_names:
            feature_names = data_manager.feature_names
        else:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        df_processed = pd.DataFrame(X, columns=feature_names)
        df_processed['target'] = y
        
        # Calculate correlations
        correlations = df_processed.corr()['target'].abs().sort_values(ascending=False)
        
        print("Top correlations with target:")
        for i, (feature, corr) in enumerate(correlations.head(10).items()):
            if feature != 'target':
                warning = ""
                if corr > 0.99:
                    warning = " 🚨 LEAKAGE!"
                elif corr > 0.95:
                    warning = " ⚠️  SUSPICIOUS!"
                print(f"  {i+1:2d}. {feature:20s}: {corr:.6f}{warning}")
        
        # Check for perfect correlations
        perfect_correlations = correlations[(correlations > 0.999) & (correlations.index != 'target')]
        if len(perfect_correlations) > 0:
            print(f"\n🚨 PERFECT CORRELATIONS DETECTED!")
            for feature, corr in perfect_correlations.items():
                print(f"   {feature}: {corr:.10f}")
            return True  # Data leakage confirmed
        
    except Exception as e:
        print(f"❌ Correlation analysis failed: {e}")
    
    # Step 4: Simple model test
    print(f"\n🧪 STEP 4: Simple Model Test")
    try:
        # Apply same preprocessing as HybridDataManager
        X_processed, y_processed = data_manager.preprocess_data(X.copy(), y.copy())
        
        print(f"After preprocessing: X={X_processed.shape}")
        print(f"Target range after preprocessing: {y_processed.min():.3f} to {y_processed.max():.3f}")
        
        # Simple train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_processed, test_size=0.2, random_state=42
        )
        
        # Train simple Ridge model
        model = Ridge(alpha=100)
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"Simple Ridge results:")
        print(f"  R²: {r2:.6f}")
        print(f"  RMSE: {rmse:.6f}")
        
        if rmse < 1e-10:
            print(f"🚨 PERFECT PREDICTIONS CONFIRMED!")
            print(f"   Prediction sample: {y_pred[:5]}")
            print(f"   Actual sample:     {y_test[:5]}")
            print(f"   Difference:        {np.abs(y_pred[:5] - y_test[:5])}")
        
        # Check feature importance for leakage
        feature_importance = np.abs(model.coef_)
        if hasattr(data_manager, 'feature_names') and data_manager.feature_names:
            feature_names = data_manager.feature_names[:len(feature_importance)]
        else:
            feature_names = [f"feature_{i}" for i in range(len(feature_importance))]
        
        importance_pairs = list(zip(feature_names, feature_importance))
        importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\nTop 5 feature importance:")
        for i, (name, importance) in enumerate(importance_pairs[:5]):
            print(f"  {i+1}. {name:20s}: {importance:.6f}")
        
    except Exception as e:
        print(f"❌ Simple model test failed: {e}")
    
    print(f"\n" + "="*50)
    print("🏁 DIAGNOSIS COMPLETE")
    print("If you see 'PERFECT CORRELATIONS' or 'PERFECT PREDICTIONS',")
    print("there's definitely data leakage in the California dataset.")
    print("The prepared_data.csv works fine, so use that instead.")


if __name__ == "__main__":
    debug_california_dataset()