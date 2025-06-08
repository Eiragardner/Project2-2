import os
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Any, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import warnings
warnings.filterwarnings('ignore')

class HybridDataManager:
    """FIXED: Hybrid data manager that REMOVES data leakage columns"""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.scaler = None
        self.feature_selector = None
        self.feature_names = None
        self.scalers = {}
        
        # CRITICAL: Define leakage columns to remove
        self.LEAKAGE_COLUMNS = [
            'Estimated neighbourhood price per m2',
            'Estimated neighbourhood price per m²',
            'estimated neighbourhood price per m2',
            'estimated neighbourhood price per m²',
            'neighbourhood_price_per_m2',
            'price_per_m2',
            'estimated_price',
            'neighbourhood_price'
        ]
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load data with hybrid processing and NO leakage"""
        self.logger.info("Loading data with hybrid processing...")
        
        if self.config.data_path and os.path.exists(self.config.data_path):
            X, y = self._load_from_file_hybrid(self.config.data_path)
        else:
            for path in ['../data/HousingPriceinBeijing.csv', 'data/HousingPriceinBeijing.csv', 'HousingPriceinBeijing.csv']:
                if os.path.exists(path):
                    X, y = self._load_from_file_hybrid(path)
                    break
            else:
                self.logger.warning("No data file found. Generating sample data.")
                X, y = self._generate_realistic_sample_data()
        
        self.logger.info(f"Loaded dataset: {X.shape[0]} samples, {X.shape[1]} features")
        self.logger.info(f"Target statistics: mean=${y.mean():,.0f}, std=${y.std():,.0f}")
        self.logger.info(f"Target range: [${y.min():,.0f}, ${y.max():,.0f}]")
        
        return X, y
    
    def _load_from_file_hybrid(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Hybrid approach: use original consolidation but add smart feature selection"""
        try:
            df = pd.read_csv(file_path)
            self.logger.info(f"Raw data shape: {df.shape}")
            
            # Find target column
            price_cols = [col for col in df.columns if 'price' in col.lower()]
            if price_cols:
                target_col = price_cols[0]
            elif 'Price' in df.columns:
                target_col = 'Price'
            else:
                target_col = df.columns[-1]
            
            y = df[target_col].values.astype(np.float32)
            feature_cols = [col for col in df.columns if col != target_col]
            
            # CRITICAL: Remove data leakage columns FIRST
            leakage_found = []
            for col in feature_cols:
                if any(leakage_term in col for leakage_term in self.LEAKAGE_COLUMNS):
                    leakage_found.append(col)
            
            if leakage_found:
                self.logger.info(f"REMOVING DATA LEAKAGE COLUMNS: {leakage_found}")
                df = df.drop(columns=leakage_found)
                feature_cols = [col for col in df.columns if col != target_col]
                self.logger.info(f"Removed {len(leakage_found)} leakage columns")
            
            # Use original consolidation approach (it worked better!)
            if self._has_too_many_binary_features(df, feature_cols):
                df = self._consolidate_features_original_way(df, feature_cols, target_col)
                feature_cols = [col for col in df.columns if col != target_col]
            
            # But add some intelligent feature engineering
            df = self._add_smart_features(df, feature_cols, target_col)
            feature_cols = [col for col in df.columns if col != target_col]
            
            # Store feature names
            self.feature_names = feature_cols
            
            # Convert to arrays
            X = df[feature_cols].select_dtypes(include=[np.number]).values.astype(np.float32)
            
            # Handle any remaining non-numeric columns
            if X.shape[1] != len(feature_cols):
                numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
                self.feature_names = list(numeric_cols)
                X = df[numeric_cols].values.astype(np.float32)
            
            self.logger.info(f"Final features: {X.shape[1]} (from {len(feature_cols)} after consolidation)")
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            return self._generate_realistic_sample_data()
    
    def _has_too_many_binary_features(self, df: pd.DataFrame, feature_cols: list) -> bool:
        """Check for too many binary features (from original)"""
        binary_count = 0
        check_limit = min(100, len(feature_cols))
        
        for col in feature_cols[:check_limit]:
            try:
                unique_vals = set(str(v).lower() for v in df[col].dropna().unique()[:10])
                if unique_vals.issubset({'true', 'false', '0', '1', 'yes', 'no', '0.0', '1.0'}):
                    binary_count += 1
            except:
                continue
        
        return binary_count > 15
    
    def _consolidate_features_original_way(self, df: pd.DataFrame, feature_cols: list, target_col: str) -> pd.DataFrame:
        """Use the original consolidation approach that worked better"""
        self.logger.info(f"Consolidating {len(feature_cols)} features using original approach...")
        
        # Separate numeric and binary columns
        numeric_cols = []
        binary_cols = []
        
        for col in feature_cols:
            try:
                if df[col].dtype in ['object', 'bool']:
                    unique_vals = set(str(v).lower() for v in df[col].dropna().unique()[:10])
                    if unique_vals.issubset({'true', 'false', '0', '1', 'yes', 'no', '0.0', '1.0'}):
                        binary_cols.append(col)
                    else:
                        # Try to convert to numeric
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        if not df[col].isna().all():
                            numeric_cols.append(col)
                else:
                    # Check if it's effectively binary
                    unique_vals = set(df[col].dropna().unique())
                    if len(unique_vals) <= 2 and unique_vals.issubset({0, 1, 0.0, 1.0, True, False}):
                        binary_cols.append(col)
                    else:
                        numeric_cols.append(col)
            except:
                continue
        
        # Start with numeric columns and target
        result_data = {}
        for col in numeric_cols:
            result_data[col] = df[col].fillna(df[col].median())
        result_data[target_col] = df[target_col]
        
        # Process binary columns in groups (original approach)
        if binary_cols:
            binary_data = {}
            for col in binary_cols:
                try:
                    binary_data[col] = df[col].astype(str).str.lower().isin(['true', '1', 'yes', '1.0']).astype(int)
                except:
                    binary_data[col] = (df[col] == 1).astype(int)
            
            binary_df = pd.DataFrame(binary_data, index=df.index)
            
            # Create consolidated groups
            n_groups = min(6, max(2, len(binary_cols) // 20))
            group_size = len(binary_cols) // n_groups
            
            for i in range(n_groups):
                start_idx = i * group_size
                end_idx = (i + 1) * group_size if i < n_groups - 1 else len(binary_cols)
                group_cols = binary_cols[start_idx:end_idx]
                
                if group_cols:
                    result_data[f'binary_group_{i+1}'] = binary_df[group_cols].sum(axis=1)
        
        result = pd.DataFrame(result_data, index=df.index)
        self.logger.info(f"Consolidated to {len(result.columns)-1} features")
        return result
    
    def _add_smart_features(self, df: pd.DataFrame, feature_cols: list, target_col: str) -> pd.DataFrame:
        """Add smart engineered features"""
        # Only add a few high-value features to avoid overfitting
        
        # Get numeric columns
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) >= 2:
            # Add 2-3 most promising feature interactions
            numeric_data = df[numeric_cols]
            
            # Calculate correlation with target to find best features
            correlations = numeric_data.corrwith(df[target_col]).abs().fillna(0)
            top_features = correlations.nlargest(3).index.tolist()
            
            # Add interaction between top 2 features
            if len(top_features) >= 2:
                feat1, feat2 = top_features[0], top_features[1]
                df[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
                
                # Add ratio feature if meaningful
                if df[feat2].abs().min() > 1e-6:  # Avoid division by zero
                    df[f'{feat1}_div_{feat2}'] = df[feat1] / (df[feat2] + 1e-6)
        
        return df
    
    def preprocess_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Hybrid preprocessing approach"""
        self.logger.info("Preprocessing data with hybrid approach...")
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=np.median(y))
        
        # Remove outliers conservatively (original approach worked better)
        X, y = self._remove_outliers_conservative(X, y)
        
        # Smart feature selection (only if too many features)
        if X.shape[1] > 20:
            X = self._select_features_smart(X, y)
        
        # Scale features
        X = self._scale_features(X)
        
        return X, y
    
    def _remove_outliers_conservative(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Conservative outlier removal (original approach)"""
        Q1, Q3 = np.percentile(y, [10, 90])
        IQR = Q3 - Q1
        lower_bound = Q1 - 2.0 * IQR
        upper_bound = Q3 + 2.0 * IQR
        
        mask = (y >= lower_bound) & (y <= upper_bound)
        n_outliers = (~mask).sum()
        
        if n_outliers > 0:
            self.logger.info(f"Removed {n_outliers} outliers ({n_outliers/len(y)*100:.1f}%)")
            return X[mask], y[mask]
        
        return X, y
    
    def _select_features_smart(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Smart feature selection using multiple methods"""
        self.logger.info(f"Applying smart feature selection from {X.shape[1]} features...")
        
        # Use both F-test and mutual information
        n_features = min(16, X.shape[1])  # Keep more features than improved version
        
        # F-test selection
        f_selector = SelectKBest(score_func=f_regression, k=n_features)
        
        # Mutual information selection
        mi_selector = SelectKBest(score_func=mutual_info_regression, k=n_features)
        
        try:
            X_f = f_selector.fit_transform(X, y)
            X_mi = mi_selector.fit_transform(X, y)
            
            # Combine selected features (union of both methods)
            f_mask = f_selector.get_support()
            mi_mask = mi_selector.get_support()
            combined_mask = f_mask | mi_mask
            
            # If too many features, prioritize F-test
            if combined_mask.sum() > 18:
                combined_mask = f_mask
            
            X_selected = X[:, combined_mask]
            self.logger.info(f"Selected {X_selected.shape[1]} features using hybrid selection")
            
            return X_selected
            
        except Exception as e:
            self.logger.warning(f"Feature selection failed: {e}, keeping all features")
            return X
    
    def _scale_features(self, X: np.ndarray) -> np.ndarray:
        """Scale features (original approach)"""
        if self.config.scaling_method == 'none':
            return X
        
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.scalers['features'] = self.scaler
        
        self.logger.info(f"Applied {self.config.scaling_method} scaling")
        return X_scaled
    
    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Split data (original approach)"""
        self.logger.info("Splitting data...")
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=self.config.test_size, 
            random_state=self.config.random_state
        )
        
        val_size_adjusted = self.config.val_size / (1 - self.config.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=self.config.random_state
        )
        
        self.logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        self.logger.info(f"Train target: mean=${y_train.mean():,.0f}, std=${y_train.std():,.0f}")
        self.logger.info(f"Test target: mean=${y_test.mean():,.0f}, std=${y_test.std():,.0f}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def _generate_realistic_sample_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate realistic sample data"""
        np.random.seed(self.config.random_state)
        n_samples = 2000
        
        # Generate correlated features
        bedrooms = np.random.poisson(3, n_samples) + 1
        bedrooms = np.clip(bedrooms, 1, 6)
        
        bathrooms = np.random.poisson(2, n_samples) + 1
        bathrooms = np.clip(bathrooms, 1, 4)
        
        sqft = np.random.normal(2000, 600, n_samples)
        sqft = np.clip(sqft, 800, 5000)
        
        year_built = np.random.normal(1990, 20, n_samples)
        year_built = np.clip(year_built, 1950, 2023)
        
        garage = np.random.randint(0, 3, n_samples)
        
        # Add some derived features
        age = 2024 - year_built
        sqft_per_bedroom = sqft / bedrooms
        
        # Create realistic price relationship
        price_base = (
            sqft * 150 +
            bedrooms * 20000 +
            bathrooms * 15000 +
            (year_built - 1950) * 1000 +
            garage * 8000 +
            np.random.normal(0, 30000, n_samples)
        )
        
        price = np.maximum(price_base, 100000)
        
        X = np.column_stack([
            bedrooms, bathrooms, sqft, year_built, garage, age, sqft_per_bedroom
        ]).astype(np.float32)
        
        y = price.astype(np.float32)
        
        self.feature_names = ['bedrooms', 'bathrooms', 'sqft', 'year_built', 'garage', 'age', 'sqft_per_bedroom']
        
        return X, y