import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
import matplotlib.pyplot as plt

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    from sklearn.ensemble import GradientBoostingRegressor as FallbackModel

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('XGBoost_Core')

class XGBoostModel:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.model_name = "XGBoost"
        self.feature_names = None

    def train(self, X_train, y_train, early_stopping_rounds=50):
        self.feature_names = X_train.columns.tolist()
        if XGBOOST_AVAILABLE:
            self.model = xgb.XGBRegressor(
                objective='reg:squarederror', eval_metric='rmse', n_estimators=500,
                learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                random_state=self.random_state, early_stopping_rounds=early_stopping_rounds
            )
            X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=self.random_state)
            self.model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        else:
            self.model = FallbackModel(n_estimators=100, random_state=self.random_state)
            self.model.fit(X_train, y_train)
        logger.info("Model training completed")

    def predict(self, X):
        if self.model is None: raise ValueError("Model not trained.")
        return self.model.predict(X)

    def evaluate(self, y_test, y_pred):
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}

    def save_model(self, path="xgboost_model.json"):
        if self.model is None: raise ValueError("Model not trained.")
        self.model.save_model(path)
        return path

    def load_model(self, path):
        if not XGBOOST_AVAILABLE: raise ImportError("XGBoost is required to load this model.")
        self.model = xgb.XGBRegressor()
        self.model.load_model(path)

        self.feature_names = self.model.get_booster().feature_names
        return self

    def plot_actual_vs_predicted(self, y_test, y_pred):
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel("Actual Prices")
        ax.set_ylabel("Predicted Prices")
        ax.set_title("Actual vs. Predicted Prices")
        ax.grid(True)
        return fig

    def plot_residuals(self, y_test, y_pred):
        fig, ax = plt.subplots()
        residuals = y_test - y_pred
        ax.scatter(y_pred, residuals, alpha=0.5)
        ax.axhline(0, color='r', linestyle='--')
        ax.set_xlabel("Predicted Prices")
        ax.set_ylabel("Residuals")
        ax.set_title("Residual Plot")
        ax.grid(True)
        return fig

    def plot_feature_importance(self):
        if self.model is None or not hasattr(self.model, 'feature_importances_'): return None
        importances = self.model.feature_importances_
        feature_df = pd.DataFrame({'Feature': self.feature_names, 'Importance': importances})
        feature_df = feature_df.sort_values(by='Importance', ascending=True).tail(20)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(feature_df['Feature'], feature_df['Importance'])
        ax.set_title("Top 20 Feature Importance")
        plt.tight_layout()
        return fig

    def plot_shap_summary(self, X_train, X_test):
        if not SHAP_AVAILABLE or not XGBOOST_AVAILABLE: return None
        try:
            explainer = shap.Explainer(self.model)
            shap_values = explainer(X_test)
            fig = plt.figure()
            shap.summary_plot(shap_values, X_test, show=False)
            plt.tight_layout()
            return fig
        except Exception as e:
            logger.warning(f"Could not generate SHAP plot for XGBoost: {e}")
            return None