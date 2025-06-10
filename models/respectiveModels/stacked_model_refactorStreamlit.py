import numpy as np
import pandas as pd
import joblib
import logging
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('StackedModel_Core')


class StackedModel:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.model_name = "Stacked Model"
        self.feature_names = None

    def train(self, X_train, y_train):
        self.feature_names = X_train.columns.tolist()

        preprocessor = ColumnTransformer(
            transformers=[('num', StandardScaler(), self.feature_names)],
            remainder='passthrough'
        )

        estimators = [
            ("LR", LinearRegression()),
            ("RF", RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                max_features='sqrt',
                random_state=self.random_state
            ))
        ]

        xgb_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'n_estimators': 250,
            'max_depth': 4,
            'learning_rate': 0.05,
            'subsample': 0.9,
            'colsample_bytree': 0.7,
            'random_state': self.random_state
        }

        stacking_regressor = StackingRegressor(
            estimators=estimators,
            final_estimator=xgb.XGBRegressor(**xgb_params),
            cv=3,
            n_jobs=-1,
            verbose=0
        )

        k_features = min(60, len(self.feature_names))

        self.model = Pipeline([
            ('preprocessor', preprocessor),
            ('feature_selection', SelectKBest(score_func=f_regression, k=k_features)),
            ('model', stacking_regressor)
        ])

        y_train_log = np.log1p(y_train)
        self.model.fit(X_train, y_train_log)
        logger.info("Model training completed")

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not trained.")
        y_pred_log = self.model.predict(X)
        y_pred = np.expm1(y_pred_log)
        return y_pred

    def evaluate(self, y_test, y_pred):
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        return {'MAE': mae, 'RMSE': rmse, 'R2': r2}

    def save_model(self, path="stacked_model.joblib"):
        if self.model is None:
            raise ValueError("Model not trained.")

        save_object = {
            'model_pipeline': self.model,
            'feature_names': self.feature_names
        }
        joblib.dump(save_object, path)
        return path

    def load_model(self, path):
        load_object = joblib.load(path)
        self.model = load_object['model_pipeline']
        self.feature_names = load_object['feature_names']
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
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, 'Feature Importance is not directly available\nfor Stacking Regressor pipelines.',
                ha='center', va='center', wrap=True, bbox=dict(boxstyle='round', fc='wheat', alpha=0.5))
        ax.set_title("Feature Importance N/A")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        return fig

    def plot_shap_summary(self, X_train, X_test):
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'SHAP plots are not available for complex\nStacking Regressor pipelines.',
                ha='center', va='center', wrap=True, bbox=dict(boxstyle='round', fc='wheat', alpha=0.5))
        ax.set_title("SHAP Summary Plot N/A")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        return fig