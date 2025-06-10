import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import shap


class LinearRegressionModel:

    def __init__(self):
        self.model = LinearRegression()
        self.model_name = "Linear Regression"
        self.feature_names = None

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.feature_names = X_train.columns.tolist()

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, y_test, y_pred):
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}

    def save_model(self, path):
        saved_object = {
            'model': self.model,
            'feature_names': self.feature_names
        }
        joblib.dump(saved_object, path)
        return path

    def load_model(self, path):
        loaded_object = joblib.load(path)
        self.model = loaded_object['model']
        self.feature_names = loaded_object['feature_names']
        return self

    def plot_actual_vs_predicted(self, y_test, y_pred):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_test, y_pred, alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
        ax.set_xlabel("Actual Prices")
        ax.set_ylabel("Predicted Prices")
        ax.set_title("Actual vs. Predicted Prices")
        ax.grid(True)
        plt.tight_layout()
        return fig

    def plot_residuals(self, y_test, y_pred):
        residuals = y_test - y_pred
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_pred, residuals, alpha=0.5)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel("Predicted Prices")
        ax.set_ylabel("Residuals")
        ax.set_title("Residual Plot")
        ax.grid(True)
        plt.tight_layout()
        return fig

    def plot_feature_importance(self):
        if not hasattr(self.model, 'coef_') or self.feature_names is None:
            return None

        importances = self.model.coef_
        feature_importance = pd.Series(importances, index=self.feature_names).sort_values(key=abs)

        fig, ax = plt.subplots(figsize=(10, 8))
        feature_importance.plot(kind='barh', ax=ax)
        ax.set_title("Feature Importance (Coefficients)")
        ax.set_xlabel("Coefficient Value")
        plt.tight_layout()
        return fig

    def plot_shap_summary(self, X_train, X_test):
        try:
            explainer = shap.LinearExplainer(self.model, X_train)
            shap_values = explainer(X_test)

            plt.figure()
            shap.summary_plot(shap_values, X_test, show=False)
            plt.tight_layout()
            return plt.gcf()
        except Exception as e:
            print(f"Could not generate SHAP plot for Linear Regression: {e}")
            return None