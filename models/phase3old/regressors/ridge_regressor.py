# phase3/regressors/ridge_regressor.py
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

class RidgeRegressor:
    """Ridge regressor with polynomial features for luxury bins"""
    def __init__(self, degree: int = 2, alpha: float = 1.0):
        self.model = make_pipeline(PolynomialFeatures(degree=degree, include_bias=False), Ridge(alpha=alpha))

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)