# phase3/regressors/lightgbm_regressor.py
import lightgbm as lgb

class LightGBMRegressor:
    """LightGBM regressor for mid-range bins"""
    def __init__(self, **kwargs):
        self.model = lgb.LGBMRegressor(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
