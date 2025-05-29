from xgboost import XGBRegressor

class XGBoostRegressor:
    def __init__(self, **kwargs):
        self.model = XGBRegressor(objective='reg:squarederror', **kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
s