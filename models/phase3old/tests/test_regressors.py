# phase3/tests/test_regressors.py
import unittest
import numpy as np
from phase3.regressors.xgboost_regressor import XGBoostRegressor
from phase3.regressors.lightgbm_regressor import LightGBMRegressor
from phase3.regressors.ridge_regressor import RidgeRegressor

class DummyData:
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10])  # perfect doubling

class TestRegressors(unittest.TestCase):
    def test_xgboost(self):
        model = XGBoostRegressor(n_estimators=10, max_depth=2)
        model.fit(DummyData.X, DummyData.y)
        preds = model.predict(DummyData.X)
        self.assertEqual(preds.shape, DummyData.y.shape)

    def test_lightgbm(self):
        model = LightGBMRegressor(n_estimators=10, max_depth=2)
        model.fit(DummyData.X, DummyData.y)
        preds = model.predict(DummyData.X)
        self.assertEqual(preds.shape, DummyData.y.shape)

    def test_ridge(self):
        model = RidgeRegressor(degree=1, alpha=1.0)
        model.fit(DummyData.X, DummyData.y)
        preds = model.predict(DummyData.X)
        self.assertEqual(preds.shape, DummyData.y.shape)
    
