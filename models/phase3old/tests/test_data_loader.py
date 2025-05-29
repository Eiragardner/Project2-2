# phase3/tests/test_data_loader.py
import unittest
import pandas as pd
from pandas.testing import assert_frame_equal
from phase3.data.data_loader import DataLoader

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        # Create a dummy DataFrame
        self.df = pd.DataFrame({
            'Feature1': [1, 2],
            'Price': [100, 200],
            'bin': [0, 1]
        })
        self.path = 'test.csv'
        self.df.to_csv(self.path, index=False)

    def tearDown(self):
        import os; os.remove(self.path)

    def test_load(self):
        loader = DataLoader(path=self.path)
        X, y, bins = loader.load()
        assert_frame_equal(X.reset_index(drop=True), self.df[['Feature1']])
        pd.testing.assert_series_equal(y.reset_index(drop=True), self.df['Price'])
        pd.testing.assert_series_equal(bins.reset_index(drop=True), self.df['bin'])

