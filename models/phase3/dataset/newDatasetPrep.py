
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
     
house_price_dataset = sklearn.datasets.fetch_california_housing()
# Loading the dataset to a pandas dataframe
house_price_dataframe = pd.DataFrame(house_price_dataset.data, columns = house_price_dataset.feature_names)
#print(house_price_dataframe.head())
house_price_dataframe['Price'] = house_price_dataset.target
house_price_dataframe.to_csv("California Dataset")

     
