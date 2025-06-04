from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb

estimators = [
    ("Ridge", Ridge(alpha=1.0)),
    ("RF", RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        max_features='sqrt',
        random_state=42
    )),
    ("KNN", KNeighborsRegressor(n_neighbors=5)),
    ("SVR", SVR(C=1.0, epsilon=0.2))
]

params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'device': 'cpu',
    'tree_method': 'hist',
    'max_depth': 4,
    'learning_rate': 0.05385798747478051,
    'subsample': 0.9803117165700693,
    'colsample_bytree': 0.6582418428502539,
    'gamma': 0.9556870921340661,
    'reg_alpha': 0.4103966675466072,
    'reg_lambda': 0.5467280896406721,
    'n_estimators': 459
}

final_estimator = xgb.XGBRegressor(**params)
#final_estimator = lgb.LGBMRegressor(
#    n_estimators=1000,
#    learning_rate=0.03,
#    max_depth=8,
#    num_leaves=31,
#    subsample=0.8,
#    colsample_bytree=0.8,
#    min_child_samples=30,
#    reg_alpha=1.0,
#    reg_lambda=1.0,
#    random_state=42
#)

model = StackingRegressor(
    estimators=estimators,
    final_estimator=final_estimator,
    cv=5,
    n_jobs=-1
)

df = pd.read_csv(Path(__file__).parent.parent.parent / "data" / "without30.csv")
df = df.select_dtypes(include=[np.number])
X = df.drop(columns=["Price"])
y = df["Price"]

# First split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae:.4f}")
print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"R^2 Score: {r2:.4f}")