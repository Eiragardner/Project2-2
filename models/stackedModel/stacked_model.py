import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from pathlib import Path
#from models.xgboost.XGBoost_core import XGBoostModel
import xgboost as xgb

df = pd.read_csv(Path(__file__).parent.parent.parent / "data" / "without30.csv")
df = df.select_dtypes(include=[np.number])
X = df.drop(columns=["Price"])
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


estimators = [
    ("LR", LinearRegression()), # Possibly change the LinReg model to Ridge?
    ("RF", RandomForestRegressor(n_estimators=100, random_state=42))
]

final_estimator=xgb.XGBRegressor(
        random_state=42,
        objective='reg:squarederror',
        eval_metric='rmse',
        max_depth=6,
        eta=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0,
        alpha=0.1,
        lambda_=1,  # Note: lambda is renamed to lambda_ in XGBRegressor
        min_child_weight=1,
        n_estimators=100,
    )

model = StackingRegressor(
    estimators=estimators,
    final_estimator=final_estimator,
    cv=5,
    n_jobs=-1,
    verbose=0
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Calculate final test set metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nFinal test set results:")
print(f"Mean Absolute Error (MAE): {mae:,.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:,.2f}")
print(f"R-squared: {r2:.2f}")