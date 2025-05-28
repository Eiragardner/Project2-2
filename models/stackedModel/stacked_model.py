import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from pathlib import Path
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(Path(__file__).parent.parent.parent / "data" / "without30.csv")
df = df.select_dtypes(include=[np.number])
X = df.drop(columns=["Price"])
y = np.log1p(df["Price"])  # log(1 + price)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


estimators = [
    ("LR", make_pipeline(StandardScaler(), LinearRegression())), # Possibly change the LinReg model to Ridge?
    ("RF", RandomForestRegressor(n_estimators=100, random_state=42))
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

model = StackingRegressor(
    estimators=estimators,
    final_estimator= xgb.XGBRegressor(**params),
    cv=5,
    n_jobs=-1,
    verbose=0
)

model.fit(X_train, y_train)
y_pred = np.expm1(model.predict(X_test))  # inverse transform
y_test_original = np.expm1(y_test)



# Calculate final test set metrics
mae = mean_absolute_error(y_test_original, y_pred)
mse = mean_squared_error(y_test_original, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_original, y_pred)

print("\nFinal test set results:")
print(f"Mean Absolute Error (MAE): {mae:,.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:,.2f}")
print(f"R-squared: {r2:.2f}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test_original, y_pred)
plt.plot([y_test_original.min(), y_test_original.max()], [y_test_original.min(), y_test_original.max()], 'r--')
plt.xlabel("Actual prices")
plt.ylabel("Predicted prices")
plt.title("Actual vs Predicted prices")
plt.grid(True)
plt.tight_layout()
#plt.show()

# Create output directory if it doesn't exist
output_dir = Path(__file__).parent.parent.parent / "outputs" / "visualisations" / "stackedModel"
output_dir.mkdir(parents=True, exist_ok=True)

plt.savefig(output_dir / "actual_vs_predicted.png", dpi=300, bbox_inches='tight')
plt.close()

residuals = y_test_original - y_pred
plt.scatter(y_pred, residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted Price")
plt.ylabel("Residual")
plt.title("Residuals vs Predicted Price")
plt.savefig(output_dir / "residuals_vs_predicted.png", dpi=300, bbox_inches='tight')
plt.close()

user_data = pd.read_csv(Path(__file__).parent.parent.parent / "to_predict.csv")
if "Price" in user_data.columns:
    user_data = user_data.drop(columns=["Price"])

user_data = user_data.select_dtypes(include=[np.number])
user_data = user_data[X_train.columns]

predicted_prices = model.predict(user_data)
user_data["Predicted Price"] = np.round(predicted_prices,2)
output_file = Path(__file__).parent.parent.parent / "outputs" / "predicted_prices_SM.csv"
user_data.to_csv(output_file, index=False)