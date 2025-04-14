import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv("prepared_data.csv")

X = df.iloc[:, [0] + list(range(2, df.shape[1]))]  
y = df.iloc[:, 1] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Random Forest Training Complete.")
print(f"Mean Absolute Error (MAE): {mae:,.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:,.2f}")
print(f"R-squared: {r2:.4f}")
print()

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual prices")
plt.ylabel("Predicted prices")
plt.title("Actual vs Predicted prices")
plt.grid(True)
plt.tight_layout()
#plt.show()

try:
    user_data = pd.read_csv("to_predict.csv")
    user_data = user_data.iloc[:, [0] + list(range(2, user_data.shape[1]))]

    predicted_prices = model.predict(user_data)
    user_data["Predicted Price"] = predicted_prices

    output_file = "predicted_prices_rf.csv"
    user_data.to_csv(output_file, index=False)
    print(f"Predicted prices saved to {output_file}")
except FileNotFoundError:
    print("No 'to_predict.csv' file found.")