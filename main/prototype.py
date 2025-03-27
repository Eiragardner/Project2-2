import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv("prepared_data.csv")
df = df.select_dtypes(include=[np.number])
X = df.drop(columns=["Price"])
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = round(r2_score(y_test, y_pred), 2)


print("Training complete.")
print(f"Mean Absolute Error (MAE): {mae:,.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:,.2f}")
print(f"R-squared: {r2}")
# Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual prices")
plt.ylabel("Predicted prices")
plt.title("Actual vs Predicted prices")
plt.grid(True)
plt.tight_layout()
plt.show()

user_data = pd.read_csv("to_predict.csv")
if "Price" in user_data.columns:
    user_data = user_data.drop(columns=["Price"])

user_data = user_data.select_dtypes(include=[np.number])
user_data = user_data[X_train.columns] 

predicted_prices = model.predict(user_data)
user_data["Predicted Price"] = np.round(predicted_prices,2)
output_file = "predicted_prices.csv"
user_data.to_csv(output_file, index=False)

