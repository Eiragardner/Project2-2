import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# GNU STP

'''
Complexity Analysis of the Linear Regression Model;

Time Complexity:
1. Training (model.fit):
    O(n * d^2), where n is the number of samples and d is the number of features.
2. Prediction (model.predict):
    O(n * d), for n predictions with d features.

Space Complexity:
1. Model Parameters:
    O(d) for storing coefficients and intercepts.
    One weight per feature plus one intercept term
2. Training Memory:
    O(n * d) for storing the feature matrix X.
    O(n) for storing the target vector y.
    
Additional Operations:
1. Cross-validation (KFold):
    Time: O(k * (n * d^2)), where k is the number of folds.
    Space: O(n * d) for each fold.
2. Data Preprocessing:
    O(n * d) for reading and filtering numeric columns
    O(n * d) for train-test split
'''

df = pd.read_csv("without30.csv")
df = df.select_dtypes(include=[np.number])
X = df.drop(columns=["Price"])
y = df["Price"]

# Perform k-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
model = LinearRegression()

# Calculate cross-validation scores
cv_scores = cross_val_score(model, X, y, cv=kfold, scoring='r2')
mae_scores = -cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_absolute_error')
rmse_scores = np.sqrt(-cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error'))

# Print cross-validation results
print("Cross-validation results:")
print(f"R² scores for each fold: {cv_scores}")
print(f"Average R²: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
print(f"Average MAE: {mae_scores.mean():.2f} (+/- {mae_scores.std() * 2:.2f})")
print(f"Average RMSE: {rmse_scores.mean():.2f} (+/- {rmse_scores.std() * 2:.2f})")

# Train final model on full training set for predictions
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
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

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual prices")
plt.ylabel("Predicted prices")
plt.title("Actual vs Predicted prices")
plt.grid(True)
plt.tight_layout()
#plt.show()

user_data = pd.read_csv("to_predict.csv")
if "Price" in user_data.columns:
    user_data = user_data.drop(columns=["Price"])

user_data = user_data.select_dtypes(include=[np.number])
user_data = user_data[X_train.columns] 

predicted_prices = model.predict(user_data)
user_data["Predicted Price"] = np.round(predicted_prices,2)
output_file = "predicted_prices.csv"
user_data.to_csv(output_file, index=False)
