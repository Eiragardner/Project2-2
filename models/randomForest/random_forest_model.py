import pandas as pd
import numpy as np
import textwrap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import shap
import os
from pathlib import Path

df = pd.read_csv(Path(__file__).parent.parent.parent / "data" / "without30.csv")

X = df.iloc[:, [0] + list(range(2, df.shape[1]))]  
y = df.iloc[:, 1] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
##NE RABOTAET(((
X_train = X_train.apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float64)
X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0)

print(X_train.dtypes)  

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    max_features='sqrt',
    random_state=42
)
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
plt.show()

importances = model.feature_importances_
features = X.columns

N = 20  
sorted_idx = np.argsort(importances)[-N:]  
importances_top = importances[sorted_idx]
features_top = features[sorted_idx]

plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), importances[sorted_idx], align='center', height=0.8)
plt.yticks(range(len(sorted_idx)), features[sorted_idx], fontsize =10, ha='right')
plt.subplots_adjust(left=0.3)
plt.xlabel("Importance")
plt.title(f"Top {N} Feature Importance")
plt.grid(axis='x', alpha=0.5)
plt.show()


residuals = y_test - y_pred

plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Prices")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted Prices")
plt.grid(True)
plt.tight_layout()
plt.show()

vis_folder = Path(__file__).parent.parent.parent / "outputs" / "visualisations" / "randomForest"
os.makedirs(vis_folder, exist_ok=True)


explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(os.path.join(vis_folder, "shap_summary_bar.png"))
plt.close()

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_train, show=False)
plt.tight_layout()
plt.savefig(os.path.join(vis_folder, "shap_summary.png"))
plt.close()

shap_abs_mean = np.abs(shap_values).mean(axis=0)
top5_idx = np.argsort(shap_abs_mean)[-5:]
for idx in top5_idx:
    feature = X_train.columns[idx]
    plt.figure(figsize=(8, 6))
    shap.dependence_plot(feature, shap_values, X_train, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_folder, f"shap_dependence_{feature}.png"))
    plt.close()

print(f"SHAP plots saved to {vis_folder}")

try:
    user_data = pd.read_csv(Path(__file__).parent.parent.parent / "to_predict.csv")
    user_data = user_data.iloc[:, [0] + list(range(2, user_data.shape[1]))]

    predicted_prices = model.predict(user_data)
    user_data["Predicted Price"] = predicted_prices

    output_file = Path(__file__).parent.parent.parent / "outputs" / "predicted_prices_rf.csv"
    user_data.to_csv(output_file, index=False)
    print(f"Predicted prices saved to {output_file}")
except FileNotFoundError:
    print("No 'to_predict.csv' file found.")