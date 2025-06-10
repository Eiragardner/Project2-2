import pandas as pd
import numpy as np
import shap
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from pathlib import Path
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load data
df = pd.read_csv(Path(__file__).parent.parent.parent / "data" / "without30.csv")
df_filtered = df[df["Price"] <= df["Price"].quantile(1)]

X = df_filtered.drop(columns=["Price"])
y = np.log1p(df_filtered["Price"])

# Include numeric + boolean features
numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()

print(f"Using {len(numeric_cols)} numeric features.")

# Preprocessing for numeric features only
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols)
    ]
)
#X = df.drop(columns=["Price"])
#y = np.log1p(df["Price"])  # log(1 + price)
#y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


estimators = [
    ("LR", LinearRegression()), # Possibly change the LinReg model to Ridge?
    ("RF", RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        max_features='sqrt',
        random_state=42
    ))
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
'''
    'max_depth': 6,
    'learning_rate': 0.04641482025331578,
    'subsample': 0.9174351376290615,
    'colsample_bytree': 0.7144730708753544,
    'gamma': 2.9463736411349157,
    'reg_alpha': 0.44507462797741115,
    'reg_lambda': 1.2744197467583638,
    'n_estimators': 499
'''

model = StackingRegressor(
    estimators=estimators,
    final_estimator= xgb.XGBRegressor(**params),
    cv=5,
    n_jobs=-1,
    verbose=0
)

def shap_score(X, y):
    modelscore = xgb.XGBRegressor(n_estimators=100, max_depth=4, random_state=42)
    modelscore.fit(X, y)

    explainer = shap.Explainer(modelscore)
    shap_values = explainer(X)  # returns a matrix of shap values (samples × features)

    # Aggregate absolute SHAP values per feature (mean absolute)
    scores = np.mean(np.abs(shap_values.values), axis=0)

    # Return scores and dummy p-values (required by SelectKBest)
    return scores, np.zeros_like(scores)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('feature_selection', SelectKBest(score_func=shap_score, k=67)),
    ('model', model)
])

#model.fit(X_train, y_train)
pipeline.fit(X_train, y_train)
#y_pred = np.expm1(model.predict(X_test))  # model predict
y_pred = np.expm1(pipeline.predict(X_test))
y_test_original = np.expm1(y_test)
#y_pred = model.predict(X_test)


# Calculate final test set metrics
mae = mean_absolute_error(y_test_original, y_pred)
mse = mean_squared_error(y_test_original, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_original, y_pred)

print("\nFinal test set results:")
print(f"Mean Absolute Error (MAE): {mae:,.2f}")
print(f"Mean Squared Error (MSE): {mse:,.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:,.2f}")
print(f"R-squared: {r2:.2f}")

'''
# Inverse transform for evaluation on actual price scale
def inverse_mae(y_true_log, y_pred_log):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    return mean_absolute_error(y_true, y_pred)

def inverse_rmse(y_true_log, y_pred_log):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    return np.sqrt(mean_squared_error(y_true, y_pred))

def inverse_r2(y_true_log, y_pred_log):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    return r2_score(y_true, y_pred)

# Wrap them as sklearn scorers
custom_mae = make_scorer(inverse_mae, greater_is_better=False)  # negative because lower is better
custom_rmse = make_scorer(inverse_rmse, greater_is_better=False)
custom_r2 = make_scorer(inverse_r2)
'''



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

# Add any missing columns and match the training data
missing_cols = set(X_train.columns) - set(user_data.columns)
for col in missing_cols:
    user_data[col] = 0

user_data = user_data[X_train.columns]

# Predict
predicted_prices = pipeline.predict(user_data)
user_data["Predicted Price"] = np.round(np.expm1(predicted_prices), 2)

output_file = Path(__file__).parent.parent.parent / "outputs" / "predicted_prices_SM.csv"
user_data.to_csv(output_file, index=False)
