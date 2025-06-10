import os

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from pathlib import Path
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import mutual_info_regression
from xgboost import XGBRegressor
import shap

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


model = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    max_features='sqrt',
    random_state=42
)

# Use inverse RMSE on the original price scale
def inverse_rmse(y_true_log, y_pred_log):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    return np.sqrt(mean_squared_error(y_true, y_pred))

def shap_score(X, y):
    modelscore = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        max_features='sqrt',
        random_state=42
    )
    modelscore.fit(X, y)

    explainer = shap.TreeExplainer(modelscore)
    shap_values = explainer(X)  # returns a matrix of shap values (samples × features)

    # Aggregate absolute SHAP values per feature (mean absolute)
    scores = np.mean(np.abs(shap_values.values), axis=0)

    # Return scores and dummy p-values (required by SelectKBest)
    return scores, np.zeros_like(scores)

rmse_scorer = make_scorer(inverse_rmse, greater_is_better=False)

# Define pipeline without fixed k
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('feature_selection', SelectKBest(shap_score)),  # default k, overridden by grid
    ('model', model)
])

# Grid search over k values
param_grid = {
    'feature_selection__k': list(range(10, 31, 1))
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring=rmse_scorer,
    verbose=2,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# Extract k values and corresponding (negative) RMSE scores
k_values = grid_search.cv_results_['param_feature_selection__k'].data
mean_test_scores = -grid_search.cv_results_['mean_test_score']  # Make RMSE positive

# Plotting
plt.figure(figsize=(10, 6))
plt.xticks(np.arange(0, 31, 1))
plt.plot(k_values, mean_test_scores, marker='o', linestyle='-')
plt.xlabel("Number of Selected Features (k)")
plt.ylabel("Cross-Validated RMSE")
plt.title("SelectKBest Performance vs. Number of Features")
plt.grid(True)
plt.tight_layout()


output_path = Path(__file__).parent.parent.parent / "outputs" / "visualisations" / "featureSelection"

if not os.path.exists(output_path):
    os.makedirs(output_path)

plt.savefig(output_path / "featureSelectionRFPrecise.png")
#plt.show()

print("\nBest parameters:")
print(grid_search.best_params_)

print(f"Best CV RMSE: {-grid_search.best_score_:,.2f}")

# Use best estimator for final test predictions
best_pipeline = grid_search.best_estimator_
y_pred = np.expm1(best_pipeline.predict(X_test))
y_test_original = np.expm1(y_test)

mae = mean_absolute_error(y_test_original, y_pred)
mse = mean_squared_error(y_test_original, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_original, y_pred)

print("\nFinal test set results (best k):")
print(f"MAE: {mae:,.2f}")
print(f"MSE: {mse:,.2f}")
print(f"RMSE: {rmse:,.2f}")
print(f"R²: {r2:.2f}")