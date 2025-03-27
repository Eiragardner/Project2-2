# Project2-2
For our prototype, we implemented a **Linear Regression** model to predict property prices.

How to Run:
1. Navigate to the `main/` directory.
2. Run the `prototype.py` script:

The script will:

    Train a linear regression model using prepared_data.csv

    Evaluate the model's performance (MAE, MSE, RMSE, R²)

    Load new data from to_predict.csv

    Generate predictions and save them to predicted_prices.csv

Project Overview

The Dutch housing market suffers from issues like overvaluation and inconsistent appraisals. Our goal is to build a reliable and explainable machine learning model that can predict property prices using features like living space, location, year built, and more.

We compare three ML models:

Linear Regression (baseline)

Random Forest

XGBoost

We also apply SHAP (SHapley Additive exPlanations) to interpret feature importance.
