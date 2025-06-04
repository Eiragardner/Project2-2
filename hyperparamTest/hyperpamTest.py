from math import sqrt
from pathlib import Path
import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, filename='optuna_xgb_search.log', filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load data
df = pd.read_csv(Path(__file__).parent.parent / "data" / "without30.csv")
df = df.select_dtypes(include=[np.number])
price_threshold = df["Price"].quantile(0.90)
df_filtered = df[df["Price"] <= price_threshold]

X = df_filtered.drop(columns=["Price"])
y = df_filtered["Price"]

#X = df.drop(columns=["Price"])
#y = df["Price"]

# Split train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to DMatrix format (required by xgb.train)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)


def objective(trial):
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'device': 'cuda',
        'tree_method': 'hist',  # GPU support
        #'predictor': 'gpu_predictor',
        'verbosity': 0,
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 2.0),
    }

    num_boost_round = trial.suggest_int('n_estimators', 100, 500)

    evals = [(dtest, 'validation')]

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=20,
        verbose_eval=False,
    )

    preds = model.predict(dtest, iteration_range=(0, model.best_iteration + 1))
    mse = mean_squared_error(y_test, preds)
    rmse = sqrt(mse)
    return rmse


# Run the Optuna study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100, timeout=18000)  # 5000 trials or 60 minutes

# Best result
print("Best RMSE:", study.best_value)
print("Best hyperparameters:", study.best_params)

# Log to file
logging.info(f"Best RMSE: {study.best_value}")
logging.info(f"Best Hyperparameters: {study.best_params}")

# Save the study results
study.trials_dataframe().to_csv("optuna_xgb_trials.csv", index=False)
