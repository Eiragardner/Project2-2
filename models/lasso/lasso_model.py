import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

class LassoModel:
    """
    Upgraded LassoModel:
      - Allows either pure LassoCV or ElasticNetCV for feature selection.
      - Uses a custom alpha grid for Lasso or ElasticNet.
      - After selecting features, trains a RandomForestRegressor on the reduced feature set.
      - Computes performance metrics (MAE, MSE, RMSE, R^2).
      - Produces and saves three plots:
          1. Bar chart of |coefficients|
          2. Actual vs Predicted (RF on reduced features)
          3. Residuals histogram (RF on reduced features)
    """

    def __init__(
        self,
        visualizations_path=None,
        random_state=42,
        lasso_alphas=None,
        elasticnet_alphas=None,
        elasticnet_l1_ratio=None,
        use_elasticnet=False
    ):
        """
        :param visualizations_path: Directory where plots will be saved.
        :param random_state: Random seed for reproducibility.
        :param lasso_alphas: List of alpha values for LassoCV. If None, defaults to [0.1, 0.01, 0.001, 0.0001].
        :param elasticnet_alphas: List of alpha values for ElasticNetCV. If None, defaults to [0.1, 0.01, 0.001].
        :param elasticnet_l1_ratio: List of l1_ratio values for ElasticNetCV. If None, defaults to [0.5, 0.7, 0.9, 1.0].
        :param use_elasticnet: If True, use ElasticNetCV instead of LassoCV for feature selection.
        """
        self.visualizations_path = visualizations_path
        self.random_state = random_state
        self.use_elasticnet = use_elasticnet

        self.lasso_alphas = lasso_alphas or [0.1, 0.01, 0.001, 0.0001]
        self.elasticnet_alphas = elasticnet_alphas or [0.1, 0.01, 0.001]
        self.elasticnet_l1_ratio = elasticnet_l1_ratio or [0.5, 0.7, 0.9, 1.0]

        self.selector = None              
        self.selected_features_ = []      
        self.feature_names_ = None         

        self.rf_model_ = None
        self.downstream_metrics_ = {}     

        self.fig_coef_ = None
        self.fig_avp_ = None
        self.fig_res_ = None

        if self.visualizations_path:
            os.makedirs(self.visualizations_path, exist_ok=True)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, feature_names: list):
        """
        1) Performs feature selection via LassoCV or ElasticNetCV on (X_train, y_train).
        2) Extracts non-zero coefficients and stores (feature_name, coef) pairs.
        3) Builds and saves a bar chart of |coefficients|.
        """
        self.feature_names_ = feature_names.copy()

        X_train_np = X_train.to_numpy(dtype=np.float64)
        y_train_np = y_train.to_numpy(dtype=np.float64)

        if self.use_elasticnet:
            self.selector = ElasticNetCV(
                alphas=self.elasticnet_alphas,
                l1_ratio=self.elasticnet_l1_ratio,
                cv=5,
                tol=1e-3, 
                max_iter=5000,
                random_state=self.random_state,
                n_jobs=-1
            )
            self.selector.fit(X_train_np, y_train_np)
            coefs = self.selector.coef_
        else:
            self.selector = LassoCV(
                alphas=self.lasso_alphas,
                cv=5,
                random_state=self.random_state,
                n_jobs=-1
            )
            self.selector.fit(X_train_np, y_train_np)
            coefs = self.selector.coef_

        selected_idx = [i for i, c in enumerate(coefs) if abs(c) > 1e-6]
        self.selected_features_ = [
            (feature_names[i], coefs[i]) for i in selected_idx
        ]
        self.selected_features_.sort(key=lambda x: abs(x[1]), reverse=True)

        self._make_and_save_coef_plot(coefs)

        return self

    def _make_and_save_coef_plot(self, coefs: np.ndarray, top_k: int = 20):
        """
        Internal helper: create and save a bar chart of the top_k absolute coefficients.
        """
        abs_coefs = np.abs(coefs)
        indices = np.argsort(abs_coefs)[::-1]
        top_k = min(len(abs_coefs), top_k)
        top_idx = indices[:top_k]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(
            range(top_k),
            abs_coefs[top_idx],
            color='slateblue',
            edgecolor='black',
            align='center'
        )
        ax.set_xticks(range(top_k))
        ax.set_xticklabels(
            np.array(self.feature_names_)[top_idx],
            rotation=45,
            ha='right',
            fontsize=10
        )
        title = "Top {} |ElasticNetCV Coefficients|".format(top_k) if self.use_elasticnet \
                else "Top {} |LassoCV Coefficients|".format(top_k)
        ax.set_title(title, fontsize=14)
        ax.set_ylabel("|Coefficient|", fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        fig.tight_layout()

        self.fig_coef_ = fig
        if self.visualizations_path:
            filename = "elasticnet_coef_bar.png" if self.use_elasticnet else "lasso_coef_bar.png"
            save_path = os.path.join(self.visualizations_path, filename)
            fig.savefig(save_path)
            plt.close(fig)

    def train_reduced_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        rf_params: dict = None
    ):
        """
        1) Trains a RandomForestRegressor on only the selected features from fit().
        2) Computes downstream performance metrics.
        3) Builds and saves:
            - Actual vs Predicted plot
            - Residuals histogram
        :param rf_params: Optional dictionary of hyperparameters for RandomForestRegressor.
        """
        if not self.selected_features_:
            self.downstream_metrics_ = {}
            self.fig_avp_ = None
            self.fig_res_ = None
            return self

        selected_names = [f for f, c in self.selected_features_]
        selected_idx = [self.feature_names_.index(f) for f in selected_names]

        X_train_red = X_train.iloc[:, selected_idx]
        X_test_red = X_test.iloc[:, selected_idx]

        params = rf_params or {
            "n_estimators": 100,
            "random_state": self.random_state,
            "n_jobs": -1
        }
        rf = RandomForestRegressor(**params)
        rf.fit(X_train_red, y_train)
        self.rf_model_ = rf

        y_pred = rf.predict(X_test_red)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        self.downstream_metrics_ = {
            "RF on Reduced MAE": mae,
            "RF on Reduced MSE": mse,
            "RF on Reduced RMSE": rmse,
            "RF on Reduced R2": r2
        }

        self._make_and_save_avp_plot(y_test, y_pred)

        self._make_and_save_residuals_plot(y_test, y_pred)

        return self

    def _make_and_save_avp_plot(self, y_true: pd.Series, y_pred: np.ndarray):
        """
        Internal helper: create and save an Actual vs Predicted scatter plot.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_true, y_pred, alpha=0.6, edgecolors='w', linewidth=0.5)
        ax.plot(
            [y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()],
            '--k',
            lw=2,
            label='Ideal Fit'
        )
        ax.set_xlabel("Actual Prices", fontsize=12)
        ax.set_ylabel("Predicted Prices", fontsize=12)
        ax.set_title("Actual vs Predicted (RF on Reduced Set)", fontsize=14)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        fig.tight_layout()

        self.fig_avp_ = fig
        if self.visualizations_path:
            save_path = os.path.join(self.visualizations_path, "reduced_rf_avp.png")
            fig.savefig(save_path)
            plt.close(fig)

    def _make_and_save_residuals_plot(self, y_true: pd.Series, y_pred: np.ndarray):
        """
        Internal helper: create and save a Residuals histogram (Actual - Predicted).
        """
        residuals = y_true - y_pred
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(residuals, kde=True, ax=ax, color='lightcoral', edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', lw=2)
        ax.set_xlabel("Residuals (Actual - Predicted)", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title("Residuals (RF on Reduced Set)", fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        fig.tight_layout()

        self.fig_res_ = fig
        if self.visualizations_path:
            save_path = os.path.join(self.visualizations_path, "reduced_rf_residuals.png")
            fig.savefig(save_path)
            plt.close(fig)

    def get_selected_features(self) -> list:
        """
        Returns a list of tuples: (feature_name, coefficient) for non-zero selector coefs.
        """
        return self.selected_features_

    def get_downstream_metrics(self) -> dict:
        """
        Returns the metrics dictionary for the RandomForest trained on reduced feature set.
        """
        return self.downstream_metrics_

    def get_plots(self):
        """
        Returns a tuple of matplotlib Figure objects:
          (fig_coef_, fig_avp_, fig_res_)
        """
        return (self.fig_coef_, self.fig_avp_, self.fig_res_)


if __name__ == "__main__":
    here = os.path.dirname(__file__) 
    project_root = os.path.abspath(os.path.join(here, os.pardir, os.pardir))
    data_path = os.path.join(project_root, "data", "without30.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Could not find {data_path}")

    df = pd.read_csv(data_path)
    X = df.iloc[:, [0] + list(range(2, df.shape[1]))]  
    y = df.iloc[:, 1] 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = X_train.apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float64)
    X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float64)

    feature_names = list(X.columns)

    vis_folder = os.path.join(here, "visualizations")
    os.makedirs(vis_folder, exist_ok=True)

    # 6) Instantiate and fit LassoModel (use_elasticnet=False for pure Lasso; set True to try ElasticNet)
    print("Running upgraded Lasso‐based feature selection…")
    lasso_fs = LassoModel(
        visualizations_path=vis_folder,
        random_state=42,
        lasso_alphas=[1, 0.1, 0.01, 0.001, 0.0001],
        elasticnet_alphas=[0.1, 0.01, 0.001],
        elasticnet_l1_ratio=[0.5, 0.7, 0.9, 1.0],
        use_elasticnet=True   # Change to True to use ElasticNetCV instead
    )
    lasso_fs.fit(X_train, y_train, feature_names)

    lasso_fs.train_reduced_model(X_train, y_train, X_test, y_test)

    selected = lasso_fs.get_selected_features()
    if selected:
        print("\nSelected Features (non-zero coefficients):")
        for feat, coef in selected:
            print(f"  • {feat:<25} → {coef: .4f}")
    else:
        print("\nSelector chose 0 features (all coefficients zero).")

    downstream_metrics = lasso_fs.get_downstream_metrics()
    if downstream_metrics:
        print("\nRandomForest on reduced feature set metrics:")
        for name, val in downstream_metrics.items():
            print(f"  {name:<25}: {val:.4f}")
    else:
        print("\nNo downstream metrics (perhaps selector chose zero features).")

    print(f"\nPlots saved to folder:\n  {vis_folder}")
    if lasso_fs.use_elasticnet:
        print("  • elasticnet_coef_bar.png")
    else:
        print("  • lasso_coef_bar.png")
    print("  • reduced_rf_avp.png")
    print("  • reduced_rf_residuals.png")

    print("\nDone.")