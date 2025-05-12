import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap  # Ensure SHAP is installed
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib
import os
import sys

# All model paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'models', 'xgboost'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'models', 'linearRegression'))  # If you modularize it
sys.path.append(os.path.join(os.path.dirname(__file__), 'models', 'randomForest'))  # If you modularize it

# Importing XGBoost class
try:
    from XGBoost_core import XGBoostModel
except ImportError as e:
    st.error(
        f"Failed to import XGBoost_core: {e}. Ensure the file is in 'models/xgboost/' and all its dependencies are installed.")
    XGBoostModel = None

# These functions return Matplotlib figure objects
def plot_actual_vs_predicted(y_true, y_pred, model_name="Model"):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors='w', linewidth=0.5)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '--k', lw=2, label='Ideal Fit')
    ax.set_xlabel('Actual Prices', fontsize=12)
    ax.set_ylabel('Predicted Prices', fontsize=12)
    ax.set_title(f'Actual vs. Predicted Prices - {model_name}', fontsize=14)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    fig.tight_layout()
    return fig


def plot_residuals(y_true, y_pred, model_name="Model"):
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(residuals, kde=True, ax=ax, color='skyblue', edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', lw=2)
    ax.set_xlabel('Residuals (Actual - Predicted)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Residual Plot - {model_name}', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    fig.tight_layout()
    return fig


def plot_feature_importance_generic(importances, feature_names, model_name="Model", top_k=20):
    if importances is None or feature_names is None:
        return None
    indices = np.argsort(importances)[::-1]
    top_k = min(len(importances), top_k)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title(f"Top {top_k} Feature Importances - {model_name}", fontsize=14)
    ax.bar(range(top_k), importances[indices][:top_k], color='lightgreen', edgecolor='black', align="center")
    ax.set_xticks(range(top_k))
    ax.set_xticklabels(np.array(feature_names)[indices][:top_k], rotation=45, ha="right", fontsize=10)
    ax.set_ylabel('Importance Score', fontsize=12)
    ax.set_xlabel('Features', fontsize=12)
    ax.set_xlim([-1, top_k])
    fig.tight_layout()
    return fig


def plot_shap_summary_generic(shap_values, X_data_df, model_name="Model"):
    if shap_values is None or X_data_df is None:
        return None
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X_data_df, feature_names=X_data_df.columns, show=False, plot_size=None)
        current_fig = plt.gcf()
        current_fig.suptitle(f"SHAP Summary Plot - {model_name}",
                             fontsize=14)
        current_fig.tight_layout(rect=[0, 0, 1, 0.96])
        return current_fig
    except Exception as e:
        st.warning(f"Could not generate SHAP plot for {model_name}: {e}")
        return None



@st.cache_data  # Cache data improves performance
def load_and_prepare_data(uploaded_file_or_path, target_column='price', is_prediction=False):
    if uploaded_file_or_path is None:
        return None, None, None, None

    try:
        df = pd.read_csv(uploaded_file_or_path)
        df_original_for_prediction = df.copy()  # Keep original for joining predictions

        if is_prediction:  # For prediction, target column might not exist or is not used
            X = df.copy()
            y = None
        elif target_column not in df.columns:
            st.error(f"Target column '{target_column}' not found in the uploaded file.")
            return None, None, None, None
        else:
            X = df.drop(target_column, axis=1)
            y = df[target_column]

        # Basic preprocessing: convert to numeric, simple imputation
        feature_names = list(X.columns)
        for col in feature_names:
            X[col] = pd.to_numeric(X[col], errors='coerce')

        imputation_values = X.mean()
        X = X.fillna(imputation_values)

        if y is not None and y.isnull().any():
            st.warning("Target column contains NaN values. Rows with NaN target have been dropped for training.")
            valid_indices = ~y.isnull()
            X = X[valid_indices]
            y = y[valid_indices]
            df_original_for_prediction = df_original_for_prediction[valid_indices]

        if X.empty or (y is not None and y.empty and not is_prediction):
            st.error("Data is empty after preprocessing or NaN handling.")
            return None, None, None, None

        return X, y, list(X.columns), imputation_values if not is_prediction else df_original_for_prediction

    except Exception as e:
        st.error(f"Error loading or preparing data: {e}")
        return None, None, None, None

# XGBoost
def run_xgboost_analysis(X_train_df, y_train_series, X_test_df, y_test_series, load_pretrained=False,
                         model_json_path=None):
    if XGBoostModel is None:
        st.error("XGBoostModel class could not be imported. XGBoost analysis cannot proceed.")
        return None, {}, None, None, None, None

    # Initialize
    # Pass None for data_path as we are providing data directly
    # Ensure results_path is valid
    xgb_model_instance = XGBoostModel(data_path=None, model_path=model_json_path,
                                      results_path='models/xgboost/visualizations_gui')
    os.makedirs('models/xgboost/visualizations_gui', exist_ok=True)  # Ensure dir exists if used by class

    if load_pretrained:
        if model_json_path and os.path.exists(model_json_path):
            st.write(f"Loading pre-trained XGBoost model from: {model_json_path}")
            xgb_model_instance.load_model(model_json_path)  # Call your class's load_model method
            if xgb_model_instance.model is None:
                st.error("Failed to load the pre-trained XGBoost model (model attribute is None).")
                return None, {}, None, None, None, None
            st.success("Pre-trained XGBoost model loaded.")

            xgb_model_instance.feature_names = list(X_test_df.columns)
        else:
            st.error(f"Pre-trained XGBoost model not found at: {model_json_path}")
            return None, {}, None, None, None, None
    else:
        st.write("Training new XGBoost model...")

        xgb_model_instance.X_train = X_train_df
        xgb_model_instance.y_train = y_train_series
        xgb_model_instance.X_test = X_test_df
        xgb_model_instance.y_test = y_test_series
        xgb_model_instance.feature_names = list(X_train_df.columns)
        xgb_model_instance.train_model(
            params={'objective': 'reg:squarederror', 'eval_metric': 'rmse'})  # Pass any default params
        st.success("XGBoost training complete.")

    # Evaluation
    metrics = xgb_model_instance.evaluate(X_test_df, y_test_series)  # Ensure this uses X_test_df, y_test_series
    y_pred_test = xgb_model_instance.predict(X_test_df)  # Ensure this uses X_test_df

    # Plots
    fig_actual_vs_pred = xgb_model_instance.plot_actual_vs_predicted(X_test_df, y_test_series, save_plot=False)
    fig_residuals = xgb_model_instance.plot_residuals(X_test_df, y_test_series, save_plot=False)
    fig_feat_imp = xgb_model_instance.plot_feature_importance(save_plot=False)

    fig_shap = xgb_model_instance.plot_shap_summary(X_train_df, save_plot=False)

    return xgb_model_instance, metrics, fig_actual_vs_pred, fig_residuals, fig_feat_imp, fig_shap


# Linear Regression
def run_linear_regression_analysis(X_train, y_train, X_test, y_test, feature_names):
    st.write("Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    st.success("Linear Regression training complete.")

    y_pred_test = model.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred_test),
        "MSE": mean_squared_error(y_test, y_pred_test),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_test)),
        "R2 Score": r2_score(y_test, y_pred_test)
    }
    fig_actual_vs_pred = plot_actual_vs_predicted(y_test, y_pred_test, "Linear Regression")
    fig_residuals = plot_residuals(y_test, y_pred_test, "Linear Regression")

    # Feature importance (coefficients for LR)
    try:
        coeffs = model.coef_
        fig_coeffs = plot_feature_importance_generic(np.abs(coeffs), feature_names, "Linear Regression (Coeff. Mag.)")
    except Exception as e:
        st.warning(f"Could not plot coefficients for Linear Regression: {e}")
        fig_coeffs = None

    return model, metrics, fig_actual_vs_pred, fig_residuals, fig_coeffs, None  # No SHAP for basic LR here


# Random Forest
def run_random_forest_analysis(X_train_df, y_train_series, X_test_df, y_test_series):
    st.write("Training Random Forest model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, oob_score=True)
    model.fit(X_train_df, y_train_series)
    st.success("Random Forest training complete.")

    y_pred_test = model.predict(X_test_df)
    metrics = {
        "MAE": mean_absolute_error(y_test_series, y_pred_test),
        "MSE": mean_squared_error(y_test_series, y_pred_test),
        "RMSE": np.sqrt(mean_squared_error(y_test_series, y_pred_test)),
        "R2 Score": r2_score(y_test_series, y_pred_test),
        "OOB Score": model.oob_score_ if hasattr(model, 'oob_score_') else 'N/A'
    }
    fig_actual_vs_pred = plot_actual_vs_predicted(y_test_series, y_pred_test, "Random Forest")
    fig_residuals = plot_residuals(y_test_series, y_pred_test, "Random Forest")
    fig_feat_imp = plot_feature_importance_generic(model.feature_importances_, X_train_df.columns, "Random Forest")

    # SHAP for Random Forest
    try:
        explainer_rf = shap.TreeExplainer(model, X_train_df)  # Background data for explainer
        shap_values_rf = explainer_rf.shap_values(X_test_df)  # Explain predictions on test set
        fig_shap = plot_shap_summary_generic(shap_values_rf, X_test_df, "Random Forest")
    except Exception as e:
        st.warning(f"Could not generate SHAP plot for Random Forest: {e}")
        fig_shap = None

    return model, metrics, fig_actual_vs_pred, fig_residuals, fig_feat_imp, fig_shap


# APP UI - Streamlit
st.set_page_config(layout="wide", page_title="House Market ML Analyzer")
st.title("🏘️ House Market Machine Learning Model Analyzer")

# --- Sidebar for Controls ---
with st.sidebar:
    st.header("⚙️ Model & Data Configuration")

    model_choice = st.selectbox("Choose Model:", ["XGBoost", "Random Forest", "Linear Regression"])

    st.subheader("Training Data")
    default_train_path = os.path.join(os.path.dirname(__file__), 'data', 'prepared_data.csv')  # Example default

    uploaded_train_file = st.file_uploader("Upload Training Data (CSV)", type="csv", key="train_uploader")

    use_default_train = st.checkbox("Use default 'prepared_data.csv'", value=True, key="default_train_check")

    train_file_source = None
    if uploaded_train_file is not None:
        train_file_source = uploaded_train_file
        st.info("Using uploaded training file.")
    elif use_default_train:
        if os.path.exists(default_train_path):
            train_file_source = default_train_path
            st.info(f"Using default training data: {default_train_path}")
        else:
            st.warning(f"Default training data '{default_train_path}' not found. Please upload a file.")
    else:
        st.info("Please upload a training CSV or select to use the default.")

    target_column = st.text_input("Target Column Name:", "Price", key="target_col_input")

    load_pretrained_xgboost = False
    if model_choice == "XGBoost":
        load_pretrained_xgboost = st.checkbox("Load Pre-trained XGBoost Model ('xgboost_model.json')",
                                              key="load_xgb_check")

    run_analysis_button = st.button("🚀 Run Analysis / Train Model", key="run_button")

    st.sidebar.markdown("---")
    st.subheader("Prediction on New Data")
    default_predict_path = os.path.join(os.path.dirname(__file__), 'to_predict.csv')  # Example default
    uploaded_predict_file = st.file_uploader("Upload Data for Prediction (CSV)", type="csv", key="predict_uploader")

    use_default_predict = st.checkbox("Use default 'to_predict.csv'", value=False, key="default_predict_check")

    predict_file_source = None
    if uploaded_predict_file is not None:
        predict_file_source = uploaded_predict_file
        st.info("Using uploaded file for prediction.")
    elif use_default_predict:
        if os.path.exists(default_predict_path):
            predict_file_source = default_predict_path
            st.info(f"Using default prediction data: {default_predict_path}")
        else:
            st.warning(f"Default prediction data '{default_predict_path}' not found.")

    predict_button = st.button("🔮 Make Predictions", key="predict_action_button")

if 'model_trained' not in st.session_state:
    st.session_state.model_trained = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = {}
if 'fig_actual_vs_pred' not in st.session_state:
    st.session_state.fig_actual_vs_pred = None
if 'fig_residuals' not in st.session_state:
    st.session_state.fig_residuals = None
if 'fig_feature_importance' not in st.session_state:
    st.session_state.fig_feature_importance = None
if 'fig_shap' not in st.session_state:
    st.session_state.fig_shap = None
if 'trained_feature_names' not in st.session_state:
    st.session_state.trained_feature_names = None
if 'imputation_values_trained' not in st.session_state:
    st.session_state.imputation_values_trained = None

if run_analysis_button:
    if train_file_source and target_column:
        st.session_state.model_trained = None
        st.spinner("Loading and preparing data...")
        X, y, feature_names, imputation_vals = load_and_prepare_data(train_file_source, target_column)

        if X is not None and y is not None:
            st.session_state.trained_feature_names = feature_names
            st.session_state.imputation_values_trained = imputation_vals

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            X_train_df = pd.DataFrame(X_train, columns=feature_names)
            X_test_df = pd.DataFrame(X_test, columns=feature_names)
            y_train_series = pd.Series(y_train, name=target_column)
            y_test_series = pd.Series(y_test, name=target_column)

            with st.spinner(f"Processing {model_choice}... This may take a moment."):
                if model_choice == "XGBoost":
                    model_path = os.path.join(os.path.dirname(__file__),
                                              'xgboost_model.json') if load_pretrained_xgboost else None
                    model_obj, metrics, fig_avp, fig_res, fig_fi, fig_s = run_xgboost_analysis(
                        X_train_df, y_train_series, X_test_df, y_test_series,
                        load_pretrained=load_pretrained_xgboost, model_json_path=model_path
                    )
                elif model_choice == "Random Forest":
                    model_obj, metrics, fig_avp, fig_res, fig_fi, fig_s = run_random_forest_analysis(
                        X_train_df, y_train_series, X_test_df, y_test_series
                    )
                elif model_choice == "Linear Regression":
                    model_obj, metrics, fig_avp, fig_res, fig_fi, fig_s = run_linear_regression_analysis(
                        X_train_df, y_train_series, X_test_df, y_test_series, feature_names
                    )
                else:
                    st.error("Invalid model choice selected.")
                    st.stop()


                st.session_state.model_trained = model_obj
                st.session_state.metrics = metrics
                st.session_state.fig_actual_vs_pred = fig_avp
                st.session_state.fig_residuals = fig_res
                st.session_state.fig_feature_importance = fig_fi
                st.session_state.fig_shap = fig_s
            st.success(f"{model_choice} analysis complete!")
        else:
            st.error("Failed to load or prepare data. Cannot proceed with analysis.")
    else:
        st.warning("Please select a training data file and specify the target column.")


if st.session_state.model_trained is not None or st.session_state.metrics:
    st.header(f"📊 Analysis Results for {model_choice}")

    tab_metrics, tab_plots = st.tabs(["Performance Metrics", "Visualizations"])

    with tab_metrics:
        if st.session_state.metrics:
            st.subheader("Evaluation Metrics")
            for metric_name, value in st.session_state.metrics.items():
                if isinstance(value, (float, np.floating)):
                    st.metric(label=metric_name, value=f"{value:.4f}")
                else:
                    st.metric(label=metric_name, value=value)
        else:
            st.info("Metrics will be shown here after running an analysis.")

    with tab_plots:
        st.subheader("Plots")
        col1, col2 = st.columns(2)
        with col1:
            if st.session_state.fig_actual_vs_pred:
                st.pyplot(st.session_state.fig_actual_vs_pred)
            else:
                st.caption("Actual vs. Predicted plot will appear here.")

            if st.session_state.fig_feature_importance:
                st.pyplot(st.session_state.fig_feature_importance)
            else:
                st.caption("Feature Importance plot will appear here.")

        with col2:
            if st.session_state.fig_residuals:
                st.pyplot(st.session_state.fig_residuals)
            else:
                st.caption("Residuals plot will appear here.")

            if st.session_state.fig_shap:
                st.pyplot(st.session_state.fig_shap)
            elif model_choice in ["Random Forest", "XGBoost"]:
                st.caption("SHAP Summary plot will appear here (if applicable).")

# Prediction Tab
if predict_button:
    if predict_file_source:
        if st.session_state.model_trained is not None:
            with st.spinner("Loading prediction data and making predictions..."):
                X_pred_processed, _, _, df_original_for_pred = load_and_prepare_data(
                    predict_file_source,
                    target_column=None,
                    is_prediction=True
                )

                if X_pred_processed is not None:
                    if st.session_state.trained_feature_names and st.session_state.imputation_values_trained is not None:
                        X_pred_aligned = pd.DataFrame(columns=st.session_state.trained_feature_names)
                        for col in st.session_state.trained_feature_names:
                            if col in X_pred_processed.columns:
                                X_pred_aligned[col] = X_pred_processed[col]
                            else:
                                X_pred_aligned[col] = np.nan

                        X_pred_aligned = X_pred_aligned.fillna(st.session_state.imputation_values_trained)

                        if X_pred_aligned.isnull().values.any():
                            st.warning(
                                "NaN values found in prediction data after aligning and imputing with training data means. Filling remaining NaNs with 0.")
                            X_pred_aligned = X_pred_aligned.fillna(0)

                    else:
                        st.error(
                            "Training feature names or imputation values not found from a previous training run. Cannot reliably align prediction data.")
                        st.stop()

                    predictions_output = None
                    current_model_object = st.session_state.model_trained

                    try:
                        if model_choice == "XGBoost" and XGBoostModel is not None and isinstance(current_model_object,
                                                                                                 XGBoostModel):
                            predictions_output = current_model_object.predict(X_pred_aligned)
                        elif hasattr(current_model_object, 'predict'):
                            predictions_output = current_model_object.predict(X_pred_aligned)
                        else:
                            st.error(
                                f"The stored model for {model_choice} is not a recognized type for prediction or does not have a 'predict' method.")
                    except Exception as e:
                        st.error(f"Error during prediction: {e}")
                        st.error(
                            "Ensure the prediction data format matches the training data format (after preprocessing).")

                    if predictions_output is not None:
                        st.header("🔮 Prediction Results")

                        results_df = df_original_for_pred.copy()

                        final_prediction_df = X_pred_aligned.copy()
                        final_prediction_df['predicted_price'] = predictions_output

                        st.dataframe(final_prediction_df.head(100))


                        @st.cache_data
                        def convert_df_to_csv(df_to_convert):
                            return df_to_convert.to_csv(index=False).encode('utf-8')


                        csv_predictions = convert_df_to_csv(final_prediction_df)
                        st.download_button(
                            label="📥 Download Predictions as CSV",
                            data=csv_predictions,
                            file_name=f"predicted_prices_{model_choice.lower().replace(' ', '_')}.csv",
                            mime="text/csv",
                        )
                        st.success("Predictions generated and available for download.")
                    else:
                        st.error("Could not generate predictions.")
                else:
                    st.error("Failed to load or prepare prediction data.")
        else:
            st.warning("Please train or load a model first before making predictions.")
    else:
        st.warning("Please upload a file for prediction or select the default prediction file.")

st.sidebar.markdown("---")
st.sidebar.info(
    "Ensure CSVs are clean. Preprocessing in this GUI is basic. For best results, use data preprocessed similarly to your model's original training.")
st.sidebar.markdown(f"Last Refreshed: {pd.Timestamp.now(tz='Europe/Amsterdam').strftime('%Y-%m-%d %H:%M:%S %Z')}")