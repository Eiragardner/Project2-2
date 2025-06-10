import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

from models.respectiveModels.XGBoost_refactorStreamlit import XGBoostModel
from models.respectiveModels.linear_regression_refactorStreamlit import LinearRegressionModel
from models.respectiveModels.random_forest_refactorStreamlit import RandomForestModel
from models.respectiveModels.stacked_model_refactorStreamlit import StackedModel

st.set_page_config(layout="wide", page_title="House Market ML Analyzer")

@st.cache_data
def load_and_prepare_data(uploaded_file_or_path, target_column='Price', is_prediction=False):
    if uploaded_file_or_path is None:
        return None, None, None, None
    try:
        df = pd.read_csv(uploaded_file_or_path)
        df_original = df.copy()

        if is_prediction:
            X = df.copy()
            y = None
        elif target_column not in df.columns:
            st.error(f"Target column '{target_column}' not found in the uploaded file.")
            return None, None, None, None
        else:
            X = df.drop(target_column, axis=1)
            y = df[target_column]

        X = X.select_dtypes(include=np.number)

        if y is not None and y.isnull().any():
            st.warning("Target column has NaNs. Rows with NaN target are dropped.")
            valid_indices = ~y.isnull()
            X, y, df_original = X[valid_indices], y[valid_indices], df_original[valid_indices]

        if X.empty:
            st.error("Data is empty after preprocessing.")
            return None, None, None, None

        return X, y, list(X.columns), df_original if is_prediction else None
    except Exception as e:
        st.error(f"Error loading or preparing data: {e}")
        return None, None, None, None


# Model factory and analysis

def get_model_instance(model_name):
    if model_name == "XGBoost":
        return XGBoostModel(), 'models/xgboost/xgboost_model.json'
    elif model_name == "Random Forest":
        return RandomForestModel(), 'models/randomForest/random_forest_model.joblib'
    elif model_name == "Linear Regression":
        return LinearRegressionModel(), 'models/linearRegression/linear_regression_model.joblib'
    elif model_name == "Stacked Model":
        return StackedModel(), 'models/stacked/stacked_model.joblib'
    else:
        return None, None


def run_analysis(model_choice, X_train, y_train, X_test, y_test, load_pretrained):
    model, model_path = get_model_instance(model_choice)
    if not model:
        st.error("Invalid model choice.")
        return [None] * 6

    # Load or train the model
    if load_pretrained:
        if os.path.exists(model_path):
            st.write(f"Loading pre-trained {model.model_name} model from: {model_path}")
            model.load_model(model_path)
            st.success("Model loaded.")
        else:
            st.error(f"Pre-trained model not found at: {model_path}")
            return [None] * 6
    else:
        st.write(f"Training new {model.model_name} model...")
        model.train(X_train, y_train)
        st.success("Training complete.")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        # Pass the correct path to the model's save method
        model.save_model(model_path)
        st.info(f"Trained model saved to {model_path}")

    model_features = model.feature_names
    if model_features is None:
        st.error(f"Model {model.model_name} lacks feature name information. Cannot proceed.")
        return [None] * 6

    imputation_series = pd.Series(st.session_state.imputation_vals)
    X_test_aligned = X_test.reindex(columns=model_features).fillna(imputation_series).fillna(0)
    X_train_aligned = X_train.reindex(columns=model_features).fillna(imputation_series).fillna(0)

    y_pred = model.predict(X_test_aligned)
    metrics = model.evaluate(y_test, y_pred)
    fig_avp = model.plot_actual_vs_predicted(y_test, y_pred)
    fig_res = model.plot_residuals(y_test, y_pred)
    fig_fi = model.plot_feature_importance()
    fig_s = model.plot_shap_summary(X_train_aligned, X_test_aligned)

    return model, metrics, fig_avp, fig_res, fig_fi, fig_s


def style_comparison_df(df):
    styled_df = df.style.highlight_min(subset=pd.IndexSlice[['MAE', 'RMSE'], :], color='lightgreen', axis=1)
    styled_df = styled_df.highlight_max(subset=pd.IndexSlice[['R2'], :], color='lightgreen', axis=1)
    return styled_df


# App ui
st.title("🏘️ House Market Machine Learning Model Analyzer")

# Initialize session state keys
for key in ['model', 'metrics', 'fig_avp', 'fig_res', 'fig_fi', 'fig_s', 'imputation_vals', 'comparison_results']:
    if key not in st.session_state:
        st.session_state[key] = None

with st.sidebar:
    st.header("⚙️ Configuration")

    analysis_mode = st.radio("Select Mode", ["Single Model Analysis", "Multi-Model Comparison"], key="mode")

    if analysis_mode == "Single Model Analysis":
        model_choice = st.selectbox("Choose Model:", ["XGBoost", "Random Forest", "Linear Regression", "Stacked Model"])
        load_pretrained = st.checkbox(f"Load Pre-trained {model_choice} Model")
    else:  # Multi-Model Comparison
        model_choices = st.multiselect("Choose Models to Compare:",
                                       ["XGBoost", "Random Forest", "Linear Regression", "Stacked Model"],
                                       default=["XGBoost", "Random Forest"])
        st.info("Comparison mode always trains new models; pre-trained models are not used.")
        load_pretrained = False

    st.subheader("Training Data")
    default_train_path = os.path.join('data', 'without30.csv')
    uploaded_train_file = st.file_uploader("Upload Training Data (CSV)", type="csv")
    use_default_train = st.checkbox("Use default training data", value=not uploaded_train_file)
    train_file_source = uploaded_train_file if uploaded_train_file else (
        default_train_path if use_default_train else None)
    target_column = st.text_input("Target Column Name:", "Price")

    run_button = st.button("🚀 Run Analysis")

    st.markdown("---")
    st.subheader("Prediction on New Data")
    st.info("Prediction is only available in 'Single Model Analysis' mode.")
    uploaded_predict_file = st.file_uploader("Upload Prediction Data (CSV)", type="csv", key="predict_uploader",
                                             disabled=(analysis_mode != "Single Model Analysis"))
    predict_button = st.button("🔮 Make Predictions", disabled=(analysis_mode != "Single Model Analysis"))

if run_button:
    st.session_state.comparison_results = None  # Reset comparison results
    st.session_state.model = None  # Reset single model results
    if train_file_source and target_column:
        with st.spinner("Loading and preparing data..."):
            X, y, _, _ = load_and_prepare_data(train_file_source, target_column)
        if X is not None and y is not None:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            imputation_values = X_train.mean()
            st.session_state.imputation_vals = imputation_values.to_dict()
            X_train = X_train.fillna(imputation_values)
            X_test = X_test.fillna(imputation_values)

            if analysis_mode == "Single Model Analysis":
                with st.spinner(f"Processing {model_choice}..."):
                    results = run_analysis(model_choice, X_train, y_train, X_test, y_test, load_pretrained)
                    st.session_state.model, st.session_state.metrics, st.session_state.fig_avp, st.session_state.fig_res, st.session_state.fig_fi, st.session_state.fig_s = results
                if st.session_state.model:
                    st.success(f"{model_choice} analysis complete!")
                else:
                    st.error("Model analysis failed.")

            else:
                if not model_choices:
                    st.warning("Please select at least one model to compare.")
                else:
                    st.session_state.comparison_results = {}
                    progress_bar = st.progress(0, text="Starting comparison...")
                    num_models = len(model_choices)
                    for i, choice in enumerate(model_choices):
                        with st.spinner(f"Training and evaluating {choice}... ({i + 1}/{num_models})"):
                            model, metrics, fig_avp, fig_res, fig_fi, fig_s = run_analysis(choice, X_train, y_train,
                                                                                           X_test, y_test,
                                                                                           load_pretrained=False)
                            if model:
                                st.session_state.comparison_results[choice] = {
                                    "metrics": metrics,
                                    "fig_avp": fig_avp,
                                    "fig_res": fig_res,
                                    "fig_fi": fig_fi,
                                    "fig_s": fig_s,
                                }
                        progress_bar.progress((i + 1) / num_models, text=f"Completed {choice} ({i + 1}/{num_models})")
                    st.success("Model comparison complete!")
        else:
            st.error("Failed to load or prepare data.")
    else:
        st.warning("Please select training data and specify target column.")

# Display Single Model Results
if st.session_state.model and analysis_mode == "Single Model Analysis":
    st.header(f"📊 Analysis Results for {st.session_state.model.model_name}")
    tab_metrics, tab_plots = st.tabs(["Performance Metrics", "Visualizations"])
    with tab_metrics:
        if st.session_state.metrics:
            st.subheader("Evaluation Metrics")
            for name, value in st.session_state.metrics.items():
                st.metric(label=name, value=f"{value:,.4f}")
    with tab_plots:
        st.subheader("Plots")
        col1, col2 = st.columns(2)
        if st.session_state.fig_avp: col1.pyplot(st.session_state.fig_avp)
        if st.session_state.fig_fi: col1.pyplot(st.session_state.fig_fi)
        if st.session_state.fig_res: col2.pyplot(st.session_state.fig_res)
        if st.session_state.fig_s: col2.pyplot(st.session_state.fig_s)

# Display Comparison Results
if st.session_state.comparison_results and analysis_mode == "Multi-Model Comparison":
    st.header("📊 Multi-Model Comparison Results")
    results = st.session_state.comparison_results

    # Metrics Table
    st.subheader("Performance Metrics Comparison")
    metrics_data = {model_name: res["metrics"] for model_name, res in results.items()}
    if metrics_data:
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(style_comparison_df(metrics_df), use_container_width=True)
    else:
        st.warning("No metrics to display.")

    # Plots
    st.subheader("Visual Comparison")
    plot_types = {"Actual vs. Predicted": "fig_avp", "Residuals": "fig_res",
                  "Feature Importance": "fig_fi", "SHAP Summary": "fig_s"}

    plot_tabs = st.tabs(list(plot_types.keys()))

    for i, (tab_name, fig_key) in enumerate(plot_types.items()):
        with plot_tabs[i]:
            cols = st.columns(len(results))
            for j, (model_name, model_results) in enumerate(results.items()):
                with cols[j]:
                    st.subheader(model_name)
                    if model_results[fig_key]:
                        st.pyplot(model_results[fig_key])
                    else:
                        st.info("Plot not available for this model.")

# Prediction Logic (only for single model mode) - untested
if predict_button and analysis_mode == "Single Model Analysis":
    if uploaded_predict_file:
        if st.session_state.model and st.session_state.imputation_vals is not None:
            with st.spinner("Making predictions..."):
                X_pred, _, _, df_original = load_and_prepare_data(uploaded_predict_file, target_column,
                                                                  is_prediction=True)
                if X_pred is not None:
                    model_features = st.session_state.model.feature_names
                    imputation_vals_series = pd.Series(st.session_state.imputation_vals)
                    X_pred_aligned = X_pred.reindex(columns=model_features).fillna(imputation_vals_series).fillna(0)

                    predictions = st.session_state.model.predict(X_pred_aligned)
                    st.header("🔮 Prediction Results")
                    results_df = df_original.copy()
                    results_df[f'predicted_{target_column}'] = predictions
                    st.dataframe(results_df)
                    csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download Predictions", csv, f"predictions.csv", "text/csv")
        else:
            st.warning("Please train or load a model first in 'Single Model Analysis' mode.")
    else:
        st.warning("Please upload a file for prediction.")