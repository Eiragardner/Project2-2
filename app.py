import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

from models.respectiveModels.XGBoost_refactorStreamlit import XGBoostModel
from models.respectiveModels.linear_regression_refactorStreamlit import LinearRegressionModel
from models.respectiveModels.random_forest_refactorStreamlit import RandomForestModel

# Importing models from the refactored versions of the codes so that the GUI can easily access the relevant information

st.set_page_config(layout="wide", page_title="House Market ML Analyzer")


# Preparing data
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


# Analysis and plotting
def run_analysis(model_choice, X_train, y_train, X_test, y_test, load_pretrained):

    # Model Factory
    if model_choice == "XGBoost":
        model = XGBoostModel()
        model_path = 'models/xgboost/xgboost_model.json'
    elif model_choice == "Random Forest":
        model = RandomForestModel()
        model_path = 'models/randomForest/random_forest_model.joblib'
    elif model_choice == "Linear Regression":
        model = LinearRegressionModel()
        model_path = 'models/linearRegression/linear_regression_model.joblib'
    else:
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
        model.save_model(model_path)
        st.info(f"Trained model saved to {model_path}")

    model_features = model.feature_names
    if model_features is None:
        st.error(f"Model {model.model_name} lacks feature name information. Cannot proceed.")
        return [None] * 6

    imputation_series = pd.Series(st.session_state.imputation_vals)

    X_test_aligned = X_test.reindex(columns=model_features).fillna(imputation_series)
    X_train_aligned = X_train.reindex(columns=model_features).fillna(imputation_series)

    if X_test_aligned.isnull().values.any():
        st.warning("NaNs found in test data after alignment. Filling with 0.")
        X_test_aligned = X_test_aligned.fillna(0)

    y_pred = model.predict(X_test_aligned)
    metrics = model.evaluate(y_test, y_pred)
    fig_avp = model.plot_actual_vs_predicted(y_test, y_pred)
    fig_res = model.plot_residuals(y_test, y_pred)
    fig_fi = model.plot_feature_importance()
    fig_s = model.plot_shap_summary(X_train_aligned, X_test_aligned)

    return model, metrics, fig_avp, fig_res, fig_fi, fig_s

# Application UI by using Streamlit
st.title("🏘️ House Market Machine Learning Model Analyzer")

with st.sidebar:
    st.header("⚙️ Configuration")
    model_choice = st.selectbox("Choose Model:", ["XGBoost", "Random Forest", "Linear Regression"])

    st.subheader("Training Data")
    default_train_path = os.path.join('data', 'without30.csv')
    uploaded_train_file = st.file_uploader("Upload Training Data (CSV)", type="csv")
    use_default_train = st.checkbox("Use default training data", value=not uploaded_train_file)

    train_file_source = uploaded_train_file if uploaded_train_file else (
        default_train_path if use_default_train else None)
    target_column = st.text_input("Target Column Name:", "Price")

    load_pretrained = st.checkbox(f"Load Pre-trained {model_choice} Model")
    run_analysis_button = st.button("🚀 Run Analysis / Train Model")

    st.markdown("---")
    st.subheader("Prediction on New Data")
    uploaded_predict_file = st.file_uploader("Upload Prediction Data (CSV)", type="csv", key="predict_uploader")
    predict_button = st.button("🔮 Make Predictions")

for key in ['model', 'metrics', 'fig_avp', 'fig_res', 'fig_fi', 'fig_s', 'imputation_vals']:
    if key not in st.session_state:
        st.session_state[key] = None

if run_analysis_button:
    if train_file_source and target_column:
        with st.spinner("Loading data..."):
            X, y, _, _ = load_and_prepare_data(train_file_source, target_column)
        if X is not None and y is not None:
            # 1. Split data FIRST
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # 2. Calculate imputation values ONLY from training data
            imputation_values = X_train.mean()
            st.session_state.imputation_vals = imputation_values.to_dict()  # Save for prediction use

            # 3. Apply imputation to both train and test sets
            X_train = X_train.fillna(imputation_values)
            X_test = X_test.fillna(imputation_values)

            with st.spinner(f"Processing {model_choice}..."):
                results = run_analysis(model_choice, X_train, y_train, X_test, y_test, load_pretrained)
                st.session_state.model, st.session_state.metrics, st.session_state.fig_avp, st.session_state.fig_res, st.session_state.fig_fi, st.session_state.fig_s = results
            st.success(f"{model_choice} analysis complete!")
        else:
            st.error("Failed to load or prepare data.")
    else:
        st.warning("Please select training data and specify target column.")

if st.session_state.model:
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
if predict_button:
    if uploaded_predict_file:
        if st.session_state.model and st.session_state.imputation_vals is not None:
            with st.spinner("Making predictions..."):
                X_pred, _, _, df_original = load_and_prepare_data(uploaded_predict_file, target_column,
                                                                  is_prediction=True)
                if X_pred is not None:
                    # Align prediction data features with the model's features
                    model_features = st.session_state.model.feature_names
                    imputation_vals_series = pd.Series(st.session_state.imputation_vals)
                    X_pred_aligned = X_pred.reindex(columns=model_features).fillna(imputation_vals_series)

                    # Final check for Nan
                    if X_pred_aligned.isnull().values.any():
                        st.warning("NaNs found in prediction data after alignment. Filling with 0.")
                        X_pred_aligned = X_pred_aligned.fillna(0)

                    predictions = st.session_state.model.predict(X_pred_aligned)
                    st.header("🔮 Prediction Results")
                    results_df = df_original.copy()
                    results_df[f'predicted_{target_column}'] = predictions
                    st.dataframe(results_df)
                    csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download Predictions", csv, f"predictions.csv", "text/csv")
        else:
            st.warning("Please train or load a model first.")
    else:
        st.warning("Please upload a file for prediction.")