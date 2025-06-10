# Project2-2: House Price Prediction
In this project we developed a prototype machine learning application that predicts house prices in the Netherlands.

---

## How to Run the App

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch the Streamlit app**
   Run the following command in your terminal:

   ```bash
   streamlit run app.py
   ```

   This will open a local Streamlit interface where you can:

   * Import the dataset
   * Select and run different ML models
   * View predictions values, plotsn and model explanations

---

## Project Overview

The Dutch housing market suffers from issues like overvaluation and inconsistent appraisals. Our goal is to build a reliable and explainable machine learning model that can predict property prices using features like living space, location, year built, and more.

---

## Models Implemented

We evaluate and compare the following ML models:
* **Linear Regression** *(Baseline model)*
* **Random Forest**
* **XGBoost**
* **Stacked Model**
* **mixed model**


We also apply SHAP (SHapley Additive exPlanations) to interpret feature importance.


To ru nthe mixed model type for example: 
python -m models.phase3.baseline_model "California Dataset.csv"
from the root folder
