"""
Main script to run XGBoost model for real estate price prediction
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import sys
import os
import argparse
from pathlib import Path

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from XGBoost_core import XGBoostModel

def main(args):
    # File paths
    prepared_data_path = args.data_path
    to_predict_path = args.predict_path
    output_path = args.output_path
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Initialize model
    model = XGBoostModel(random_state=args.random_state)
    
    # Load and split data
    print("Loading data...")
    X_train, X_test, y_train, y_test = model.load_data(
        prepared_data_path, 
        target_column=args.target_column,
        test_size=args.test_size
    )
    
    # Train model
    print("Training XGBoost model...")
    model.train(X_train, y_train, early_stopping_rounds=args.early_stopping)
    
    # Evaluate model
    print("Evaluating model...")
    metrics, y_pred = model.evaluate(X_test, y_test)
    
    # Print metrics
    print("\nModel Evaluation:")
    print(f"Mean Absolute Error (MAE): {metrics['MAE']:,.2f}")
    print(f"Root Mean Squared Error (RMSE): {metrics['RMSE']:,.2f}")
    print(f"R-squared: {round(metrics['R2'], 4)}")
    print(f"Mean Percentage Error: {metrics['Mean Percentage Error']:,.2f}%")
    print(f"Median Percentage Error: {metrics['Median Percentage Error']:,.2f}%")
    
    # Plot actual vs predicted if visualization is enabled
    if args.visualize:
        print("Generating visualizations...")
        
        # Create visualization directory
        viz_dir = Path("visualizations")
        viz_dir.mkdir(exist_ok=True)
        
        # Actual vs Predicted plot
        model.plot_actual_vs_predicted(y_test, y_pred, save_path=viz_dir / "actual_vs_predicted.png")
        print("Actual vs Predicted plot saved")
        
        # Residuals plot
        model.plot_residuals(y_test, y_pred, save_path=viz_dir / "residuals.png")
        print("Residuals plot saved")
        
        # Feature importance plot
        model.plot_feature_importance(top_n=10, save_path=viz_dir / "feature_importance.png")
        print("Feature importance plot saved")
        
        # SHAP plots if available
        shap_summary = model.plot_shap_summary(X_test, save_path=viz_dir / "shap_summary.png")
        if shap_summary:
            print("SHAP summary plot saved")
    
    # Feature importance
    importance_df = model.get_feature_importance(top_n=10)
    print("\nTop 10 Most Important Features:")
    for i, row in importance_df.iterrows():
        print(f"{row['Feature']}: {row['Importance']:.4f}")
    
    # Generate predictions for new data if predict path is provided
    if to_predict_path and os.path.exists(to_predict_path):
        print("\nGenerating predictions for new data...")
        try:
            output_file = model.save_predictions(to_predict_path, output_path)
            print(f"Predictions saved to {output_file}")
        except Exception as e:
            print(f"Error generating predictions: {str(e)}")
    
    # Save model if requested
    if args.save_model:
        model_path = args.model_path or "xgboost_model.json"
        try:
            saved_path = model.save_model(model_path)
            print(f"Model saved to {saved_path}")
        except Exception as e:
            print(f"Error saving model: {str(e)}")
    
    return model, X_train, X_test, y_train, y_test, y_pred, metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run XGBoost model for real estate price prediction")
    parser.add_argument("--data_path", type=str, default="without30.csv", help="Path to prepared data CSV")
    parser.add_argument("--predict_path", type=str, default="to_predict.csv", help="Path to data to predict CSV")
    parser.add_argument("--output_path", type=str, default="predicted_prices.csv", help="Path to save predictions")
    parser.add_argument("--target_column", type=str, default="Price", help="Target column name")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test size for train-test split")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility")
    parser.add_argument("--early_stopping", type=int, default=50, help="Early stopping rounds")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    parser.add_argument("--save_model", action="store_true", help="Save the trained model")
    parser.add_argument("--model_path", type=str, default=None, help="Path to save the model")
    
    args = parser.parse_args()
    main(args)