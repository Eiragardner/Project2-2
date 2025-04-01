import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
from pathlib import Path
import matplotlib.pyplot as plt

# Try to import xgboost, use a fallback if not available
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost is not installed. Using fallback implementation.")
    print("To install XGBoost, run: pip install xgboost")
    # Fallback model that mimics XGBoost interface
    from sklearn.ensemble import GradientBoostingRegressor as FallbackModel

# Try to import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP is not installed. Feature importance explanations will be limited.")
    print("To install SHAP, run: pip install shap")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('XGBoost_Core')

class XGBoostModel:
    """
    XGBoost implementation for real estate price prediction
    with fallback to sklearn's GradientBoostingRegressor if XGBoost is not available
    """
    
    def __init__(self, random_state=42):
        """Initialize the XGBoost model with default parameters"""
        self.random_state = random_state
        self.model = None
        self.feature_importance = None
        self.shap_values = None
        self.features = None
        self.params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'random_state': random_state
        }
        logger.info("XGBoost model initialized")
    
    def load_data(self, data_path, target_column='Price', test_size=0.2):
        """
        Load and split the prepared data
        
        Parameters:
        data_path (str): Path to the CSV file with prepared data
        target_column (str): Column name for the target variable
        test_size (float): Proportion of data to use for testing
        
        Returns:
        tuple: X_train, X_test, y_train, y_test
        """
        try:
            data = pd.read_csv(data_path)
            # Keep only numerical columns as in the prototype
            data = data.select_dtypes(include=[np.number])
            logger.info(f"Data loaded successfully from {data_path}")
            logger.info(f"Data shape: {data.shape}")
            
            if target_column not in data.columns:
                raise ValueError(f"Target column '{target_column}' not found in data")
            
            # Split features and target
            X = data.drop(columns=[target_column])
            y = data[target_column]
            
            # Store feature names for later use
            self.features = X.columns.tolist()
            
            # Split data into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state
            )
            
            logger.info(f"Data split: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def train(self, X_train, y_train, early_stopping_rounds=50):
        """
        Train the XGBoost model
        
        Parameters:
        X_train: Training features
        y_train: Training target
        early_stopping_rounds: Number of rounds for early stopping
        
        Returns:
        self: The trained model instance
        """
        try:
            if XGBOOST_AVAILABLE:
                # Use XGBoost if available
                # Create validation set for early stopping
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=self.random_state
                )
                
                # Create DMatrix objects for XGBoost
                dtrain = xgb.DMatrix(X_tr, label=y_tr)
                dval = xgb.DMatrix(X_val, label=y_val)
                
                # Set up parameters
                params = {
                    'objective': 'reg:squarederror',
                    'eval_metric': 'rmse',
                    'max_depth': 6,
                    'eta': 0.05,  # learning_rate
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'gamma': 0,
                    'alpha': 0.1,  # reg_alpha
                    'lambda': 1,  # reg_lambda
                    'min_child_weight': 1,
                    'seed': self.random_state
                }
                
                # Set up evaluation list
                evallist = [(dval, 'eval')]
                
                # Train using the lower-level XGBoost API (which accepts early_stopping_rounds)
                bst = xgb.train(
                    params, 
                    dtrain, 
                    num_boost_round=500,
                    evals=evallist,
                    early_stopping_rounds=early_stopping_rounds,
                    verbose_eval=True
                )
                
                # Convert to sklearn interface for consistency with the rest of the code
                self.model = xgb.XGBRegressor(
                    objective='reg:squarederror',
                    eval_metric='rmse',
                    n_estimators=bst.best_iteration,
                    learning_rate=0.05,
                    max_depth=6,
                    min_child_weight=1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    gamma=0,
                    reg_alpha=0.1,
                    reg_lambda=1,
                    random_state=self.random_state
                )
                
                # Fit the model without early stopping (just to set the internal state)
                self.model.fit(X_train, y_train)
                
                # Use the raw model from xgb.train for predictions
                self.raw_model = bst
                
                # Store best iteration
                logger.info(f"Best iteration: {bst.best_iteration}")
            else:
                # Use fallback model if XGBoost is not available
                self.model = FallbackModel(
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    random_state=self.random_state
                )
                
                self.model.fit(X_train, y_train)
            
            logger.info(f"Training model using {'XGBoost' if XGBOOST_AVAILABLE else 'GradientBoostingRegressor'}")
            
            # Calculate feature importance
            self.feature_importance = dict(zip(
                X_train.columns,
                self.model.feature_importances_
            ))
            
            # Calculate SHAP values if available
            if SHAP_AVAILABLE and XGBOOST_AVAILABLE:
                try:
                    # Create SHAP explainer
                    explainer = shap.Explainer(self.model)
                    self.shap_values = explainer(X_train)
                    logger.info("SHAP values calculated")
                except Exception as shap_error:
                    logger.warning(f"Error calculating SHAP values: {str(shap_error)}")
                    self.shap_values = None
            
            logger.info("Model training completed")
            return self
            
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise
    
    def predict(self, X):
        """
        Make predictions with the trained model
        
        Parameters:
        X: Features to predict on
        
        Returns:
        array: Predicted values
        """
        if self.model is None:
            logger.error("Model has not been trained yet")
            raise ValueError("Model must be trained before making predictions")
        
        try:
            # Make predictions
            if XGBOOST_AVAILABLE and hasattr(self, 'raw_model'):
                # Use the raw model for predictions if available
                dtest = xgb.DMatrix(X)
                predictions = self.raw_model.predict(dtest)
            else:
                # Use the sklearn interface
                predictions = self.model.predict(X)
            
            logger.info(f"Predictions made for {len(predictions)} samples")
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data
        
        Parameters:
        X_test: Test features
        y_test: Test target
        
        Returns:
        dict: Dictionary with evaluation metrics
        """
        if self.model is None:
            logger.error("Model has not been trained yet")
            raise ValueError("Model must be trained before evaluation")
        
        try:
            # Make predictions
            y_pred = self.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            # Calculate percentage error
            percentage_error = np.abs((y_test - y_pred) / y_test) * 100
            mean_percentage_error = np.mean(percentage_error)
            median_percentage_error = np.median(percentage_error)
            
            metrics = {
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'R2': r2,
                'Mean Percentage Error': mean_percentage_error,
                'Median Percentage Error': median_percentage_error
            }
            
            logger.info("Model evaluation results:")
            for metric, value in metrics.items():
                logger.info(f"    {metric}: {value:.4f}")
            
            return metrics, y_pred
            
        except Exception as e:
            logger.error(f"Error in model evaluation: {str(e)}")
            raise
    
    def get_feature_importance(self, top_n=None):
        """
        Get feature importance from the trained model
        
        Parameters:
        top_n (int, optional): Number of top features to return
        
        Returns:
        DataFrame: Feature importance scores
        """
        if self.model is None or not self.feature_importance:
            logger.error("Model not trained or features not available")
            raise ValueError("Model must be trained before getting feature importance")
        
        try:
            # Create DataFrame
            importance_df = pd.DataFrame({
                'Feature': list(self.feature_importance.keys()),
                'Importance': list(self.feature_importance.values())
            })
            
            # Sort by importance
            importance_df = importance_df.sort_values('Importance', ascending=False)
            
            # Filter top N if specified
            if top_n is not None:
                importance_df = importance_df.head(top_n)
            
            return importance_df
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            raise
    
    def plot_feature_importance(self, top_n=10, save_path=None):
        """
        Plot feature importance
        
        Parameters:
        top_n (int): Number of top features to plot
        save_path (str, optional): Path to save the plot
        
        Returns:
        Figure: Matplotlib figure
        """
        importance_df = self.get_feature_importance(top_n)
        
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['Feature'][::-1], importance_df['Importance'][::-1])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Top {top_n} Feature Importance')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Feature importance plot saved to {save_path}")
        
        return plt.gcf()
    
    def plot_shap_summary(self, X_data=None, plot_type="bar", max_display=10, save_path=None):
        """
        Plot SHAP summary
        
        Parameters:
        X_data (DataFrame, optional): Data to use for SHAP values (uses training data if None)
        plot_type (str): Type of plot ("bar", "beeswarm", "violin")
        max_display (int): Maximum number of features to display
        save_path (str, optional): Path to save the plot
        
        Returns:
        Figure: Matplotlib figure or None if SHAP is not available
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP is not available. Cannot plot SHAP summary.")
            return None
        
        if self.shap_values is None:
            logger.warning("SHAP values have not been calculated. Cannot plot SHAP summary.")
            return None
        
        try:
            plt.figure(figsize=(10, 8))
            
            if plot_type == "bar":
                shap.plots.bar(self.shap_values, max_display=max_display, show=False)
            elif plot_type == "beeswarm":
                shap.plots.beeswarm(self.shap_values, max_display=max_display, show=False)
            elif plot_type == "violin":
                shap.plots.violin(self.shap_values, max_display=max_display, show=False)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                logger.info(f"SHAP summary plot saved to {save_path}")
            
            return plt.gcf()
        except Exception as e:
            logger.warning(f"Error plotting SHAP summary: {str(e)}")
            return None
    
    def plot_shap_dependence(self, feature_idx, interaction_idx="auto", save_path=None):
        """
        Plot SHAP dependence plot
        
        Parameters:
        feature_idx (str or int): Feature to plot
        interaction_idx (str or int): Feature to use for interaction
        save_path (str, optional): Path to save the plot
        
        Returns:
        Figure: Matplotlib figure or None if SHAP is not available
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP is not available. Cannot plot SHAP dependence.")
            return None
        
        if self.shap_values is None:
            logger.warning("SHAP values have not been calculated. Cannot plot SHAP dependence.")
            return None
        
        try:
            plt.figure(figsize=(10, 6))
            shap.plots.scatter(self.shap_values[:, feature_idx], color=self.shap_values[:, interaction_idx], show=False)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                logger.info(f"SHAP dependence plot saved to {save_path}")
            
            return plt.gcf()
        except Exception as e:
            logger.warning(f"Error plotting SHAP dependence: {str(e)}")
            return None
    
    def save_predictions(self, user_data_path, output_path="predicted_prices.csv"):
        """
        Generate and save predictions for new data
        
        Parameters:
        user_data_path (str): Path to the CSV file with data to predict
        output_path (str): Path where predictions will be saved
        
        Returns:
        str: Path to the saved predictions
        """
        if self.model is None:
            logger.error("Model has not been trained yet")
            raise ValueError("Model must be trained before making predictions")
        
        try:
            # Load user data
            user_data = pd.read_csv(user_data_path)
            
            # Drop Price column if it exists
            if "Price" in user_data.columns:
                user_data = user_data.drop(columns=["Price"])
            
            # Select only numerical columns
            user_data = user_data.select_dtypes(include=[np.number])
            
            # Ensure we only use features that were in the training data
            missing_features = set(self.features) - set(user_data.columns)
            extra_features = set(user_data.columns) - set(self.features)
            
            if missing_features:
                logger.warning(f"Missing features in prediction data: {missing_features}")
                # Add missing features with zeros
                for feature in missing_features:
                    user_data[feature] = 0
            
            if extra_features:
                logger.warning(f"Extra features in prediction data will be ignored: {extra_features}")
                # Remove extra features
                user_data = user_data.drop(columns=list(extra_features))
            
            # Ensure column order matches training data
            user_data = user_data[self.features]
            
            # Make predictions
            predicted_prices = self.predict(user_data)
            
            # Add predictions to user data
            result_df = pd.DataFrame(user_data)
            result_df["Predicted_Price"] = np.round(predicted_prices, 2)
            
            # Save to CSV
            result_df.to_csv(output_path, index=False)
            
            logger.info(f"Predictions saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving predictions: {str(e)}")
            raise
    
    def hyperparameter_tuning(self, X_train, y_train, param_grid=None, cv=5):
        """
        Perform hyperparameter tuning using GridSearchCV
        
        Parameters:
        X_train: Training features
        y_train: Training target
        param_grid (dict, optional): Parameter grid for tuning
        cv (int): Number of cross-validation folds
        
        Returns:
        dict: Best parameters
        """
        if not XGBOOST_AVAILABLE:
            logger.warning("XGBoost is not available. Cannot perform hyperparameter tuning.")
            return None
        
        try:
            from sklearn.model_selection import GridSearchCV
            
            if param_grid is None:
                # Use a smaller parameter grid for faster execution
                param_grid = {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 6],
                    'min_child_weight': [1, 3],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0]
                }
            
            # Create base model
            base_model = xgb.XGBRegressor(
                objective='reg:squarederror',
                random_state=self.random_state
            )
            
            # Create GridSearchCV
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                scoring='neg_mean_squared_error',
                cv=cv,
                verbose=1,
                n_jobs=-1
            )
            
            # Fit GridSearchCV
            grid_search.fit(X_train, y_train)
            
            # Get best parameters
            best_params = grid_search.best_params_
            
            logger.info(f"Best parameters: {best_params}")
            return best_params
            
        except Exception as e:
            logger.error(f"Error in hyperparameter tuning: {str(e)}")
            raise
    
    def plot_actual_vs_predicted(self, y_test, y_pred, save_path=None):
        """
        Plot actual vs predicted values
        
        Parameters:
        y_test: Actual target values
        y_pred: Predicted target values
        save_path (str, optional): Path to save the plot
        
        Returns:
        Figure: Matplotlib figure
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        
        # Plot the perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.xlabel('Actual Prices')
        plt.ylabel('Predicted Prices')
        plt.title('Actual vs Predicted Prices')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Actual vs predicted plot saved to {save_path}")
        
        return plt.gcf()
    
    def plot_residuals(self, y_test, y_pred, save_path=None):
        """
        Plot residuals
        
        Parameters:
        y_test: Actual target values
        y_pred: Predicted target values
        save_path (str, optional): Path to save the plot
        
        Returns:
        Figure: Matplotlib figure
        """
        residuals = y_test - y_pred
        
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        
        plt.xlabel('Predicted Prices')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Residuals plot saved to {save_path}")
        
        return plt.gcf()
    
    def save_model(self, model_path="xgboost_model.json"):
        """
        Save the trained model
        
        Parameters:
        model_path (str): Path to save the model
        
        Returns:
        str: Path to the saved model
        """
        if self.model is None:
            logger.error("Model has not been trained yet")
            raise ValueError("Model must be trained before saving")
        
        try:
            if XGBOOST_AVAILABLE and hasattr(self, 'raw_model'):
                # Save raw XGBoost model
                self.raw_model.save_model(model_path)
            elif XGBOOST_AVAILABLE:
                # Save sklearn XGBoost model
                self.model.save_model(model_path)
            else:
                # Save sklearn model
                import joblib
                joblib.dump(self.model, model_path)
            
            logger.info(f"Model saved to {model_path}")
            return model_path
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, model_path):
        """
        Load a saved model
        
        Parameters:
        model_path (str): Path to the saved model
        
        Returns:
        self: The loaded model instance
        """
        try:
            if XGBOOST_AVAILABLE:
                # Try to load as raw model first
                try:
                    self.raw_model = xgb.Booster()
                    self.raw_model.load_model(model_path)
                    
                    # Also create an XGBRegressor instance for compatibility
                    self.model = xgb.XGBRegressor()
                    # We can't properly set all parameters, but this ensures the interface works
                    
                    logger.info(f"Model loaded from {model_path} as Booster")
                except Exception:
                    # Try as XGBRegressor
                    self.model = xgb.XGBRegressor()
                    self.model.load_model(model_path)
                    logger.info(f"Model loaded from {model_path} as XGBRegressor")
            else:
                # Load sklearn model
                import joblib
                self.model = joblib.load(model_path)
                logger.info(f"Model loaded from {model_path}")
            
            return self
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

# Main execution
if __name__ == "__main__":
    try:
        # Initialize model
        model = XGBoostModel()
        
        # Load and prepare data
        X_train, X_test, y_train, y_test = model.load_data("prepared_data.csv")
        
        # Train model
        model.train(X_train, y_train)
        
        # Evaluate model
        metrics, y_pred = model.evaluate(X_test, y_test)
        
        # Print metrics
        print("Training complete.")
        print(f"Mean Absolute Error (MAE): {metrics['MAE']:,.2f}")
        print(f"Root Mean Squared Error (RMSE): {metrics['RMSE']:,.2f}")
        print(f"R-squared: {metrics['R2']:.2f}")
        print(f"Mean Percentage Error: {metrics['Mean Percentage Error']:,.2f}%")
        
        # Generate predictions for new data
        try:
            model.save_predictions("to_predict.csv")
            print("Predictions saved to predicted_prices.csv")
        except Exception as pred_error:
            print(f"Could not generate predictions: {str(pred_error)}")
        
        # Optional: Save model
        try:
            model.save_model()
            print("Model saved to xgboost_model.json")
        except Exception as save_error:
            print(f"Could not save model: {str(save_error)}")
        
    except Exception as e:
        print(f"Error: {str(e)}")