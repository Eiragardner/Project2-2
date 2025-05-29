# phase3/main.py
import os
import sys
import numpy as np
from typing import Dict, Any
import torch

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.phase3.core.config import MoEConfig
from models.phase3.core.logger import MoELogger
from models.phase3.data.data_manager import DataManager
from models.phase3.binning.bin_optimizer import BinOptimizer
from models.phase3.experts.expert_factory import ExpertFactory
from models.phase3.gates.neural_gate import GateTrainer
from models.phase3.evaluation.evaluator import ModelEvaluator
from models.phase3.ensemble.ensemble_methods import EnsembleManager


class EnhancedMoETrainer:
    """Main trainer class for Enhanced Mixture of Experts"""
    
    def __init__(self, config: MoEConfig):
        self.config = config
        self.logger = MoELogger(config.log_file)
        
        # Initialize components
        self.data_manager = DataManager(config, self.logger)
        self.bin_optimizer = BinOptimizer(config, self.logger)
        self.expert_factory = ExpertFactory(config, self.logger)
        self.gate_trainer = GateTrainer(config, self.logger)
        self.evaluator = ModelEvaluator(self.logger)
        self.ensemble_manager = EnsembleManager(config, self.logger)
        
        # Storage for trained components
        self.bin_boundaries = None
        self.trained_experts = None
        self.gate_model = None
        self.n_bins = None
    
    def train(self) -> Dict[str, Any]:
        """Full training pipeline"""
        self.logger.info("Starting Enhanced MoE Training Pipeline")
        
        # 1. Load and preprocess data
        X, y = self.data_manager.load_data()
        X, y = self.data_manager.preprocess_data(X, y)
        
        # 2. Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.data_manager.split_data(X, y)
        
        # 3. Optimize binning strategy
        self.n_bins, self.bin_boundaries = self.bin_optimizer.optimize_bins(y_train)
        
        # 4. Assign samples to bins
        train_bin_assignments = np.digitize(y_train, self.bin_boundaries)
        val_bin_assignments = np.digitize(y_val, self.bin_boundaries)
        test_bin_assignments = np.digitize(y_test, self.bin_boundaries)
        
        # 5. Select and train expert models
        selected_experts = self.expert_factory.select_experts_for_bins(
            X_train, y_train, train_bin_assignments, self.n_bins
        )
        self.trained_experts = self.expert_factory.train_experts(
            X_train, y_train, train_bin_assignments, selected_experts
        )
        
        # 6. Generate expert predictions for gate training
        train_expert_preds = self._get_expert_predictions(X_train, train_bin_assignments)
        val_expert_preds = self._get_expert_predictions(X_val, val_bin_assignments)
        
        # 7. Train gating network
        self.gate_model = self.gate_trainer.train_gate(
            X_train, X_val, y_train, y_val, train_expert_preds, val_expert_preds
        )
        
        # 8. Generate final predictions
        test_expert_preds = self._get_expert_predictions(X_test, test_bin_assignments)
        final_predictions = self._predict_with_gate(X_test, test_expert_preds)
        
        # 9. Evaluate model
        evaluation_results = self.evaluator.evaluate(
            y_test, final_predictions, test_bin_assignments
        )
        
        # 10. Create ensemble predictions if configured
        if len(self.config.ensemble_methods) > 0:
            base_predictions = {
                'moe': final_predictions,
                'expert_avg': np.mean(test_expert_preds, axis=1)
            }
            ensemble_preds = self.ensemble_manager.create_ensemble_predictions(
                base_predictions, X_train, y_train, X_test
            )
            
            # Evaluate ensemble methods
            for method, preds in ensemble_preds.items():
                self.logger.info(f"\n=== {method.upper()} Ensemble Results ===")
                self.evaluator.evaluate(y_test, preds, test_bin_assignments)
        
        # 11. Save model
        self._save_model()
        
        self.logger.info("Training completed successfully!")
        
        return {
            'config': self.config,
            'evaluation': evaluation_results,
            'n_bins': self.n_bins,
            'bin_boundaries': self.bin_boundaries
        }
    
    def _get_expert_predictions(self, X: np.ndarray, bin_assignments: np.ndarray) -> np.ndarray:
        """Get predictions from all expert models"""
        n_samples = X.shape[0]
        expert_preds = np.zeros((n_samples, self.n_bins))
        
        for bin_idx, expert in enumerate(self.trained_experts):
            try:
                preds = expert.predict(X)
                expert_preds[:, bin_idx] = preds
            except Exception as e:
                self.logger.warning(f"Expert {bin_idx} prediction failed: {e}")
                # Use mean prediction as fallback
                expert_preds[:, bin_idx] = np.mean(expert_preds[:, :bin_idx], axis=1) if bin_idx > 0 else 0
        
        return expert_preds
    
    def _predict_with_gate(self, X: np.ndarray, expert_preds: np.ndarray) -> np.ndarray:
        """Generate final predictions using the gating network"""
        X_tensor = torch.from_numpy(X).float().to(self.gate_trainer.device)
        expert_preds_tensor = torch.from_numpy(expert_preds).float().to(self.gate_trainer.device)
        
        self.gate_model.eval()
        with torch.no_grad():
            predictions = self.gate_model(X_tensor, expert_preds_tensor)
        
        return predictions.cpu().numpy()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data"""
        if self.gate_model is None or self.trained_experts is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Scale features if scaler was used during training
        if hasattr(self.data_manager, 'scalers') and 'features' in self.data_manager.scalers:
            X = self.data_manager.scalers['features'].transform(X)
        
        # Assign to bins (dummy assignment for prediction)
        bin_assignments = np.zeros(X.shape[0], dtype=int)
        
        # Get expert predictions
        expert_preds = self._get_expert_predictions(X, bin_assignments)
        
        # Get final predictions
        return self._predict_with_gate(X, expert_preds)
    
    def _save_model(self):
        """Save trained model components"""
        os.makedirs(self.config.model_dir, exist_ok=True)
        
        # Save gate model
        if self.gate_model is not None:
            torch.save(self.gate_model.state_dict(), 
                      os.path.join(self.config.model_dir, 'gate_model.pth'))
        
        # Save other components using pickle
        import pickle
        
        model_components = {
            'config': self.config,
            'bin_boundaries': self.bin_boundaries,
            'n_bins': self.n_bins,
            'trained_experts': self.trained_experts,
            'scalers': self.data_manager.scalers if hasattr(self.data_manager, 'scalers') else None
        }
        
        with open(os.path.join(self.config.model_dir, 'model_components.pkl'), 'wb') as f:
            pickle.dump(model_components, f)
        
        self.logger.info(f"Model saved to {self.config.model_dir}")


def main():
    """Main execution function"""
    # Create configuration
    config = MoEConfig(
        min_bin_size=50,
        max_bins=6,
        min_bins=3,
        gate_epochs=150,
        gate_lr=1e-3,
        early_stopping_patience=15,
        binning_strategy='adaptive',
        expert_selection_strategy='performance_based',
        ensemble_methods=['voting', 'stacking'],
        scaling_method='robust'
    )
    
    # Initialize and train
    trainer = EnhancedMoETrainer(config)
    results = trainer.train()
    
    print("\n=== Training Summary ===")
    print(f"Number of bins: {results['n_bins']}")
    print(f"Overall RMSE: {results['evaluation']['overall']['rmse']:.4f}")
    print(f"Overall R²: {results['evaluation']['overall']['r2']:.4f}")
    
    return trainer, results


if __name__ == "__main__":
    trainer, results = main()