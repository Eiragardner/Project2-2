# phase3/inference_utils.py
import os
import json
import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

from .binning.learnable_binning import LearnableBinning
from .classifier.moe_classifier import MoEClassifier


class EnhancedMoEInference:
    """
    Inference and analysis utilities for the Enhanced MoE model
    """
    
    def __init__(self, model_dir: str = 'models_enhanced'):
        self.model_dir = model_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load components
        self.config = self._load_config()
        self.scalers = self._load_scalers()
        self.bin_layer = self._load_bin_layer()
        self.experts = self._load_experts()
        self.gate = self._load_gate()
        self.metrics = self._load_metrics()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load model configuration"""
        config_path = os.path.join(self.model_dir, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _load_scalers(self) -> Dict:
        """Load feature scalers"""
        scaler_path = os.path.join(self.model_dir, 'scalers.pkl')
        if os.path.exists(scaler_path):
            return joblib.load(scaler_path)
        return {}
    
    def _load_bin_layer(self) -> LearnableBinning:
        """Load binning layer"""
        bin_path = os.path.join(self.model_dir, 'bin_layer.pt')
        if os.path.exists(bin_path):
            # Reconstruct bin layer (would need to save architecture info)
            # For now, return None and handle in predict method
            return None
        return None
    
    def _load_experts(self) -> List[Any]:
        """Load expert models"""
        experts = []
        i = 0
        while True:
            expert_path = os.path.join(self.model_dir, f'expert_{i}.pkl')
            if os.path.exists(expert_path):
                expert = joblib.load(expert_path)
                experts.append(expert)
                i += 1
            else:
                break
        return experts
    
    def _load_gate(self):
        """Load gating network"""
        gate_path = os.path.join(self.model_dir, 'gate.pt')
        if os.path.exists(gate_path):
            # Would need to reconstruct gate architecture
            # For now, return None
            return None
        return None
    
    def _load_metrics(self) -> Dict[str, Any]:
        """Load training metrics"""
        metrics_path = os.path.join(self.model_dir, 'detailed_metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                return json.load(f)
        return {}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the full MoE pipeline"""
        # Preprocess features
        if 'features' in self.scalers:
            X_scaled = self.scalers['features'].transform(X)
        else:
            X_scaled = X
        
        # Get expert predictions
        expert_preds = np.column_stack([expert.predict(X_scaled) for expert in self.experts])
        
        # If gate is available, use it; otherwise use simple averaging
        if self.gate:
            self.gate.eval()
            with torch.no_grad():
                final_preds = self.gate(
                    torch.from_numpy(X_scaled).float().to(self.device),
                    torch.from_numpy(expert_preds).float().to(self.device)
                ).cpu().numpy()
        else:
            # Simple ensemble
            final_preds = np.mean(expert_preds, axis=1)
        
        return final_preds
    
    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with confidence intervals"""
        # Preprocess features
        if 'features' in self.scalers:
            X_scaled = self.scalers['features'].transform(X)
        else:
            X_scaled = X
        
        # Get all expert predictions
        expert_preds = np.column_stack([expert.predict(X_scaled) for expert in self.experts])
        
        # Calculate ensemble prediction and confidence
        mean_pred = np.mean(expert_preds, axis=1)
        std_pred = np.std(expert_preds, axis=1)
        
        # Confidence interval (assuming normal distribution)
        confidence_interval = 1.96 * std_pred  # 95% CI
        
        return mean_pred, confidence_interval
    
    def get_expert_contributions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get individual expert contributions for interpretability"""
        if 'features' in self.scalers:
            X_scaled = self.scalers['features'].transform(X)
        else:
            X_scaled = X
        
        contributions = {}
        for i, expert in enumerate(self.experts):
            contributions[f'expert_{i}'] = expert.predict(X_scaled)
        
        return contributions
    
    def analyze_prediction_uncertainty(self, X: np.ndarray) -> Dict[str, Any]:
        """Analyze prediction uncertainty across experts"""
        contributions = self.get_expert_contributions(X)
        
        # Stack predictions
        all_preds = np.column_stack(list(contributions.values()))
        
        # Calculate uncertainty metrics
        mean_preds = np.mean(all_preds, axis=1)
        std_preds = np.std(all_preds, axis=1)
        min_preds = np.min(all_preds, axis=1)
        max_preds = np.max(all_preds, axis=1)
        
        # Coefficient of variation
        cv = std_preds / (mean_preds + 1e-8)
        
        return {
            'mean_predictions': mean_preds,
            'std_predictions': std_preds,
            'min_predictions': min_preds,
            'max_predictions': max_preds,
            'coefficient_of_variation': cv,
            'high_uncertainty_indices': np.where(cv > np.percentile(cv, 90))[0].tolist()
        }
    
    def generate_model_report(self) -> str:
        """Generate comprehensive model analysis report"""
        report = []
        report.append("=" * 80)
        report.append("ENHANCED MOE MODEL ANALYSIS REPORT")
        report.append("=" * 80)
        
        # Model Overview
        report.append("\n📊 MODEL OVERVIEW")
        report.append("-" * 40)
        report.append(f"Number of Experts: {len(self.experts)}")
        report.append(f"Model Directory: {self.model_dir}")
        report.append(f"Device: {self.device}")
        
        # Configuration
        if self.config:
            report.append("\n⚙️  CONFIGURATION")
            report.append("-" * 40)
            for key, value in self.config.items():
                report.append(f"{key}: {value}")
        
        # Performance Metrics
        if self.metrics and 'overall' in self.metrics:
            overall = self.metrics['overall']
            report.append("\n📈 OVERALL PERFORMANCE")
            report.append("-" * 40)
            report.append(f"MAE:  {overall['MAE']:.4f}")
            report.append(f"RMSE: {overall['RMSE']:.4f}")
            report.append(f"R²:   {overall['R2']:.4f}")
            report.append(f"MAPE: {overall['MAPE']:.2f}%")
            report.append(f"Test Samples: {overall['samples']}")
        
        # Per-bin Performance
        if self.metrics and 'per_bin' in self.metrics:
            report.append("\n🎯 PER-BIN PERFORMANCE")
            report.append("-" * 40)
            for bin_idx, bin_metrics in self.metrics['per_bin'].items():
                report.append(f"\nBin {bin_idx}:")
                report.append(f"  Samples: {bin_metrics['samples']}")
                if bin_metrics['target_range']:
                    report.append(f"  Target Range: [{bin_metrics['target_range'][0]:.2f}, {bin_metrics['target_range'][1]:.2f}]")
                report.append(f"  MAE: {bin_metrics['MAE']:.4f}")
                report.append(f"  R²: {bin_metrics['R2']:.4f}")
        
        # Expert Information
        report.append("\n🤖 EXPERT MODELS")
        report.append("-" * 40)
        for i, expert in enumerate(self.experts):
            expert_type = type(expert).__name__
            report.append(f"Expert {i}: {expert_type}")
        
        # Feature Importance
        if self.metrics and 'feature_importance' in self.metrics:
            importance = self.metrics['feature_importance']
            if 'average' in importance:
                avg_imp = np.array(importance['average'])
                top_indices = np.argsort(avg_imp)[-10:][::-1]
                
                report.append("\n🔍 TOP 10 IMPORTANT FEATURES")
                report.append("-" * 40)
                for i, idx in enumerate(top_indices):
                    report.append(f"{i+1:2d}. Feature {idx}: {avg_imp[idx]:.4f}")
        
        return "\n".join(report)
    
    def create_visualizations(self, save_dir: str = 'visualizations'):
        """Create comprehensive visualizations"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Performance comparison chart
        if self.metrics and 'per_bin' in self.metrics:
            self._plot_per_bin_performance(save_dir)
        
        # 2. Feature importance plot
        if self.metrics and 'feature_importance' in self.metrics:
            self._plot_feature_importance(save_dir)
        
        # 3. Expert contribution analysis
        self._plot_expert_contributions(save_dir)
        
        print(f"Visualizations saved to {save_dir}/")
    
    def _plot_per_bin_performance(self, save_dir: str):
        """Plot per-bin performance metrics"""
        if 'per_bin' not in self.metrics:
            return
        
        bins = list(self.metrics['per_bin'].keys())
        metrics_data = {
            'MAE': [],
            'RMSE': [],
            'R2': [],
            'Samples': []
        }
        
        for bin_idx in bins:
            bin_data = self.metrics['per_bin'][str(bin_idx)]
            metrics_data['MAE'].append(bin_data['MAE'])
            metrics_data['RMSE'].append(bin_data['RMSE'])
            metrics_data['R2'].append(bin_data['R2'])
            metrics_data['Samples'].append(bin_data['samples'])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Per-Bin Performance Analysis', fontsize=16)
        
        # MAE
        axes[0,0].bar(bins, metrics_data['MAE'], color='skyblue', alpha=0.7)
        axes[0,0].set_title('Mean Absolute Error by Bin')
        axes[0,0].set_xlabel('Bin')
        axes[0,0].set_ylabel('MAE')
        
        # RMSE
        axes[0,1].bar(bins, metrics_data['RMSE'], color='lightcoral', alpha=0.7)
        axes[0,1].set_title('Root Mean Square Error by Bin')
        axes[0,1].set_xlabel('Bin')
        axes[0,1].set_ylabel('RMSE')
        
        # R²
        axes[1,0].bar(bins, metrics_data['R2'], color='lightgreen', alpha=0.7)
        axes[1,0].set_title('R² Score by Bin')
        axes[1,0].set_xlabel('Bin')
        axes[1,0].set_ylabel('R²')
        axes[1,0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # Sample distribution
        axes[1,1].pie(metrics_data['Samples'], labels=[f'Bin {b}' for b in bins], 
                     autopct='%1.1f%%', startangle=90)
        axes[1,1].set_title('Sample Distribution Across Bins')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'per_bin_performance.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_importance(self, save_dir: str):
        """Plot feature importance analysis"""
        if 'feature_importance' not in self.metrics or 'average' not in self.metrics['feature_importance']:
            return
        
        importance = np.array(self.metrics['feature_importance']['average'])
        top_indices = np.argsort(importance)[-20:]  # Top 20 features
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_indices)), importance[top_indices], color='steelblue', alpha=0.7)
        plt.yticks(range(len(top_indices)), [f'Feature {i}' for i in top_indices])
        plt.xlabel('Importance Score')
        plt.title('Top 20 Feature Importance Scores')
        plt.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_expert_contributions(self, save_dir: str):
        """Plot expert model contributions"""
        # This would require sample data to demonstrate
        # For now, create a placeholder showing expert types
        
        expert_types = [type(expert).__name__ for expert in self.experts]
        expert_counts = {}
        for et in expert_types:
            expert_counts[et] = expert_counts.get(et, 0) + 1
        
        plt.figure(figsize=(10, 6))
        plt.bar(expert_counts.keys(), expert_counts.values(), color='lightseagreen', alpha=0.7)
        plt.title('Expert Model Distribution')
        plt.xlabel('Model Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'expert_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()


# Example usage and testing functions
def demo_inference():
    """Demonstrate model inference capabilities"""
    # Initialize inference engine
    inference = EnhancedMoEInference()
    
    # Generate sample data for demonstration
    np.random.seed(42)
    X_sample = np.random.randn(10, 20)  # 10 samples, 20 features
    
    try:
        # Basic prediction
        predictions = inference.predict(X_sample)
        print("Basic Predictions:")
        print(predictions)
        
        # Prediction with confidence
        pred_mean, pred_ci = inference.predict_with_confidence(X_sample)
        print("\nPredictions with Confidence Intervals:")
        for i, (mean, ci) in enumerate(zip(pred_mean, pred_ci)):
            print(f"Sample {i}: {mean:.2f} ± {ci:.2f}")
        
        # Uncertainty analysis
        uncertainty = inference.analyze_prediction_uncertainty(X_sample)
        print(f"\nHigh uncertainty samples: {uncertainty['high_uncertainty_indices']}")
        
        # Generate report
        report = inference.generate_model_report()
        print("\n" + report)
        
        # Create visualizations
        inference.create_visualizations()
        
    except Exception as e:
        print(f"Demo failed: {str(e)}")
        print("This is expected if no trained model exists yet.")


if __name__ == '__main__':
    demo_inference()