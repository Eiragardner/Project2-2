# phase3/evaluation/evaluator.py 
from typing import Dict, Any, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from ..core.logger import MoELogger
import numpy as np

class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self, logger: MoELogger):
        self.logger = logger
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray,
                bin_assignments: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Comprehensive model evaluation"""
        metrics = {}
        
        # Overall metrics
        metrics['overall'] = self._calculate_overall_metrics(y_true, y_pred)
        
        # Per-bin metrics if available
        if bin_assignments is not None:
            metrics['per_bin'] = self._calculate_bin_metrics(y_true, y_pred, bin_assignments)
        
        # Distribution analysis
        metrics['distribution'] = self._analyze_prediction_distribution(y_true, y_pred)
        
        # Error analysis
        metrics['errors'] = self._analyze_errors(y_true, y_pred)
        
        self._log_metrics(metrics)
        return metrics
    
    def _calculate_overall_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate overall performance metrics"""
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100,
            'max_error': np.max(np.abs(y_true - y_pred))
        }
    
    def _calculate_bin_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                              bin_assignments: np.ndarray) -> Dict[int, Dict[str, float]]:
        """Calculate metrics for each bin"""
        bin_metrics = {}
        unique_bins = np.unique(bin_assignments)
        
        for bin_idx in unique_bins:
            mask = bin_assignments == bin_idx
            if mask.sum() > 0:
                y_true_bin = y_true[mask]
                y_pred_bin = y_pred[mask]
                
                bin_metrics[int(bin_idx)] = {
                    'samples': int(mask.sum()),
                    'mse': float(mean_squared_error(y_true_bin, y_pred_bin)),
                    'mae': float(mean_absolute_error(y_true_bin, y_pred_bin)),
                    'r2': float(r2_score(y_true_bin, y_pred_bin)) if len(y_true_bin) > 1 else 0.0,
                    'mean_target': float(y_true_bin.mean()),
                    'std_target': float(y_true_bin.std())
                }
        
        return bin_metrics
    
    def _analyze_prediction_distribution(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Analyze prediction distribution characteristics"""
        return {
            'pred_mean': float(y_pred.mean()),
            'pred_std': float(y_pred.std()),
            'true_mean': float(y_true.mean()),
            'true_std': float(y_true.std()),
            'correlation': float(np.corrcoef(y_true, y_pred)[0, 1]),
            'bias': float((y_pred - y_true).mean())
        }
    
    def _analyze_errors(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Analyze error characteristics"""
        errors = y_pred - y_true
        abs_errors = np.abs(errors)
        
        return {
            'error_mean': float(errors.mean()),
            'error_std': float(errors.std()),
            'error_median': float(np.median(errors)),
            'abs_error_mean': float(abs_errors.mean()),
            'abs_error_median': float(np.median(abs_errors)),
            'error_95th_percentile': float(np.percentile(abs_errors, 95)),
            'error_99th_percentile': float(np.percentile(abs_errors, 99))
        }
    
    def _log_metrics(self, metrics: Dict[str, Any]):
        """Log evaluation metrics"""
        self.logger.info("=== Model Evaluation Results ===")
        
        overall = metrics['overall']
        self.logger.info(f"Overall Performance:")
        self.logger.info(f"  RMSE: {overall['rmse']:.4f}")
        self.logger.info(f"  MAE: {overall['mae']:.4f}")
        self.logger.info(f"  R²: {overall['r2']:.4f}")
        self.logger.info(f"  MAPE: {overall['mape']:.2f}%")
        
        if 'per_bin' in metrics:
            self.logger.info(f"\nPer-bin Performance:")
            for bin_idx, bin_metrics in metrics['per_bin'].items():
                self.logger.info(f"  Bin {bin_idx}: RMSE={np.sqrt(bin_metrics['mse']):.4f}, "
                               f"R²={bin_metrics['r2']:.4f}, Samples={bin_metrics['samples']}")

