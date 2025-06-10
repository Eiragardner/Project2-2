from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import os
import numpy as np
from pathlib import Path

@dataclass
class ImprovedMoEConfig:
    """IMPROVED Configuration class for MoE training with better defaults"""
    min_bin_size: int = 200
    max_bins: int = 4
    min_bins: int = 2
    test_size: float = 0.2
    val_size: float = 0.15
    random_state: int = 42

    adaptive_bin_sizing: bool = True
    min_samples_per_bin_ratio: float = 0.05
    balance_bins: bool = True

    cv_folds: int = 5
    gate_epochs: int = 150
    gate_lr: float = 5e-4
    early_stopping_patience: int = 15
    optimization_trials: int = 50

    use_neural_gate: bool = True
    gate_learning_rate: float = 2e-4
    gate_hidden_dim: Optional[int] = 32
    gate_dropout: float = 0.2
    gate_weight_decay: float = 1e-3
    gate_patience: int = 10
    gate_clip_grad: float = 1.0

    use_ensemble_gate: bool = True
    ensemble_size: int = 3
    ensemble_temperature: float = 0.8

    expert_regularization: float = 0.1
    expert_early_stopping: bool = True
    expert_validation_split: float = 0.2

    scaling_method: str = 'robust'
    binning_strategy: str = 'balanced_quantile'
    expert_selection_strategy: str = 'ensemble_based'
    ensemble_methods: List[str] = field(default_factory=lambda: ['neural_voting', 'adaptive_stacking'])

    use_progressive_training: bool = True
    stability_threshold: float = 0.95
    variance_penalty: float = 0.1

    model_dir: str = 'models_enhanced'
    log_file: str = 'training_improved.log'
    ROOT = Path(__file__).resolve().parents[2]
    data_path: str = str(ROOT / "data" / "California Dataset.csv")

    def get_optimal_bins(self, n_samples: int, n_features: int) -> int:
        max_bins_by_sample_size = max(2, n_samples // self.min_bin_size)
        max_bins_by_features = max(2, min(6, 20 // max(1, n_features // 5)))
        if n_samples > 15000:
            max_bins_by_dataset = 4
        else:
            max_bins_by_dataset = 3
        optimal_bins = min(max_bins_by_sample_size, max_bins_by_features, max_bins_by_dataset, self.max_bins)
        return max(self.min_bins, optimal_bins)

    def get_progressive_bin_schedule(self, n_samples: int) -> List[int]:
        optimal = self.get_optimal_bins(n_samples, 8)
        if optimal <= 2:
            return [2]
        elif optimal <= 3:
            return [2, 3]
        elif optimal <= 4:
            return [2, 3, 4]
        else:
            return [2, 3, 4, optimal]

    def adjust_for_performance(self, current_bins: int, performance_ratio: float) -> int:
        if performance_ratio >= self.stability_threshold:
            return min(current_bins + 1, self.max_bins)
        else:
            return max(current_bins - 1, self.min_bins)

    @classmethod
    def for_california_dataset(cls) -> 'ImprovedMoEConfig':
        config = cls()
        config.min_bin_size = 300
        config.max_bins = 4
        config.gate_hidden_dim = 16
        config.gate_dropout = 0.3
        config.use_ensemble_gate = True
        config.ensemble_size = 5
        config.gate_learning_rate = 1e-4
        config.gate_epochs = 100
        config.early_stopping_patience = 20
        return config

    @classmethod
    def for_small_dataset(cls, n_samples: int) -> 'ImprovedMoEConfig':
        config = cls()
        if n_samples < 1000:
            config.min_bin_size = 50
            config.max_bins = 2
            config.gate_hidden_dim = 8
            config.use_ensemble_gate = False
        elif n_samples < 5000:
            config.min_bin_size = 100
            config.max_bins = 3
            config.gate_hidden_dim = 16
            config.ensemble_size = 3
        return config

    def __post_init__(self):
        if self.test_size + self.val_size >= 1.0:
            raise ValueError(f"test_size ({self.test_size}) + val_size ({self.val_size}) = "
                             f"{self.test_size + self.val_size:.2f} must be < 1.0")
        if self.min_bins > self.max_bins:
            raise ValueError(f"min_bins ({self.min_bins}) must be <= max_bins ({self.max_bins})")
        if self.min_bin_size < 30:
            print(f"⚠️  Warning: min_bin_size ({self.min_bin_size}) is very small. "
                  f"Consider using at least 100 for stable experts.")
        if self.gate_learning_rate <= 0 or self.gate_learning_rate > 0.1:
            raise ValueError(f"gate_learning_rate ({self.gate_learning_rate}) should be in (0, 0.1]")
        if self.gate_hidden_dim and self.gate_hidden_dim > 128:
            print(f"⚠️  Warning: gate_hidden_dim ({self.gate_hidden_dim}) is large. "
                  f"Consider using 16-64 for stability.")

def analyze_bin_performance_degradation(results_by_bins: Dict[int, float]) -> Dict[str, Any]:
    analysis = {
        'performance_trend': 'decreasing' if len(results_by_bins) > 1 and 
                           list(results_by_bins.values())[-1] < list(results_by_bins.values())[0] else 'stable',
        'max_performance': max(results_by_bins.values()),
        'optimal_bins': max(results_by_bins.keys(), key=lambda k: results_by_bins[k]),
        'degradation_rate': 0
    }
    if len(results_by_bins) > 1:
        performances = list(results_by_bins.values())
        bins = list(results_by_bins.keys())
        total_degradation = performances[0] - performances[-1]
        bin_increase = bins[-1] - bins[0]
        analysis['degradation_rate'] = total_degradation / bin_increase if bin_increase > 0 else 0
        reasons = []
        if analysis['degradation_rate'] > 0.05:
            reasons.append("HIGH_DEGRADATION: Each additional bin reduces performance by "
                           f"{analysis['degradationation_rate']:.1%}")
        if analysis['optimal_bins'] <= 3:
            reasons.append("SMALL_OPTIMAL: Dataset works best with ≤3 bins (data sparsity issue)")
        if max(performances) - min(performances) > 0.2:
            reasons.append("HIGH_VARIANCE: Large performance differences between bin counts")
        analysis['reasons'] = reasons
        analysis['recommendation'] = f"Use {analysis['optimal_bins']} bins for best performance"
    return analysis

def suggest_improvements(config: ImprovedMoEConfig, analysis: Dict[str, Any]) -> List[str]:
    suggestions = []
    if analysis.get('optimal_bins', 4) <= 2:
        suggestions.append("🔧 INCREASE min_bin_size to 400+ (more data per expert)")
        suggestions.append("🔧 USE ensemble experts instead of single experts per bin")
    if analysis.get('degradation_rate', 0) > 0.1:
        suggestions.append("🔧 ENABLE progressive training (start with 2 bins, grow carefully)")
        suggestions.append("🔧 ADD bin balancing to ensure equal-sized bins")
    if config.gate_hidden_dim and config.gate_hidden_dim > 32:
        suggestions.append("🔧 REDUCE gate_hidden_dim to 16-32 (prevent overfitting)")
    if not config.use_ensemble_gate:
        suggestions.append("🔧 ENABLE ensemble gates for stability")
    if config.gate_dropout < 0.2:
        suggestions.append("🔧 INCREASE gate_dropout to 0.2-0.3 (more regularization)")
    if config.gate_epochs > 150:
        suggestions.append("🔧 REDUCE gate_epochs to 100-150 (prevent overfitting)")
    if config.min_bin_size < 200:
        suggestions.append("🔧 INCREASE min_bin_size to 200+ for California dataset")
    return suggestions



if __name__ == "__main__":
    demo_improved_config()
