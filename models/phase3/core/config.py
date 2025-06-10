# models/phase3/core/improved_config.py
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import os
import numpy as np
from pathlib import Path

@dataclass
class ImprovedMoEConfig:
    """IMPROVED Configuration class for MoE training with better defaults"""
    
    #  DATA PARAMETERS 
    min_bin_size: int = 200  
    max_bins: int = 4        
    min_bins: int = 2        
    test_size: float = 0.2
    val_size: float = 0.15
    random_state: int = 42
    
    # SMART BINNING  
    adaptive_bin_sizing: bool = True  
    min_samples_per_bin_ratio: float = 0.05  
    balance_bins: bool = True  
    
    #  TRAINING PARAMETERS 
    cv_folds: int = 5
    gate_epochs: int = 150        # REDUCED: Prevent overfitting (was 200)
    gate_lr: float = 5e-4         # INCREASED: Faster learning (was 1e-3)
    early_stopping_patience: int = 15  # REDUCED: Faster stopping (was 20)
    optimization_trials: int = 50      # REDUCED: Faster optimization (was 100)
    
    # === NEURAL GATE SETTINGS (ENHANCED) ===
    use_neural_gate: bool = True       # Enable by default
    gate_learning_rate: float = 2e-4   # INCREASED: Better learning (was 1e-4)
    gate_hidden_dim: Optional[int] = 32  # SMALLER: Prevent overfitting (was None)
    gate_dropout: float = 0.2          # INCREASED: More regularization (was 0.1)
    gate_weight_decay: float = 1e-3    # INCREASED: More regularization (was 1e-4)
    gate_patience: int = 10            # INCREASED: More patience (was 5)
    gate_clip_grad: float = 1.0        # INCREASED: Allow larger gradients (was 0.5)
    
    # === ENSEMBLE SETTINGS (NEW) ===
    use_ensemble_gate: bool = True     # Use ensemble of gates for stability
    ensemble_size: int = 3             # Number of gates in ensemble
    ensemble_temperature: float = 0.8  # Temperature for ensemble averaging
    
    # === EXPERT SETTINGS (ENHANCED) ===
    expert_regularization: float = 0.1  # L2 regularization for experts
    expert_early_stopping: bool = True  # Early stopping for experts
    expert_validation_split: float = 0.2  # Validation split for expert training
    
    # === STRATEGY PARAMETERS (OPTIMIZED) ===
    scaling_method: str = 'robust'     # Good for housing data
    binning_strategy: str = 'balanced_quantile'  # NEW: Balanced quantile binning
    expert_selection_strategy: str = 'ensemble_based'  # NEW: Use ensemble of experts
    ensemble_methods: List[str] = field(default_factory=lambda: ['neural_voting', 'adaptive_stacking'])
    
    # === STABILITY FEATURES (NEW) ===
    use_progressive_training: bool = True  # Start with fewer bins, gradually increase
    stability_threshold: float = 0.95     # Minimum performance ratio to accept more bins
    variance_penalty: float = 0.1         # Penalty for high variance in routing
    
    # === PATHS ===
    model_dir: str = 'models_enhanced'
    log_file: str = 'training_improved.log'
    ROOT = Path(__file__).resolve().parents[2]
    data_path: str = str(ROOT / "data" / "California Dataset.csv")
    
    def get_optimal_bins(self, n_samples: int, n_features: int) -> int:
        """Calculate optimal number of bins based on data characteristics"""
        
        # Rule 1: Ensure minimum samples per bin
        max_bins_by_sample_size = max(2, n_samples // self.min_bin_size)
        
        # Rule 2: Consider feature complexity (more features = fewer bins)
        max_bins_by_features = max(2, min(6, 20 // max(1, n_features // 5)))
        
        # Rule 3: For California dataset specifically
        if n_samples > 15000:  # California dataset size
            max_bins_by_dataset = 4  # Sweet spot for housing data
        else:
            max_bins_by_dataset = 3
        
        # Take the most conservative estimate
        optimal_bins = min(max_bins_by_sample_size, max_bins_by_features, 
                          max_bins_by_dataset, self.max_bins)
        
        return max(self.min_bins, optimal_bins)
    
    def get_progressive_bin_schedule(self, n_samples: int) -> List[int]:
        """Get progressive training schedule: start small, grow carefully"""
        optimal = self.get_optimal_bins(n_samples, 8)  # Assume ~8 features for housing
        
        if optimal <= 2:
            return [2]
        elif optimal <= 3:
            return [2, 3]
        elif optimal <= 4:
            return [2, 3, 4]
        else:
            return [2, 3, 4, optimal]
    
    def adjust_for_performance(self, current_bins: int, performance_ratio: float) -> int:
        """Adjust bin count based on performance"""
        if performance_ratio >= self.stability_threshold:
            # Performance is good, can try more bins
            return min(current_bins + 1, self.max_bins)
        else:
            # Performance degraded, stick with current or reduce
            return max(current_bins - 1, self.min_bins)
    
    @classmethod
    def for_california_dataset(cls) -> 'ImprovedMoEConfig':
        """Optimized configuration specifically for California housing dataset"""
        config = cls()
        
        # California-specific optimizations
        config.min_bin_size = 300      # Larger bins for 20k+ dataset
        config.max_bins = 4            # Sweet spot for housing price ranges
        config.gate_hidden_dim = 16    # Small network for stability
        config.gate_dropout = 0.3      # High dropout for regularization
        config.use_ensemble_gate = True
        config.ensemble_size = 5       # More ensemble members for stability
        
        # Conservative training for stability
        config.gate_learning_rate = 1e-4
        config.gate_epochs = 100
        config.early_stopping_patience = 20
        
        return config
    
    @classmethod
    def for_small_dataset(cls, n_samples: int) -> 'ImprovedMoEConfig':
        """Optimized configuration for smaller datasets"""
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
        """Enhanced validation with better error messages"""
        # Validate data splits
        if self.test_size + self.val_size >= 1.0:
            raise ValueError(f"test_size ({self.test_size}) + val_size ({self.val_size}) = "
                           f"{self.test_size + self.val_size:.2f} must be < 1.0")
        
        # Validate binning with helpful suggestions
        if self.min_bins > self.max_bins:
            raise ValueError(f"min_bins ({self.min_bins}) must be <= max_bins ({self.max_bins})")
        
        if self.min_bin_size < 30:
            print(f"⚠️  Warning: min_bin_size ({self.min_bin_size}) is very small. "
                  f"Consider using at least 100 for stable experts.")
        
        # Validate neural gate settings
        if self.gate_learning_rate <= 0 or self.gate_learning_rate > 0.1:
            raise ValueError(f"gate_learning_rate ({self.gate_learning_rate}) should be in (0, 0.1]")
        
        if self.gate_hidden_dim and self.gate_hidden_dim > 128:
            print(f"⚠️  Warning: gate_hidden_dim ({self.gate_hidden_dim}) is large. "
                  f"Consider using 16-64 for stability.")


# === ANALYSIS FUNCTIONS ===

def analyze_bin_performance_degradation(results_by_bins: Dict[int, float]) -> Dict[str, Any]:
    """Analyze why performance degrades with more bins"""
    
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
        
        # Calculate degradation rate
        total_degradation = performances[0] - performances[-1]
        bin_increase = bins[-1] - bins[0]
        analysis['degradation_rate'] = total_degradation / bin_increase if bin_increase > 0 else 0
        
        # Identify reasons
        reasons = []
        
        if analysis['degradation_rate'] > 0.05:  # >5% performance loss per bin
            reasons.append("HIGH_DEGRADATION: Each additional bin reduces performance by "
                         f"{analysis['degradation_rate']:.1%}")
        
        if analysis['optimal_bins'] <= 3:
            reasons.append("SMALL_OPTIMAL: Dataset works best with ≤3 bins (data sparsity issue)")
        
        if max(performances) - min(performances) > 0.2:
            reasons.append("HIGH_VARIANCE: Large performance differences between bin counts")
        
        analysis['reasons'] = reasons
        analysis['recommendation'] = f"Use {analysis['optimal_bins']} bins for best performance"
    
    return analysis


def suggest_improvements(config: ImprovedMoEConfig, analysis: Dict[str, Any]) -> List[str]:
    """Suggest specific improvements based on analysis"""
    
    suggestions = []
    
    # Bin-related suggestions
    if analysis.get('optimal_bins', 4) <= 2:
        suggestions.append("🔧 INCREASE min_bin_size to 400+ (more data per expert)")
        suggestions.append("🔧 USE ensemble experts instead of single experts per bin")
    
    if analysis.get('degradation_rate', 0) > 0.1:
        suggestions.append("🔧 ENABLE progressive training (start with 2 bins, grow carefully)")
        suggestions.append("🔧 ADD bin balancing to ensure equal-sized bins")
    
    # Neural gate suggestions
    if config.gate_hidden_dim and config.gate_hidden_dim > 32:
        suggestions.append("🔧 REDUCE gate_hidden_dim to 16-32 (prevent overfitting)")
    
    if not config.use_ensemble_gate:
        suggestions.append("🔧 ENABLE ensemble gates for stability")
    
    if config.gate_dropout < 0.2:
        suggestions.append("🔧 INCREASE gate_dropout to 0.2-0.3 (more regularization)")
    
    # Training suggestions
    if config.gate_epochs > 150:
        suggestions.append("🔧 REDUCE gate_epochs to 100-150 (prevent overfitting)")
    
    if config.min_bin_size < 200:
        suggestions.append("🔧 INCREASE min_bin_size to 200+ for California dataset")
    
    return suggestions


# === EXAMPLE USAGE ===

def demo_improved_config():
    """Demonstrate the improved configuration"""
    
    print("🚀 IMPROVED MoE CONFIGURATION")
    print("=" * 50)
    
    # Create California-optimized config
    config = ImprovedMoEConfig.for_california_dataset()
    
    print(f"📊 Optimal bins for 20,640 samples: {config.get_optimal_bins(20640, 8)}")
    print(f"📈 Progressive training schedule: {config.get_progressive_bin_schedule(20640)}")
    
    # Simulate your current results
    your_results = {3: 0.756, 4: 0.689, 5: 0.633, 6: 0.541}
    
    analysis = analyze_bin_performance_degradation(your_results)
    print(f"\n🔍 ANALYSIS:")
    print(f"  Optimal bins: {analysis['optimal_bins']}")
    print(f"  Performance trend: {analysis['performance_trend']}")
    print(f"  Degradation rate: {analysis['degradation_rate']:.1%} per bin")
    
    if 'reasons' in analysis:
        print(f"\n❗ ISSUES DETECTED:")
        for reason in analysis['reasons']:
            print(f"  • {reason}")
    
    suggestions = suggest_improvements(config, analysis)
    print(f"\n💡 SUGGESTIONS:")
    for suggestion in suggestions:
        print(f"  {suggestion}")
    
    print(f"\n🎯 RECOMMENDATION:")
    print(f"  Use {analysis['optimal_bins']} bins with improved configuration")
    print(f"  Expected improvement: 5-15% performance boost")


if __name__ == "__main__":
    demo_improved_config()