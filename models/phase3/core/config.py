# phase3/core/config.py
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import os
from pathlib import Path

@dataclass
class MoEConfig:
    """Configuration class for MoE training"""
    # Data parameters
    min_bin_size: int = 50
    max_bins: int = 8
    min_bins: int = 3
    test_size: float = 0.2
    val_size: float = 0.15
    random_state: int = 42
    
    # Training parameters
    cv_folds: int = 5
    gate_epochs: int = 200
    gate_lr: float = 1e-3
    early_stopping_patience: int = 20
    optimization_trials: int = 100
    
    # Neural Gate settings (NEW!)
    use_neural_gate: bool = False
    gate_learning_rate: float = 0.0001  # More conservative than gate_lr
    gate_hidden_dim: Optional[int] = None  # Auto-sized if None
    gate_dropout: float = 0.1
    gate_weight_decay: float = 1e-4
    gate_patience: int = 5
    gate_clip_grad: float = 0.5
    
    # Strategy parameters
    scaling_method: str = 'robust'  # 'standard', 'robust', 'none'
    binning_strategy: str = 'adaptive'  # 'quantile', 'kmeans', 'adaptive'
    expert_selection_strategy: str = 'performance_based'  # 'random', 'specialized', 'performance_based'
    ensemble_methods: List[str] = field(default_factory=lambda: ['voting', 'stacking'])
    
    # Paths
    model_dir: str = 'models_enhanced'
    log_file: str = 'training.log'
    ROOT = Path(__file__).resolve().parents[2]
    data_path: str = str(ROOT / "data" / "without30.csv")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MoEConfig':
        """Create config from dictionary"""
        return cls(**{k: v for k, v in config_dict.items() if hasattr(cls, k)})
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            field.name: getattr(self, field.name) 
            for field in self.__dataclass_fields__.values()
        }
    
    def enable_neural_gate(self):
        """Enable neural gate with default settings"""
        self.use_neural_gate = True
        return self
    
    def disable_neural_gate(self):
        """Disable neural gate"""
        self.use_neural_gate = False
        return self
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        # Validate data splits
        if self.test_size + self.val_size >= 1.0:
            raise ValueError("test_size + val_size must be < 1.0")
        
        # Validate binning
        if self.min_bins > self.max_bins:
            raise ValueError("min_bins must be <= max_bins")
        
        # Validate neural gate settings
        if self.gate_epochs < 1:
            raise ValueError("gate_epochs must be >= 1")
        
        if self.gate_learning_rate <= 0:
            raise ValueError("gate_learning_rate must be > 0")