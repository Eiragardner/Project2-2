# phase3/core/config.py
from dataclasses import dataclass, field
from typing import Dict, Any, List
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
    
    # Strategy parameters
    scaling_method: str = 'robust'  # 'standard', 'robust', 'none'
    binning_strategy: str = 'adaptive'  # 'quantile', 'kmeans', 'adaptive'
    expert_selection_strategy: str = 'performance_based'  # 'random', 'specialized', 'performance_based'
    ensemble_methods: List[str] = field(default_factory=lambda: ['voting', 'stacking'])
    
    # Paths
    model_dir: str = 'models_enhanced'
    log_file: str = 'training.log'
    ROOT = Path(__file__).resolve().parents[2]
    data_path: str = str(ROOT / "data" / "prepared_data.csv")
    
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

