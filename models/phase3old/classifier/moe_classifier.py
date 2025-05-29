# phase3/classifier/moe_classifier.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MoEClassifier(nn.Module):
    """
    Lightweight gating controller:
    - Linear layer to compute per-bin logits
    - Softmax to derive weights
    - Weighted sum over bin_outputs to yield final prediction
    """
    def __init__(self, feature_dim: int, num_bins: int):
        super().__init__()
        self.gate = nn.Linear(feature_dim, num_bins)

    def forward(self, features: torch.Tensor, bin_outputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Tensor of shape [B, feature_dim]
            bin_outputs: Tensor of shape [B, num_bins]
        Returns:
            Tensor of shape [B]
        """
        logits = self.gate(features)           # [B, num_bins]
        weights = F.softmax(logits, dim=1)     # [B, num_bins]
        return (weights * bin_outputs).sum(dim=1)  # [B]
