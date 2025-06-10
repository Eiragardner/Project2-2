# phase3/binning/learnable_binning.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnableBinning(nn.Module):
    """
    Soft, trainable binning layer with Gaussian-like assignments.
    Args:
        num_bins: number of bins
        init_edges: 1D tensor of initial bin edges (length=num_bins)
        init_alpha: initial temperature (scalar)
    """
    def __init__(self, num_bins: int, init_edges: torch.Tensor, init_alpha: float = 1.0):
        super().__init__()
        self.num_bins = num_bins
        self.edges = nn.Parameter(init_edges.clone().float())
        self.alpha = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B] or [B,1]
        if x.dim() == 2 and x.size(1) == 1:
            x_flat = x.view(-1)
        elif x.dim() == 1:
            x_flat = x
        else:
            x_flat = x[:, 0]
        diffs = x_flat.unsqueeze(1) - self.edges.unsqueeze(0)  # [B, num_bins]
        logits = -self.alpha * diffs.pow(2)
        return F.softmax(logits, dim=1)  # [B, num_bins]