# phase3/binning/bin_optimizer.py
import numpy as np
from typing import Tuple, List
from sklearn.cluster import KMeans
from ..core.config import MoEConfig
from ..core.logger import MoELogger

class BinOptimizer:
    """Optimizes binning strategy for target values"""
    
    def __init__(self, config: MoEConfig, logger: MoELogger):
        self.config = config
        self.logger = logger
    
    def optimize_bins(self, y: np.ndarray) -> Tuple[int, np.ndarray]:
        """Find optimal number of bins and boundaries"""
        self.logger.info("Optimizing binning strategy...")
        
        n_samples = len(y)
        max_bins = min(
            self.config.max_bins, 
            n_samples // self.config.min_bin_size
        )
        max_bins = max(max_bins, self.config.min_bins)
        
        best_score = -np.inf
        best_bins = self.config.min_bins
        best_boundaries = None
        
        for n_bins in range(self.config.min_bins, max_bins + 1):
            boundaries = self._create_boundaries(y, n_bins)
            score = self._evaluate_binning(y, boundaries, n_bins)
            
            if score > best_score:
                best_score = score
                best_bins = n_bins
                best_boundaries = boundaries
        
        self.logger.info(f"Optimal bins: {best_bins}, Score: {best_score:.4f}")
        return best_bins, best_boundaries
    
    def _create_boundaries(self, y: np.ndarray, n_bins: int) -> np.ndarray:
        """Create bin boundaries based on strategy"""
        if self.config.binning_strategy == 'quantile':
            return self._quantile_boundaries(y, n_bins)
        elif self.config.binning_strategy == 'kmeans':
            return self._kmeans_boundaries(y, n_bins)
        else:  # adaptive
            return self._adaptive_boundaries(y, n_bins)
    
    def _quantile_boundaries(self, y: np.ndarray, n_bins: int) -> np.ndarray:
        """Create quantile-based boundaries"""
        return np.percentile(y, np.linspace(0, 100, n_bins + 1))[1:-1]
    
    def _kmeans_boundaries(self, y: np.ndarray, n_bins: int) -> np.ndarray:
        """Create KMeans-based boundaries"""
        try:
            kmeans = KMeans(
                n_clusters=n_bins, 
                random_state=self.config.random_state, 
                n_init=10
            )
            clusters = kmeans.fit_predict(y.reshape(-1, 1))
            
            boundaries = []
            for i in range(n_bins - 1):
                mask_curr = clusters == i
                mask_next = clusters == i + 1
                if mask_curr.any() and mask_next.any():
                    boundary = (y[mask_curr].max() + y[mask_next].min()) / 2
                    boundaries.append(boundary)
            
            return np.array(sorted(boundaries))
        except:
            # Fallback to quantile if KMeans fails
            return self._quantile_boundaries(y, n_bins)
    
    def _adaptive_boundaries(self, y: np.ndarray, n_bins: int) -> np.ndarray:
        """Create adaptive boundaries combining quantile and density"""
        # Start with quantile boundaries
        q_boundaries = self._quantile_boundaries(y, n_bins)
        
        # Find density peaks
        hist, edges = np.histogram(y, bins=n_bins * 3)
        peak_indices = self._find_density_peaks(hist)
        
        if len(peak_indices) >= n_bins - 1:
            density_boundaries = edges[peak_indices[:n_bins-1]]
            # Weighted combination
            boundaries = 0.7 * q_boundaries + 0.3 * density_boundaries
            return np.sort(boundaries)
        
        return q_boundaries
    
    def _find_density_peaks(self, hist: np.ndarray) -> List[int]:
        """Find peaks in histogram"""
        peaks = []
        for i in range(1, len(hist) - 1):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                peaks.append(i)
        return peaks
    
    def _evaluate_binning(self, y: np.ndarray, boundaries: np.ndarray, n_bins: int) -> float:
        """Evaluate binning quality using multiple criteria"""
        bin_assignments = np.digitize(y, boundaries)
        scores = []
        
        # Balance score
        bin_counts = np.bincount(bin_assignments, minlength=n_bins)
        if bin_counts.mean() > 0:
            balance_score = 1.0 - np.std(bin_counts) / np.mean(bin_counts)
            scores.append(balance_score * 0.3)
        
        # Separation score
        bin_means = []
        for i in range(n_bins):
            mask = bin_assignments == i
            if mask.sum() > 0:
                bin_means.append(y[mask].mean())
        
        if len(bin_means) > 1:
            separation_score = np.std(bin_means) / np.std(y)
            scores.append(separation_score * 0.4)
        
        # Homogeneity score
        homogeneity_scores = []
        for i in range(n_bins):
            mask = bin_assignments == i
            if mask.sum() > 1:
                bin_std = y[mask].std()
                homogeneity_scores.append(1.0 - bin_std / np.std(y))
        
        if homogeneity_scores:
            homogeneity_score = np.mean(homogeneity_scores)
            scores.append(homogeneity_score * 0.3)
        
        return sum(scores) if scores else 0.0

