# phase3/binning/binning_optimizer.py
import numpy as np
from sklearn.cluster import KMeans
from typing import Dict, Tuple, List, Any
import logging

class BinningOptimizer:
    """
    Determines optimal binning strategies and boundaries.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def determine_optimal_bins(self, y: np.ndarray) -> Tuple[int, np.ndarray]:
        """Dynamically determine optimal number of bins and boundaries"""
        self.logger.info("Determining optimal binning strategy...")
        
        n_samples = len(y)
        max_bins = min(self.config['max_bins'], n_samples // self.config['min_bin_size'])
        max_bins = max(max_bins, self.config['min_bins'])
        
        best_score = -np.inf
        best_bins = self.config['min_bins']
        best_boundaries = None
        
        for n_bins in range(self.config['min_bins'], max_bins + 1):
            boundaries = self._compute_boundaries(y, n_bins)
            
            # Evaluate binning quality
            bin_assignments = np.digitize(y, boundaries)
            score = self._evaluate_binning_quality(y, bin_assignments, n_bins)
            
            if score > best_score:
                best_score = score
                best_bins = n_bins
                best_boundaries = boundaries
        
        self.logger.info(f"Optimal bins: {best_bins}, Score: {best_score:.4f}")
        return best_bins, best_boundaries
    
    def _compute_boundaries(self, y: np.ndarray, n_bins: int) -> np.ndarray:
        """Compute bin boundaries based on strategy"""
        if self.config['binning_strategy'] == 'quantile':
            boundaries = np.percentile(y, np.linspace(0, 100, n_bins + 1))[1:-1]
        elif self.config['binning_strategy'] == 'kmeans':
            boundaries = self._kmeans_boundaries(y, n_bins)
        else:  # adaptive
            boundaries = self._adaptive_boundaries(y, n_bins)
        
        return boundaries
    
    def _kmeans_boundaries(self, y: np.ndarray, n_bins: int) -> np.ndarray:
        """Compute boundaries using k-means clustering"""
        kmeans = KMeans(n_clusters=n_bins, random_state=self.config['random_state'], n_init=10)
        clusters = kmeans.fit_predict(y.reshape(-1, 1))
        boundaries = []
        
        for i in range(n_bins - 1):
            mask_curr = clusters == i
            mask_next = clusters == i + 1
            if mask_curr.any() and mask_next.any():
                boundary = (y[mask_curr].max() + y[mask_next].min()) / 2
                boundaries.append(boundary)
        
        return np.array(sorted(boundaries))
    
    def _adaptive_boundaries(self, y: np.ndarray, n_bins: int) -> np.ndarray:
        """Combine quantile and density-based approaches"""
        # Quantile boundaries
        q_boundaries = np.percentile(y, np.linspace(0, 100, n_bins + 1))[1:-1]
        
        # Density-based adjustment
        hist, edges = np.histogram(y, bins=n_bins * 3)
        peak_indices = self._find_density_peaks(hist)
        
        if len(peak_indices) >= n_bins - 1:
            density_boundaries = edges[peak_indices[:n_bins-1]]
            # Weighted combination
            boundaries = 0.7 * q_boundaries + 0.3 * density_boundaries
        else:
            boundaries = q_boundaries
        
        return boundaries
    
    def _find_density_peaks(self, hist: np.ndarray) -> List[int]:
        """Find peaks in density histogram"""
        peaks = []
        for i in range(1, len(hist) - 1):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                peaks.append(i)
        return peaks
    
    def _evaluate_binning_quality(self, y: np.ndarray, bin_assignments: np.ndarray, n_bins: int) -> float:
        """Evaluate the quality of binning using multiple criteria"""
        scores = []
        
        # 1. Balance score (prefer balanced bins)
        bin_counts = np.bincount(bin_assignments, minlength=n_bins)
        if np.mean(bin_counts) > 0:
            balance_score = 1.0 - np.std(bin_counts) / np.mean(bin_counts)
            scores.append(balance_score * 0.3)
        
        # 2. Separation score (prefer well-separated bins)
        bin_means = [y[bin_assignments == i].mean() for i in range(n_bins) if (bin_assignments == i).sum() > 0]
        if len(bin_means) > 1:
            separation_score = np.std(bin_means) / np.std(y)
            scores.append(separation_score * 0.4)
        
        # 3. Homogeneity score (prefer homogeneous bins)
        homogeneity_scores = []
        for i in range(n_bins):
            if (bin_assignments == i).sum() > 1:
                bin_std = y[bin_assignments == i].std()
                homogeneity_scores.append(1.0 - bin_std / np.std(y))
        if homogeneity_scores:
            homogeneity_score = np.mean(homogeneity_scores)
            scores.append(homogeneity_score * 0.3)
        
        return sum(scores) if scores else 0.0