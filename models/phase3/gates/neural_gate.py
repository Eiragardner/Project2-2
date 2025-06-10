# Fixed Neural Gate classes
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

class EnhancedMoEGate(nn.Module):
    """COMPLETELY FIXED: Neural gate with proper architecture"""
    
    def __init__(self, feature_dim: int, num_bins: int, hidden_dim: int = None):
        super().__init__()
        
        # FIXED: Very simple architecture for stability
        if hidden_dim is None:
            hidden_dim = max(4, min(12, feature_dim))
        
        self.feature_dim = feature_dim
        self.num_bins = num_bins
        
        # FIXED: Minimal architecture to prevent overfitting
        self.layers = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_bins)
        )
        
        # Initialize weights conservatively
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)  # Conservative gain
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, features: torch.Tensor, bin_outputs: torch.Tensor) -> torch.Tensor:
        """FIXED: Proper forward pass"""
        # Get gate weights
        logits = self.layers(features)
        weights = torch.softmax(logits, dim=1)
        
        # FIXED: Ensure shapes are compatible
        if weights.shape != bin_outputs.shape:
            raise ValueError(f"Shape mismatch: weights {weights.shape} vs outputs {bin_outputs.shape}")
        
        # Weighted sum
        output = (weights * bin_outputs).sum(dim=1)
        
        return output


class GateTrainer:
    """COMPLETELY FIXED: Gate trainer with proper loss scaling"""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
    
    def train_gate(self, X_train: np.ndarray, X_val: np.ndarray,
                   y_train: np.ndarray, y_val: np.ndarray,
                   train_preds: np.ndarray, val_preds: np.ndarray) -> EnhancedMoEGate:
        """FIXED: Train gate with proper loss scaling"""
        self.logger.info("Training enhanced gating network...")
        
        n_features = X_train.shape[1]
        n_bins = train_preds.shape[1]
        
        # Initialize gate
        gate = EnhancedMoEGate(n_features, n_bins).to(self.device)
        
        # FIXED: Conservative training settings
        optimizer = torch.optim.Adam(gate.parameters(), lr=0.0001, weight_decay=1e-4)
        
        # FIXED: Scale targets for better training
        y_mean = np.mean(y_train)
        y_std = np.std(y_train)
        
        def scale_targets(y):
            return (y - y_mean) / y_std
        
        def unscale_targets(y_scaled):
            return y_scaled * y_std + y_mean
        
        # Scale targets
        y_train_scaled = scale_targets(y_train)
        y_val_scaled = scale_targets(y_val)
        train_preds_scaled = scale_targets(train_preds)
        val_preds_scaled = scale_targets(val_preds)
        
        # Convert to tensors
        X_train_t = torch.from_numpy(X_train.astype(np.float32)).to(self.device)
        X_val_t = torch.from_numpy(X_val.astype(np.float32)).to(self.device)
        y_train_t = torch.from_numpy(y_train_scaled.astype(np.float32)).to(self.device)
        y_val_t = torch.from_numpy(y_val_scaled.astype(np.float32)).to(self.device)
        train_preds_t = torch.from_numpy(train_preds_scaled.astype(np.float32)).to(self.device)
        val_preds_t = torch.from_numpy(val_preds_scaled.astype(np.float32)).to(self.device)
        
        criterion = nn.MSELoss()
        
        # Training loop
        best_val_loss = float('inf')
        patience = 0
        max_epochs = min(30, self.config.gate_epochs)
        
        for epoch in range(max_epochs):
            # Training
            gate.train()
            optimizer.zero_grad()
            
            train_out = gate(X_train_t, train_preds_t)
            train_loss = criterion(train_out, y_train_t)
            
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(gate.parameters(), 0.5)
            optimizer.step()
            
            # Validation
            gate.eval()
            with torch.no_grad():
                val_out = gate(X_val_t, val_preds_t)
                val_loss = criterion(val_out, y_val_t).item()
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = 0
                best_state = gate.state_dict().copy()
            else:
                patience += 1
                if patience >= 5:
                    break
            
            if epoch % 10 == 0:
                # Unscale for logging
                train_loss_unscaled = train_loss.item() * (y_std ** 2)
                val_loss_unscaled = val_loss * (y_std ** 2)
                self.logger.info(f"Epoch {epoch}: Train={train_loss_unscaled:.0f}, Val={val_loss_unscaled:.0f}")
        
        # Load best state
        if 'best_state' in locals():
            gate.load_state_dict(best_state)
        
        return gate
    
    def predict_with_gate(self, gate: EnhancedMoEGate, X: np.ndarray, 
                         expert_predictions: np.ndarray) -> np.ndarray:
        """Make predictions using the trained gate"""
        gate.eval()
        
        with torch.no_grad():
            X_t = torch.from_numpy(X.astype(np.float32)).to(self.device)
            preds_t = torch.from_numpy(expert_predictions.astype(np.float32)).to(self.device)
            
            output = gate(X_t, preds_t)
            return output.cpu().numpy()