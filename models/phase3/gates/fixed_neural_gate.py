# models/phase3/gates/fixed_neural_gate.py
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

class FixedNeuralGate(nn.Module):
    """FIXED: Neural gate with proper scaling and architecture"""
    
    def __init__(self, feature_dim: int, num_experts: int, hidden_dim: int = None):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = max(8, min(32, feature_dim))
        
        self.feature_dim = feature_dim
        self.num_experts = num_experts
        
        # Simple but effective architecture
        self.gate_network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_experts)
        )
        
        # Conservative weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)  # Very conservative
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, features: torch.Tensor, expert_predictions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [batch_size, feature_dim] - input features
            expert_predictions: [batch_size, num_experts] - predictions from each expert
        Returns:
            weighted_prediction: [batch_size] - final blended prediction
        """
        # Get gate logits from features
        gate_logits = self.gate_network(features)  # [batch_size, num_experts]
        
        # Convert to probabilities (weights)
        gate_weights = torch.softmax(gate_logits, dim=1)  # [batch_size, num_experts]
        
        # Weighted sum of expert predictions
        weighted_prediction = torch.sum(gate_weights * expert_predictions, dim=1)  # [batch_size]
        
        return weighted_prediction, gate_weights


class FixedGateTrainer:
    """FIXED: Gate trainer with proper loss scaling and training"""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
    
    def train_gate(self, X_train: np.ndarray, X_val: np.ndarray,
                   y_train: np.ndarray, y_val: np.ndarray,
                   train_expert_preds: np.ndarray, val_expert_preds: np.ndarray) -> FixedNeuralGate:
        """
        FIXED: Train gate with proper scaling
        
        Args:
            X_train/X_val: Input features [samples, features]
            y_train/y_val: True targets [samples]
            train_expert_preds/val_expert_preds: Expert predictions [samples, num_experts]
        """
        self.logger.info("Training FIXED neural gating network...")
        
        n_features = X_train.shape[1]
        n_experts = train_expert_preds.shape[1]
        
        # Initialize gate
        hidden_dim = getattr(self.config, 'gate_hidden_dim', None)
        gate = FixedNeuralGate(n_features, n_experts, hidden_dim).to(self.device)
        
        # FIXED: Use config parameters properly
        lr = getattr(self.config, 'gate_learning_rate', 1e-4)
        weight_decay = getattr(self.config, 'gate_weight_decay', 1e-4)
        optimizer = torch.optim.Adam(gate.parameters(), lr=lr, weight_decay=weight_decay)
        
        # FIXED: DON'T scale expert predictions - they're already in target space!
        # Only normalize for numerical stability if needed
        y_scale = max(np.std(y_train), 1.0)  # Prevent division by very small numbers
        
        # Convert to tensors - NO SCALING OF EXPERT PREDICTIONS!
        X_train_t = torch.from_numpy(X_train.astype(np.float32)).to(self.device)
        X_val_t = torch.from_numpy(X_val.astype(np.float32)).to(self.device)
        y_train_t = torch.from_numpy(y_train.astype(np.float32)).to(self.device)
        y_val_t = torch.from_numpy(y_val.astype(np.float32)).to(self.device)
        
        # Expert predictions stay in original scale
        train_preds_t = torch.from_numpy(train_expert_preds.astype(np.float32)).to(self.device)
        val_preds_t = torch.from_numpy(val_expert_preds.astype(np.float32)).to(self.device)
        
        # Use relative loss for better training
        def relative_loss(pred, target):
            # Relative error is more stable for price prediction
            relative_error = (pred - target) / (target.abs() + y_scale * 0.1)
            return torch.mean(relative_error ** 2)
        
        # Training loop
        best_val_loss = float('inf')
        patience = 0
        max_epochs = getattr(self.config, 'gate_epochs', 100)
        patience_limit = getattr(self.config, 'gate_patience', 10)
        clip_grad = getattr(self.config, 'gate_clip_grad', 1.0)
        
        train_losses = []
        val_losses = []
        
        for epoch in range(max_epochs):
            # Training phase
            gate.train()
            optimizer.zero_grad()
            
            train_pred, train_weights = gate(X_train_t, train_preds_t)
            train_loss = relative_loss(train_pred, y_train_t)
            
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(gate.parameters(), clip_grad)
            optimizer.step()
            
            # Validation phase
            gate.eval()
            with torch.no_grad():
                val_pred, val_weights = gate(X_val_t, val_preds_t)
                val_loss = relative_loss(val_pred, y_val_t)
            
            train_losses.append(train_loss.item())
            val_losses.append(val_loss.item())
            
            # Early stopping
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                patience = 0
                best_state = gate.state_dict().copy()
            else:
                patience += 1
                if patience >= patience_limit:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            # Logging
            if epoch % 20 == 0 or epoch < 10:
                # Convert back to interpretable scale
                train_rmse = np.sqrt(train_loss.item()) * y_scale
                val_rmse = np.sqrt(val_loss.item()) * y_scale
                self.logger.info(f"Epoch {epoch:3d}: Train RMSE={train_rmse:8.0f}, Val RMSE={val_rmse:8.0f}")
                
                # Log weight statistics
                if epoch % 40 == 0:
                    weights_mean = train_weights.mean(dim=0).detach().cpu().numpy()
                    weights_str = ", ".join([f"{w:.3f}" for w in weights_mean])
                    self.logger.info(f"         Avg expert weights: [{weights_str}]")
        
        # Load best state
        if 'best_state' in locals():
            gate.load_state_dict(best_state)
            self.logger.info(f"Loaded best model with validation RMSE: {np.sqrt(best_val_loss) * y_scale:.0f}")
        
        return gate
    
    def predict_with_gate(self, gate: FixedNeuralGate, X: np.ndarray, 
                         expert_predictions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions using the trained gate"""
        gate.eval()
        
        with torch.no_grad():
            X_t = torch.from_numpy(X.astype(np.float32)).to(self.device)
            expert_preds_t = torch.from_numpy(expert_predictions.astype(np.float32)).to(self.device)
            
            predictions, weights = gate(X_t, expert_preds_t)
            
            return predictions.cpu().numpy(), weights.cpu().numpy()


class FeatureBasedBinning:
    """Alternative: Feature-based binning that doesn't need target values"""
    
    def __init__(self, logger):
        self.logger = logger
        self.bin_rules = None
        
    def create_feature_bins(self, X_train: np.ndarray, y_train: np.ndarray, 
                          feature_names: list, n_bins: int = 3):
        """Create binning rules based on features, not targets"""
        
        # Example rules for real estate
        # You can customize these based on your feature names
        rules = []
        
        # Try to find sqft feature
        sqft_idx = self._find_feature_idx(feature_names, ['sqft', 'area', 'size'])
        bedroom_idx = self._find_feature_idx(feature_names, ['bedroom', 'bed', 'room'])
        
        if sqft_idx is not None:
            sqft_values = X_train[:, sqft_idx]
            sqft_thresholds = np.percentile(sqft_values, [33, 67])
            
            rules.append({
                'feature_idx': sqft_idx,
                'feature_name': feature_names[sqft_idx],
                'thresholds': sqft_thresholds,
                'type': 'continuous'
            })
            
        # Add more rules based on your specific features
        self.bin_rules = rules
        self.logger.info(f"Created {len(rules)} feature-based binning rules")
        
        return self
    
    def assign_bins(self, X: np.ndarray) -> np.ndarray:
        """Assign bins based on features"""
        n_samples = X.shape[0]
        bin_assignments = np.zeros(n_samples, dtype=int)
        
        if not self.bin_rules:
            return bin_assignments  # All go to bin 0
            
        for rule in self.bin_rules:
            feature_values = X[:, rule['feature_idx']]
            thresholds = rule['thresholds'] 
            
            # Simple binning logic
            bin_assignments = np.where(
                feature_values < thresholds[0], 0,
                np.where(feature_values < thresholds[1], 1, 2)
            )
            
        return bin_assignments
    
    def _find_feature_idx(self, feature_names: list, search_terms: list) -> int:
        """Find feature index by searching for terms"""
        for i, name in enumerate(feature_names):
            for term in search_terms:
                if term.lower() in name.lower():
                    return i
        return None