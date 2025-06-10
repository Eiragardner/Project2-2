# models/phase3/gates/target_aware_neural_gate.py
"""
Clean Target-Aware Neural Gate Architecture

Contains only the neural network architecture and core training methods.
No execution logic - that goes in neural_baseline_model.py
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


class TargetAwareNeuralGate(nn.Module):
    """
    Neural gate that learns soft bin assignments based on features
    
    Learns to route properties to price-range specialists using features only,
    mimicking the successful target-based binning approach.
    """
    
    def __init__(self, feature_dim: int, num_bins: int, hidden_dim: int = 32):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_bins = num_bins
        
        # Feature encoder (learns representations)
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1)
        )
        
        # Bin assignment network (learns which bin each property belongs to)
        self.bin_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_bins)
        )
        
        # Initialize conservatively
        self._init_weights()
    
    def _init_weights(self):
        """Conservative weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, features: torch.Tensor, bin_expert_predictions: torch.Tensor, 
                temperature: float = 2.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: Learn to route properties to appropriate price-range experts
        
        Args:
            features: [batch_size, feature_dim] - property features
            bin_expert_predictions: [batch_size, num_bins] - predictions from bin-specific experts
            temperature: Controls softmax sharpness
        
        Returns:
            prediction: [batch_size] - final prediction
            bin_weights: [batch_size, num_bins] - soft bin assignments
        """
        # Extract features
        encoded_features = self.feature_encoder(features)
        
        # Get bin assignment logits
        bin_logits = self.bin_classifier(encoded_features)
        
        # Apply temperature scaling
        bin_logits = bin_logits / temperature
        
        # Convert to soft bin assignments (probabilities)
        bin_weights = torch.softmax(bin_logits, dim=1)
        
        # Weighted combination of bin-specific expert predictions
        prediction = torch.sum(bin_weights * bin_expert_predictions, dim=1)
        
        return prediction, bin_weights


class TargetAwareGateTrainer:
    """
    Trainer for the target-aware neural gate
    
    Handles the training loop and optimization, but doesn't manage data or evaluation.
    """
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def train_gate(self, X_train: np.ndarray, X_val: np.ndarray,
                   y_train: np.ndarray, y_val: np.ndarray,
                   train_bin_preds: np.ndarray, val_bin_preds: np.ndarray,
                   train_boundaries: np.ndarray) -> TargetAwareNeuralGate:
        """
        Train the target-aware gate
        
        Args:
            X_train/X_val: Input features
            y_train/y_val: True targets  
            train_bin_preds/val_bin_preds: Bin expert predictions
            train_boundaries: Bin boundaries for supervision
        """
        self.logger.info("Training target-aware neural gate...")
        
        n_features = X_train.shape[1]
        n_bins = train_bin_preds.shape[1]
        
        # Create gate
        gate = TargetAwareNeuralGate(n_features, n_bins).to(self.device)
        
        # Training parameters
        lr = getattr(self.config, 'gate_learning_rate', 3e-5)
        weight_decay = getattr(self.config, 'gate_weight_decay', 1e-3)
        epochs = getattr(self.config, 'gate_epochs', 150)
        patience_limit = getattr(self.config, 'gate_patience', 20)
        
        optimizer = torch.optim.AdamW(gate.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=10, factor=0.8
        )
        
        # Prepare tensors
        X_train_t = torch.from_numpy(X_train.astype(np.float32)).to(self.device)
        X_val_t = torch.from_numpy(X_val.astype(np.float32)).to(self.device)
        y_train_t = torch.from_numpy(y_train.astype(np.float32)).to(self.device)
        y_val_t = torch.from_numpy(y_val.astype(np.float32)).to(self.device)
        train_preds_t = torch.from_numpy(train_bin_preds.astype(np.float32)).to(self.device)
        val_preds_t = torch.from_numpy(val_bin_preds.astype(np.float32)).to(self.device)
        
        # Create target bin assignments for supervision
        train_target_bins = np.digitize(y_train, train_boundaries)
        train_target_bins_t = torch.from_numpy(train_target_bins.astype(np.int64)).to(self.device)
        
        # Loss functions
        mse_criterion = nn.MSELoss()
        ce_criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_val_loss = float('inf')
        patience = 0
        best_state = None
        
        for epoch in range(epochs):
            # Temperature annealing
            progress = epoch / epochs
            temperature = 3.0 * (1 - progress) + 1.0 * progress
            
            # Training phase
            gate.train()
            optimizer.zero_grad()
            
            train_pred, train_bin_weights = gate(X_train_t, train_preds_t, temperature)
            
            # Main prediction loss
            pred_loss = mse_criterion(train_pred, y_train_t)
            
            # Bin assignment supervision (teach gate to match target-based binning)
            bin_logits = gate.bin_classifier(gate.feature_encoder(X_train_t))
            bin_supervision_loss = ce_criterion(bin_logits, train_target_bins_t)
            
            # Combined loss (prediction is primary, bin supervision is secondary)
            total_loss = pred_loss + 0.1 * bin_supervision_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(gate.parameters(), 1.0)
            optimizer.step()
            
            # Validation phase
            gate.eval()
            with torch.no_grad():
                val_pred, val_bin_weights = gate(X_val_t, val_preds_t, temperature)
                val_loss = mse_criterion(val_pred, y_val_t).item()
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = 0
                best_state = gate.state_dict().copy()
            else:
                patience += 1
                if patience >= patience_limit:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            # Logging
            if epoch % 25 == 0:
                pred_rmse = np.sqrt(pred_loss.item()) * np.std(y_train)
                val_rmse = np.sqrt(val_loss) * np.std(y_train)
                bin_loss_val = bin_supervision_loss.item()
                
                self.logger.info(f"Epoch {epoch:3d}: Pred RMSE={pred_rmse:8.0f}, "
                               f"Val RMSE={val_rmse:8.0f}, Bin Loss={bin_loss_val:.4f}")
                
                if epoch % 50 == 0:
                    # Log bin utilization
                    weights_mean = train_bin_weights.mean(dim=0).detach().cpu().numpy()
                    weights_str = ", ".join([f"{w:.3f}" for w in weights_mean])
                    self.logger.info(f"         Bin weights: [{weights_str}]")
        
        # Load best model
        if best_state is not None:
            gate.load_state_dict(best_state)
            best_rmse = np.sqrt(best_val_loss) * np.std(y_train)
            self.logger.info(f"Loaded best model with validation RMSE: {best_rmse:.0f}")
        
        return gate
    
    def predict_with_gate(self, gate: TargetAwareNeuralGate, X: np.ndarray, 
                         bin_predictions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions using the trained gate"""
        gate.eval()
        
        with torch.no_grad():
            X_t = torch.from_numpy(X.astype(np.float32)).to(self.device)
            bin_preds_t = torch.from_numpy(bin_predictions.astype(np.float32)).to(self.device)
            
            predictions, bin_weights = gate(X_t, bin_preds_t, temperature=1.0)
            
            return predictions.cpu().numpy(), bin_weights.cpu().numpy()