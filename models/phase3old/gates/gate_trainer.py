# phase3/gates/gate_trainer.py
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any
import logging

class EnhancedMoEGate(nn.Module):
    """Enhanced MoE gate with additional layers"""
    
    def __init__(self, feature_dim: int, num_bins: int, hidden_dim: int = 64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_bins)
        )
        
    def forward(self, features: torch.Tensor, bin_outputs: torch.Tensor) -> torch.Tensor:
        logits = self.layers(features)
        weights = torch.softmax(logits, dim=1)
        return (weights * bin_outputs).sum(dim=1)

class GateTrainer:
    """
    Handles training of the gating network.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(__name__)
    
    def train_gate(self, X_train: np.ndarray, X_val: np.ndarray, 
                   y_train: np.ndarray, y_val: np.ndarray,
                   train_preds: np.ndarray, val_preds: np.ndarray) -> nn.Module:
        """Train enhanced gating network with validation and early stopping"""
        self.logger.info("Training enhanced gating network...")
        
        n_features = X_train.shape[1]
        n_bins = train_preds.shape[1]
        
        gate = EnhancedMoEGate(n_features, n_bins).to(self.device)
        optimizer = torch.optim.AdamW(gate.parameters(), 
                                     lr=self.config['gate_lr'], 
                                     weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                              patience=10, 
                                                              factor=0.5)
        criterion = nn.MSELoss()
        
        # Convert to tensors
        X_train_t = torch.from_numpy(X_train).float().to(self.device)
        X_val_t = torch.from_numpy(X_val).float().to(self.device)
        y_train_t = torch.from_numpy(y_train).float().to(self.device)
        y_val_t = torch.from_numpy(y_val).float().to(self.device)
        train_preds_t = torch.from_numpy(train_preds).float().to(self.device)
        val_preds_t = torch.from_numpy(val_preds).float().to(self.device)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        for epoch in range(self.config['gate_epochs']):
            # Training
            gate.train()
            optimizer.zero_grad()
            train_out = gate(X_train_t, train_preds_t)
            train_loss = criterion(train_out, y_train_t)
            train_loss.backward()
            optimizer.step()
            
            # Validation
            gate.eval()
            with torch.no_grad():
                val_out = gate(X_val_t, val_preds_t)
                val_loss = criterion(val_out, y_val_t)
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = gate.state_dict().copy()
            else:
                patience_counter += 1
            
            if epoch % 20 == 0:
                self.logger.info(f"Epoch {epoch}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")
            
            if patience_counter >= self.config['early_stopping_patience']:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Load best state
        if best_state:
            gate.load_state_dict(best_state)
        
        return gate