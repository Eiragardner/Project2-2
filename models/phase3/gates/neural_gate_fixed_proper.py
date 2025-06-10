# models/phase3/gates/neural_gate_fixed_proper.py
# Fixed implementation that learns INPUT-SPECIFIC expert weights
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler


class ProperNeuralGate(nn.Module):
    """Neural gate that learns INPUT-SPECIFIC expert weights"""
    
    def __init__(self, n_features, n_experts, hidden=64, dropout=0.1):
        super().__init__()
        self.n_experts = n_experts
        
        # Simple but effective architecture
        self.gate_net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_experts)
        )
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """Forward pass - returns INPUT-SPECIFIC weights"""
        logits = self.gate_net(x)
        # Softmax to get probability distribution over experts
        weights = F.softmax(logits, dim=1)
        return weights


class FixedNeuralGateTrainer:
    """Fixed trainer that actually learns input-specific gating"""
    
    def __init__(self, n_features, experts, device='cpu'):
        self.experts = experts
        self.n_experts = len(experts)
        self.device = torch.device(device)
        self.scaler = StandardScaler()
        
        # Create the neural gate
        self.gate = ProperNeuralGate(n_features, self.n_experts).to(self.device)
        
    def train(self, X_train, y_train, X_val, y_val, epochs=200, lr=1e-3):
        """Train gate to learn input-specific expert selection"""
        
        print("🔧 Training neural gate for input-specific expert weighting...")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Convert to tensors
        X_train_t = torch.from_numpy(X_train_scaled.astype(np.float32)).to(self.device)
        X_val_t = torch.from_numpy(X_val_scaled.astype(np.float32)).to(self.device)
        y_train_t = torch.from_numpy(y_train.astype(np.float32)).to(self.device)
        y_val_t = torch.from_numpy(y_val.astype(np.float32)).to(self.device)
        
        # Get expert predictions (these are FIXED during gate training)
        train_expert_preds = self._get_expert_predictions(X_train)
        val_expert_preds = self._get_expert_predictions(X_val)
        
        train_preds_t = torch.from_numpy(train_expert_preds.astype(np.float32)).to(self.device)
        val_preds_t = torch.from_numpy(val_expert_preds.astype(np.float32)).to(self.device)
        
        # Optimizer
        optimizer = torch.optim.Adam(self.gate.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.gate.train()
            optimizer.zero_grad()
            
            # Get INPUT-SPECIFIC weights from neural gate
            weights = self.gate(X_train_t)  # [batch_size, n_experts]
            
            # Compute weighted predictions
            # weights: [batch_size, n_experts], train_preds_t: [batch_size, n_experts]
            weighted_preds = torch.sum(weights * train_preds_t, dim=1)  # [batch_size]
            
            # Loss: MSE between weighted predictions and targets
            loss = F.mse_loss(weighted_preds, y_train_t)
            
            # Optional: Add small regularization to encourage diversity
            # Entropy regularization (higher entropy = more diverse weights)
            entropy_reg = -torch.mean(torch.sum(weights * torch.log(weights + 1e-8), dim=1))
            total_loss = loss + 0.01 * entropy_reg
            
            total_loss.backward()
            optimizer.step()
            
            # Validation phase
            if epoch % 10 == 0:
                self.gate.eval()
                with torch.no_grad():
                    val_weights = self.gate(X_val_t)
                    val_weighted_preds = torch.sum(val_weights * val_preds_t, dim=1)
                    val_loss = F.mse_loss(val_weighted_preds, y_val_t).item()
                
                scheduler.step(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if epoch % 50 == 0:
                    print(f"Epoch {epoch}: train_loss={loss.item():.6f}, val_loss={val_loss:.6f}")
                
                # Early stopping
                if patience_counter >= 30:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        print(f"✅ Gate training completed, best_val_loss = {best_val_loss:.6f}")
        return best_val_loss
    
    def _get_expert_predictions(self, X):
        """Get predictions from all experts"""
        predictions = np.zeros((len(X), self.n_experts))
        for i, expert in enumerate(self.experts):
            try:
                predictions[:, i] = expert.predict(X)
            except Exception as e:
                print(f"Warning: Expert {i} prediction failed: {e}")
                # Fallback to mean of other experts
                other_preds = [self.experts[j].predict(X) for j in range(self.n_experts) if j != i]
                if other_preds:
                    predictions[:, i] = np.mean(other_preds, axis=0)
                else:
                    predictions[:, i] = np.mean(X, axis=1)  # Very basic fallback
        return predictions
    
    def predict(self, X):
        """Make predictions with input-specific expert weighting"""
        X_scaled = self.scaler.transform(X)
        expert_preds = self._get_expert_predictions(X)
        
        X_t = torch.from_numpy(X_scaled.astype(np.float32)).to(self.device)
        expert_preds_t = torch.from_numpy(expert_preds.astype(np.float32)).to(self.device)
        
        self.gate.eval()
        with torch.no_grad():
            # Get INPUT-SPECIFIC weights
            weights = self.gate(X_t)  # [batch_size, n_experts]
            
            # Compute weighted predictions
            predictions = torch.sum(weights * expert_preds_t, dim=1)  # [batch_size]
        
        return predictions.cpu().numpy()
    
    def analyze_weights(self, X_sample, sample_indices=None):
        """Analyze how weights vary across different inputs"""
        X_scaled = self.scaler.transform(X_sample)
        X_t = torch.from_numpy(X_scaled.astype(np.float32)).to(self.device)
        
        self.gate.eval()
        with torch.no_grad():
            weights = self.gate(X_t).cpu().numpy()
        
        print(f"\n📊 Input-Specific Weight Analysis:")
        print(f"Weight statistics across {len(X_sample)} samples:")
        
        for expert_idx in range(self.n_experts):
            expert_weights = weights[:, expert_idx]
            print(f"Expert {expert_idx}: mean={expert_weights.mean():.3f}, "
                  f"std={expert_weights.std():.3f}, "
                  f"min={expert_weights.min():.3f}, "
                  f"max={expert_weights.max():.3f}")
        
        # Show some individual examples
        if sample_indices is None:
            sample_indices = np.random.choice(len(X_sample), min(5, len(X_sample)), replace=False)
        
        print(f"\n📊 Individual Sample Weights:")
        for i, idx in enumerate(sample_indices):
            print(f"Sample {idx}: {weights[idx]}")
        
        return weights