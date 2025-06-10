# models/phase3/gates/perfect_routing_neural_gate.py
"""
Perfect Routing Neural Gate - Option A

This neural gate learns to mimic the exact hard assignment logic from the baseline.
Instead of blending experts, it learns to route each house to the correct expert.

Key insight: The baseline works because experts are specialists who only see their price range.
The neural gate should learn to replicate this perfect routing from features alone.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error


class PerfectRoutingNeuralGate(nn.Module):
    """
    Neural gate that learns to route houses to the correct price-range expert
    
    Instead of blending, this learns: features → bin assignment → use that bin's expert
    """
    
    def __init__(self, feature_dim: int, num_bins: int):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_bins = num_bins
        
        # Feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.1),
        )
        
        # Bin classifier (learns which price range this house belongs to)
        self.bin_classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, num_bins)
        )
        
        # Conservative initialization
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, features: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Forward pass: Learn which bin each house belongs to
        
        Args:
            features: [batch_size, feature_dim]
            temperature: Controls softmax sharpness
        
        Returns:
            bin_probabilities: [batch_size, num_bins] - probability for each bin
        """
        # Extract features
        encoded_features = self.feature_encoder(features)
        
        # Get bin assignment logits
        bin_logits = self.bin_classifier(encoded_features)
        
        # Apply temperature scaling
        bin_logits = bin_logits / temperature
        
        # Convert to bin probabilities
        bin_probabilities = torch.softmax(bin_logits, dim=1)
        
        return bin_probabilities


class PerfectRoutingGateTrainer:
    """
    Trainer for perfect routing neural gate
    
    Key difference: Trains to predict BIN ASSIGNMENT, not final price
    """
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Scalers
        self.feature_scaler = StandardScaler()
        
    def train_gate(self, X_train: np.ndarray, X_val: np.ndarray,
                   y_train: np.ndarray, y_val: np.ndarray,
                   train_bins: np.ndarray, val_bins: np.ndarray,
                   boundaries: np.ndarray, trained_experts: list) -> PerfectRoutingNeuralGate:
        """
        Train gate to learn perfect routing (features → bin assignment)
        
        Args:
            X_train/X_val: Input features
            y_train/y_val: True targets (for evaluation only)
            train_bins/val_bins: Target bin assignments (THIS IS WHAT WE LEARN)
            boundaries: Bin boundaries for inference
            trained_experts: List of trained specialist experts
        """
        self.logger.info("Training perfect routing neural gate...")
        self.logger.info("Goal: Learn features → bin assignment (mimic hard routing)")
        
        # Store for later use
        self.boundaries = boundaries
        self.trained_experts = trained_experts
        
        # Scale features
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_val_scaled = self.feature_scaler.transform(X_val)
        
        # Log target distribution
        self._log_routing_targets(train_bins, val_bins, boundaries)
        
        # Create gate
        n_features = X_train_scaled.shape[1]
        n_bins = len(trained_experts)
        
        gate = PerfectRoutingNeuralGate(n_features, n_bins).to(self.device)
        
        # Training parameters
        lr = getattr(self.config, 'gate_learning_rate', 1e-3)
        epochs = getattr(self.config, 'gate_epochs', 200)
        
        optimizer = torch.optim.AdamW(gate.parameters(), lr=lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Loss function for classification
        criterion = nn.CrossEntropyLoss()
        
        # Convert to tensors
        X_train_t = torch.from_numpy(X_train_scaled.astype(np.float32)).to(self.device)
        X_val_t = torch.from_numpy(X_val_scaled.astype(np.float32)).to(self.device)
        train_bins_t = torch.from_numpy(train_bins.astype(np.int64)).to(self.device)
        val_bins_t = torch.from_numpy(val_bins.astype(np.int64)).to(self.device)
        
        # Training loop
        best_val_acc = 0.0
        best_val_r2 = -np.inf
        patience = 0
        best_state = None
        
        for epoch in range(epochs):
            # Temperature annealing (start high, reduce for sharp decisions)
            progress = epoch / epochs
            temperature = 2.0 * (1 - progress) + 0.5 * progress
            
            # Training phase
            gate.train()
            optimizer.zero_grad()
            
            # Get bin probabilities
            bin_probs = gate(X_train_t, temperature)
            
            # Classification loss (learn to predict correct bin)
            loss = criterion(torch.log(bin_probs + 1e-8), train_bins_t)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gate.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            # Validation phase
            gate.eval()
            with torch.no_grad():
                val_bin_probs = gate(X_val_t, temperature)
                val_loss = criterion(torch.log(val_bin_probs + 1e-8), val_bins_t)
                
                # Calculate bin assignment accuracy
                predicted_bins = torch.argmax(val_bin_probs, dim=1)
                bin_accuracy = (predicted_bins == val_bins_t).float().mean().item()
                
                # Calculate R² using the routed predictions
                val_r2 = self._calculate_routing_r2(
                    val_bin_probs.cpu().numpy(), X_val, y_val
                )
            
            # Track best model based on R² (most important metric)
            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                best_val_acc = bin_accuracy
                patience = 0
                best_state = gate.state_dict().copy()
            else:
                patience += 1
                if patience >= 25:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            # Logging
            if epoch % 25 == 0:
                self.logger.info(f"Epoch {epoch:3d}: Loss={loss.item():.4f}, "
                               f"Bin Acc={bin_accuracy:.3f}, R²={val_r2:.4f}, Temp={temperature:.2f}")
                
                if epoch % 50 == 0:
                    # Log bin prediction distribution
                    bin_pred_dist = predicted_bins.cpu().numpy()
                    bin_counts = [np.sum(bin_pred_dist == i) for i in range(n_bins)]
                    self.logger.info(f"         Predicted bin counts: {bin_counts}")
        
        # Load best model
        if best_state is not None:
            gate.load_state_dict(best_state)
            self.logger.info(f"Loaded best model: Bin Acc={best_val_acc:.3f}, R²={best_val_r2:.4f}")
        
        return gate
    
    def _log_routing_targets(self, train_bins: np.ndarray, val_bins: np.ndarray, boundaries: np.ndarray):
        """Log information about routing targets"""
        n_bins = len(boundaries) + 1
        
        self.logger.info("Routing target analysis:")
        self.logger.info(f"  Bin boundaries: {[f'${b:,.0f}' for b in boundaries]}")
        
        # Log train distribution
        train_counts = [np.sum(train_bins == i) for i in range(n_bins)]
        self.logger.info(f"  Train bin distribution: {train_counts}")
        
        # Log val distribution
        val_counts = [np.sum(val_bins == i) for i in range(n_bins)]
        self.logger.info(f"  Val bin distribution: {val_counts}")
        
        # Check for imbalanced bins
        min_count = min(train_counts)
        max_count = max(train_counts)
        if max_count > 3 * min_count:
            self.logger.warning(f"⚠️  Imbalanced bins detected: {min_count} to {max_count}")
            self.logger.warning("    This may hurt routing accuracy")
    
    def _calculate_routing_r2(self, bin_probs: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> float:
        """Calculate R² by routing predictions through appropriate experts"""
        try:
            # Get hard bin assignments (most likely bin for each house)
            predicted_bins = np.argmax(bin_probs, axis=1)
            
            # Route each house to its predicted expert
            predictions = np.zeros(len(y_val))
            
            for bin_idx in range(len(self.trained_experts)):
                # Find houses assigned to this bin
                mask = predicted_bins == bin_idx
                
                if mask.sum() > 0:
                    # Get predictions from this bin's expert
                    try:
                        bin_predictions = self.trained_experts[bin_idx].predict(X_val[mask])
                        predictions[mask] = bin_predictions
                    except Exception as e:
                        # If expert fails, use mean prediction
                        predictions[mask] = np.mean(y_val)
                        self.logger.debug(f"Expert {bin_idx} failed: {e}")
            
            # Calculate R²
            r2 = r2_score(y_val, predictions)
            return r2
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate routing R²: {e}")
            return -999.0
    
    def predict_with_gate(self, gate: PerfectRoutingNeuralGate, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions using the trained routing gate"""
        
        # Scale inputs
        X_scaled = self.feature_scaler.transform(X)
        
        gate.eval()
        with torch.no_grad():
            X_t = torch.from_numpy(X_scaled.astype(np.float32)).to(self.device)
            
            # Get bin probabilities (use low temperature for sharp decisions)
            bin_probs = gate(X_t, temperature=0.1)
            bin_probs_np = bin_probs.cpu().numpy()
            
            # Get hard bin assignments
            predicted_bins = np.argmax(bin_probs_np, axis=1)
            
            # Route each prediction to the appropriate expert
            predictions = np.zeros(len(X))
            
            for bin_idx in range(len(self.trained_experts)):
                mask = predicted_bins == bin_idx
                
                if mask.sum() > 0:
                    # Use this bin's specialist expert
                    bin_predictions = self.trained_experts[bin_idx].predict(X[mask])
                    predictions[mask] = bin_predictions
            
            return predictions, bin_probs_np


def test_perfect_routing_gate():
    """Test the perfect routing neural gate"""
    print("🚀 TESTING PERFECT ROUTING NEURAL GATE")
    print("="*50)
    print("Goal: Learn features → bin assignment → use specialist expert")
    print("="*50)
    
    try:
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        from models.phase3.core.config import MoEConfig
        from models.phase3.core.logger import MoELogger
        from models.phase3.data.hybrid_data_manager import HybridDataManager
        from models.phase3.experts.enhanced_expert_factory import EnhancedExpertFactory
        
        # Configuration
        config = MoEConfig(
            min_bin_size=30,
            scaling_method='robust',
            random_state=42,
            gate_epochs=150,
            gate_learning_rate=1e-3
        )
        
        logger = MoELogger(config.log_file)
        
        # Load data
        logger.info("Loading data...")
        data_manager = HybridDataManager(config, logger)
        X, y = data_manager.load_data()
        X, y = data_manager.preprocess_data(X, y)
        X_train, X_val, X_test, y_train, y_val, y_test = data_manager.split_data(X, y)
        
        # Test with 3 bins (where experts are good)
        n_bins = 3
        print(f"\n🔍 Testing {n_bins} bins (good expert quality):")
        
        # Create bins and train experts (EXACT same as baseline)
        boundaries = np.percentile(y_train, np.linspace(0, 100, n_bins + 1))[1:-1]
        train_bins = np.digitize(y_train, boundaries)
        val_bins = np.digitize(y_val, boundaries)
        test_bins = np.digitize(y_test, boundaries)
        
        # Train specialist experts (EXACT same as baseline)
        expert_factory = EnhancedExpertFactory(config, logger)
        selected_experts = expert_factory.select_experts_for_bins(
            X_train, y_train, train_bins, n_bins
        )
        trained_experts = expert_factory.train_experts(
            X_train, y_train, train_bins, selected_experts
        )
        
        # Get baseline performance (EXACT same as baseline)
        baseline_predictions = np.zeros(len(y_test))
        for bin_idx in range(n_bins):
            test_mask = test_bins == bin_idx
            if test_mask.sum() > 0 and bin_idx < len(trained_experts):
                bin_pred = trained_experts[bin_idx].predict(X_test[test_mask])
                baseline_predictions[test_mask] = bin_pred
        
        baseline_r2 = r2_score(y_test, baseline_predictions)
        baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_predictions))
        
        print(f"   Baseline (hard routing): R² = {baseline_r2:.4f}, RMSE = ${baseline_rmse:,.0f}")
        
        # Train perfect routing neural gate
        print(f"\n🧠 Training neural gate to learn perfect routing:")
        
        gate_trainer = PerfectRoutingGateTrainer(config, logger)
        gate = gate_trainer.train_gate(
            X_train, X_val, y_train, y_val,
            train_bins, val_bins, boundaries, trained_experts
        )
        
        # Test neural gate
        neural_predictions, bin_probabilities = gate_trainer.predict_with_gate(gate, X_test)
        
        neural_r2 = r2_score(y_test, neural_predictions)
        neural_rmse = np.sqrt(mean_squared_error(y_test, neural_predictions))
        
        # Results
        print(f"\n📊 ROUTING COMPARISON:")
        print(f"   Baseline (target-based): R² = {baseline_r2:.4f}, RMSE = ${baseline_rmse:,.0f}")
        print(f"   Neural (feature-based):  R² = {neural_r2:.4f}, RMSE = ${neural_rmse:,.0f}")
        
        improvement = ((neural_r2 - baseline_r2) / baseline_r2) * 100 if baseline_r2 > 0 else 0
        
        if neural_r2 > baseline_r2 * 0.95:  # Within 5% is excellent
            print(f"   🎉 SUCCESS: Neural gate learns perfect routing! ({improvement:+.1f}%)")
        elif neural_r2 > baseline_r2 * 0.85:  # Within 15% is good
            print(f"   ✅ GOOD: Neural gate nearly matches baseline ({improvement:+.1f}%)")
        else:
            print(f"   ⚠️  Neural gate needs improvement ({improvement:+.1f}%)")
        
        # Analyze routing accuracy
        predicted_test_bins = np.argmax(bin_probabilities, axis=1)
        routing_accuracy = np.mean(predicted_test_bins == test_bins)
        
        print(f"\n🎯 ROUTING ANALYSIS:")
        print(f"   Bin assignment accuracy: {routing_accuracy:.1%}")
        
        # Show confusion matrix
        print(f"   Routing confusion (predicted vs actual):")
        for actual_bin in range(n_bins):
            actual_mask = test_bins == actual_bin
            if actual_mask.sum() > 0:
                predicted_for_actual = predicted_test_bins[actual_mask]
                distribution = [np.sum(predicted_for_actual == pred_bin) for pred_bin in range(n_bins)]
                percentages = [f"{count}/{actual_mask.sum()}" for count in distribution]
                print(f"     Actual bin {actual_bin}: {percentages}")
        
        # Success criteria
        if routing_accuracy > 0.8 and neural_r2 > baseline_r2 * 0.9:
            print(f"\n🏆 PERFECT ROUTING ACHIEVED!")
            print(f"   Neural gate successfully learns feature-based routing")
            print(f"   Ready for production deployment (no target needed)")
        elif routing_accuracy > 0.6:
            print(f"\n👍 Good routing learned, needs fine-tuning")
        else:
            print(f"\n🔧 Routing needs work - try different architecture or features")
        
        return {
            'baseline_r2': baseline_r2,
            'neural_r2': neural_r2,
            'routing_accuracy': routing_accuracy,
            'improvement': improvement,
            'bin_probabilities': bin_probabilities
        }
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = test_perfect_routing_gate()