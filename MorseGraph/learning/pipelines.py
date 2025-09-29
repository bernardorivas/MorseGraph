"""
Training pipelines for complex workflows.

This module manages scenarios where reduction and dynamics are learned 
simultaneously (end-to-end training).
"""

from typing import Optional, Tuple
import numpy as np

# PyTorch-based pipelines
try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    
    class AutoencoderPipeline:
        """
        Manages joint training of autoencoder and dynamics models.
        
        Optimizes combined loss: Reconstruction Loss + Dynamics Prediction Loss
        """
        
        def __init__(self, encoder: nn.Module, decoder: nn.Module, dynamics: nn.Module,
                     learning_rate: float = 0.001):
            """
            :param encoder: PyTorch encoder model
            :param decoder: PyTorch decoder model  
            :param dynamics: PyTorch latent dynamics model
            :param learning_rate: Learning rate for optimization
            """
            self.encoder = encoder
            self.decoder = decoder
            self.dynamics = dynamics
            self.lr = learning_rate
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.encoder.to(self.device)
            self.decoder.to(self.device)
            self.dynamics.to(self.device)
            
            # Combined optimizer for all parameters
            all_params = (list(encoder.parameters()) + 
                         list(decoder.parameters()) + 
                         list(dynamics.parameters()))
            self.optimizer = torch.optim.Adam(all_params, lr=self.lr)
            
            # Loss functions
            self.recon_loss_fn = nn.MSELoss()
            self.dynamics_loss_fn = nn.MSELoss()
            
            self._fitted = False
        
        def fit(self, X: np.ndarray, Y: np.ndarray, epochs: int = 100, 
                dynamics_weight: float = 0.5, batch_size: int = 32,
                validation_split: Optional[float] = None) -> dict:
            """
            Train the complete pipeline end-to-end.
            
            :param X: Current state data of shape (n_samples, n_features)
            :param Y: Next state data of shape (n_samples, n_features)
            :param epochs: Number of training epochs
            :param dynamics_weight: Weight for dynamics loss (0-1)
            :param batch_size: Batch size for training
            :param validation_split: Fraction of data for validation
            :return: Training history dictionary
            """
            # Prepare data
            X_tensor = torch.FloatTensor(X)
            Y_tensor = torch.FloatTensor(Y)
            
            # Split validation if requested
            if validation_split:
                n_val = int(len(X) * validation_split)
                indices = torch.randperm(len(X))
                train_indices, val_indices = indices[n_val:], indices[:n_val]
                
                X_train, Y_train = X_tensor[train_indices], Y_tensor[train_indices]
                X_val, Y_val = X_tensor[val_indices], Y_tensor[val_indices]
                
                val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=batch_size)
            else:
                X_train, Y_train = X_tensor, Y_tensor
                val_loader = None
            
            # Training data loader
            train_loader = DataLoader(TensorDataset(X_train, Y_train), 
                                    batch_size=batch_size, shuffle=True)
            
            # Training history
            history = {'train_loss': [], 'val_loss': []}
            
            # Training loop
            for epoch in range(epochs):
                train_loss = self._train_epoch(train_loader, dynamics_weight)
                history['train_loss'].append(train_loss)
                
                if val_loader:
                    val_loss = self._validate_epoch(val_loader, dynamics_weight)
                    history['val_loss'].append(val_loss)
                    
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}/{epochs}: train_loss={train_loss:.6f}"
                          + (f", val_loss={val_loss:.6f}" if val_loader else ""))
            
            self._fitted = True
            return history
        
        def _train_epoch(self, data_loader: DataLoader, dynamics_weight: float) -> float:
            """Train for one epoch."""
            self.encoder.train()
            self.decoder.train()
            self.dynamics.train()
            
            total_loss = 0
            for x_t, x_tau in data_loader:
                x_t = x_t.to(self.device)
                x_tau = x_tau.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                z_t = self.encoder(x_t)
                z_tau_pred = self.dynamics(z_t)
                x_t_recon = self.decoder(z_t)
                
                # Losses
                recon_loss = self.recon_loss_fn(x_t_recon, x_t)
                
                # For dynamics loss, encode the true next state
                with torch.no_grad():
                    z_tau_true = self.encoder(x_tau)
                dynamics_loss = self.dynamics_loss_fn(z_tau_pred, z_tau_true)
                
                # Combined loss
                loss = (1 - dynamics_weight) * recon_loss + dynamics_weight * dynamics_loss
                
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            return total_loss / len(data_loader)
        
        def _validate_epoch(self, data_loader: DataLoader, dynamics_weight: float) -> float:
            """Validate for one epoch."""
            self.encoder.eval()
            self.decoder.eval() 
            self.dynamics.eval()
            
            total_loss = 0
            with torch.no_grad():
                for x_t, x_tau in data_loader:
                    x_t = x_t.to(self.device)
                    x_tau = x_tau.to(self.device)
                    
                    # Forward pass
                    z_t = self.encoder(x_t)
                    z_tau_pred = self.dynamics(z_t)
                    x_t_recon = self.decoder(z_t)
                    z_tau_true = self.encoder(x_tau)
                    
                    # Losses
                    recon_loss = self.recon_loss_fn(x_t_recon, x_t)
                    dynamics_loss = self.dynamics_loss_fn(z_tau_pred, z_tau_true)
                    loss = (1 - dynamics_weight) * recon_loss + dynamics_weight * dynamics_loss
                    
                    total_loss += loss.item()
            
            return total_loss / len(data_loader)
        
        def transform(self, X: np.ndarray) -> np.ndarray:
            """Transform data to latent space using trained encoder."""
            if not self._fitted:
                raise ValueError("Pipeline must be fitted before transform")
                
            self.encoder.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(self.device)
                Z_tensor = self.encoder(X_tensor)
                return Z_tensor.cpu().numpy()
        
        def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
            """Transform latent data back to original space using trained decoder."""
            if not self._fitted:
                raise ValueError("Pipeline must be fitted before inverse_transform")
                
            self.decoder.eval()
            with torch.no_grad():
                Z_tensor = torch.FloatTensor(Z).to(self.device)
                X_tensor = self.decoder(Z_tensor)
                return X_tensor.cpu().numpy()
        
        def predict_dynamics(self, Z: np.ndarray) -> np.ndarray:
            """Predict next latent state using trained dynamics model."""
            if not self._fitted:
                raise ValueError("Pipeline must be fitted before predict_dynamics")
                
            self.dynamics.eval()
            with torch.no_grad():
                Z_tensor = torch.FloatTensor(Z).to(self.device)
                Z_next_tensor = self.dynamics(Z_tensor)
                return Z_next_tensor.cpu().numpy()

except ImportError:
    class AutoencoderPipeline:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for AutoencoderPipeline")