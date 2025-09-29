"""
Dimension reduction abstractions and implementations.

This module handles the geometry (encoding/decoding) component of the ML pipeline,
providing a unified interface for various dimension reduction techniques.
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

class AbstractReducer(ABC):
    """Abstract base class for dimension reduction techniques."""
    
    @abstractmethod
    def fit(self, X: np.ndarray, Y: Optional[np.ndarray] = None):
        """
        Learn the transformation from data.
        
        :param X: Input data of shape (n_samples, n_features)
        :param Y: Optional target data for dynamics-aware reduction
        """
        pass
    
    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data to the latent space.
        
        :param X: Input data of shape (n_samples, n_features)
        :return: Transformed data of shape (n_samples, latent_dim)
        """
        pass
    
    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """
        Transform data back from latent space (optional).
        
        :param Z: Latent data of shape (n_samples, latent_dim)
        :return: Reconstructed data of shape (n_samples, n_features)
        """
        raise NotImplementedError("Inverse transform not implemented for this reducer")


class IdentityReducer(AbstractReducer):
    """Pass-through reducer for baseline comparisons (Z = X)."""
    
    def fit(self, X: np.ndarray, Y: Optional[np.ndarray] = None):
        """Identity transform requires no fitting."""
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return input unchanged."""
        return X.copy()
    
    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """Return input unchanged."""
        return Z.copy()


# Sklearn-based implementations
try:
    from sklearn.base import BaseEstimator
    from sklearn.decomposition import PCA
    from sklearn.manifold import Isomap
    
    class SklearnReducer(AbstractReducer):
        """Wrapper for sklearn dimension reduction models."""
        
        def __init__(self, sklearn_model: BaseEstimator):
            """
            :param sklearn_model: Any sklearn model with fit/transform interface
            """
            self.model = sklearn_model
            self._fitted = False
        
        def fit(self, X: np.ndarray, Y: Optional[np.ndarray] = None):
            """Fit the sklearn model."""
            self.model.fit(X)
            self._fitted = True
            return self
        
        def transform(self, X: np.ndarray) -> np.ndarray:
            """Transform using the fitted sklearn model."""
            if not self._fitted:
                raise ValueError("Model must be fitted before transform")
            return self.model.transform(X)
        
        def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
            """Use inverse_transform if available."""
            if not hasattr(self.model, 'inverse_transform'):
                raise NotImplementedError(f"{type(self.model).__name__} does not support inverse_transform")
            return self.model.inverse_transform(Z)

except ImportError:
    class SklearnReducer(AbstractReducer):
        def __init__(self, *args, **kwargs):
            raise ImportError("scikit-learn is required for SklearnReducer")


# PyTorch-based implementations  
try:
    import torch
    from torch import nn
    
    class AEReducer(AbstractReducer):
        """Autoencoder-based dimension reduction using PyTorch."""
        
        def __init__(self, encoder: nn.Module, decoder: nn.Module, learning_rate: float = 0.001):
            """
            :param encoder: PyTorch encoder model
            :param decoder: PyTorch decoder model  
            :param learning_rate: Learning rate for training
            """
            self.encoder = encoder
            self.decoder = decoder
            self.lr = learning_rate
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.encoder.to(self.device)
            self.decoder.to(self.device)
            self._fitted = False
        
        def fit(self, X: np.ndarray, Y: Optional[np.ndarray] = None):
            """Train the autoencoder on reconstruction loss."""
            # Simple training loop (can be extended)
            optimizer = torch.optim.Adam(
                list(self.encoder.parameters()) + list(self.decoder.parameters()),
                lr=self.lr
            )
            loss_fn = nn.MSELoss()
            
            # Convert to tensor
            X_tensor = torch.FloatTensor(X).to(self.device)
            
            # Basic training (simplified)
            self.encoder.train()
            self.decoder.train()
            
            for epoch in range(100):  # Fixed epochs for simplicity
                optimizer.zero_grad()
                z = self.encoder(X_tensor)
                x_recon = self.decoder(z)
                loss = loss_fn(x_recon, X_tensor)
                loss.backward()
                optimizer.step()
            
            self._fitted = True
            return self
        
        def transform(self, X: np.ndarray) -> np.ndarray:
            """Encode data to latent space."""
            if not self._fitted:
                raise ValueError("Model must be fitted before transform")
                
            self.encoder.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(self.device)
                Z_tensor = self.encoder(X_tensor)
                return Z_tensor.cpu().numpy()
        
        def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
            """Decode data from latent space."""
            if not self._fitted:
                raise ValueError("Model must be fitted before inverse_transform")
                
            self.decoder.eval()
            with torch.no_grad():
                Z_tensor = torch.FloatTensor(Z).to(self.device)
                X_tensor = self.decoder(Z_tensor)
                return X_tensor.cpu().numpy()

except ImportError:
    class AEReducer(AbstractReducer):
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for AEReducer")