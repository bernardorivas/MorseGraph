"""
Latent dynamics abstractions and implementations.

This module handles learning the evolution rule g: Z → Z in the latent space
from latent data transitions.
"""

from abc import ABC, abstractmethod
import numpy as np

class AbstractLatentDynamics(ABC):
    """Abstract base class for learning dynamics in latent space."""
    
    @abstractmethod
    def fit(self, Z_X: np.ndarray, Z_Y: np.ndarray):
        """
        Train the model on latent transitions.
        
        :param Z_X: Current latent states of shape (n_samples, latent_dim)
        :param Z_Y: Next latent states of shape (n_samples, latent_dim)
        """
        pass
    
    @abstractmethod
    def predict(self, Z: np.ndarray) -> np.ndarray:
        """
        Predict the next latent state.
        
        :param Z: Current latent state of shape (n_samples, latent_dim)
        :return: Predicted next latent state of shape (n_samples, latent_dim)
        """
        pass


# Sklearn-based implementations
try:
    from sklearn.base import BaseEstimator
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    
    class SklearnRegressionDynamics(AbstractLatentDynamics):
        """Wrapper for sklearn regression models as dynamics."""
        
        def __init__(self, sklearn_model: BaseEstimator):
            """
            :param sklearn_model: Any sklearn regressor with fit/predict interface
            """
            self.model = sklearn_model
            self._fitted = False
        
        def fit(self, Z_X: np.ndarray, Z_Y: np.ndarray):
            """Fit the sklearn regressor."""
            self.model.fit(Z_X, Z_Y)
            self._fitted = True
            return self
        
        def predict(self, Z: np.ndarray) -> np.ndarray:
            """Predict using the fitted sklearn model."""
            if not self._fitted:
                raise ValueError("Model must be fitted before predict")
            return self.model.predict(Z)

except ImportError:
    class SklearnRegressionDynamics(AbstractLatentDynamics):
        def __init__(self, *args, **kwargs):
            raise ImportError("scikit-learn is required for SklearnRegressionDynamics")


# Linear dynamics (DMD)
class DMDDynamics(AbstractLatentDynamics):
    """Dynamic Mode Decomposition - learns linear dynamics A such that Z_Y = A @ Z_X."""
    
    def __init__(self, regularization: float = 1e-6):
        """
        :param regularization: Small value added to diagonal for numerical stability
        """
        self.regularization = regularization
        self.A = None
        self._fitted = False
    
    def fit(self, Z_X: np.ndarray, Z_Y: np.ndarray):
        """
        Learn the linear operator A using least squares.
        
        Solves: Z_Y = A @ Z_X for A
        """
        # Add regularization for numerical stability
        eye = np.eye(Z_X.shape[1]) * self.regularization
        self.A = Z_Y.T @ Z_X @ np.linalg.pinv(Z_X.T @ Z_X + eye)
        self._fitted = True
        return self
    
    def predict(self, Z: np.ndarray) -> np.ndarray:
        """Apply the learned linear operator."""
        if not self._fitted:
            raise ValueError("Model must be fitted before predict")
        return (self.A @ Z.T).T


# PyTorch-based implementations
try:
    import torch
    from torch import nn
    
    class MLPDynamics(AbstractLatentDynamics):
        """Multi-layer perceptron dynamics using PyTorch."""
        
        def __init__(self, latent_dim: int, hidden_dim: int = 64, num_layers: int = 2, 
                     learning_rate: float = 0.001, epochs: int = 100):
            """
            :param latent_dim: Dimension of the latent space
            :param hidden_dim: Hidden layer dimension
            :param num_layers: Number of hidden layers
            :param learning_rate: Learning rate for training
            :param epochs: Number of training epochs
            """
            self.latent_dim = latent_dim
            self.epochs = epochs
            self.lr = learning_rate
            
            # Build MLP
            layers = []
            layers.append(nn.Linear(latent_dim, hidden_dim))
            layers.append(nn.ReLU())
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, latent_dim))
            
            self.model = nn.Sequential(*layers)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self._fitted = False
        
        def fit(self, Z_X: np.ndarray, Z_Y: np.ndarray):
            """Train the MLP on latent transitions."""
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            loss_fn = nn.MSELoss()
            
            # Convert to tensors
            Z_X_tensor = torch.FloatTensor(Z_X).to(self.device)
            Z_Y_tensor = torch.FloatTensor(Z_Y).to(self.device)
            
            self.model.train()
            for epoch in range(self.epochs):
                optimizer.zero_grad()
                Z_Y_pred = self.model(Z_X_tensor)
                loss = loss_fn(Z_Y_pred, Z_Y_tensor)
                loss.backward()
                optimizer.step()
            
            self._fitted = True
            return self
        
        def predict(self, Z: np.ndarray) -> np.ndarray:
            """Predict next latent state using trained MLP."""
            if not self._fitted:
                raise ValueError("Model must be fitted before predict")
                
            self.model.eval()
            with torch.no_grad():
                Z_tensor = torch.FloatTensor(Z).to(self.device)
                Z_next_tensor = self.model(Z_tensor)
                return Z_next_tensor.cpu().numpy()

except ImportError:
    class MLPDynamics(AbstractLatentDynamics):
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for MLPDynamics")