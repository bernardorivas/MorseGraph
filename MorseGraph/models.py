'''
Defines the neural network architectures for the MorseGraph library.

This module contains the PyTorch models for the optional machine learning
components, including the Encoder, Decoder, and the LatentDynamics model.
These are only available if PyTorch is installed.
'''

try:
    import torch
    from torch import nn
    from typing import Optional

    class Encoder(nn.Module):
        """
        Encodes high-dimensional state into a low-dimensional latent space.

        Args:
            input_dim: Dimension of input space
            latent_dim: Dimension of latent space
            hidden_dim: Width of hidden layers (default: 64)
            num_layers: Number of hidden layers (default: 3)
            output_activation: Optional output activation ('tanh', 'sigmoid', or None)
                             None = no activation (recommended for unnormalized data)
        """
        def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 64,
                     num_layers: int = 3, output_activation: Optional[str] = None):
            super().__init__()
            layers = []

            # Input layer
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())

            # Hidden layers
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())

            # Output layer
            layers.append(nn.Linear(hidden_dim, latent_dim))

            # Optional output activation
            if output_activation == 'tanh':
                layers.append(nn.Tanh())
            elif output_activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif output_activation is not None:
                raise ValueError(f"Unknown activation: {output_activation}. Use 'tanh', 'sigmoid', or None")

            self.encoder = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.encoder(x)

    class Decoder(nn.Module):
        """
        Decodes a latent space representation back to the original state space.

        Args:
            latent_dim: Dimension of latent space
            output_dim: Dimension of output space
            hidden_dim: Width of hidden layers (default: 64)
            num_layers: Number of hidden layers (default: 3)
            output_activation: Optional output activation ('tanh', 'sigmoid', or None)
                             'sigmoid' for data normalized to [0, 1]
                             'tanh' for data normalized to [-1, 1]
                             None for unnormalized data (recommended for general use)
        """
        def __init__(self, latent_dim: int, output_dim: int, hidden_dim: int = 64,
                     num_layers: int = 3, output_activation: Optional[str] = None):
            super().__init__()
            layers = []

            # Input layer
            layers.append(nn.Linear(latent_dim, hidden_dim))
            layers.append(nn.ReLU())

            # Hidden layers
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())

            # Output layer
            layers.append(nn.Linear(hidden_dim, output_dim))

            # Optional output activation
            if output_activation == 'tanh':
                layers.append(nn.Tanh())
            elif output_activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif output_activation is not None:
                raise ValueError(f"Unknown activation: {output_activation}. Use 'tanh', 'sigmoid', or None")

            self.decoder = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.decoder(x)

    class LatentDynamics(nn.Module):
        """
        Models the dynamics in the low-dimensional latent space.

        Args:
            latent_dim: Dimension of latent space
            hidden_dim: Width of hidden layers (default: 64)
            num_layers: Number of hidden layers (default: 3)
            output_activation: Optional output activation ('tanh', 'sigmoid', or None)
                             None = no activation (recommended for general dynamics learning)
        """
        def __init__(self, latent_dim: int, hidden_dim: int = 64,
                     num_layers: int = 3, output_activation: Optional[str] = None):
            super().__init__()
            layers = []

            # Input layer
            layers.append(nn.Linear(latent_dim, hidden_dim))
            layers.append(nn.ReLU())

            # Hidden layers
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())

            # Output layer
            layers.append(nn.Linear(hidden_dim, latent_dim))

            # Optional output activation
            if output_activation == 'tanh':
                layers.append(nn.Tanh())
            elif output_activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif output_activation is not None:
                raise ValueError(f"Unknown activation: {output_activation}. Use 'tanh', 'sigmoid', or None")

            self.dynamics = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.dynamics(x)

except ImportError:
    # If torch is not installed, create dummy classes that raise an error upon instantiation.
    class Encoder:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for ML models. Please install it via `pip install torch` or `pip install morsegraph[ml]`.")

    class Decoder:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for ML models. Please install it via `pip install torch` or `pip install morsegraph[ml]`.")

    class LatentDynamics:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for ML models. Please install it via `pip install torch` or `pip install morsegraph[ml]`.")
