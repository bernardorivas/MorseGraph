'''
Defines the neural network architectures for the MorseGraph library.

This module contains the PyTorch models for the optional machine learning
components, including the Encoder, Decoder, and the LatentDynamics model.
These are only available if PyTorch is installed.
'''

try:
    import torch
    from torch import nn

    class Encoder(nn.Module):
        """Encodes high-dimensional state into a low-dimensional latent space."""
        def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 64, num_layers: int = 2):
            super().__init__()
            layers = []
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU(True))
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU(True))
            layers.append(nn.Linear(hidden_dim, latent_dim))
            layers.append(nn.Tanh())
            self.encoder = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.encoder(x)

    class Decoder(nn.Module):
        """Decodes a latent space representation back to the original state space."""
        def __init__(self, latent_dim: int, output_dim: int, hidden_dim: int = 64, num_layers: int = 2):
            super().__init__()
            layers = []
            layers.append(nn.Linear(latent_dim, hidden_dim))
            layers.append(nn.ReLU(True))
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU(True))
            layers.append(nn.Linear(hidden_dim, output_dim))
            layers.append(nn.Sigmoid())  # Assumes input data is normalized to [0, 1]
            self.decoder = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.decoder(x)

    class LatentDynamics(nn.Module):
        """Models the dynamics in the low-dimensional latent space."""
        def __init__(self, latent_dim: int, hidden_dim: int = 64, num_layers: int = 2):
            super().__init__()
            layers = []
            layers.append(nn.Linear(latent_dim, hidden_dim))
            layers.append(nn.ReLU(True))
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU(True))
            layers.append(nn.Linear(hidden_dim, latent_dim))
            layers.append(nn.Tanh())
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
