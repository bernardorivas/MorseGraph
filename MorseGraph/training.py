'''
Defines the training pipeline for the learned dynamics models.

This module provides a `Training` class to handle the training loop, loss
computation, and model saving for the autoencoder and latent dynamics models.
'''

import os
import torch
from torch.utils.data import DataLoader

# Use try-except for type hinting
try:
    from .models import Encoder, Decoder, LatentDynamics
except ImportError:
    from morsegraph.models import Encoder, Decoder, LatentDynamics


class Training:
    """
    Manages the training process for the autoencoder and latent dynamics models.
    (Placeholder for future implementation).
    """

    def __init__(self, encoder: Encoder, decoder: Decoder, dynamics: LatentDynamics, learning_rate: float = 0.001):
        """
        :param encoder: The Encoder model.
        :param decoder: The Decoder model.
        :param dynamics: The LatentDynamics model.
        :param learning_rate: The learning rate for the optimizer.
        """
        if not hasattr(torch, 'nn'):
            raise ImportError("PyTorch is required for Training. Please install it.")

        self.encoder = encoder
        self.decoder = decoder
        self.dynamics = dynamics
        self.lr = learning_rate

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.dynamics.to(self.device)

        # Combine all model parameters for the optimizer
        all_params = list(encoder.parameters()) + list(decoder.parameters()) + list(dynamics.parameters())
        self.optimizer = torch.optim.Adam(all_params, lr=self.lr)

        # Define loss functions
        self.reconstruction_loss_fn = torch.nn.MSELoss()
        self.dynamics_loss_fn = torch.nn.MSELoss()

    def train(self, train_loader: DataLoader, epochs: int, val_loader: Optional[DataLoader] = None, dynamics_weight: float = 0.5):
        """
        Runs the main training loop.

        :param train_loader: DataLoader for the training dataset.
        :param epochs: Number of epochs to train for.
        :param val_loader: Optional DataLoader for a validation dataset.
        :param dynamics_weight: Weight for the dynamics loss (0 to 1).
        """
        print("Training process started...")
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}:")
            self._train_one_epoch(train_loader, dynamics_weight)
            if val_loader:
                self._validate(val_loader, dynamics_weight)
        print("Training finished.")

    def _train_one_epoch(self, data_loader: DataLoader, dynamics_weight: float):
        """
        Performs a single epoch of training.
        """
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
            z_tau_predicted = self.dynamics(z_t)
            x_t_reconstructed = self.decoder(z_t)

            # Reconstruction loss
            recon_loss = self.reconstruction_loss_fn(x_t_reconstructed, x_t)

            # Dynamics loss
            with torch.no_grad():
                z_tau_actual = self.encoder(x_tau)
            dyn_loss = self.dynamics_loss_fn(z_tau_predicted, z_tau_actual)

            # Combined loss
            loss = (1 - dynamics_weight) * recon_loss + dynamics_weight * dyn_loss
            
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(data_loader)
        print(f"    Training Loss: {avg_loss:.6f}")

    def _validate(self, data_loader: DataLoader, dynamics_weight: float):
        """
        Performs validation on the validation set.
        """
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
                z_tau_predicted = self.dynamics(z_t)
                x_t_reconstructed = self.decoder(z_t)
                z_tau_actual = self.encoder(x_tau)

                # Losses
                recon_loss = self.reconstruction_loss_fn(x_t_reconstructed, x_t)
                dyn_loss = self.dynamics_loss_fn(z_tau_predicted, z_tau_actual)
                loss = (1 - dynamics_weight) * recon_loss + dynamics_weight * dyn_loss
                total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        print(f"    Validation Loss: {avg_loss:.6f}")

    def save_models(self, directory: str):
        """
        Saves the trained models to the specified directory.
        """
        os.makedirs(directory, exist_ok=True)
        torch.save(self.encoder.state_dict(), os.path.join(directory, 'encoder.pt'))
        torch.save(self.decoder.state_dict(), os.path.join(directory, 'decoder.pt'))
        torch.save(self.dynamics.state_dict(), os.path.join(directory, 'dynamics.pt'))
        print(f"Models saved to {directory}")

    def load_models(self, directory: str):
        """
        Loads model weights from the specified directory.
        """
        self.encoder.load_state_dict(torch.load(os.path.join(directory, 'encoder.pt')))
        self.decoder.load_state_dict(torch.load(os.path.join(directory, 'decoder.pt')))
        self.dynamics.load_state_dict(torch.load(os.path.join(directory, 'dynamics.pt')))
        print(f"Models loaded from {directory}")
