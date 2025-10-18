'''
Defines the training pipeline for the learned dynamics models.

This module provides a `Training` class to handle the training loop, loss
computation, and model saving for the autoencoder and latent dynamics models.
'''

import os
from typing import Optional
import torch
from torch.utils.data import DataLoader
from .models import Encoder, Decoder, LatentDynamics


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


# =============================================================================
# Data Normalization Utilities
# =============================================================================

def needs_normalization(config):
    """
    Determine if normalization is needed based on activation functions.

    Normalization is automatically enabled if any network uses tanh or sigmoid.

    Args:
        config: Configuration object or dict with activation settings

    Returns:
        bool: True if normalization should be applied

    Example:
        >>> config = ExperimentConfig(output_activation='tanh')
        >>> if needs_normalization(config):
        ...     norm_params = compute_normalization_params(data)
    """
    # Handle both dict and object configs
    if hasattr(config, '__dict__'):
        config_dict = config.__dict__
    else:
        config_dict = config

    activations_to_check = [
        config_dict.get('encoder_activation'),
        config_dict.get('decoder_activation'),
        config_dict.get('latent_dynamics_activation'),
        config_dict.get('output_activation')
    ]

    bounded_activations = {'tanh', 'sigmoid'}

    for act in activations_to_check:
        if act in bounded_activations:
            return True

    return False


def compute_normalization_params(data, method='minmax'):
    """
    Compute normalization parameters from training data.

    Args:
        data: Training data array of shape (N, D)
        method: 'minmax' for [-1, 1] scaling, 'minmax_01' for [0, 1] scaling

    Returns:
        dict: Normalization parameters

    Example:
        >>> import numpy as np
        >>> data = np.random.randn(1000, 3)
        >>> params = compute_normalization_params(data, method='minmax')
    """
    import numpy as np

    if method == 'minmax':
        data_min = np.min(data, axis=0)
        data_max = np.max(data, axis=0)
        data_range = data_max - data_min
        data_range[data_range == 0] = 1.0

        return {
            'method': 'minmax',
            'min': data_min.tolist(),
            'max': data_max.tolist(),
            'range': data_range.tolist()
        }
    elif method == 'minmax_01':
        data_min = np.min(data, axis=0)
        data_max = np.max(data, axis=0)
        data_range = data_max - data_min
        data_range[data_range == 0] = 1.0

        return {
            'method': 'minmax_01',
            'min': data_min.tolist(),
            'max': data_max.tolist(),
            'range': data_range.tolist()
        }
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def normalize_data(data, params):
    """Normalize data using precomputed parameters."""
    import numpy as np

    if params['method'] == 'minmax':
        data_min = np.array(params['min'])
        data_range = np.array(params['range'])
        return 2.0 * (data - data_min) / data_range - 1.0
    elif params['method'] == 'minmax_01':
        data_min = np.array(params['min'])
        data_range = np.array(params['range'])
        return (data - data_min) / data_range
    else:
        raise ValueError(f"Unknown normalization method: {params['method']}")


def denormalize_data(data, params):
    """Denormalize data back to original scale."""
    import numpy as np

    if params['method'] == 'minmax':
        data_min = np.array(params['min'])
        data_range = np.array(params['range'])
        return (data + 1.0) * data_range / 2.0 + data_min
    elif params['method'] == 'minmax_01':
        data_min = np.array(params['min'])
        data_range = np.array(params['range'])
        return data * data_range + data_min
    else:
        raise ValueError(f"Unknown normalization method: {params['method']}")


# =============================================================================
# Complete Training Pipeline
# =============================================================================

def train_autoencoder_dynamics(
    x_train,
    y_train,
    x_val,
    y_val,
    config,
    verbose=True,
    progress_interval=50
):
    """
    Train autoencoder and latent dynamics models with full pipeline.

    This function implements the complete training workflow including:
    - Data normalization (if needed based on activations)
    - Model initialization
    - Three-loss training: reconstruction, dynamics reconstruction, dynamics consistency
    - Early stopping and learning rate scheduling
    - Validation monitoring

    Args:
        x_train: Training current states (N_train, D)
        y_train: Training next states (N_train, D)
        x_val: Validation current states (N_val, D)
        y_val: Validation next states (N_val, D)
        config: ExperimentConfig instance with architecture and training settings
        verbose: Print progress messages
        progress_interval: Print progress every N epochs

    Returns:
        Dictionary with:
            - 'encoder': Trained encoder model
            - 'decoder': Trained decoder model
            - 'latent_dynamics': Trained latent dynamics model
            - 'device': torch device used
            - 'train_losses': Dictionary of training losses by epoch
            - 'val_losses': Dictionary of validation losses by epoch
            - 'final_epoch': Final epoch number
            - 'training_time': Total training time in seconds
            - 'norm_params': Normalization parameters (if used)

    Example:
        >>> from MorseGraph.utils import ExperimentConfig, generate_map_trajectory_data
        >>> config = ExperimentConfig(latent_dim=2, hidden_dim=32)
        >>> X, Y, _ = generate_map_trajectory_data(my_map, 1000, 20, domain)
        >>> # Split into train/val
        >>> split = int(0.8 * len(X))
        >>> result = train_autoencoder_dynamics(
        ...     X[:split], Y[:split], X[split:], Y[split:], config
        ... )
    """
    import numpy as np
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    import time
    from .utils import count_parameters, format_time

    # Data normalization
    use_normalization = needs_normalization(config)
    if use_normalization:
        # Determine normalization method
        decoder_activation = (config.decoder_activation if config.decoder_activation is not None
                            else config.output_activation)

        if decoder_activation == 'sigmoid':
            norm_method = 'minmax_01'
            norm_range_str = '[0, 1]'
        else:
            norm_method = 'minmax'
            norm_range_str = '[-1, 1]'

        if verbose:
            print(f"\nData Normalization:")
            print(f"  Detected bounded activations (tanh/sigmoid)")
            print(f"  Normalizing data to {norm_range_str} range")

        norm_params = compute_normalization_params(np.vstack([x_train, y_train]), method=norm_method)
        x_train = normalize_data(x_train, norm_params)
        y_train = normalize_data(y_train, norm_params)
        x_val = normalize_data(x_val, norm_params)
        y_val = normalize_data(y_val, norm_params)

        if verbose:
            print(f"  Data range after normalization: [{x_train.min():.3f}, {x_train.max():.3f}]")
    else:
        norm_params = None
        if verbose:
            print(f"\nNo bounded activations detected - skipping normalization")

    # Convert to tensors
    x_train_tensor = torch.FloatTensor(x_train)
    y_train_tensor = torch.FloatTensor(y_train)
    x_val_tensor = torch.FloatTensor(x_val)
    y_val_tensor = torch.FloatTensor(y_val)

    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Model setup
    encoder = Encoder(
        config.input_dim,
        config.latent_dim,
        config.hidden_dim,
        config.num_layers,
        output_activation=config.encoder_activation or config.output_activation
    ).to(device)
    decoder = Decoder(
        config.latent_dim,
        config.input_dim,
        config.hidden_dim,
        config.num_layers,
        output_activation=config.decoder_activation or config.output_activation
    ).to(device)
    latent_dynamics = LatentDynamics(
        config.latent_dim,
        config.hidden_dim,
        config.num_layers,
        output_activation=config.latent_dynamics_activation or config.output_activation
    ).to(device)

    # Count parameters
    encoder_params = count_parameters(encoder)
    decoder_params = count_parameters(decoder)
    dynamics_params = count_parameters(latent_dynamics)
    total_params = encoder_params + decoder_params + dynamics_params

    if verbose:
        print(f"\nModel Architecture:")
        print(f"  Encoder:         {encoder_params:,} parameters")
        print(f"  Decoder:         {decoder_params:,} parameters")
        print(f"  Latent Dynamics: {dynamics_params:,} parameters")
        print(f"  Total:           {total_params:,} parameters")
        print(f"  Training data:   {len(x_train):,} samples")
        print(f"  Device:          {device}")

    # Training setup
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    parameters = list(encoder.parameters()) + list(decoder.parameters()) + list(latent_dynamics.parameters())
    optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    mse_loss = torch.nn.MSELoss()

    # Loss weights
    w_recon = config.w_recon
    w_dyn_recon = config.w_dyn_recon
    w_dyn_cons = config.w_dyn_cons

    # Loss tracking
    train_losses = {'reconstruction': [], 'dynamics_recon': [], 'dynamics_consistency': [], 'total': []}
    val_losses = {'reconstruction': [], 'dynamics_recon': [], 'dynamics_consistency': [], 'total': []}

    # Training loop
    if verbose:
        print(f"\nTraining {config.num_epochs} epochs (progress shown every {progress_interval} epochs)...")

    training_start_time = time.time()
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(config.num_epochs):
        # Training phase
        encoder.train()
        decoder.train()
        latent_dynamics.train()

        epoch_loss_recon = 0.0
        epoch_loss_dyn_recon = 0.0
        epoch_loss_dyn_cons = 0.0
        epoch_loss_total = 0.0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            z_t = encoder(x_batch)
            x_t_recon = decoder(z_t)
            z_t_next_pred = latent_dynamics(z_t)
            x_t_next_pred = decoder(z_t_next_pred)
            z_t_next_true = encoder(y_batch)

            loss_recon = mse_loss(x_t_recon, x_batch)
            loss_dyn_recon = mse_loss(x_t_next_pred, y_batch)
            loss_dyn_cons = mse_loss(z_t_next_pred, z_t_next_true)
            loss_total = w_recon * loss_recon + w_dyn_recon * loss_dyn_recon + w_dyn_cons * loss_dyn_cons

            loss_total.backward()
            optimizer.step()

            epoch_loss_recon += loss_recon.item()
            epoch_loss_dyn_recon += loss_dyn_recon.item()
            epoch_loss_dyn_cons += loss_dyn_cons.item()
            epoch_loss_total += loss_total.item()

        # Average training losses
        num_batches = len(train_loader)
        train_losses['reconstruction'].append(epoch_loss_recon / num_batches)
        train_losses['dynamics_recon'].append(epoch_loss_dyn_recon / num_batches)
        train_losses['dynamics_consistency'].append(epoch_loss_dyn_cons / num_batches)
        train_losses['total'].append(epoch_loss_total / num_batches)

        # Validation phase
        encoder.eval()
        decoder.eval()
        latent_dynamics.eval()

        val_loss_recon = 0.0
        val_loss_dyn_recon = 0.0
        val_loss_dyn_cons = 0.0
        val_loss_total = 0.0

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                z_t = encoder(x_batch)
                x_t_recon = decoder(z_t)
                z_t_next_pred = latent_dynamics(z_t)
                x_t_next_pred = decoder(z_t_next_pred)
                z_t_next_true = encoder(y_batch)

                loss_recon = mse_loss(x_t_recon, x_batch)
                loss_dyn_recon = mse_loss(x_t_next_pred, y_batch)
                loss_dyn_cons = mse_loss(z_t_next_pred, z_t_next_true)
                loss_total = w_recon * loss_recon + w_dyn_recon * loss_dyn_recon + w_dyn_cons * loss_dyn_cons

                val_loss_recon += loss_recon.item()
                val_loss_dyn_recon += loss_dyn_recon.item()
                val_loss_dyn_cons += loss_dyn_cons.item()
                val_loss_total += loss_total.item()

        # Average validation losses
        num_batches = len(val_loader)
        val_losses['reconstruction'].append(val_loss_recon / num_batches)
        val_losses['dynamics_recon'].append(val_loss_dyn_recon / num_batches)
        val_losses['dynamics_consistency'].append(val_loss_dyn_cons / num_batches)
        val_losses['total'].append(val_loss_total / num_batches)

        # Print progress
        if verbose and (epoch + 1) % progress_interval == 0:
            print(f"  Epoch {epoch + 1}/{config.num_epochs}")
            print(f"    Train: Recon={train_losses['reconstruction'][-1]:.6f} | "
                  f"DynRecon={train_losses['dynamics_recon'][-1]:.6f} | "
                  f"DynCons={train_losses['dynamics_consistency'][-1]:.6f} | "
                  f"Total={train_losses['total'][-1]:.6f}")
            print(f"    Val:   Recon={val_losses['reconstruction'][-1]:.6f} | "
                  f"DynRecon={val_losses['dynamics_recon'][-1]:.6f} | "
                  f"DynCons={val_losses['dynamics_consistency'][-1]:.6f} | "
                  f"Total={val_losses['total'][-1]:.6f}")

        # Early stopping check
        current_val_loss = val_losses['total'][-1]
        if current_val_loss < best_val_loss - config.min_delta:
            best_val_loss = current_val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        scheduler.step(current_val_loss)

        if patience_counter >= config.early_stopping_patience:
            if verbose:
                print(f"  Early stopping triggered at epoch {epoch + 1}")
            break

    total_training_time = time.time() - training_start_time
    if verbose:
        print(f"  Training completed in {format_time(total_training_time)}")

    return {
        'encoder': encoder,
        'decoder': decoder,
        'latent_dynamics': latent_dynamics,
        'device': device,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'final_epoch': epoch + 1,
        'training_time': total_training_time,
        'norm_params': norm_params,
    }
