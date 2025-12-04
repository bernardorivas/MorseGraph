"""
Training visualization functions.

This module provides functions for visualizing training progress and
encoder-decoder roundtrip mappings.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional
import torch

from .utils import finalize_plot


def plot_training_curves(train_losses, val_losses, output_path=None):
    """
    Plot training and validation loss curves.

    Args:
        train_losses: Dict with keys 'reconstruction', 'dynamics_recon', 'dynamics_consistency', 'total'
        val_losses: Dict with same keys as train_losses
        output_path: Path to save figure (if None, displays instead)

    Example:
        >>> plot_training_curves(train_losses, val_losses, 'training_curves.png')
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    loss_types = {
        'total': 'Total Loss',
        'reconstruction': 'Reconstruction: ||D(E(x)) - x||²',
        'dynamics_recon': 'Dynamics Recon: ||D(G(E(x))) - f(x)||²',
        'dynamics_consistency': 'Dynamics Cons: ||G(E(x)) - E(f(x))||²'
    }
    
    # a helper to zoom on the y-axis to the interesting part
    def set_dynamic_ylim(ax, train_loss_history, val_loss_history):
        num_epochs = len(train_loss_history)
        start_epoch = int(num_epochs * 0.2) if num_epochs > 10 else 0
        final_losses = train_loss_history[start_epoch:] + val_loss_history[start_epoch:]
        if len(final_losses) > 0:
            upper_limit = max(final_losses) * 1.2
            ax.set_ylim(bottom=0, top=upper_limit + 1e-6)
        else:
            ax.set_ylim(bottom=0)

    plot_order = ['total', 'reconstruction', 'dynamics_recon', 'dynamics_consistency']

    for i, key in enumerate(plot_order):
        if key in train_losses and key in val_losses:
            ax = axes[i]
            epochs = range(1, len(train_losses[key]) + 1)
            ax.plot(epochs, train_losses[key], label='Train')
            ax.plot(epochs, val_losses[key], label='Val')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title(loss_types.get(key, key))
            ax.legend()
            ax.grid(True, alpha=0.3)
            set_dynamic_ylim(ax, train_losses[key], val_losses[key])

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def _prepare_encoder_decoder_data(X, encoder, decoder, num_samples, latent_grid_points):
    """Prepare data for encoder-decoder roundtrip visualization."""
    # Sample data if needed
    if len(X) > num_samples:
        indices = np.random.choice(len(X), num_samples, replace=False)
        X_sample = X[indices]
    else:
        X_sample = X

    # Get device
    try:
        device = next(encoder.parameters()).device
    except Exception:
        device = 'cpu'
        
    X_tensor = torch.tensor(X_sample, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        # E(X)
        Z_sample = encoder(X_tensor).cpu().numpy()
        
        # D(E(X)) - Reconstruction
        X_recon = decoder(encoder(X_tensor)).cpu().numpy()
        
        # Prepare Latent Grid
        if latent_grid_points is None:
            z_min = Z_sample.min(axis=0)
            z_max = Z_sample.max(axis=0)
            padding = 0.1 * (z_max - z_min)
            z_min -= padding
            z_max += padding
            
            grid_res = 20
            x = np.linspace(z_min[0], z_max[0], grid_res)
            y = np.linspace(z_min[1], z_max[1], grid_res)
            xx, yy = np.meshgrid(x, y)
            latent_grid = np.column_stack([xx.ravel(), yy.ravel()])
        else:
            latent_grid = latent_grid_points

        latent_grid_tensor = torch.tensor(latent_grid, dtype=torch.float32).to(device)
        
        # D(Z_grid)
        X_latent_decoded = decoder(latent_grid_tensor).cpu().numpy()
        
        # E(D(Z_grid)) - Consistency
        Z_latent_reencoded = encoder(decoder(latent_grid_tensor)).cpu().numpy()
    
    return X_sample, Z_sample, X_recon, latent_grid, X_latent_decoded, Z_latent_reencoded


def _plot_3d_subplot(ax, data, title, color='blue', marker='o', alpha=0.5):
    """Plot 3D scatter subplot."""
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=color, marker=marker, s=10, alpha=alpha)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)


def _plot_2d_subplot(ax, data, title, color='red', marker='o', alpha=0.5):
    """Plot 2D scatter subplot."""
    ax.scatter(data[:, 0], data[:, 1], c=color, marker=marker, s=10, alpha=alpha)
    ax.set_xlabel('Latent 1')
    ax.set_ylabel('Latent 2')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def plot_encoder_decoder_roundtrip(X, encoder, decoder, output_path=None, num_samples=500, latent_grid_points=None):
    """
    Plot a 3x2 grid visualization of the encoder-decoder mappings.
    
    Rows:
    1. Original Grid (3D) | Encoded Grid (2D)
    2. Decoded Latent Grid (3D) | Latent Grid (2D)
    3. Reconstructed Original (3D) | Re-encoded Latent (2D)
    
    Args:
        X: Input data (N, D) - Original data sample
        encoder: Encoder model
        decoder: Decoder model
        output_path: Path to save figure
        num_samples: Number of points to use for original data sample
        latent_grid_points: Optional pre-generated latent grid points. If None, generated from bounds of E(X).
    """
    from mpl_toolkits.mplot3d import Axes3D

    # Prepare data
    X_sample, Z_sample, X_recon, latent_grid, X_latent_decoded, Z_latent_reencoded = \
        _prepare_encoder_decoder_data(X, encoder, decoder, num_samples, latent_grid_points)

    # Create figure
    fig = plt.figure(figsize=(16, 18))
    
    # Row 1: Original and Encoded
    ax1 = fig.add_subplot(3, 2, 1, projection='3d')
    _plot_3d_subplot(ax1, X_sample, "Original Data (X)", color='blue')
    ax2 = fig.add_subplot(3, 2, 2)
    _plot_2d_subplot(ax2, Z_sample, "Encoded Data E(X)", color='red')

    # Row 2: Decoded Latent Grid and Latent Grid
    ax3 = fig.add_subplot(3, 2, 3, projection='3d')
    _plot_3d_subplot(ax3, X_latent_decoded, "Decoded Latent Grid D(Z_grid)", color='green')
    ax4 = fig.add_subplot(3, 2, 4)
    _plot_2d_subplot(ax4, latent_grid, "Latent Grid (Z_grid)", color='purple')

    # Row 3: Reconstructed and Re-encoded
    ax5 = fig.add_subplot(3, 2, 5, projection='3d')
    _plot_3d_subplot(ax5, X_recon, "Reconstructed D(E(X))", color='orange')
    ax6 = fig.add_subplot(3, 2, 6)
    _plot_2d_subplot(ax6, Z_latent_reencoded, "Re-encoded Latent E(D(Z_grid))", color='brown')

    finalize_plot(fig, None, output_path, should_close=True)

