"""
Attractor visualization functions.

This module provides functions for visualizing chaotic attractors alongside
Morse set barycenters in both physical and latent space.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Dict, Optional

from .utils import (
    setup_figure_and_axes,
    finalize_plot,
    configure_3d_axes,
    normalize_colormap,
    extract_trajectory_data,
    sample_trajectories,
    compute_encoded_barycenters,
)


def _plot_3d_attractor_with_barycenters(
    ax,
    attractor_data: np.ndarray,
    barycenters_3d: Dict[int, list],
    domain_bounds,
    labels: Dict[str, str],
    title: str,
    attractor_color: str = 'cyan',
    attractor_alpha: float = 0.5,
    attractor_size: float = 0.1,
):
    """
    Plot 3D attractor points with Morse set barycenters overlaid.

    Args:
        ax: Matplotlib 3D axes
        attractor_data: Attractor points (N, 3)
        barycenters_3d: Dict mapping Morse set index to list of barycenters
        domain_bounds: [[lower], [upper]] bounds
        labels: Dict with 'x', 'y', 'z' axis labels
        title: Plot title
        attractor_color: Color for attractor points
        attractor_alpha: Transparency for attractor
        attractor_size: Marker size for attractor points
    """
    # Plot attractor points
    if attractor_data is not None and len(attractor_data) > 0:
        ax.scatter(
            attractor_data[:, 0], attractor_data[:, 1], attractor_data[:, 2],
            c=attractor_color, s=attractor_size, alpha=attractor_alpha,
            label='Attractor', rasterized=True
        )

    # Overlay Morse set barycenters with cool colormap
    if barycenters_3d:
        max_idx = max(barycenters_3d.keys()) if barycenters_3d else 1
        cmap = cm.cool

        for morse_idx in sorted(barycenters_3d.keys()):
            pts = barycenters_3d[morse_idx]
            if pts:
                pts_arr = np.array(pts)
                if pts_arr.ndim == 1:
                    pts_arr = pts_arr.reshape(1, -1)
                color = cmap(morse_idx / max(max_idx, 1))
                ax.scatter(
                    pts_arr[:, 0], pts_arr[:, 1], pts_arr[:, 2],
                    color=color, marker='o', s=200,
                    edgecolors='black', linewidths=1.5,
                    label=f'Morse Set {morse_idx}', zorder=10
                )

    configure_3d_axes(ax, domain_bounds, labels, title, fontsize=12)
    ax.legend(loc='upper left', fontsize=9)


def _plot_latent_attractor_with_barycenters(
    ax,
    z_density: np.ndarray,
    z_attractor: np.ndarray,
    barycenters_latent: Dict[int, np.ndarray],
    latent_bounds,
    title: str,
    attractor_color: str = 'cyan',
    attractor_alpha: float = 0.5,
    attractor_size: float = 0.1,
    density_alpha: float = 0.05,
):
    """
    Plot 2D latent space with density background, attractor, and barycenters.

    Args:
        ax: Matplotlib axes
        z_density: Encoded uniform density samples (N, 2) - background
        z_attractor: Encoded attractor points (M, 2) - foreground
        barycenters_latent: Dict mapping Morse set index to encoded barycenters (K, 2)
        latent_bounds: [[lower], [upper]] bounds
        title: Plot title
        attractor_color: Color for attractor points
        attractor_alpha: Transparency for attractor
        attractor_size: Marker size for attractor points
        density_alpha: Transparency for density background
    """
    # Background: density from uniform sampling (manifold structure)
    if z_density is not None and len(z_density) > 0:
        ax.scatter(
            z_density[:, 0], z_density[:, 1],
            c='black', s=1, alpha=density_alpha,
            label='Latent Manifold', rasterized=True
        )

    # Foreground: attractor projection
    if z_attractor is not None and len(z_attractor) > 0:
        ax.scatter(
            z_attractor[:, 0], z_attractor[:, 1],
            c=attractor_color, s=attractor_size, alpha=attractor_alpha,
            rasterized=True
        )

    # Overlay encoded Morse set barycenters
    if barycenters_latent:
        max_idx = max(barycenters_latent.keys()) if barycenters_latent else 1
        cmap = cm.cool

        for morse_idx in sorted(barycenters_latent.keys()):
            pts = barycenters_latent[morse_idx]
            if pts is not None and len(pts) > 0:
                pts_arr = np.array(pts)
                if pts_arr.ndim == 1:
                    pts_arr = pts_arr.reshape(1, -1)
                color = cmap(morse_idx / max(max_idx, 1))
                ax.scatter(
                    pts_arr[:, 0], pts_arr[:, 1],
                    color=color, marker='o', s=200,
                    edgecolors='black', linewidths=1.5,
                    label=f'Morse Set {morse_idx}', zorder=10
                )

    ax.set_xlim(latent_bounds[0][0], latent_bounds[1][0])
    ax.set_ylim(latent_bounds[0][1], latent_bounds[1][1])
    ax.set_xlabel('Latent Dim 1', fontsize=12)
    ax.set_ylabel('Latent Dim 2', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper left', fontsize=9)


def _plot_loss_curves(ax, loss_history: Dict[str, list], title: str = "Training Loss"):
    """
    Plot training loss curves.

    Args:
        ax: Matplotlib axes
        loss_history: Dict with loss names as keys and lists of values
        title: Plot title
    """
    for name, values in loss_history.items():
        if values:
            ax.plot(values, label=name, alpha=0.7)

    ax.set_yscale('log')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (Log Scale)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)


def plot_attractor_barycenter_comparison(
    trajectory_data,
    barycenters_3d: Dict[int, list],
    encoder,
    device,
    domain_bounds,
    latent_bounds,
    output_path: Optional[str] = None,
    title_prefix: str = "",
    n_density_samples: int = 100000,
    n_attractor_samples: int = 100000,
    n_trajectories: int = 1000,
    tail_fraction: float = 0.5,
    loss_history: Optional[Dict[str, list]] = None,
    labels: Optional[Dict[str, str]] = None,
    attractor_color: str = 'cyan',
    dynamics_func=None,
):
    """
    Create 3-panel comparison showing attractor + Morse set barycenters.

    Panel A (3D): Chaotic attractor points + Morse set barycenters
    Panel B (2D Latent): Density background + attractor projection + encoded barycenters
    Panel C: Training loss curves (if provided)

    Args:
        trajectory_data: np.ndarray of shape (N, n_points, 3) from trajectory generation
        barycenters_3d: Dict mapping Morse set index to list of 3D barycenter coordinates
        encoder: PyTorch encoder model
        device: PyTorch device
        domain_bounds: [[lower_x, lower_y, lower_z], [upper_x, upper_y, upper_z]]
        latent_bounds: [[z0_min, z1_min], [z0_max, z1_max]]
        output_path: Path to save figure
        title_prefix: Prefix for subplot titles
        n_density_samples: Number of uniform samples for density background
        n_attractor_samples: Max number of attractor points to plot
        n_trajectories: Number of trajectories to use for attractor
        tail_fraction: Fraction of trajectory to use as tail (attractor)
        loss_history: Optional dict with training loss history
        labels: Dict with 'x', 'y', 'z' axis labels
        attractor_color: Color for attractor points
        dynamics_func: Optional dynamics function for generating fresh trajectories

    Returns:
        Dict with encoded barycenters and statistics
    """
    import torch

    if labels is None:
        labels = {'x': 'X', 'y': 'Y', 'z': 'Z'}

    domain_bounds_np = np.array(domain_bounds)
    latent_bounds_np = np.array(latent_bounds)

    # Extract attractor data from trajectories (use tails only)
    traj_data = extract_trajectory_data(trajectory_data, use_tail_only=True, tail_fraction=tail_fraction)
    if traj_data is not None:
        traj_data = sample_trajectories(traj_data, n_trajectories)
        # Flatten to points
        attractor_3d = traj_data.reshape(-1, 3)
        # Subsample if too many points
        if len(attractor_3d) > n_attractor_samples:
            idx = np.random.choice(len(attractor_3d), n_attractor_samples, replace=False)
            attractor_3d = attractor_3d[idx]
    else:
        attractor_3d = np.array([])

    # Generate uniform density samples
    np.random.seed(42)
    density_samples_3d = np.random.uniform(
        domain_bounds_np[0], domain_bounds_np[1], (n_density_samples, 3)
    )

    # Encode to latent space
    encoder.eval()
    with torch.no_grad():
        # Density samples
        z_density = encoder(
            torch.tensor(density_samples_3d, dtype=torch.float32).to(device)
        ).cpu().numpy()

        # Attractor points
        if len(attractor_3d) > 0:
            z_attractor = encoder(
                torch.tensor(attractor_3d, dtype=torch.float32).to(device)
            ).cpu().numpy()
        else:
            z_attractor = np.array([])

    # Encode barycenters
    barycenters_latent = compute_encoded_barycenters(barycenters_3d, encoder, device)

    # Create figure
    fig = plt.figure(figsize=(24, 8))

    # Panel A: 3D Physical Space
    ax1 = fig.add_subplot(131, projection='3d')
    _plot_3d_attractor_with_barycenters(
        ax1, attractor_3d, barycenters_3d, domain_bounds,
        labels, f"{title_prefix}Attractor & Morse Sets (3D)",
        attractor_color=attractor_color
    )

    # Panel B: 2D Latent Space
    ax2 = fig.add_subplot(132)
    _plot_latent_attractor_with_barycenters(
        ax2, z_density, z_attractor, barycenters_latent, latent_bounds,
        f"{title_prefix}Latent Space Projection",
        attractor_color=attractor_color
    )

    # Panel C: Loss curves or statistics
    ax3 = fig.add_subplot(133)
    if loss_history:
        _plot_loss_curves(ax3, loss_history, f"{title_prefix}Training Loss")
    else:
        # Show statistics instead
        ax3.axis('off')
        stats_text = f"Attractor Analysis\n\n"
        stats_text += f"Attractor points: {len(attractor_3d)}\n"
        stats_text += f"Morse sets: {len(barycenters_3d)}\n"
        stats_text += f"Total barycenters: {sum(len(v) for v in barycenters_3d.values())}\n\n"
        stats_text += f"Domain bounds:\n"
        stats_text += f"  X: [{domain_bounds[0][0]:.2f}, {domain_bounds[1][0]:.2f}]\n"
        stats_text += f"  Y: [{domain_bounds[0][1]:.2f}, {domain_bounds[1][1]:.2f}]\n"
        stats_text += f"  Z: [{domain_bounds[0][2]:.2f}, {domain_bounds[1][2]:.2f}]\n\n"
        stats_text += f"Latent bounds:\n"
        stats_text += f"  Z1: [{latent_bounds[0][0]:.2f}, {latent_bounds[1][0]:.2f}]\n"
        stats_text += f"  Z2: [{latent_bounds[0][1]:.2f}, {latent_bounds[1][1]:.2f}]"
        ax3.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                family='monospace', transform=ax3.transAxes)
        ax3.set_title(f"{title_prefix}Analysis Statistics", fontsize=14)

    plt.tight_layout()
    finalize_plot(fig, ax=None, output_path=output_path, should_close=True)

    return {
        'barycenters_latent': barycenters_latent,
        'n_attractor_points': len(attractor_3d),
        'n_morse_sets': len(barycenters_3d),
    }
