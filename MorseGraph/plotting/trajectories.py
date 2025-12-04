"""
Trajectory visualization functions.

This module provides functions for visualizing trajectory simulations
in both original and latent spaces.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from typing import Dict, List, Optional
from mpl_toolkits.mplot3d import Axes3D

from .utils import finalize_plot


def _plot_time_series_subplots(fig, gs, trajectories_3d, colors, labels, title_prefix):
    """Plot time series for each coordinate in left column."""
    coord_names = [labels['x'], labels['y'], labels['z']]
    for i, coord_name in enumerate(coord_names):
        ax = fig.add_subplot(gs[i, 0])
        for j, traj in enumerate(trajectories_3d):
            steps = np.arange(len(traj))
            ax.plot(steps, traj[:, i], color=colors[j], alpha=0.7,
                   label=f'Trajectory {j+1}')
        ax.set_xlabel('Iteration')
        ax.set_ylabel(coord_name)
        ax.set_title(f'{title_prefix}{coord_name} vs Time')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc='best', fontsize=8)


def _plot_3d_phase_portrait(ax, trajectories_3d, colors, labels, title_prefix):
    """Plot 3D phase portrait with trajectory paths and markers."""
    for j, traj in enumerate(trajectories_3d):
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
              color=colors[j], alpha=0.7, linewidth=1.5)
        # Mark starting point
        ax.scatter([traj[0, 0]], [traj[0, 1]], [traj[0, 2]],
                 color=colors[j], s=120, marker='o', edgecolors='white', linewidths=1.5)
        # Mark ending point with different marker
        ax.scatter([traj[-1, 0]], [traj[-1, 1]], [traj[-1, 2]],
                 color=colors[j], s=120, marker='s', edgecolors='white', linewidths=1.5)
    ax.set_xlabel(labels['x'], fontsize=11)
    ax.set_ylabel(labels['y'], fontsize=11)
    ax.set_zlabel(labels['z'], fontsize=11)
    ax.set_title(f'{title_prefix}3D Phase Portrait', fontsize=12, fontweight='bold')


def _plot_2d_latent_portrait(ax, trajectories_latent, colors, n_trajectories, title_prefix):
    """Plot 2D latent phase portrait with trajectory paths and markers."""
    for j, traj in enumerate(trajectories_latent):
        ax.plot(traj[:, 0], traj[:, 1],
              color=colors[j], alpha=0.7, linewidth=1.5)
        # Mark starting point
        ax.scatter([traj[0, 0]], [traj[0, 1]],
                 color=colors[j], s=120, marker='o', edgecolors='white', linewidths=1.5,
                 label=f'Traj {j+1}' if j < 3 else None)
        # Mark ending point
        ax.scatter([traj[-1, 0]], [traj[-1, 1]],
                 color=colors[j], s=120, marker='s', edgecolors='white', linewidths=1.5)
    ax.set_xlabel('Latent Dim 0', fontsize=11)
    ax.set_ylabel('Latent Dim 1', fontsize=11)
    ax.set_title(f'{title_prefix}2D Latent Phase Portrait', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # Add legend to latent plot
    if n_trajectories <= 5:
        ax.legend(loc='best', fontsize=8)


def plot_trajectory_analysis(
    trajectories_3d,
    trajectories_latent,
    output_path=None,
    title_prefix="",
    labels=None,
    n_steps=None
):
    """
    Visualize trajectory simulations in both original and latent space.

    Creates a multi-panel figure showing:
    - Left column: Time series for each coordinate
    - Right column: Large phase portraits (3D original, 2D latent)

    This helps verify:
    - Numerical stability of dynamics
    - Trajectory behavior consistency
    - Correspondence between 3D and latent dynamics

    Args:
        trajectories_3d: List of 3D trajectories, each shape (n_steps, 3)
        trajectories_latent: List of 2D latent trajectories, each shape (n_steps, 2)
        output_path: Path to save figure (if None, displays instead)
        title_prefix: Prefix for subplot titles
        labels: Dict with keys 'x', 'y', 'z' for axis labels (optional)
        n_steps: Number of steps to plot (if None, plots all)

    Returns:
        None

    Example:
        >>> # Generate some trajectories
        >>> traj_3d = [trajectory1, trajectory2, trajectory3]  # Each shape (100, 3)
        >>> traj_latent = [latent_traj1, latent_traj2, latent_traj3]  # Each shape (100, 2)
        >>> plot_trajectory_analysis(
        ...     traj_3d, traj_latent,
        ...     output_path='results/trajectories.png',
        ...     title_prefix='Ives Model - ',
        ...     labels={'x': 'log(M)', 'y': 'log(A)', 'z': 'log(D)'}
        ... )
    """
    if labels is None:
        labels = {'x': 'X', 'y': 'Y', 'z': 'Z'}

    n_trajectories = len(trajectories_3d)
    colors = cm.viridis(np.linspace(0, 1, n_trajectories))

    # Truncate if needed
    if n_steps is not None:
        trajectories_3d = [traj[:n_steps] for traj in trajectories_3d]
        trajectories_latent = [traj[:n_steps] for traj in trajectories_latent]

    # Create figure with custom grid layout
    # Left column: 3 time series plots
    # Right column: 3D plot (spans 2 rows) + 2D plot (spans 1 row)
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 2, figure=fig, width_ratios=[1, 1], height_ratios=[1, 1, 1])

    # Plot time series (left column)
    _plot_time_series_subplots(fig, gs, trajectories_3d, colors, labels, title_prefix)

    # Plot 3D phase portrait (right column, spans top 2 rows)
    ax_3d = fig.add_subplot(gs[0:2, 1], projection='3d')
    _plot_3d_phase_portrait(ax_3d, trajectories_3d, colors, labels, title_prefix)

    # Plot 2D latent phase portrait (right column, bottom row)
    ax_latent = fig.add_subplot(gs[2, 1])
    _plot_2d_latent_portrait(ax_latent, trajectories_latent, colors, n_trajectories, title_prefix)

    finalize_plot(fig, None, output_path, should_close=True)

