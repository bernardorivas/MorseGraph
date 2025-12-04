"""
Latent space utilities for computing bounds and working with encoded data.

This module provides functions for computing bounding boxes in latent space
and related operations on encoded data.
"""

import numpy as np
from typing import List


def compute_latent_bounds(
    encoded_data: np.ndarray,
    padding_factor: float = 1.2
) -> np.ndarray:
    """
    Compute bounding box for latent space with padding.

    Takes encoded data points and computes a bounding box that contains
    all points, then expands it by a padding factor to ensure coverage.

    Args:
        encoded_data: Array of shape (N, latent_dim)
        padding_factor: Factor to expand bounds by (default: 1.2 for 20% padding)

    Returns:
        bounds: Array of shape (2, latent_dim) for [[xmin, ymin, ...], [xmax, ymax, ...]]

    Example:
        >>> encoded = encoder(data)  # Shape: (1000, 2)
        >>> bounds = compute_latent_bounds(encoded, padding_factor=1.5)
        >>> # bounds might be [[-3.5, -2.1], [3.5, 2.1]]
    """
    mins = encoded_data.min(axis=0)
    maxs = encoded_data.max(axis=0)

    center = (mins + maxs) / 2
    range_vec = (maxs - mins) * padding_factor / 2

    return np.array([center - range_vec, center + range_vec])


def compute_latent_bounds_from_data(z_data: np.ndarray, padding: float = 1.1) -> List[List[float]]:
    """
    Compute padded bounds for latent data.
    
    Similar to compute_latent_bounds but returns a list format suitable for
    grid initialization.

    Args:
        z_data: Array of shape (N, latent_dim) with encoded data points
        padding: Padding factor (default: 1.1 for 10% padding)

    Returns:
        List of [min_bounds, max_bounds] where each is a list of floats

    Example:
        >>> z_train = encoder(X_train).cpu().numpy()
        >>> bounds = compute_latent_bounds_from_data(z_train, padding=1.2)
        >>> # bounds = [[-2.5, -1.8], [2.5, 1.8]]
    """
    mins = z_data.min(axis=0)
    maxs = z_data.max(axis=0)
    
    ranges = maxs - mins
    centers = (maxs + mins) / 2
    
    padded_ranges = ranges * padding
    padded_mins = centers - padded_ranges / 2
    padded_maxs = centers + padded_ranges / 2
    
    return [padded_mins.tolist(), padded_maxs.tolist()]


def generate_3d_grid_for_encoding(
    domain_bounds: np.ndarray,
    subdiv: int,
    input_dim: int
) -> np.ndarray:
    """
    Generate a uniform grid in the original space for encoding to latent space.
    
    Creates a grid of points uniformly distributed across the domain bounds,
    suitable for encoding with an encoder model.
    
    Args:
        domain_bounds: Domain bounds of shape (2, input_dim) for [[mins], [maxs]]
        subdiv: Subdivision level - grid will have 2^subdiv points per dimension
        input_dim: Dimensionality of the input space
        
    Returns:
        grid_points: Array of shape (2^(subdiv*input_dim), input_dim) with grid points
        
    Example:
        >>> bounds = np.array([[-2, -2, -2], [2, 2, 2]])
        >>> grid = generate_3d_grid_for_encoding(bounds, subdiv=5, input_dim=3)
        >>> # grid.shape = (32768, 3)  # 2^(5*3) = 2^15 = 32768 points
    """
    # Number of points per dimension
    n_per_dim = 2 ** subdiv
    
    # Create meshgrid
    dims = []
    for i in range(input_dim):
        dims.append(np.linspace(domain_bounds[0, i], domain_bounds[1, i], n_per_dim))
    
    # Create meshgrid and flatten
    mesh = np.meshgrid(*dims, indexing='ij')
    grid_points = np.stack([m.flatten() for m in mesh], axis=1)
    
    return grid_points.astype(np.float32)


__all__ = [
    'compute_latent_bounds',
    'compute_latent_bounds_from_data',
    'generate_3d_grid_for_encoding',
]

