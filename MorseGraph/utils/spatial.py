"""
Spatial filtering utilities for grid operations.

This module provides functions for filtering grid boxes based on spatial
proximity to data points.
"""

import numpy as np
from scipy.spatial import cKDTree
from typing import List


def filter_boxes_near_data(
    grid,
    data_points: np.ndarray,
    epsilon_radius: float
) -> List[int]:
    """
    Return list of box indices that contain or are epsilon-close to data points.

    Uses cKDTree for efficient spatial queries to identify which grid boxes
    are within epsilon distance of any data point. Useful for restricting
    computations to regions with data coverage.

    Args:
        grid: Grid object (UniformGrid or AdaptiveGrid)
        data_points: Array of shape (N, D) with data point coordinates
        epsilon_radius: Maximum distance for a box to be considered "near" data

    Returns:
        List of box indices that are within epsilon_radius of any data point

    Example:
        >>> from MorseGraph.grids import UniformGrid
        >>> grid = UniformGrid(bounds=np.array([[-5, -5], [5, 5]]), divisions=[100, 100])
        >>> data = np.random.randn(1000, 2)
        >>> active_boxes = filter_boxes_near_data(grid, data, epsilon_radius=0.5)
        >>> print(f"Active boxes: {len(active_boxes)} out of {len(grid.get_boxes())}")
    """
    tree = cKDTree(data_points)

    # Get all box centers
    all_boxes = grid.get_boxes()
    box_centers = (all_boxes[:, 0, :] + all_boxes[:, 1, :]) / 2.0

    # Query the tree for all box centers at once
    distances, _ = tree.query(box_centers, k=1)

    # Find active boxes where distance is within the radius
    active_indices = np.where(distances <= epsilon_radius)[0]

    return active_indices.tolist()


__all__ = [
    'filter_boxes_near_data',
]

