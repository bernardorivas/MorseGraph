'''
Defines the grid abstraction for the MorseGraph library.

This module provides the `AbstractGrid` interface for different subdivision
strategies and concrete implementations like `UniformGrid` and `AdaptiveGrid`.
'''

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
import itertools


class AbstractGrid(ABC):
    """
    Abstract base class for state space subdivision strategies.
    """

    def __init__(self, bounds: np.ndarray):
        """
        :param bounds: A numpy array of shape (D, 2) defining the domain,
                       where D is the dimension and each row is [min, max].
        """
        if not isinstance(bounds, np.ndarray) or bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError("Bounds must be a numpy array of shape (D, 2).")
        self.bounds = bounds
        self.D = bounds.shape[0]

    @abstractmethod
    def get_indices(self) -> List[int]:
        """Returns the list of active box indices (e.g., leaf nodes in an adaptive grid)."""
        pass

    @abstractmethod
    def get_box_coordinates(self, index: int) -> np.ndarray:
        """Converts a box index to its geometric coordinates [min, max] for each dimension."""
        pass

    @abstractmethod
    def find_intersections(self, image_box: np.ndarray) -> List[int]:
        """Finds the indices of the active boxes that intersect the given image box."""
        pass

    @abstractmethod
    def subdivide(self, indices_to_subdivide: Optional[List[int]] = None) -> bool:
        """
        Subdivides the grid.

        If indices_to_subdivide is None, it should perform a uniform subdivision if supported.
        If a list of indices is provided, it should perform an adaptive subdivision if supported.

        :return: True if the grid changed, False otherwise (e.g., max depth reached).
        """
        pass


class UniformGrid(AbstractGrid):
    """
    A standard, uniform grid subdivision of the state space.
    """

    def __init__(self, bounds: np.ndarray, subdivisions: List[int]):
        super().__init__(bounds)
        if len(subdivisions) != self.D:
            raise ValueError("Length of subdivisions must match dimension of bounds.")
        self.subdivisions = np.array(subdivisions, dtype=int)
        self._update_grid_properties()

    def _update_grid_properties(self):
        """Helper to re-calculate grid properties after subdivision."""
        self.total_boxes = np.prod(self.subdivisions)
        self.box_widths = (self.bounds[:, 1] - self.bounds[:, 0]) / self.subdivisions

    def get_indices(self) -> List[int]:
        return list(range(self.total_boxes))

    def get_box_coordinates(self, index: int) -> np.ndarray:
        if not (0 <= index < self.total_boxes):
            raise IndexError("Box index out of range.")
        
        grid_coords = np.array(np.unravel_index(index, self.subdivisions))
        lower_bounds = self.bounds[:, 0] + grid_coords * self.box_widths
        upper_bounds = lower_bounds + self.box_widths
        return np.stack([lower_bounds, upper_bounds], axis=1)

    def find_intersections(self, image_box: np.ndarray) -> List[int]:
        # Clip the image box to the grid bounds to handle intersections at the boundary
        clipped_min = np.maximum(image_box[:, 0], self.bounds[:, 0])
        clipped_max = np.minimum(image_box[:, 1], self.bounds[:, 1])

        # If the clipped box has no volume, it's outside the grid
        if np.any(clipped_min >= clipped_max):
            return []

        # Calculate the range of grid coordinates (multi-indices) the box covers
        min_multi_index = np.floor((clipped_min - self.bounds[:, 0]) / self.box_widths).astype(int)
        max_multi_index = np.floor((clipped_max - self.bounds[:, 0]) / self.box_widths).astype(int)

        # Clamp indices to be within the valid grid range
        min_multi_index = np.maximum(0, min_multi_index)
        max_multi_index = np.minimum(self.subdivisions - 1, max_multi_index)

        # Create an iterator for all multi-indices in the intersection range
        iter_ranges = [range(min_multi_index[d], max_multi_index[d] + 1) for d in range(self.D)]
        
        # Convert multi-indices to linear indices
        intersections = [np.ravel_multi_index(multi_index, self.subdivisions) for multi_index in itertools.product(*iter_ranges)]
        
        return intersections

    def subdivide(self, indices_to_subdivide: Optional[List[int]] = None) -> bool:
        # Uniform grid subdivision doubles the resolution in each dimension
        self.subdivisions *= 2
        self._update_grid_properties()
        return True


class AdaptiveGrid(AbstractGrid):
    """
    An adaptive grid that supports local refinement (e.g., a Quadtree/Octree).
    (Placeholder for future implementation).
    """

    def __init__(self, bounds: np.ndarray, max_depth: int = 10):
        super().__init__(bounds)
        self.max_depth = max_depth
        # The implementation would require a tree data structure.
        # For now, we raise NotImplementedError.
        raise NotImplementedError("AdaptiveGrid is not yet implemented.")

    def get_indices(self) -> List[int]:
        raise NotImplementedError()

    def get_box_coordinates(self, index: int) -> np.ndarray:
        raise NotImplementedError()

    def find_intersections(self, image_box: np.ndarray) -> List[int]:
        raise NotImplementedError()

    def subdivide(self, indices_to_subdivide: Optional[List[int]] = None) -> bool:
        raise NotImplementedError()
