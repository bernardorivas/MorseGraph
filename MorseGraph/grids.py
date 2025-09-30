from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple

class AbstractGrid(ABC):
    """
    Abstract base class for a grid that discretizes the state space.
    """

    @abstractmethod
    def get_boxes(self) -> np.ndarray:
        """
        Return all boxes in the grid.
        
        :return: A numpy array of shape (N, 2, D) where N is the number of boxes
                 and D is the dimension of the state space.
        """
        pass

    @abstractmethod
    def box_to_indices(self, box: np.ndarray) -> np.ndarray:
        """
        Find the indices of the grid boxes that intersect with the given box.

        :param box: A numpy array of shape (2, D) representing the lower and
                    upper bounds of a box.
        :return: A numpy array of integer indices.
        """
        pass

    @abstractmethod
    def subdivide(self, indices: np.ndarray = None):
        """
        Subdivide the grid. If indices are given, only subdivide the specified
        boxes. If indices is None, perform a global subdivision.
        """
        pass
    
    @abstractmethod
    def dilate_indices(self, indices: np.ndarray, radius: int = 1) -> np.ndarray:
        """
        Expand a set of box indices to include their spatial neighbors.
        
        This implements the "Grid Dilation" strategy for rigorous outer approximation
        by expanding in discrete grid space rather than continuous phase space.
        
        :param indices: Original box indices
        :param radius: Neighborhood radius (1 = immediate neighbors, 2 = second-order, etc.)
        :return: Expanded set of indices including neighbors
        """
        pass


class UniformGrid(AbstractGrid):
    """
    A uniform grid on a rectangular domain.
    """

    def __init__(self, bounds: np.ndarray, divisions: np.ndarray):
        """
        :param bounds: A numpy array of shape (2, D) defining the rectangular
                       domain, where D is the dimension.
        :param divisions: A numpy array of shape (D,) with the number of
                          divisions along each axis.
        """
        self.bounds = bounds
        self.divisions = np.array(divisions).astype(int)
        self.dim = bounds.shape[1]
        self.box_size = (bounds[1] - bounds[0]) / self.divisions
        self._boxes = self._create_boxes()

    def _create_boxes(self) -> np.ndarray:
        """
        Create the grid boxes based on the current divisions.
        """
        ranges = [range(d) for d in self.divisions]
        indices = np.array(np.meshgrid(*ranges)).T.reshape(-1, self.dim)
        
        lower_bounds = self.bounds[0] + indices * self.box_size
        upper_bounds = lower_bounds + self.box_size
        
        return np.stack([lower_bounds, upper_bounds], axis=1)

    def get_boxes(self) -> np.ndarray:
        """
        Return all boxes in the grid.
        """
        return self._boxes

    def box_to_indices(self, box: np.ndarray) -> np.ndarray:
        """
        Find the indices of the grid boxes that intersect with the given box.
        """
        # Clip the box to the grid bounds
        clipped_box = np.clip(box, self.bounds[0], self.bounds[1])

        # If the clipped box is empty, there are no intersections
        if np.any(clipped_box[0] >= clipped_box[1]):
            return np.array([], dtype=int)

        # Calculate the range of indices in each dimension
        min_indices = np.floor((clipped_box[0] - self.bounds[0]) / self.box_size).astype(int)
        max_indices = np.ceil((clipped_box[1] - self.bounds[0]) / self.box_size).astype(int) - 1
        
        # Clip indices to be within the valid range
        min_indices = np.clip(min_indices, 0, self.divisions - 1)
        max_indices = np.clip(max_indices, 0, self.divisions - 1)

        # Generate all indices within the range
        ranges = [range(min_i, max_i + 1) for min_i, max_i in zip(min_indices, max_indices)]
        
        # Check if any range is empty
        if any(r.start > r.stop for r in ranges):
            return np.array([], dtype=int)
            
        grid_coords = np.array(np.meshgrid(*ranges)).T.reshape(-1, self.dim)

        # Convert grid coordinates to flat indices
        return np.ravel_multi_index(grid_coords.T, self.divisions)

    def subdivide(self, indices: np.ndarray = None):
        """
        Subdivide the grid. For a uniform grid, this is a global operation.
        The number of divisions is doubled in each dimension.
        """
        if indices is not None:
            # Uniform grid only supports global subdivision
            pass
        self.divisions *= 2
        self.box_size = (self.bounds[1] - self.bounds[0]) / self.divisions
        self._boxes = self._create_boxes()
    
    def dilate_indices(self, indices: np.ndarray, radius: int = 1) -> np.ndarray:
        """
        Expand a set of box indices to include their spatial neighbors.
        
        For UniformGrid, neighbors are found using spatial indexing on the regular grid.
        
        :param indices: Original box indices  
        :param radius: Neighborhood radius (1 = immediate neighbors, 2 = second-order, etc.)
        :return: Expanded set of indices including neighbors
        """
        if len(indices) == 0:
            return indices
            
        # Convert flat indices to grid coordinates
        grid_coords = np.array(np.unravel_index(indices, self.divisions)).T
        
        # Generate all neighbor offsets within the given radius
        # Create a hypercube of offsets from -radius to +radius in each dimension
        offset_ranges = [range(-radius, radius + 1) for _ in range(self.dim)]
        offsets = np.array(np.meshgrid(*offset_ranges)).T.reshape(-1, self.dim)
        
        # Expand each original coordinate by all offsets
        expanded_coords = []
        for coord in grid_coords:
            neighbor_coords = coord[None, :] + offsets  # Broadcasting
            expanded_coords.append(neighbor_coords)
        
        # Concatenate all expanded coordinates
        all_coords = np.vstack(expanded_coords)
        
        # Filter coordinates to be within grid bounds
        valid_mask = np.all((all_coords >= 0) & (all_coords < self.divisions), axis=1)
        valid_coords = all_coords[valid_mask]
        
        # Remove duplicates and convert back to flat indices
        unique_coords = np.unique(valid_coords, axis=0)
        dilated_indices = np.ravel_multi_index(unique_coords.T, self.divisions)
        
        return dilated_indices


class AdaptiveGrid(AbstractGrid):
    """
    An adaptive grid that supports local refinement using a tree structure.
    
    This implementation uses a tree-based approach where leaves represent
    active boxes. Supports efficient spatial indexing and neighbor finding.
    """
    
    class TreeNode:
        """Internal node representation for the adaptive grid tree."""
        
        def __init__(self, bounds: np.ndarray, depth: int = 0, max_depth: int = 10):
            self.bounds = bounds  # Shape (2, D)
            self.depth = depth
            self.max_depth = max_depth
            self.dim = bounds.shape[1]
            
            # Tree structure
            self.children = None  # Will be a list of 2^D children when subdivided
            self.is_leaf = True
            self.index = None  # Flat index for leaf nodes
            
            # Compute center for spatial queries
            self.center = (bounds[0] + bounds[1]) / 2
            self.size = bounds[1] - bounds[0]
        
        def subdivide(self):
            """Subdivide this node into 2^D children."""
            if not self.is_leaf or self.depth >= self.max_depth:
                return False
            
            # Create 2^D children
            num_children = 2 ** self.dim
            self.children = []
            
            # Generate all combinations of lower/upper bounds for each dimension
            for i in range(num_children):
                # Binary representation determines which bound to use per dimension
                child_bounds = np.zeros_like(self.bounds)
                for d in range(self.dim):
                    if (i >> d) & 1:  # Use upper bound
                        child_bounds[0, d] = self.center[d]
                        child_bounds[1, d] = self.bounds[1, d]
                    else:  # Use lower bound
                        child_bounds[0, d] = self.bounds[0, d]
                        child_bounds[1, d] = self.center[d]
                
                child = AdaptiveGrid.TreeNode(child_bounds, self.depth + 1, self.max_depth)
                self.children.append(child)
            
            self.is_leaf = False
            return True
        
        def intersects_box(self, box: np.ndarray) -> bool:
            """Check if this node's bounds intersect with the given box."""
            return np.all(self.bounds[0] <= box[1]) and np.all(box[0] <= self.bounds[1])
        
        def contains_point(self, point: np.ndarray) -> bool:
            """Check if this node contains the given point."""
            return np.all(self.bounds[0] <= point) and np.all(point <= self.bounds[1])
    
    def __init__(self, bounds: np.ndarray, max_depth: int = 10):
        """
        :param bounds: Overall domain bounds of shape (2, D)
        :param max_depth: Maximum tree depth for subdivision
        """
        self.bounds = bounds
        self.dim = bounds.shape[1]
        self.max_depth = max_depth
        
        # Initialize with single root node
        self.root = self.TreeNode(bounds, 0, max_depth)
        self.leaves = [self.root]
        self.root.index = 0
        self.leaf_map = {0: self.root}
    
    def update_leaf_map(self):
        self.leaf_map = {leaf.index: leaf for leaf in self.leaves}

    def get_boxes(self) -> np.ndarray:
        """Return all active boxes (leaf node bounds)."""
        if not self.leaves:
            return np.empty((0, 2, self.dim))
        
        # Sort leaves by index to ensure consistent ordering
        sorted_leaves = sorted(self.leaves, key=lambda leaf: leaf.index)
        boxes = np.array([leaf.bounds for leaf in sorted_leaves])
        return boxes
    
    def box_to_indices(self, box: np.ndarray) -> np.ndarray:
        """Find leaf node indices that intersect with the given box."""
        intersecting_indices = []
        
        def traverse(node):
            if node.intersects_box(box):
                if node.is_leaf:
                    intersecting_indices.append(node.index)
                else:
                    for child in node.children:
                        traverse(child)
        
        traverse(self.root)
        return np.array(intersecting_indices, dtype=int)

    def get_boxes_by_index(self, indices: List[int]) -> np.ndarray:
        """Return the bounds of boxes for the given indices."""
        boxes = []
        for i in indices:
            if i in self.leaf_map:
                boxes.append(self.leaf_map[i].bounds)
        return np.array(boxes)

    def subdivide(self, indices: np.ndarray = None) -> dict:
        """
        Subdivide specified leaf nodes. If indices is None, subdivide all leaves.
        
        :param indices: Indices of leaf nodes to subdivide
        :return: A dictionary mapping the index of each subdivided parent box
                 to a list of the indices of its new children.
        """
        if indices is None:
            nodes_to_subdivide = list(self.leaves)
        else:
            nodes_to_subdivide = [self.leaf_map[i] for i in indices if i in self.leaf_map]

        if not nodes_to_subdivide:
            return {}

        new_children_map = {}
        next_index = max(self.leaf_map.keys()) + 1

        leaves_to_remove = []
        children_to_add = []

        for leaf in nodes_to_subdivide:
            parent_index = leaf.index
            if leaf.subdivide():
                leaves_to_remove.append(leaf)
                
                child_indices = []
                for child in leaf.children:
                    child.index = next_index
                    children_to_add.append(child)
                    child_indices.append(next_index)
                    next_index += 1
                
                new_children_map[parent_index] = child_indices

        if new_children_map:
            for leaf in leaves_to_remove:
                self.leaves.remove(leaf)
            self.leaves.extend(children_to_add)
            self.update_leaf_map()

        return new_children_map
    
    def dilate_indices(self, indices: np.ndarray, radius: int = 1) -> np.ndarray:
        """
        Expand indices to include spatial neighbors using tree traversal.
        
        For AdaptiveGrid, neighbors are found by spatial proximity in the tree structure.
        """
        if len(indices) == 0:
            return indices
        
        original_nodes = [self.leaf_map[i] for i in indices if i in self.leaf_map]
        
        neighbor_nodes = set()
        
        # For each original node, find neighbors within the radius
        for node in original_nodes:
            self._find_neighbors_recursive(node, self.root, radius, neighbor_nodes)
        
        # Convert neighbor nodes to indices
        neighbor_indices = []
        for neighbor in neighbor_nodes:
            if neighbor.index is not None:
                neighbor_indices.append(neighbor.index)
        
        return np.array(neighbor_indices, dtype=int)
    
    def _find_neighbors_recursive(self, target_node: 'TreeNode', current_node: 'TreeNode', 
                                 radius: int, neighbors: set):
        """
        Recursively find neighbors within the given radius.
        
        This uses a simple distance-based criterion - neighbors are nodes whose
        centers are within radius * typical_box_size of the target center.
        """
        if current_node.is_leaf:
            # Check if this leaf is within radius
            distance = np.linalg.norm(current_node.center - target_node.center)
            typical_size = np.mean(target_node.size)
            
            if distance <= radius * typical_size:
                neighbors.add(current_node)
        else:
            # Recurse into children that might contain neighbors
            for child in current_node.children:
                # Simple pruning: only recurse if child might contain neighbors
                max_distance = np.linalg.norm(child.size) / 2  # Half diagonal of child
                center_distance = np.linalg.norm(child.center - target_node.center)
                typical_size = np.mean(target_node.size)
                
                if center_distance - max_distance <= radius * typical_size:
                    self._find_neighbors_recursive(target_node, child, radius, neighbors)