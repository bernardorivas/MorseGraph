from abc import ABC, abstractmethod
import numpy as np
from typing import Callable
import itertools
from scipy.spatial import cKDTree
from scipy.integrate import solve_ivp

class Dynamics(ABC):
    """
    Abstract base class for a dynamical system.
    """
    @abstractmethod
    def __call__(self, box: np.ndarray) -> np.ndarray:
        """
        Apply the dynamics to a box in the state space.

        :param box: A numpy array of shape (2, D) representing the lower and upper
                    bounds of a D-dimensional box.
        :return: A numpy array of shape (2, D) representing the bounding box
                 of the image of the input box under the dynamics.
        """
        pass

    def get_active_boxes(self, grid) -> np.ndarray:
        """
        Return the indices of boxes that have meaningful dynamics.
        
        By default, all boxes are considered active. Subclasses can override
        this to filter out boxes where dynamics cannot be computed.
        
        :param grid: The grid to check for active boxes
        :return: Array of active box indices
        """
        return np.arange(len(grid.get_boxes()))

class BoxMapFunction(Dynamics):
    """
    A dynamical system defined by an explicit function.
    """
    def __init__(self, map_f: Callable[[np.ndarray], np.ndarray], epsilon: float = 1e-6, 
                 evaluation_method: str = "corners", num_random_points: int = 10):
        """
        :param map_f: The function defining the dynamics. It takes a D-dimensional
                      point and returns a D-dimensional point.
        :param epsilon: The bloating factor to guarantee an outer approximation.
        :param evaluation_method: Method for evaluating the function on the box.
                                 Options: "corners", "center", "random"
        :param num_random_points: Number of random points to use when evaluation_method="random"
        """
        self.map_f = map_f
        self.epsilon = epsilon
        self.evaluation_method = evaluation_method
        self.num_random_points = num_random_points
        
        if evaluation_method not in ["corners", "center", "random"]:
            raise ValueError("evaluation_method must be one of: 'corners', 'center', 'random'")

    def __call__(self, box: np.ndarray) -> np.ndarray:
        """
        Computes a bounding box of the image of the input box under the map.

        The bounding box is computed by sampling points from the input box based on
        the specified evaluation method, applying the map to these sample points, 
        and then computing the bounding box of the resulting points. This bounding 
        box is then "bloated" by epsilon.

        :param box: A numpy array of shape (2, D) representing the lower and upper
                    bounds of a D-dimensional box.
        :return: A numpy array of shape (2, D) representing the bloated
                 bounding box of the image.
        """
        dim = box.shape[1]
        
        # Generate sample points based on evaluation method
        if self.evaluation_method == "corners":
            # Generate all 2^D corners of the box
            corner_points = list(itertools.product(*zip(box[0], box[1])))
            # Add the center of the box
            center_point = (box[0] + box[1]) / 2
            sample_points = np.array(corner_points + [center_point])
            
        elif self.evaluation_method == "center":
            # Use only the center of the box
            center_point = (box[0] + box[1]) / 2
            sample_points = np.array([center_point])
            
        elif self.evaluation_method == "random":
            # Generate random points inside the box
            sample_points = np.random.uniform(
                low=box[0], 
                high=box[1], 
                size=(self.num_random_points, dim)
            )
        
        # Apply the map to the sample points
        image_points = np.array([self.map_f(p) for p in sample_points])

        # Compute the bounding box of the image points
        min_bounds = np.min(image_points, axis=0)
        max_bounds = np.max(image_points, axis=0)

        # Bloat the bounding box
        min_bounds -= self.epsilon
        max_bounds += self.epsilon

        return np.array([min_bounds, max_bounds])


class BoxMapData(Dynamics):
    """
    A data-driven dynamical system optimized for uniform grids.
    
    Supports:
    - Input perturbation: B(x, input_epsilon) 
    - Output perturbation: B(f(x), output_epsilon)
    - Grid dilation: expand to neighboring boxes
    
    Handles empty boxes with different strategies:
    - 'interpolate': Use neighboring boxes to estimate dynamics (default)
    - 'outside': Map empty boxes outside the domain  
    - 'terminate': Raise error if empty boxes are encountered
    
    This implementation assumes uniform grid spacing and pre-assigns 
    data points to grid boxes for performance.
    """
    def __init__(self, X: np.ndarray, Y: np.ndarray, grid,
                 input_distance_metric='L1',          # L1 (manhattan) or L2 (euclidean)
                 output_distance_metric='L1',         # L1 (manhattan) or L2 (euclidean)
                 input_epsilon: float = None,         # Can be scalar or array
                 output_epsilon: float = None,        # Can be scalar or array
                 map_empty: str = 'interpolate',
                 k_neighbors: int = 5,
                 force_interpolation: bool = False,
                 output_enclosure: str = 'box_enclosure'):
        """
        Distance metrics:
        - 'L1': Manhattan/box-based neighborhoods (touches faces only)
        - 'L2': Euclidean/ball-based neighborhoods (includes corners)

        Epsilon behavior:
        - None: defaults to per-dimension cell size (grid.box_size)
        - Scalar: uniform radius in all dimensions
        - Array shape (D,): per-dimension radii for axis-aligned neighborhoods

        Note on L1 vs touching neighbors:
        - L1 with radius=1 gives face-adjacent boxes (6 in 3D)
        - L2 with radius=box_diagonal includes corner-touching boxes

        :param k_neighbors: Number of nearest neighbors to use for interpolation (default: 5)
        :param force_interpolation: If True, apply interpolation strategy to ALL boxes,
                                   creating smooth continuous dynamics across the entire domain.
                                   If False, only interpolate boxes without data (default: False)
        :param output_enclosure: Strategy for converting output points to grid boxes.
                                Options:
                                - 'box_enclosure' (default): Filled rectangular region [min_idx, max_idx].
                                  Most common, computes bounding box then fills rectangle.
                                - 'box_union': Sparse union of boxes near each B(yi, eps).
                                  More conservative, respects epsilon in discrete box space.
        """
        self.X = X
        self.Y = Y
        self.grid = grid
        self.map_empty = map_empty
        self.input_distance_metric = input_distance_metric
        self.output_distance_metric = output_distance_metric
        self.k_neighbors = k_neighbors
        self.force_interpolation = force_interpolation
        self.output_enclosure = output_enclosure

        # Validate output_enclosure
        valid_enclosures = ['box_enclosure', 'box_union']
        if output_enclosure not in valid_enclosures:
            raise ValueError(f"output_enclosure must be one of {valid_enclosures}, got '{output_enclosure}'")

        # Handle epsilon as scalar or vector
        if input_epsilon is None:
            # Default: use per-dimension cell size
            self.input_epsilon = grid.box_size.copy()
        elif np.isscalar(input_epsilon):
            self.input_epsilon = np.full(grid.dim, input_epsilon)
        else:
            input_epsilon = np.array(input_epsilon)
            if input_epsilon.shape != (grid.dim,):
                raise ValueError(f"Epsilon must be scalar or array of shape ({grid.dim},)")
            self.input_epsilon = input_epsilon

        if output_epsilon is None:
            # Default: use per-dimension cell size
            self.output_epsilon = grid.box_size.copy()
        elif np.isscalar(output_epsilon):
            self.output_epsilon = np.full(grid.dim, output_epsilon)
        else:
            output_epsilon = np.array(output_epsilon)
            if output_epsilon.shape != (grid.dim,):
                raise ValueError(f"Epsilon must be scalar or array of shape ({grid.dim},)")
            self.output_epsilon = output_epsilon

        # Pre-assign each data point to its grid box
        self._assign_points_to_boxes()

    def _get_boxes_in_epsilon_neighborhood(self, point: np.ndarray, 
                                           epsilon: np.ndarray, 
                                           metric: str) -> np.ndarray:
        """
        Find all boxes that intersect with epsilon-neighborhood of point.
        
        Handles three cases:
        1. epsilon << box_size: May return 0 boxes (point in box center, epsilon tiny)
        2. epsilon ~ box_size: Returns few boxes (1-27 in 3D for L2)
        3. epsilon >> box_size: Returns many boxes
        
        :param point: Center point of neighborhood
        :param epsilon: Per-dimension radii (array of shape (D,))
        :param metric: 'L1' or 'L2'
        :return: Array of box indices that intersect the neighborhood
        """
        if metric == 'L1':
            # Axis-aligned rectangular neighborhood
            # Intersection = boxes whose bounds overlap with [point-eps, point+eps]
            neighborhood_box = np.array([
                point - epsilon,
                point + epsilon
            ])
            return self.grid.box_to_indices(neighborhood_box)
        
        elif metric == 'L2':
            # Ball-based neighborhood (includes corners)
            # Find all boxes within Euclidean distance
            all_boxes = self.grid.get_boxes()
            box_centers = (all_boxes[:, 0, :] + all_boxes[:, 1, :]) / 2
            
            # Check if box center or any corner is within epsilon ball
            # More conservative: check if box intersects ball
            distances = np.linalg.norm(box_centers - point, axis=1)
            box_half_diag = np.linalg.norm(self.grid.box_size) / 2
            
            # Box intersects ball if center_distance <= epsilon_radius + box_half_diagonal
            epsilon_radius = np.linalg.norm(epsilon)
            intersecting = distances <= (epsilon_radius + box_half_diag)
            
            return np.where(intersecting)[0]
        
        else:
            raise ValueError(f"Unknown metric: {metric}. Use 'L1' or 'L2'.")

    def _assign_points_to_boxes(self):
        """Pre-compute which data points belong to each grid box."""
        # Convert points to grid indices
        point_indices = self._points_to_grid_indices(self.X)
        
        # Create mapping from box index to list of data point indices
        self.box_to_points = {}
        for i, box_idx in enumerate(point_indices):
            if box_idx != -1:  # Valid grid box
                if box_idx not in self.box_to_points:
                    self.box_to_points[box_idx] = []
                self.box_to_points[box_idx].append(i)

    def _points_to_grid_indices(self, points: np.ndarray) -> np.ndarray:
        """
        Convert points to flat grid indices.
        
        :param points: Array of shape (N, D) with point coordinates
        :return: Array of shape (N,) with flat grid indices (-1 for points outside grid)
        """
        # Check if points are within grid bounds
        in_bounds = np.all((points >= self.grid.bounds[0]) & 
                          (points <= self.grid.bounds[1]), axis=1)
        
        # Calculate grid coordinates
        grid_coords = np.floor((points - self.grid.bounds[0]) / self.grid.box_size).astype(int)
        
        # Clip to valid range
        grid_coords = np.clip(grid_coords, 0, self.grid.divisions - 1)
        
        # Convert to flat indices
        flat_indices = np.full(len(points), -1, dtype=int)
        valid_mask = in_bounds
        
        if np.any(valid_mask):
            flat_indices[valid_mask] = np.ravel_multi_index(
                grid_coords[valid_mask].T, self.grid.divisions
            )
        
        return flat_indices

    def __call__(self, box: np.ndarray) -> np.ndarray:
        """
        Compute image of box under data-driven map.

        Uses the output_enclosure strategy to convert output points to grid boxes:

        - 'box_enclosure' (default): Computes bounding box of all yi ± epsilon,
          then grid.box_to_indices() fills the rectangular index region.
          Result: Filled rectangle [min_idx, max_idx].

        - 'box_union': For each yi, finds boxes intersecting B(yi, epsilon),
          then returns union of all such boxes.
          Result: Sparse union (more conservative).

        :param box: A numpy array of shape (2, D) representing the lower and upper
                    bounds of a D-dimensional box.
        :return: A numpy array of shape (2, D) representing the bounding box
                 of the image.
        """

        # If force_interpolation is True, always use interpolation strategy
        if self.force_interpolation:
            # Determine interpolation strategy
            if self.map_empty == 'interpolate':
                strategy = 'k_nearest_points'  # default
            elif self.map_empty in ['k_nearest_points', 'k_nearest_boxes', 'n_box_neighborhood']:
                strategy = self.map_empty
            else:
                # For 'outside' or 'terminate', fall back to k_nearest_points
                strategy = 'k_nearest_points'

            return self._interpolate_from_neighbors(box, strategy=strategy)

        # Get data points in epsilon-neighborhood of box
        box_center = (box[0] + box[1]) / 2

        if self.input_distance_metric == 'L1':
            # Expand box by input_epsilon, find intersecting boxes
            expanded_box = np.array([
                box[0] - self.input_epsilon,
                box[1] + self.input_epsilon
            ])
            relevant_boxes = self.grid.box_to_indices(expanded_box)
        else:
            # Use epsilon-ball around box center
            relevant_boxes = self._get_boxes_in_epsilon_neighborhood(
                box_center, self.input_epsilon, self.input_distance_metric
            )

        # Collect data from these boxes
        all_point_indices = []
        for box_idx in relevant_boxes:
            if box_idx in self.box_to_points:
                all_point_indices.extend(self.box_to_points[box_idx])

        if not all_point_indices:
            return self._handle_empty_box(box)

        # Get corresponding image points
        image_points = self.Y[all_point_indices]

        if self.output_enclosure == 'box_enclosure':
            # Box enclosure: bounding box + output_epsilon bloating
            # grid.box_to_indices() will fill the rectangular index region
            min_bounds = np.min(image_points, axis=0) - self.output_epsilon
            max_bounds = np.max(image_points, axis=0) + self.output_epsilon
            return np.array([min_bounds, max_bounds])

        elif self.output_enclosure == 'box_union':
            # Box union: find all boxes near each image point, take union
            all_target_boxes = set()

            for y_point in image_points:
                nearby_boxes = self._get_boxes_in_epsilon_neighborhood(
                    y_point, self.output_epsilon, self.output_distance_metric
                )
                all_target_boxes.update(nearby_boxes)

            if not all_target_boxes:
                # Fallback to box_enclosure mode
                min_bounds = np.min(image_points, axis=0) - self.output_epsilon
                max_bounds = np.max(image_points, axis=0) + self.output_epsilon
                return np.array([min_bounds, max_bounds])

            # Return bounding box of union of all target boxes
            target_box_indices = np.array(list(all_target_boxes))
            target_boxes = self.grid.get_boxes()[target_box_indices]

            union_min = np.min(target_boxes[:, 0, :], axis=0)
            union_max = np.max(target_boxes[:, 1, :], axis=0)

            return np.array([union_min, union_max])

        else:
            raise ValueError(f"Unknown output_enclosure: {self.output_enclosure}")
    
    def _handle_empty_box(self, box: np.ndarray) -> np.ndarray:
        """
        Handle empty boxes according to the specified strategy.

        :param box: The empty box
        :return: Image of the empty box according to strategy
        """
        if self.map_empty == 'terminate':
            raise ValueError(f"Box {box.flatten()} has no data points (empty image)")

        elif self.map_empty == 'outside':
            # Map to a box outside the grid domain
            # Use a box that's clearly outside the domain bounds
            margin = np.max(self.grid.bounds[1] - self.grid.bounds[0])
            outside_point = self.grid.bounds[1] + margin
            return np.array([outside_point, outside_point + 0.1 * margin])

        elif self.map_empty == 'interpolate':
            # Default interpolation strategy (k_nearest_points)
            return self._interpolate_from_neighbors(box, strategy='k_nearest_points')

        elif self.map_empty in ['k_nearest_points', 'k_nearest_boxes', 'n_box_neighborhood']:
            # Specific interpolation strategy
            return self._interpolate_from_neighbors(box, strategy=self.map_empty)

        else:
            raise ValueError(f"Unknown map_empty strategy: {self.map_empty}")
    
    def _interpolate_from_neighbors(self, box: np.ndarray,
                               strategy: str = 'k_nearest_points') -> np.ndarray:
        """
        Interpolate dynamics for boxes with no data.

        Strategies:
        - 'k_nearest_points': Use k-nearest data points in X
        - 'k_nearest_boxes': Use k-nearest boxes with data
        - 'n_box_neighborhood': Expanding neighborhood search
        """
        box_center = (box[0] + box[1]) / 2

        if strategy == 'k_nearest_points':
            # Find k nearest data points in X
            k = min(self.k_neighbors, len(self.X))
            distances = np.linalg.norm(self.X - box_center, axis=1)
            nearest_idx = np.argpartition(distances, k-1)[:k]

            # Use their corresponding Y values
            image_points = self.Y[nearest_idx]

            # Return bounding box expanded by output_epsilon
            min_bounds = np.min(image_points, axis=0) - self.output_epsilon
            max_bounds = np.max(image_points, axis=0) + self.output_epsilon
            return np.array([min_bounds, max_bounds])

        elif strategy == 'k_nearest_boxes':
            # Find k nearest boxes with data (by box center distance)
            all_boxes = self.grid.get_boxes()
            all_box_centers = (all_boxes[:, 0, :] + all_boxes[:, 1, :]) / 2

            # Find boxes that have data
            boxes_with_data = list(self.box_to_points.keys())

            if not boxes_with_data:
                # No data anywhere - map to outside
                margin = np.max(self.grid.bounds[1] - self.grid.bounds[0])
                outside_point = self.grid.bounds[1] + margin
                return np.array([outside_point, outside_point + 0.1 * margin])

            # Compute distances from empty box center to all boxes with data
            data_box_centers = all_box_centers[boxes_with_data]
            distances = np.linalg.norm(data_box_centers - box_center, axis=1)

            # Find k nearest (or fewer if we don't have k boxes)
            k = min(self.k_neighbors, len(boxes_with_data))
            nearest_box_indices = np.argpartition(distances, k-1)[:k]
            nearest_box_ids = [boxes_with_data[i] for i in nearest_box_indices]

            # Collect Y points from these boxes
            image_points = []
            for box_id in nearest_box_ids:
                point_indices = self.box_to_points[box_id]
                image_points.extend(self.Y[point_indices])

            image_points = np.array(image_points)

            # Compute bounding box + output_epsilon
            min_bounds = np.min(image_points, axis=0) - self.output_epsilon
            max_bounds = np.max(image_points, axis=0) + self.output_epsilon
            result_box = np.array([min_bounds, max_bounds])

            # Check if genuinely outside domain
            if self._is_box_outside_domain(result_box):
                # Map to outside
                margin = np.max(self.grid.bounds[1] - self.grid.bounds[0])
                outside_point = self.grid.bounds[1] + margin
                return np.array([outside_point, outside_point + 0.1 * margin])

            return result_box

        elif strategy == 'n_box_neighborhood':
            # Expanding neighborhood search using grid structure
            # Similar to L1 epsilon strategy but for empty boxes
            return self._interpolate_n_box_neighborhood(box)

        else:
            raise ValueError(f"Unknown interpolation strategy: {strategy}")

    def _get_neighbor_box_indices(self, box_idx: int) -> list:
        """
        Get all boxes that touch the given box (share face, edge, or corner).
        
        In 2D: up to 8 neighbors (like chess king moves)
        In 3D: up to 26 neighbors
        
        :param box_idx: Index of the box
        :return: List of neighboring box indices
        """
        # Convert linear index to grid coordinates
        grid_coords = np.unravel_index(box_idx, self.grid.divisions)
        neighbors = []
        
        # Generate all offset combinations (-1, 0, 1) in each dimension
        # but exclude (0, 0, ..., 0) which is the box itself
        dim = self.grid.dim
        offsets = np.array(np.meshgrid(*([[-1, 0, 1]] * dim))).T.reshape(-1, dim)
        
        for offset in offsets:
            if np.all(offset == 0):
                continue  # Skip the box itself
            
            neighbor_coords = np.array(grid_coords) + offset
            
            # Check if neighbor is within grid bounds
            if np.all(neighbor_coords >= 0) and np.all(neighbor_coords < self.grid.divisions):
                neighbor_idx = np.ravel_multi_index(tuple(neighbor_coords), self.grid.divisions)
                neighbors.append(neighbor_idx)
        
        return neighbors

    def _interpolate_n_box_neighborhood(self, box: np.ndarray) -> np.ndarray:
        """
        Find outputs using expanding neighborhood search (BFS).

        For a box without data:
        1. Look at all boxes that intersect/touch it
        2. If they have data/outputs, collect those outputs
        3. If not, expand to their neighbors (BFS)
        4. Continue until finding boxes with data

        Only returns "outside" if all found outputs are outside domain.
        """
        # Get the box index for this box
        box_center = (box[0] + box[1]) / 2

        # Find which box index this corresponds to
        # (Assuming box aligns with grid - it should since we're processing grid boxes)
        grid_coords = np.floor((box_center - self.grid.bounds[0]) / self.grid.box_size).astype(int)
        grid_coords = np.clip(grid_coords, 0, self.grid.divisions - 1)
        start_box_idx = np.ravel_multi_index(tuple(grid_coords), self.grid.divisions)

        # BFS to find nearest boxes with data
        from collections import deque
        visited = set()
        queue = deque([start_box_idx])
        visited.add(start_box_idx)

        while queue:
            current_idx = queue.popleft()

            # Check if this box has data
            if current_idx in self.box_to_points:
                # Found a box with data! Collect its output
                point_indices = self.box_to_points[current_idx]
                image_points = self.Y[point_indices]

                # Compute bounding box + output_epsilon
                min_bounds = np.min(image_points, axis=0) - self.output_epsilon
                max_bounds = np.max(image_points, axis=0) + self.output_epsilon
                result_box = np.array([min_bounds, max_bounds])

                # Only return outside if result is genuinely outside domain
                if self._is_box_outside_domain(result_box):
                    # Map to outside
                    margin = np.max(self.grid.bounds[1] - self.grid.bounds[0])
                    outside_point = self.grid.bounds[1] + margin
                    return np.array([outside_point, outside_point + 0.1 * margin])

                return result_box

            # No data in this box, expand to neighbors
            neighbors = self._get_neighbor_box_indices(current_idx)
            for neighbor_idx in neighbors:
                if neighbor_idx not in visited:
                    visited.add(neighbor_idx)
                    queue.append(neighbor_idx)

        # Should never reach here if grid is connected, but just in case:
        # Map to outside
        margin = np.max(self.grid.bounds[1] - self.grid.bounds[0])
        outside_point = self.grid.bounds[1] + margin
        return np.array([outside_point, outside_point + 0.1 * margin])




    def _is_box_outside_domain(self, box: np.ndarray) -> bool:
        """
        Check if a bounding box is entirely outside the grid domain.
        
        :param box: Bounding box of shape (2, D)
        :return: True if box doesn't intersect with grid domain
        """
        # Box is outside if:
        # - Its max bound is less than domain min (box is below/left of domain)
        # - Its min bound is greater than domain max (box is above/right of domain)
        
        # Check if there's any intersection
        # No intersection if box_max < domain_min OR box_min > domain_max in any dimension
        domain_min = self.grid.bounds[0]
        domain_max = self.grid.bounds[1]
        
        # Check each dimension
        for d in range(self.grid.dim):
            if box[1, d] < domain_min[d] or box[0, d] > domain_max[d]:
                # No overlap in this dimension -> box is outside
                return True
        
        return False

    def _get_relevant_box_indices(self, box: np.ndarray) -> np.ndarray:
        """
        Get all box indices relevant for the given box, including dilation.
        
        :param box: Input box to find relevant indices for
        :return: Array of relevant box indices
        """
        # Find primary box indices that intersect with the (possibly expanded) input box
        primary_indices = self.grid.box_to_indices(box)
        
        # Apply grid dilation if specified
        if self.dilation_radius > 0:
            dilated_indices = self.grid.dilate_indices(primary_indices, self.dilation_radius)
            return dilated_indices
        else:
            return primary_indices

    def get_active_boxes(self, grid) -> np.ndarray:
        """
        Return boxes to be processed during computation.

        Returns all boxes when:
        - force_interpolation=True (for continuous dynamics)
        - map_empty='terminate' (to detect empty boxes)
        - map_empty='interpolate' or other interpolation strategies (for CMGDB compatibility)

        Otherwise returns only boxes that contain data points.

        :param grid: The grid (should match self.grid)
        :return: Array of box indices to process
        """
        if self.force_interpolation or self.map_empty == 'terminate':
            # Process all boxes for continuous interpolation or to detect empty ones
            return np.arange(len(grid.get_boxes()))
        elif self.map_empty in ['interpolate', 'k_nearest_points', 'k_nearest_boxes', 'n_box_neighborhood']:
            # Process ALL boxes for interpolation modes (matches CMGDB behavior)
            # This ensures continuous dynamics across the entire domain
            return np.arange(len(grid.get_boxes()))
        else:
            # map_empty='outside': only process boxes with data
            return np.array(list(self.box_to_points.keys()), dtype=int)





class BoxMapODE(Dynamics):
    """
    A dynamical system defined by an ordinary differential equation.
    """
    def __init__(self, ode_f: Callable[[float, np.ndarray], np.ndarray], tau: float, epsilon: float = 0):
        """
        :param ode_f: The function defining the ODE, f(t, y).
        :param tau: The integration time.
        :param epsilon: The bloating factor.
        """
        self.ode_f = ode_f
        self.tau = tau
        self.epsilon = epsilon

    def __call__(self, box: np.ndarray) -> np.ndarray:
        """
        Computes the bounding box of the image of the input box under the ODE flow.

        :param box: A numpy array of shape (2, D).
        :return: A numpy array of shape (2, D) for the bloated bounding box of the image.
        """
        dim = box.shape[1]

        # Sample points from the box (corners and center)
        corner_points = list(itertools.product(*zip(box[0], box[1])))
        center_point = (box[0] + box[1]) / 2
        sample_points = np.array(corner_points + [center_point])

        # Integrate the ODE for each sample point
        image_points = []
        for p in sample_points:
            sol = solve_ivp(self.ode_f, [0, self.tau], p, t_eval=[self.tau])
            image_points.append(sol.y[:, -1])

        image_points = np.array(image_points)

        # Compute the bounding box of the final points
        min_bounds = np.min(image_points, axis=0)
        max_bounds = np.max(image_points, axis=0)

        # Bloat the bounding box
        min_bounds -= self.epsilon
        max_bounds += self.epsilon

        return np.array([min_bounds, max_bounds])


# =============================================================================
# Machine Learning-Based Dynamics (Optional PyTorch Dependency)
# =============================================================================

try:
    import torch
    from typing import Union, Optional, Set

    class BoxMapLearnedLatent(Dynamics):
        """
        Learned latent dynamics G: R^d -> R^d using a neural network.

        Supports:
        1. Full Domain Evaluation (Method 2)
        2. Restricted Domain Evaluation (Method 3) via allowed_indices
        3. Rigorous Padding (epsilon L-infinity ball)
        """
        def __init__(self, 
                     latent_dynamics_model, 
                     device, 
                     padding: Union[float, np.ndarray] = 0.0,
                     allowed_indices: Optional[Set[int]] = None):
            """
            :param latent_dynamics_model: PyTorch model G(z)
            :param device: torch device
            :param padding: Epsilon for output bloating (scalar or per-dimension)
            :param allowed_indices: If Set[int], computation is restricted to these box indices.
                                  If None, computation assumes full domain (Method 2).
            """
            if torch is None:
                raise ImportError("PyTorch required for BoxMapLearnedLatent")
                
            self.G = latent_dynamics_model
            self.device = device
            self.padding = padding
            self.allowed_indices = allowed_indices

        def get_active_boxes(self, grid) -> np.ndarray:
            """
            If restricted domain is active, return only allowed indices.
            Otherwise, return all grid indices.
            """
            if self.allowed_indices is not None:
                # Convert set to sorted numpy array for determinism
                return np.array(sorted(list(self.allowed_indices)))
            return np.arange(len(grid.get_boxes()))

        def __call__(self, box: np.ndarray) -> np.ndarray:
            """
            Compute F(box) = BoundingBox(G(samples)) + epsilon.
            """
            # Sample points (Corners + Center + Edge Midpoints for better coverage)
            samples = self._sample_box_points(box)
            
            # Forward pass
            samples_tensor = torch.FloatTensor(samples).to(self.device)
            with torch.no_grad():
                images_tensor = self.G(samples_tensor)
            images = images_tensor.cpu().numpy()
            
            # Compute bounds
            min_b = np.min(images, axis=0)
            max_b = np.max(images, axis=0)
            
            # Apply L-infinity padding
            min_b -= self.padding
            max_b += self.padding
            
            return np.array([min_b, max_b])

        def _sample_box_points(self, box: np.ndarray) -> np.ndarray:
            """
            Generate representative points: 2^D corners, center, and 2*D face centers.
            """
            dim = box.shape[1]
            
            # 1. Corners
            corner_points = list(itertools.product(*zip(box[0], box[1])))
            
            # 2. Center
            center = (box[0] + box[1]) / 2
            
            # 3. Face centers (optional but recommended for non-linear maps)
            # Construct points that are at the center of each face
            face_centers = []
            for d in range(dim):
                p_min = center.copy(); p_min[d] = box[0, d]
                p_max = center.copy(); p_max[d] = box[1, d]
                face_centers.extend([p_min, p_max])
                
            return np.array(corner_points + [center] + face_centers)

except ImportError:
    # If torch is not installed, create a dummy class
    class BoxMapLearnedLatent:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is required for BoxMapLearnedLatent. "
                "Please install it via `pip install torch` or `pip install morsegraph[ml]`."
            )