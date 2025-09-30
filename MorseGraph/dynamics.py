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
    
    Supports flexible error models:
    - Input perturbation: B(x, input_epsilon) 
    - Output perturbation: B(f(x), output_epsilon)
    - Grid dilation: expand to neighboring boxes
    
    This implementation assumes uniform grid spacing and pre-assigns 
    data points to grid boxes for performance.
    """
    def __init__(self, X: np.ndarray, Y: np.ndarray, grid, 
                 input_epsilon: float = 0, output_epsilon: float = 0,
                 dilation_radius: int = 0):
        """
        :param X: A numpy array of shape (N, D) of input data points.
        :param Y: A numpy array of shape (N, D) of output data points.
        :param grid: The UniformGrid instance used for discretization.
        :param input_epsilon: Input domain uncertainty - expand input boxes by this amount.
        :param output_epsilon: Output domain uncertainty - bloat output bounding boxes by this amount.
        :param dilation_radius: Grid dilation radius - include neighboring boxes (0 = no dilation).
        """
        self.X = X
        self.Y = Y
        self.input_epsilon = input_epsilon
        self.output_epsilon = output_epsilon
        self.dilation_radius = dilation_radius
        self.grid = grid
        
        # Pre-assign each data point to its grid box
        self._assign_points_to_boxes()

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
        Computes a bounding box of the image under the data-driven map.
        
        Supports
        1. Input perturbation: expands input box by input_epsilon
        2. Output perturbation: expands output image by output_epsilon  
        3. Grid dilation: include n-neighboring boxes in computation
        
        :param box: A numpy array of shape (2, D) representing the lower and upper
                    bounds of a D-dimensional box.
        :return: A numpy array of shape (2, D) representing the bloated
                 bounding box of the image.
        """
        # Apply input perturbation if specified
        if self.input_epsilon > 0:
            expanded_box = np.array([
                box[0] - self.input_epsilon,
                box[1] + self.input_epsilon
            ])
        else:
            expanded_box = box.copy()
        
        # Find which grid boxes to consider
        box_indices = self._get_relevant_box_indices(expanded_box)
        
        # Collect all relevant data points
        all_point_indices = []
        for box_idx in box_indices:
            if box_idx in self.box_to_points:
                all_point_indices.extend(self.box_to_points[box_idx])
        
        # Since this method should only be called on active boxes,
        # we should always have data points. But handle the edge case.
        if not all_point_indices:
            # This should not happen for active boxes, but return empty box if it does
            return np.array([[np.inf, np.inf], [-np.inf, -np.inf]])
        
        # Get the corresponding points in Y
        image_points = self.Y[all_point_indices]
        
        # Compute the bounding box of the image points
        min_bounds = np.min(image_points, axis=0)
        max_bounds = np.max(image_points, axis=0)
        
        # Apply output perturbation
        if self.output_epsilon > 0:
            min_bounds -= self.output_epsilon
            max_bounds += self.output_epsilon
        
        return np.array([min_bounds, max_bounds])
    
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
        Return only the boxes that contain data points.
        
        :param grid: The grid (should match self.grid)
        :return: Array of box indices that contain at least one data point
        """
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


class LearnedMapDynamics(Dynamics):
    """
    A dynamical system defined by a learned function in latent space.
    
    This class wraps a learned dynamics model (implementing AbstractLatentDynamics)
    into the rigorous BoxMap framework. The dynamics operate entirely in the latent
    space - the associated grid must also be defined in the latent space dimensions.
    """
    
    def __init__(self, dynamics_model, bloating: float = 1e-6):
        """
        :param dynamics_model: Any model implementing AbstractLatentDynamics interface
                              (has predict method)
        :param bloating: Epsilon expansion factor for outer approximation
        """
        # Check if the model has the expected interface
        if not hasattr(dynamics_model, 'predict'):
            raise ValueError("dynamics_model must have a 'predict' method")
        
        self.dynamics_model = dynamics_model
        self.epsilon = bloating
    
    def __call__(self, latent_box: np.ndarray) -> np.ndarray:
        """
        Apply learned dynamics to a box in latent space.
        
        :param latent_box: A numpy array of shape (2, D) representing bounds 
                          in the D-dimensional latent space
        :return: Bounding box of the image under learned dynamics
        """
        dim = latent_box.shape[1]
        
        # Sample points from the latent box (corners and center)
        corner_points = list(itertools.product(*zip(latent_box[0], latent_box[1])))
        center_point = (latent_box[0] + latent_box[1]) / 2
        sample_points = np.array(corner_points + [center_point])
        
        # Apply the learned dynamics
        image_points = self.dynamics_model.predict(sample_points)
        
        # Compute bounding box of image points
        min_bounds = np.min(image_points, axis=0)
        max_bounds = np.max(image_points, axis=0)
        
        # Apply bloating for outer approximation
        min_bounds -= self.epsilon
        max_bounds += self.epsilon
        
        return np.array([min_bounds, max_bounds])


class LinearMapDynamics(Dynamics):
    """
    Optimized dynamics for linear models (e.g., from DMD).
    
    Since linear dynamics are defined by a matrix A (Z_next = A @ Z_current),
    we can optimize by only mapping the corners of the input box.
    """
    
    def __init__(self, linear_matrix: np.ndarray, bloating: float = 1e-6):
        """
        :param linear_matrix: The linear transformation matrix A of shape (D, D)
        :param bloating: Epsilon expansion factor for outer approximation
        """
        self.A = linear_matrix
        self.epsilon = bloating
    
    def __call__(self, latent_box: np.ndarray) -> np.ndarray:
        """
        Apply linear dynamics to a box in latent space.
        
        For linear dynamics Z_next = A @ Z, we only need to map the corners
        since the image of a box under linear transformation is determined
        by the images of its corners.
        
        :param latent_box: Box in latent space of shape (2, D)
        :return: Bounding box of the image under linear dynamics
        """
        dim = latent_box.shape[1]
        
        # Generate all corners of the box
        corner_points = list(itertools.product(*zip(latent_box[0], latent_box[1])))
        corner_points = np.array(corner_points)
        
        # Apply linear transformation: image = A @ corners.T
        image_points = (self.A @ corner_points.T).T
        
        # Compute bounding box
        min_bounds = np.min(image_points, axis=0)
        max_bounds = np.max(image_points, axis=0)
        
        # Apply bloating
        min_bounds -= self.epsilon
        max_bounds += self.epsilon
        
        return np.array([min_bounds, max_bounds])


class GridDilatedDynamics(Dynamics):
    """
    A wrapper that applies grid dilation for rigorous outer approximation.
    
    This class implements the "Grid Dilation" strategy by expanding results
    in discrete grid space rather than continuous phase space.
    """
    
    def __init__(self, base_dynamics: Dynamics, grid, dilation_radius: int = 1):
        """
        :param base_dynamics: The underlying dynamics to wrap
        :param grid: The AbstractGrid instance used for discretization
        :param dilation_radius: Number of neighboring layers to include
        """
        self.base_dynamics = base_dynamics
        self.grid = grid
        self.radius = dilation_radius
    
    def __call__(self, box: np.ndarray) -> np.ndarray:
        """
        Apply base dynamics and then dilate the result in grid space.
        
        :param box: Input box in phase space
        :return: Union of boxes after grid dilation
        """
        # 1. Apply base dynamics to get target box
        target_box = self.base_dynamics(box)
        
        # 2. Find grid indices that intersect with target box
        target_indices = self.grid.box_to_indices(target_box)
        
        # 3. Apply grid dilation to expand the indices
        dilated_indices = self.grid.dilate_indices(target_indices, self.radius)
        
        # 4. Get all boxes corresponding to dilated indices
        if len(dilated_indices) == 0:
            return target_box  # Return original if no valid indices
            
        all_boxes = self.grid.get_boxes()
        dilated_boxes = all_boxes[dilated_indices]
        
        # 5. Compute the union bounding box of all dilated boxes
        all_min_bounds = dilated_boxes[:, 0, :]  # Shape: (n_boxes, dim)
        all_max_bounds = dilated_boxes[:, 1, :]
        
        union_min = np.min(all_min_bounds, axis=0)
        union_max = np.max(all_max_bounds, axis=0)
        
        return np.array([union_min, union_max])