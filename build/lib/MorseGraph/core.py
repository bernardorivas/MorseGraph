'''
Defines the core Model class for the MorseGraph library.

This module contains the `Model` class, which links a `Dynamics` object with a
`Grid` object to compute the state transition graph (map graph).
'''

import networkx as nx
import numpy as np

# Use try-except for type hinting to avoid circular imports if structured differently
try:
    from .dynamics import Dynamics
    from .grids import AbstractGrid
except ImportError:
    # This allows for type hinting even if the file is run standalone
    from morsegraph.dynamics import Dynamics
    from morsegraph.grids import AbstractGrid


class Model:
    """
    The core model that links a dynamical system with a state space subdivision.
    """

    def __init__(self, dynamics_obj: Dynamics, grid: AbstractGrid):
        """
        :param dynamics_obj: An instance of a class that inherits from `Dynamics`.
        :param grid: An instance of a class that inherits from `AbstractGrid`.
        """
        self.dynamics = dynamics_obj
        self.grid = grid
        self.D = grid.D
        self.bounds = grid.bounds

        if hasattr(self.dynamics, 'D') and self.dynamics.D != self.D:
            raise ValueError(
                f"Dimension mismatch: Grid has dimension D={self.D}, "
                f"but Dynamics object has dimension D={self.dynamics.D}."
            )

    def compute_map_graph(self) -> nx.DiGraph:
        """
        Computes the combinatorial multivalued map (the Map Graph) at the current
        grid resolution.

        The map graph is a directed graph where each node represents a box in the
        grid. An edge from box A to box B exists if the image of box A under the
        dynamics intersects box B.

        :return: A networkx.DiGraph representing the map graph.
        """
        map_graph = nx.DiGraph()
        active_indices = self.grid.get_indices()
        map_graph.add_nodes_from(active_indices)

        for box_index in active_indices:
            box_coords = self.grid.get_box_coordinates(box_index)

            # Apply the dynamics to get the image of the box
            image_box = self.dynamics(box_coords)

            # Handle invalid images (e.g., from failed ODE integrations or no data)
            if image_box is None or np.isnan(image_box).any():
                continue

            # Find all grid boxes that intersect the image box
            intersecting_boxes = self.grid.find_intersections(image_box)

            # Add directed edges to the map graph
            # For adaptive grids, it's crucial to only add edges to active boxes (leaves)
            active_indices_set = set(active_indices)
            for target_box in intersecting_boxes:
                if target_box in active_indices_set:
                    map_graph.add_edge(box_index, target_box)

        return map_graph
