import networkx as nx
import numpy as np
from joblib import Parallel, delayed

from .dynamics import Dynamics
from .grids import AbstractGrid


class Model:
    """
    The core engine that connects dynamics and grids.
    """

    def __init__(self, grid: AbstractGrid, dynamics: Dynamics):
        """
        :param grid: The grid that discretizes the state space.
        :param dynamics: The dynamical system.
        """
        self.grid = grid
        self.dynamics = dynamics

    def compute_box_map(self, n_jobs: int = -1) -> nx.DiGraph:
        """
        Compute the BoxMap, which represents the discrete dynamical system on grid boxes.

        This method iterates over active boxes in the grid (those with meaningful dynamics),
        computes the image of each box under the dynamics, and finds which other grid boxes 
        intersect with this image. The result is a directed graph representing the BoxMap
        where edges indicate possible transitions between boxes.

        :param n_jobs: The number of jobs to run in parallel. -1 means using all
                       available CPUs.
        :return: A directed graph representing the BoxMap, where nodes are box 
                 indices and edges represent possible transitions.
        """
        boxes = self.grid.get_boxes()
        active_box_indices = self.dynamics.get_active_boxes(self.grid)

        def compute_adjacencies(i):
            image_box = self.dynamics(boxes[i])
            adj = self.grid.box_to_indices(image_box)
            return i, adj

        results = Parallel(n_jobs=n_jobs)(
            delayed(compute_adjacencies)(i) for i in active_box_indices
        )

        graph = nx.DiGraph()
        # Only add active boxes as nodes
        graph.add_nodes_from(active_box_indices)

        for i, adj in results:
            for j in adj:
                # Only add edges to active boxes (j might be inactive)
                if j in active_box_indices:
                    graph.add_edge(i, j)

        return graph