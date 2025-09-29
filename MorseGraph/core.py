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

    def compute_map_graph(self, n_jobs: int = -1) -> nx.DiGraph:
        """
        Compute the map graph, which represents the transitions between grid boxes.

        This method iterates over all boxes in the grid, computes the image of
        each box under the dynamics, and finds which other grid boxes intersect
        with this image. An edge is created in the graph from the source box
        to each of the intersecting boxes.

        :param n_jobs: The number of jobs to run in parallel. -1 means using all
                       available CPUs.
        :return: A directed graph where nodes are box indices and edges
                 represent transitions.
        """
        boxes = self.grid.get_boxes()
        num_boxes = len(boxes)

        def compute_adjacencies(i):
            image_box = self.dynamics(boxes[i])
            adj = self.grid.box_to_indices(image_box)
            return i, adj

        results = Parallel(n_jobs=n_jobs)(
            delayed(compute_adjacencies)(i) for i in range(num_boxes)
        )

        graph = nx.DiGraph()
        graph.add_nodes_from(range(num_boxes))

        for i, adj in results:
            for j in adj:
                graph.add_edge(i, j)

        return graph