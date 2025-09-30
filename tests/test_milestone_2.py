import numpy as np
import networkx as nx
from MorseGraph.dynamics import BoxMapData, BoxMapODE
from MorseGraph.grids import UniformGrid
from MorseGraph.core import Model
from MorseGraph.analysis import compute_morse_graph, compute_basins_of_attraction

def test_box_map_data():
    # Create a simple dataset
    X = np.array([[0.25, 0.25], [0.75, 0.75]])
    Y = np.array([[0.3, 0.3], [0.7, 0.7]])

    # Create a grid
    bounds = np.array([[0, 0], [1, 1]])
    divisions = np.array([2, 2])
    grid = UniformGrid(bounds, divisions)

    # Create a BoxMapData object with small output_epsilon to handle point data
    dynamics = BoxMapData(X, Y, grid, output_epsilon=0.01)

    # Create a model
    model = Model(grid, dynamics)

    # Compute the map graph
    map_graph = model.compute_box_map()

    # There should be 2 nodes (only active boxes) and 2 edges
    assert map_graph.number_of_nodes() == 2
    assert map_graph.number_of_edges() == 2

    # Check the edges
    # Box 0 contains (0.25, 0.25), so its image is (0.3, 0.3) which is in box 0
    # Box 3 contains (0.75, 0.75), so its image is (0.7, 0.7) which is in box 3
    assert map_graph.has_edge(0, 0)
    assert map_graph.has_edge(3, 3)

def test_box_map_ode():
    # Define a simple ODE that moves everything to the right
    def ode_f(t, y):
        return np.array([0.1, 0])

    # Create a grid
    bounds = np.array([[0, 0], [1, 1]])
    divisions = np.array([2, 2])
    grid = UniformGrid(bounds, divisions)

    # Create a BoxMapODE object
    dynamics = BoxMapODE(ode_f, tau=1.0)

    # Create a model
    model = Model(grid, dynamics)

    # Compute the map graph
    map_graph = model.compute_box_map()

    # Check expected transitions for rightward motion
    # Box 0 ([0, 0.5] x [0, 0.5]) maps to boxes 0 and 2 (moves right and overlaps both)
    assert map_graph.has_edge(0, 0)
    assert map_graph.has_edge(0, 2)
    # Box 1 ([0, 0.5] x [0.5, 1]) maps to boxes 1 and 3 (moves right and overlaps both)
    assert map_graph.has_edge(1, 1)
    assert map_graph.has_edge(1, 3)

def test_basins_of_attraction():
    # Create a simple map graph
    map_graph = nx.DiGraph()
    map_graph.add_edges_from([(0, 1), (1, 1), (2, 1), (3, 3)])

    # Compute the Morse graph
    morse_graph = compute_morse_graph(map_graph)

    # Compute the basins of attraction
    basins = compute_basins_of_attraction(morse_graph, map_graph)

    # There are two attractors: {1} and {3}
    attractor1 = frozenset([1])
    attractor3 = frozenset([3])

    assert len(basins) == 2
    assert attractor1 in basins
    assert attractor3 in basins

    # Check the basins
    assert basins[attractor1] == {0, 1, 2}
    assert basins[attractor3] == {3}
