import numpy as np
import matplotlib.pyplot as plt

# Import the necessary components from the MorseGraph library
from MorseGraph.grids import UniformGrid
from MorseGraph.dynamics import BoxMapData
from MorseGraph.core import Model
from MorseGraph.analysis import compute_morse_graph
from MorseGraph.plot import plot_morse_graph, plot_morse_sets

def test_data_driven_example():
    """
    Test based on examples/2_data_driven.ipynb
    """
    def henon_map(x, a=1.4, b=0.3):
        """ Standard Henon map. """
        x_next = 1 - a * x[:, 0]**2 + x[:, 1]
        y_next = b * x[:, 0]
        return np.column_stack([x_next, y_next])

    # Define the domain and number of sample points
    lower_bounds = np.array([-1.5, -0.4])
    upper_bounds = np.array([1.5, 0.4])
    num_points = 5000

    # Generate random points X and their images Y
    X = np.random.uniform(low=lower_bounds, high=upper_bounds, size=(num_points, 2))
    Y = henon_map(X)

    # Define the grid parameters
    subdivisions = [20, 20]
    domain = np.array([[-1.5, 1.5], [-0.4, 0.4]])

    # 1. Create the dynamics object from our data
    # We add a small bloat_factor to ensure outer approximation.
    dynamics = BoxMapData(X, Y, bloat_factor=0.1)

    # 2. Create the grid
    grid = UniformGrid(bounds=domain, subdivisions=subdivisions)

    # 3. Create the model which connects the grid and dynamics
    model = Model(dynamics, grid)

    # 4. Compute the state transition graph (map graph)
    map_graph = model.compute_map_graph()

    # 5. Compute the Morse graph from the map graph
    morse_graph, morse_sets = compute_morse_graph(map_graph)

    assert len(morse_graph) > 0

    # 6. Visualize the Results
    try:
        plot_morse_graph(morse_graph, morse_sets, output_path='examples/output/morse_graph_data_driven.png')
    except ImportError:
        print("Warning: pygraphviz is not installed. Cannot plot Morse graph.")

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_morse_sets(grid, morse_sets, ax=ax)
    ax.set_title("Morse Sets for Data-Driven Henon Map")
    plt.savefig("examples/output/henon_morse_sets_data_driven.png")

if __name__ == "__main__":
    test_data_driven_example()
