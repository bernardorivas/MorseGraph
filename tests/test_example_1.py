import numpy as np
import matplotlib.pyplot as plt

# Import the necessary components
from MorseGraph.grids import UniformGrid
from MorseGraph.dynamics import BoxMapFunction
from MorseGraph.core import Model
from MorseGraph.plot import plot_morse_graph, plot_morse_sets
from MorseGraph.analysis import compute_morse_graph

def test_henon_map_example():
    """
    Test based on examples/1_map_dynamics.ipynb
    """
    def henon_map(x, a=1.4, b=0.3):
        """ Standard Henon map, vectorized. """
        x_next = 1 - a * x[:, 0]**2 + x[:, 1]
        y_next = b * x[:, 0]
        return np.column_stack([x_next, y_next])

    # Define the domain for the grid
    domain = np.array([[-1.5, 1.5], [-0.4, 0.4]])

    # 1. Create the dynamics object
    dynamics = BoxMapFunction(
        func=henon_map,
        dimension=2,
        bloat_factor=0.0
    )

    # 2. Create a grid
    subdivisions = [100, 100]
    grid = UniformGrid(bounds=domain, subdivisions=subdivisions)

    # 3. Create the model
    model = Model(dynamics, grid)

    # 4. Compute the Morse Graph
    map_graph = model.compute_map_graph()
    morse_graph, morse_sets = compute_morse_graph(map_graph)
    print(f"Computed Morse graph with {len(morse_graph)} nodes.")

    # 5. Visualize the Results
    # Plot the Morse graph (saves to a file)
    plot_morse_graph(morse_graph, morse_sets, output_path='examples/output/morse_graph.png')

    try:
        import pygraphviz as pgv
        A = pgv.AGraph(directed=True, strict=True, rankdir='TB')

        for i, node_id in enumerate(morse_graph.nodes()):
            scc = morse_sets.get(node_id, set())
            node_label = f"M({node_id})\n({len(scc)} boxes)"
            A.add_node(node_id, label=node_label, shape="ellipse")

        for u, v in morse_graph.edges():
            A.add_edge(u, v)
        
        print("Morse graph (graphviz format):")
        print(A.to_string())

    except ImportError:
        print("Warning: pygraphviz is not installed. Cannot print Morse graph.")
        print("Please install it via `pip install pygraphviz` (requires graphviz system library).")


    # Plot the Morse sets on the grid
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_morse_sets(grid, morse_sets, ax=ax)
    ax.set_title("Morse Sets for the Hénon Map")
    plt.savefig("examples/output/henon_morse_sets.png") # Save the figure instead of showing it

if __name__ == "__main__":
    test_henon_map_example()
