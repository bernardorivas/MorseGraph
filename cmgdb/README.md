# CMGDB (Conley Morse Graph Database)

## Project Overview

*   **Summary:** CMGDB is a C++ library with Python bindings designed for the rigorous analysis of discrete dynamical systems using combinatorial and topological methods. Its core functionality is to compute the Conley-Morse graph of a system by discretizing the state space and analyzing the resulting transition graph.
*   **Mathematical Problem:** This library addresses the problem of understanding the global dynamics of a system $f: X \to X$. It provides tools for computing the Conley-Morse graph of a discrete dynamical system defined on a grid, which reveals the hierarchical relationships between the invariant sets (attractors, repellers, etc.).
*   **Intended Audience:** This tool is intended for researchers in applied mathematics, computational topology, and systems biology who analyze complex dynamical systems and require a robust, combinatorial description of their global structure.

## Mathematical Foundations

This section provides the theoretical underpinnings of the software.

*   **Theory:** The algorithms are based on Conley-Morse theory, which simplifies the study of a dynamical system by representing its gradient-like flow structure as a directed acyclic graph.
    *   **State Space Discretization:** The state space $X$ is partitioned into a finite collection of hyperrectangles (boxes), forming a grid. This process transforms the state space into a finite, combinatorial representation.
    *   **Box Map:** A multi-valued map $F$ is constructed on the grid. An edge exists from a box $B_i$ to a box $B_j$ if the image of $B_i$ under the dynamics, $f(B_i)$, intersects $B_j$. This creates a directed graph, known as the *box map* (or transition graph), which is a combinatorial representation of the dynamics.
    *   **Morse Graph:** First, it computes the condensation graph (CG) of the transition graph. That's an acyclic direted graph, which is equivalent to a partial order on the vertices. The non-trivial strongly connected components are used as vertices for the Morse Graph (MG). The MG is the minimal directed graph whose edges induce the same ordering of CG. For example, if (a,b) and (b,c) are edges, there is no need for edge (a,c), even if there is a path from SCC_a to SCC_c.
    *   **Morse Sets:** The box-regions in the original state space $X$ corresponding to the union of boxes in an SCC are called *Morse sets*. These sets represent regions of recurrent or chain-recurrent behavior, such as fixed points, periodic orbits, or chaotic attractors.
    *   **Conley-Morse Graph:** By annotating the Morse Graph with the Conley Index of each node, one obtains the Conley-Morse Graph (CMG). 
    *   **Attractor Lattice:** The Conley-Morse graph provides a combinatorial representation of the system's attractor lattice, showing the hierarchical relationships between attractors, repellers, and other invariant sets.

## Technical Implementation and Architecture

*   **Core Algorithms:**
    *   **Discretization Method:** The implementation uses a cubical grid for state space discretization. It supports both uniform grids and adaptive mesh refinement, where regions of interest (potential Morse sets) are subdivided to a higher resolution.
    *   **Graph Construction:** The transition graph is built by rigorously evaluating the image of each grid box. The library supports dynamics defined by explicit functions (`BoxMap`) or data from simulations (`BoxMapData`). Interval arithmetic may be used for rigorous enclosures of the map's image.
    *   **SCC Computation:** The library uses algorithms from the CHOMP (Computational Homology Project) library to efficiently find the strongly connected components of the large transition graph, which is a standard step in combinatorial Conley theory.

*   **Codebase Structure:**
    *   `src/CMGDB/_cmgdb/CMGDB.cpp`: The core C++ backend, which is wrapped using `pybind11` to create the Python extension module `_cmgdb`.
    *   `src/CMGDB/_cmgdb/include/`: Contains the C++ header files for core data structures and algorithms, such as `Model.h`, `MorseGraph.h`, `Grid.h`, and `Map.h`.
    *   `src/CMGDB/`: The Python package directory containing user-facing helper functions and classes that wrap the C++ backend (e.g., `PlotMorseSets.py`, `BoxMapData.py`).
    *   `examples/`: A collection of Jupyter notebooks and Python scripts demonstrating various use cases, from simple maps to data-driven and ODE-based systems.

*   **Dependencies:**
    *   **Python:** `numpy`, `matplotlib`, `graphviz`
    *   **C++:** A C++11 compatible compiler.
    *   **External Libraries:** [Boost](https://www.boost.org/), [GMP](https://gmplib.org/), and the [Succinct Data Structure Library (SDSL)](https://github.com/simongog/sdsl-lite).

## Installation and Quick Start

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/marciogameiro/CMGDB.git
    cd CMGDB
    ```

2.  **Install dependencies:** Ensure you have a C++ compiler, Boost, GMP, and SDSL installed on your system.

3.  **Install the package:**
    You can install the package using `pip`, which will compile the C++ source.
    ```bash
    pip install ./cmgdb
    ```
    Alternatively, use the provided shell script:
    ```bash
    ./install.sh
    ```

### Minimal Working Example

This example computes the Conley-Morse graph for the 2D Leslie map.

```python
import CMGDB
import math
import matplotlib

# 1. Define the dynamical system
def f(x):
    th1 = 19.6
    th2 = 23.68
    return [(th1 * x[0] + th2 * x[1]) * math.exp(-0.1 * (x[0] + x[1])), 0.7 * x[0]]

# Define a box map that rigorously encloses the image of a rectangle
def F(rect):
    return CMGDB.BoxMap(f, rect, padding=True)

# 2. Define the model for computation
subdiv_min = 20
subdiv_max = 30
lower_bounds = [-0.001, -0.001]
upper_bounds = [90.0, 70.0]
model = CMGDB.Model(subdiv_min, subdiv_max, lower_bounds, upper_bounds, F)

# 3. Run the analysis
morse_graph, map_graph = CMGDB.ComputeMorseGraph(model)

# 4. Visualize the output
# Plot the Conley-Morse Graph
CMGDB.PlotMorseGraph(morse_graph)

# Plot the Morse Sets in the phase space
CMGDB.PlotMorseSets(morse_graph, cmap=matplotlib.cm.cool)
```

## API Reference for Interoperability Analysis

### `CMGDB.Model(subdiv_min, subdiv_max, lower_bounds, upper_bounds, F)`

*   **Purpose:** Creates a model object that encapsulates all parameters for the Morse graph computation.
*   **Parameters:**
    *   `subdiv_min` (`int`): The minimum level of subdivision for the adaptive grid.
    *   `subdiv_max` (`int`): The maximum level of subdivision for refining potential Morse sets.
    *   `lower_bounds` (`list[float]`): The lower corner of the state space domain to be analyzed.
    *   `upper_bounds` (`list[float]`): The upper corner of the state space domain.
    *   `F` (`callable`): A Python function that takes a rectangle (list of bounds) and returns its image as a rectangle.
*   **Return Value:**
    *   `model` (`CMGDB.Model`): An object to be passed to the computation functions.
*   **Example Snippet:**
    ```python
    import CMGDB
    # Define a map F
    def F(rect): return rect
    model = CMGDB.Model(10, 12, [0, 0], [1, 1], F)
    print(model)
    ```

### `CMGDB.ComputeMorseGraph(model)`

*   **Purpose:** Computes the Conley-Morse graph for the dynamical system defined in the model.
*   **Parameters:**
    *   `model` (`CMGDB.Model`): The model object specifying the system and computation parameters.
*   **Return Value:**
    *   `morse_graph` (`CMGDB.MorseGraph`): An object representing the computed Conley-Morse graph.
    *   `map_graph` (`CMGDB.MapGraph`): An object representing the underlying transition graph on the final grid.
*   **Example Snippet:**
    ```python
    import CMGDB
    def F(rect): return rect
    model = CMGDB.Model(10, 12, [0, 0], [1, 1], F)
    morse_graph, map_graph = CMGDB.ComputeMorseGraph(model)
    print(f"Found {morse_graph.num_vertices()} Morse sets.")
    ```

### `CMGDB.ComputeConleyMorseGraph(model)`

*   **Purpose:** Computes the Conley-Morse graph and annotates it with the homological Conley index for each Morse set.
*   **Parameters:**
    *   `model` (`CMGDB.Model`): The model object specifying the system and computation parameters.
*   **Return Value:**
    *   `morse_graph` (`CMGDB.MorseGraph`): An annotated object representing the Conley-Morse graph.
    *   `map_graph` (`CMGDB.MapGraph`): The underlying transition graph.
*   **Example Snippet:**
    ```python
    import CMGDB
    def F(rect): return rect
    model = CMGDB.Model(8, 10, [0, 0], [1, 1], F)
    morse_graph, map_graph = CMGDB.ComputeConleyMorseGraph(model)
    # Print the Conley index of the first Morse set
    print(morse_graph.annotations(0))
    ```

### `CMGDB.PlotMorseGraph(morse_graph)`

*   **Purpose:** Generates a visualization of the Conley-Morse graph using Graphviz.
*   **Parameters:**
    *   `morse_graph` (`CMGDB.MorseGraph`): The computed Morse graph object.
*   **Return Value:**
    *   `source` (`graphviz.Source`): A Graphviz object that can be rendered in a Jupyter notebook or saved to a file.
*   **Example Snippet:**
    ```python
    import CMGDB
    def F(rect): return rect
    model = CMGDB.Model(10, 12, [0, 0], [1, 1], F)
    morse_graph, _ = CMGDB.ComputeMorseGraph(model)
    # This will display the graph in a Jupyter environment
    CMGDB.PlotMorseGraph(morse_graph)
    ```

### `CMGDB.PlotMorseSets(morse_graph)`

*   **Purpose:** Creates a 2D plot of the Morse sets in the phase space using Matplotlib.
*   **Parameters:**
    *   `morse_graph` (`CMGDB.MorseGraph`): The computed Morse graph object.
    *   `proj_dims` (`list[int]`, optional): A list of two indices specifying the dimensions to project onto for higher-dimensional systems.
*   **Return Value:**
    *   None. A Matplotlib plot is displayed.
*   **Example Snippet:**
    ```python
    import CMGDB
    import matplotlib
    def F(rect): return rect
    model = CMGDB.Model(10, 12, [0, 0], [1, 1], F)
    morse_graph, _ = CMGDB.ComputeMorseGraph(model)
    CMGDB.PlotMorseSets(morse_graph, cmap=matplotlib.cm.viridis)
    ```