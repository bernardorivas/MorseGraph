# MorseGraph Project Plan & To-Do List

**Project Goal:** To create a modern, lightweight, and extensible Python library for computing and analyzing Morse graphs of dynamical systems. The library will draw inspiration from `cmgdb`, `hybrid_boxmap`, and `morals` but will prioritize a pure Python core and exclude heavy topological computations like homology.

---

### **Phase 1: Foundational Analysis and High-Level Planning**

This phase solidifies the core principles and architectural decisions for the new library by synthesizing the best aspects of the existing packages.

*   **1.1. Review and Synthesize Existing Packages:**
    *   **1.1.1. `cmgdb` Analysis:**
        *   [x] **Summary:** A C++/Python library for computing Conley-Morse graphs, including homology. Uses a custom C++ backend for performance.
        *   **Actionable Insights:**
            *   Adopt the core concepts of `BoxMap` from data/functions and `Model` to represent the discretized system.
            *   **Decision:** Avoid the C++ backend and dependencies (Boost, GMP, SDSL) to maintain a "lightweight" pure Python package. The homology computation (`ComputeConleyMorseGraph`) will be excluded as requested.
    *   **1.1.2. `hybrid_boxmap` Analysis:**
        *   [x] **Summary:** A pure Python library for analyzing hybrid systems. Uses `networkx` for graph algorithms and `scipy` for ODE integration. Excellent, modular structure.
        *   **Actionable Insights:**
            *   Adopt its pure Python approach. `networkx` is perfect for graph analysis (SCCs, condensation, reachability).
            *   The `HybridSystem` class is a good model for defining dynamics, but we will generalize it to a `Dynamics` ABC as proposed to support maps and data-driven models directly.
            *   Incorporate the "bloating" concept from `HybridBoxMap.compute` to ensure outer approximations.
            *   Leverage its parallel computation pattern (`joblib`) for performance.
    *   **1.1.3. `morals` Analysis:**
        *   [x] **Summary:** A library that uses autoencoders (PyTorch) to learn a low-dimensional latent space and then applies Morse graph analysis (via `cmgdb`) to understand the dynamics.
        *   **Actionable Insights:**
            *   The new `MorseGraph` library should support a "learned dynamics" model. The structure proposed with `models.py` and `training.py` is directly aligned with this.
            *   This implies a dependency on a deep learning framework. `torch` is a reasonable choice, as used in `morals`.

*   **1.2. Finalize Core Design Principles:**
    *   [x] **Principle 1: Pure Python First.** The core library (`core`, `grids`, `analysis`, `dynamics`) will be pure Python, using `numpy` and `networkx`. This makes it lightweight and easy to install.
    *   [x] **Principle 2: Optional Deep Learning Dependency.** The learned dynamics functionality (`models`, `training`) will depend on `torch`. This should be an optional dependency (e.g., `pip install morsegraph[ml]`).
    *   [x] **Principle 3: Modular and Extensible.** Adhere to the proposed structure. The `Dynamics` and `AbstractGrid` abstract base classes (ABCs) are key to extensibility.
    *   [x] **Principle 4: Focus on Morse Graphs, Not Homology.** The core analysis will be finding Strongly Connected Components (SCCs) to build the Morse graph and computing Regions of Attraction (RoAs). No Conley index computations.

---

### **Phase 2: Detailed Project Scaffolding and API Design**

This phase details the specific modules and their contents, based on the initial plan and insights from the other packages.

*   **2.1. Project Scaffolding:**
    *   [x] Create the directory structure:
        ```
        MorseGraph/
        ├── morsegraph/
        │   ├── __init__.py
        │   ├── analysis.py
        │   ├── core.py
        │   ├── dynamics.py
        │   ├── grids.py
        │   ├── models.py
        │   ├── plot.py
        │   └── training.py
        ├── examples/
        ├── tests/
        ├── README.md
        └── setup.py
        ```
    *   [x] Initialize `setup.py` (or `pyproject.toml`) defining dependencies: `numpy`, `scipy`, `networkx`, `matplotlib`. Define an optional `[ml]` extra for `torch`.

*   **2.2. Module-by-Module API Plan:**

    *   **`morsegraph/dynamics.py` (The "How" - Dynamics Abstraction)**
        *   [x] **`Dynamics` (ABC):** An abstract class defining the interface for any dynamical system.
            *   `__call__(self, box: np.ndarray) -> np.ndarray`: Takes a box `[min_x, min_y, ..., max_x, max_y, ...]` and returns its image as a bounding box.
        *   [x] **`BoxMapData(Dynamics)`:** For dynamics from a dataset `(x, f(x))`.
            *   `__init__(self, X, Y, ...)`: Takes dataset `(X, f(X))`.
            *   `__call__`: Implements the logic to find the image of a box by finding all points in `X` within the box and computing the bounding box of their images in `Y`. Include bloating.
        *   [x] **`BoxMapODE(Dynamics)`:** For dynamics from an Ordinary Differential Equation.
            *   `__init__(self, ode_func, dimension, tau, ...)`: Takes an ODE function `f(t, x)`, dimension, and time horizon `tau`.
            *   `__call__`: Implements the time-`tau` map. It should sample points in the input box (corners, center, etc.), integrate them for time `tau` using `scipy.integrate.solve_ivp`, and compute the bounding box of the final points. Include bloating.
        *   [x] **`BoxMapFunction(Dynamics)`:** For dynamics from an explicit map `f: R^D -> R^D`.
            *   Implement as described in the original markdown file.
        *   [x] **`LearnedDynamics(Dynamics)`:** For dynamics from a trained neural network.
            *   `__init__(self, encoder, dynamics_model, decoder, ...)`: Takes trained PyTorch models.
            *   `__call__`: Implements `decode(dynamics_model(encode(box)))`. The logic for mapping a box through the autoencoder needs to be defined (e.g., map corners and take bounding box in latent space).

    *   **`morsegraph/grids.py` (The "Where" - Grid Abstraction)**
        *   [x] **`AbstractGrid` (ABC):** The interface for all grid structures, as defined in the original markdown.
        *   [x] **`UniformGrid(AbstractGrid)`:** A standard, non-adaptive grid. `subdivide()` will refine the entire grid.
        *   [x] **`AdaptiveGrid(AbstractGrid)`:** A tree-based grid (Quadtree/Octree) that supports local refinement. `subdivide(indices)` will only split the specified boxes.

    *   **`morsegraph/core.py` (The "Engine")**
        *   [x] **`Model` class:** The central class that connects dynamics to a grid.
            *   `__init__(self, dynamics: Dynamics, grid: AbstractGrid)`
            *   `compute_map_graph(self) -> nx.DiGraph`: The core function that builds the state transition graph.

    *   **`morsegraph/analysis.py` (The "Results")**
        *   [x] **`compute_morse_graph(map_graph)`:** Uses `networkx` to find the Strongly Connected Components (SCCs) and build the condensation graph.
        *   [x] **`compute_basins_of_attraction(...)`:** Identifies attractors and uses backward reachability on the `map_graph` to find their basins.
        *   [x] **`iterative_morse_computation(model)`:** The adaptive refinement loop.

    *   **`morsegraph/plot.py` (The "Visuals")**
        *   [x] **`plot_morse_graph(...)`:** Visualizes the Morse graph.
        *   [x] **`plot_morse_sets(...)`:** Creates a 2D plot of the grid, coloring the boxes that belong to Morse sets.
        *   [x] **`plot_basins_of_attraction(...)`:** Creates a 2D plot showing the basins of attraction.

    *   **`morsegraph/models.py` & `training.py` (The "Learning")**
        *   [x] Plan simple `torch.nn.Module` classes for `Encoder`, `Decoder`, and `LatentDynamics`.
        *   [x] Plan a `Training` class in `training.py` to manage the training loop.

---

### **Phase 3: Implementation and Testing Strategy**

This phase provides a roadmap for building and validating the library incrementally.

*   **3.1. Staged Implementation Plan:**
    *   [x] **Milestone 1 (Core Workflow):** Implement the non-adaptive core.
        1.  `dynamics.py`: `Dynamics` ABC and `BoxMapFunction`.
        2.  `grids.py`: `AbstractGrid` ABC and `UniformGrid`.
        3.  `core.py`: `Model` class.
        4.  `analysis.py`: `compute_morse_graph`.
        5.  **Goal:** Be able to take a simple function and a uniform grid, and compute a basic Morse graph.
    *   [x] **Milestone 2 (Expanded Capabilities):**
        1.  Add `BoxMapData` and `BoxMapODE` to `dynamics.py`.
        2.  Implement `compute_basins_of_attraction` in `analysis.py`.
        3.  Implement all plotting functions in `plot.py`.
        4.  **Goal:** Support all basic dynamics types and produce all key visualizations.
    *   [x] **Milestone 3 (Advanced Features):**
        1.  Implement `AdaptiveGrid` and `iterative_morse_computation`.
        2.  **Goal:** Enable adaptive analysis for focusing computational effort on complex regions.
    *   [x] **Milestone 4 (ML Integration):**
        1.  Implement `models.py`, `training.py`, and `LearnedDynamics`.
        2.  **Goal:** Support the full "learn-then-analyze" workflow.

*   **3.2. Testing and Examples:**
    *   [x] **Unit Tests:** For each function and class, create corresponding tests in the `tests/` directory (e.g., `test_grid.py`, `test_analysis.py`).
    *   [x] **Integration Tests (Examples):** Create a series of Jupyter notebooks in `examples/` for each use case. These will serve as excellent documentation and integration tests.
        1.  **`1_map_dynamics.ipynb`:** Demonstrate `BoxMapFunction` with a simple 2D map (e.g., Henon map).
        2.  **`2_data_driven.ipynb`:** Demonstrate `BoxMapData` with a sample dataset.
        3.  **`3_ode_dynamics.ipynb`:** Demonstrate `BoxMapODE` with a simple 2D ODE (e.g., bistable system).
        4.  **`4_adaptive_refinement.ipynb`:** Showcase `AdaptiveGrid` and `iterative_morse_computation`.
        5.  **`5_learned_dynamics.ipynb`:** Walk through the full ML workflow: training an autoencoder and analyzing the learned latent dynamics.
