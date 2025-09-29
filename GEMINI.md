# GEMINI.md - MorseGraph Project

## Project Overview

This repository contains the `MorseGraph` project, a lightweight, modern, and extensible Python library for computing and analyzing Morse graphs of dynamical systems. The goal is to provide a pure-Python core for easy installation and use, with optional extensions for more complex scenarios like machine learning-based dynamics.

The repository is structured as a monorepo and includes several key packages:

*   **`MorseGraph/`**: The primary, new library being developed. It is designed to be modular and focuses on computing Morse graphs from various types of dynamical systems (defined by functions, ODEs, data, or learned models).
*   **`cmgdb/`**: A legacy C++/Python library for computing Conley-Morse graphs. It serves as a source of inspiration for the core concepts.
*   **`hybrid_boxmap/`**: A pure Python library for analyzing hybrid systems. Its modular, `networkx`-based approach to graph analysis is a key architectural influence for `MorseGraph`.
*   **`morals/`**: A research library that combines PyTorch autoencoders with Morse graph analysis to study dynamics in a learned latent space. This inspires the optional machine learning integration in `MorseGraph`.

The core philosophy of the new `MorseGraph` library is to be "pure Python first," using standard libraries like `numpy`, `scipy`, and `networkx` for its main functionality, while allowing for optional, heavy dependencies like `torch` for advanced features.

## Building and Running

The primary `morsegraph` package is designed for easy installation and use within the provided virtual environment.

### Installation

To install the `morsegraph` package and its core dependencies, run the following command from the root of the repository. The `-e` flag installs it in "editable" mode, so changes to the source code are immediately available.

```bash
pip install -e .
```

For machine learning functionalities, which require PyTorch, install the package with the `[ml]` extra:

```bash
pip install -e ".[ml]"
```

### Running Examples

The `examples/` directory contains Jupyter notebooks that demonstrate the library's key features:

*   `1_map_dynamics.ipynb`: Basic Morse graph from a simple function map.
*   `2_data_driven.ipynb`: Analysis of a system from a dataset.
*   `3_ode_dynamics.ipynb`: Analysis of a system defined by an ODE.
*   `4_adaptive_refinement.ipynb`: Using adaptive grids for more efficient computation.
*   `5_learned_dynamics.ipynb`: The full machine learning workflow.

To run these, start a Jupyter server from the root of the project.

## Development Conventions

*   **Modular Architecture**: The `MorseGraph` library is broken down into clear, single-responsibility modules:
    *   `dynamics.py`: Defines the "how" (the system's evolution).
    *   `grids.py`: Defines the "where" (the state space discretization).
    *   `core.py`: The engine that connects dynamics and grids (`Model` class).
    *   `analysis.py`: Computes the results (Morse graphs, basins of attraction).
    *   `plot.py`: Handles visualization.
    *   `models.py` & `training.py`: Manages the optional ML components.
*   **Extensibility**: The library uses Abstract Base Classes (ABCs) for `Dynamics` and `AbstractGrid` to make it easy to add new types of systems or grid structures.
*   **Testing**:
    *   **Unit Tests**: Located in the `tests/` directory, they should be created for each new function or class.
    *   **Integration Tests**: The Jupyter notebooks in the `examples/` directory serve as integration tests and usage documentation.
*   **Dependencies**: The core library aims to be lightweight. New dependencies should be added thoughtfully, and heavy dependencies like `torch` must be optional.
