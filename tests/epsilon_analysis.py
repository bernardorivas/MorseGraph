#!/usr/bin/env python3

import numpy as np
from MorseGraph.grids import UniformGrid

# Replicate the example setup
divisions = np.array([128, 128])
domain = np.array([[-2.5, -0.5], [2.5, 0.5]])
grid = UniformGrid(bounds=domain, divisions=divisions)

print("Grid analysis:")
print(f"Domain: {domain}")
print(f"Divisions: {divisions}")
print(f"Box size: {grid.box_size}")
print(f"Total boxes: {np.prod(divisions)}")

# Analyze epsilon impact
epsilons = [1e-4, 1e-3, 1e-2, 1e-1]

for eps in epsilons:
    # How many box widths is this epsilon?
    eps_in_box_units = eps / grid.box_size
    print(f"\nEpsilon {eps}:")
    print(f"  As fraction of box size: {eps_in_box_units}")
    
    # Estimate how many neighboring boxes this would affect
    # For 2D, roughly (2*ceil(eps/box_size) + 1)^2 boxes
    neighbors_per_dim = 2 * np.ceil(eps_in_box_units) + 1
    total_neighbors = np.prod(neighbors_per_dim)
    print(f"  Approximate neighboring boxes affected: {total_neighbors}")