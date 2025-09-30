#!/usr/bin/env python3

import numpy as np
from MorseGraph.grids import UniformGrid
from MorseGraph.dynamics import BoxMapData

# Replicate the example setup
def henon_map(x, a=1.4, b=0.3):
    """Standard Henon map, vectorized."""
    x_next = 1 - a * x[:, 0]**2 + x[:, 1]
    y_next = b * x[:, 0]
    return np.column_stack([x_next, y_next])

# Generate sample data (smaller for debugging)
lower_bounds = np.array([-2.5, -0.5])
upper_bounds = np.array([2.5, 0.5])
num_points = 100  # Much smaller for debugging
X = np.random.uniform(low=lower_bounds, high=upper_bounds, size=(num_points, 2))
Y = henon_map(X)

# Create grid (smaller for debugging)
divisions = np.array([4, 4])  # Much smaller
domain = np.array([[-2.5, -0.5], [2.5, 0.5]])
grid = UniformGrid(bounds=domain, divisions=divisions)

print(f"Grid has {len(grid.get_boxes())} boxes")
print(f"Data has {len(X)} points")

# Create dynamics
dynamics = BoxMapData(X, Y, grid, input_epsilon=1e-1, output_epsilon=1e-1)

print(f"Points assigned to {len(dynamics.box_to_points)} boxes")

# Test a few boxes
boxes = grid.get_boxes()
for i in range(min(3, len(boxes))):
    print(f"\nTesting box {i}: {boxes[i]}")
    
    # Test with input perturbation
    if dynamics.input_epsilon > 0:
        expanded_box = np.array([
            boxes[i][0] - dynamics.input_epsilon,
            boxes[i][1] + dynamics.input_epsilon
        ])
        print(f"Expanded box: {expanded_box}")
        
        # Check what indices this gives
        relevant_indices = dynamics._get_relevant_box_indices(expanded_box)
        print(f"Relevant indices: {relevant_indices} (count: {len(relevant_indices)})")
        
    result = dynamics(boxes[i])
    print(f"Result: {result}")