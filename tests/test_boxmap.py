#!/usr/bin/env python3

import numpy as np
from MorseGraph.grids import UniformGrid
from MorseGraph.dynamics import BoxMapData

# Simple test data
X = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
Y = np.array([[0.1, 0.1], [0.6, 0.6], [1.1, 1.1]])

# Simple grid
domain = np.array([[0.0, 0.0], [1.0, 1.0]])
divisions = np.array([2, 2])
grid = UniformGrid(bounds=domain, divisions=divisions)

print("Grid boxes:")
boxes = grid.get_boxes()
for i, box in enumerate(boxes):
    print(f"  Box {i}: {box}")

# Test BoxMapData
dynamics = BoxMapData(X, Y, grid, input_epsilon=0.1, output_epsilon=0.1)

print(f"\nBox to points mapping: {dynamics.box_to_points}")

# Test one box
test_box = boxes[0]
print(f"\nTesting box {test_box}")
result = dynamics(test_box)
print(f"Result: {result}")