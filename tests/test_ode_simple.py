#!/usr/bin/env python3

import numpy as np
from MorseGraph.grids import UniformGrid
from MorseGraph.dynamics import BoxMapODE

def simple_ode(t, x):
    """Simple linear ODE: dx/dt = -x, dy/dt = -y"""
    return np.array([-x[0], -x[1]])

# Simple test
domain = np.array([[-1.0, -1.0], [1.0, 1.0]])
divisions = np.array([2, 2])  # Very small grid
grid = UniformGrid(bounds=domain, divisions=divisions)

print("Testing BoxMapODE with simple linear system...")
print(f"Grid has {len(grid.get_boxes())} boxes")

# Test BoxMapODE
dynamics = BoxMapODE(simple_ode, tau=0.1, epsilon=0.01)

# Test one box
boxes = grid.get_boxes()
test_box = boxes[0]
print(f"Testing box: {test_box}")
print("Computing image...")
result = dynamics(test_box)
print(f"Result: {result}")

print("BoxMapODE test completed successfully!")