
import numpy as np
from functools import partial
import CMGDB
from MorseGraph.systems import ives_model_log
from ives_modules.config import Config

# Equilibrium point (in log10 scale) from ives_modules/morse_graph.py
equilibrium = np.array([0.792107, 0.209010, 0.376449])

# Define the domain
lower_bounds = np.array(Config.LOWER_BOUNDS)
upper_bounds = np.array(Config.UPPER_BOUNDS)

# Define the number of subdivisions per axis
subdivisions = [10, 10, 10]

# Calculate the size of each box
box_sizes = (upper_bounds - lower_bounds) / (2**np.array(subdivisions))

# Find the coordinates of the box containing the equilibrium
box_coords = np.floor((equilibrium - lower_bounds) / box_sizes).astype(int)

# Get the bounds of the box
box_lower = lower_bounds + box_coords * box_sizes
box_upper = lower_bounds + (box_coords + 1) * box_sizes

# The box is represented by a rect in CMGDB
rect = np.concatenate([box_lower, box_upper])

print("="*80)
print("Evaluating f(box) around the equilibrium point")
print("="*80)
print(f"Equilibrium point (log scale): {equilibrium}")
print(f"Domain lower bounds: {lower_bounds}")
print(f"Domain upper bounds: {upper_bounds}")
print(f"Subdivisions per axis (exponent): {subdivisions}")
print(f"Total boxes: 2^{sum(subdivisions)} = {2**sum(subdivisions)}")
print(f"Box coordinates of equilibrium: {box_coords}")
print(f"Box bounds (rect): {rect}")
print("="*80)

# Configure ives_model_log
f = partial(ives_model_log, r1=Config.R1, r2=Config.R2, c=Config.C, d=Config.D, p=Config.P, q=Config.Q, offset=Config.LOG_OFFSET)

# Define the BoxMap
# The result of the box map is the image of the box
image_rect = CMGDB.BoxMap(f, rect)

print("f(box) result:")
print(f"Image box bounds: {image_rect}")
print("="*80)
