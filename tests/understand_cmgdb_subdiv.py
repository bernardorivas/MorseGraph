#!/usr/bin/env python3
"""
Understand how CMGDB subdivision parameter works.
"""

import CMGDB
import numpy as np

def simple_map(rect):
    """Simple identity-like map that keeps things in domain."""
    dim = 3
    center = [(rect[i] + rect[i+dim])/2 for i in range(dim)]
    # Scale down slightly to stay in domain
    result = [c * 0.9 for c in center]
    return result + [r + 0.01 for r in result]  # lower + upper

print("="*80)
print("CMGDB SUBDIVISION INTERPRETATION TEST")
print("="*80)

# Test with different subdiv values
for subdiv_min in [4, 6, 8, 10]:
    print(f"\nTesting subdiv_min = {subdiv_min}")
    print("-" * 40)

    model = CMGDB.Model(
        subdiv_min,
        subdiv_min,  # min = max for uniform-like behavior
        [-1, -1, -1],
        [1, 1, 1],
        simple_map
    )

    mg, mapg = CMGDB.ComputeMorseGraph(model)

    # Count all boxes in all Morse sets
    total_boxes = 0
    for i in range(mg.num_vertices()):
        boxes = mg.morse_set_boxes(i)
        total_boxes += len(boxes)

    # Estimate boxes per dimension
    boxes_per_dim_est = total_boxes ** (1/3)

    print(f"  Total boxes in Morse sets: {total_boxes}")
    print(f"  Estimated boxes per dim: {boxes_per_dim_est:.2f}")
    print(f"  If 2^subdiv per dim: {2**subdiv_min} per dim → {(2**subdiv_min)**3:,} total")
    print(f"  If 2^subdiv total: {2**subdiv_min} total")
    print(f"  If subdiv = depth in tree: varies by adaptive splitting")

print("\n" + "="*80)
print("INTERPRETATION:")
print("="*80)
print("Based on the results above, CMGDB subdiv likely means:")
print("...")
