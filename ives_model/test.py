#!/usr/bin/env python3
"""
Quick test to verify the equilibrium point for the Ives model in log scale.

For an equilibrium point z*, we should have f(z*) = z* (fixed point).
"""

import numpy as np
from MorseGraph.systems import ives_model_log
from ives_modules import Config

# Equilibrium point (in log10 scale)
equilibrium = np.array([0.792, 0.209, 0.376]) #  in log scale, which corresponds to [6.19, 1.62, 2.38]

print("="*80)
print("Testing Ives Model Equilibrium Point")
print("="*80)
print(f"\nEquilibrium point (log scale): {equilibrium}")
print(f"  log(midge)    = {equilibrium[0]:.4f}")
print(f"  log(algae)    = {equilibrium[1]:.4f}")
print(f"  log(detritus) = {equilibrium[2]:.4f}")

# Evaluate f(equilibrium)
f_equilibrium = ives_model_log(
    equilibrium,
    r1=Config.R1,
    r2=Config.R2,
    c=Config.C,
    d=Config.D,
    p=Config.P,
    q=Config.Q,
    offset=Config.LOG_OFFSET
)

print(f"\nf(equilibrium) (log scale): {f_equilibrium}")
print(f"  log(midge)    = {f_equilibrium[0]:.4f}")
print(f"  log(algae)    = {f_equilibrium[1]:.4f}")
print(f"  log(detritus) = {f_equilibrium[2]:.4f}")

# Compute error
error = f_equilibrium - equilibrium
error_norm = np.linalg.norm(error)

print(f"\nError: f(equilibrium) - equilibrium = {error}")
print(f"  Δ midge      = {error[0]:.6e}")
print(f"  Δ algae      = {error[1]:.6e}")
print(f"  Δ detritus   = {error[2]:.6e}")
print(f"\nError norm (L2): {error_norm:.6e}")

# Check if it's a fixed point (within tolerance)
tolerance = 1e-3
is_equilibrium = error_norm < tolerance

print(f"\nIs equilibrium (tolerance={tolerance})? {is_equilibrium}")

if is_equilibrium:
    print("✓ PASS: Point is a valid equilibrium")
else:
    print("✗ FAIL: Point is NOT a valid equilibrium")
    print(f"  Error norm {error_norm:.6e} exceeds tolerance {tolerance}")

print("="*80)
