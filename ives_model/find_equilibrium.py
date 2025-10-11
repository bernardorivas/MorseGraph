#!/usr/bin/env python3
"""
Find the equilibrium point for the Ives model in log scale using numerical optimization.
"""

import numpy as np
from scipy.optimize import fsolve, root
from MorseGraph.systems import ives_model_log
from ives_modules import Config

print("="*80)
print("Finding Equilibrium for Ives Model (log scale)")
print("="*80)

# Define the fixed point equation: f(x) - x = 0
def fixed_point_residual(x):
    """Returns f(x) - x, which should be zero at equilibrium."""
    f_x = ives_model_log(
        x,
        r1=Config.R1,
        r2=Config.R2,
        c=Config.C,
        d=Config.D,
        p=Config.P,
        q=Config.Q,
        offset=Config.LOG_OFFSET
    )
    return f_x - x

# Try multiple initial guesses
initial_guesses = [
    np.array([0.792, 0.209, 0.376]),  # in log-scale, otherwise [6.19, 1.62, 2.38]
    np.array([0.0, 0.0, 0.0]),        # Origin
]

print("\nTrying multiple initial guesses...\n")

best_solution = None
best_error = float('inf')

for i, x0 in enumerate(initial_guesses):
    print(f"Initial guess {i+1}: {x0}")

    try:
        # Use fsolve (Levenberg-Marquardt)
        solution = fsolve(fixed_point_residual, x0, full_output=True)
        x_eq = solution[0]
        info = solution[1]

        # Verify the solution
        residual = fixed_point_residual(x_eq)
        error = np.linalg.norm(residual)

        print(f"  Solution: {x_eq}")
        print(f"  Error: {error:.6e}")

        if error < best_error:
            best_error = error
            best_solution = x_eq

        print()

    except Exception as e:
        print(f"  Failed: {e}\n")

print("="*80)
print("BEST SOLUTION FOUND")
print("="*80)

if best_solution is not None:
    print(f"\nEquilibrium (log scale): {best_solution}")
    print(f"  log(midge)    = {best_solution[0]:.6f}")
    print(f"  log(algae)    = {best_solution[1]:.6f}")
    print(f"  log(detritus) = {best_solution[2]:.6f}")

    # Verify
    f_eq = ives_model_log(
        best_solution,
        r1=Config.R1,
        r2=Config.R2,
        c=Config.C,
        d=Config.D,
        p=Config.P,
        q=Config.Q,
        offset=Config.LOG_OFFSET
    )

    print(f"\nf(equilibrium): {f_eq}")
    print(f"  log(midge)    = {f_eq[0]:.6f}")
    print(f"  log(algae)    = {f_eq[1]:.6f}")
    print(f"  log(detritus) = {f_eq[2]:.6f}")

    error = np.linalg.norm(f_eq - best_solution)
    print(f"\nFinal error: {error:.6e}")

    if error < 1e-6:
        print("✓ VERIFIED: Valid equilibrium point found!")
    else:
        print("⚠ WARNING: Solution may not be accurate")

    # Convert to original space
    original_space = 10**best_solution - Config.LOG_OFFSET
    print(f"\nEquilibrium (original space):")
    print(f"  midge     = {original_space[0]:.6f}")
    print(f"  algae     = {original_space[1]:.6f}")
    print(f"  detritus  = {original_space[2]:.6f}")

    print(f"\nPython array (for easy copy):")
    print(f"equilibrium_point = np.array([{best_solution[0]:.6f}, {best_solution[1]:.6f}, {best_solution[2]:.6f}])")
else:
    print("No solution found!")

print("="*80)
