#!/usr/bin/env python3
"""
Parameter sweep for detecting bistability in the Ives ecological model.

This script searches for parameter values (specifically the consumption coefficient c)
where the system exhibits bistability: coexistence of a stable equilibrium and a
period-12 periodic orbit.

Usage:
    python tests/ives_parameter.py --quick          # Quick test (~10 parameters)
    python tests/ives_parameter.py --refined        # Refined sweep (100-500 parameters)
    python tests/ives_parameter.py --num-params 50  # Custom number of parameters

Output:
    results/bistable_parameters.json  # Detailed bistability data
    results/parameter_sweep.csv       # All parameters + classifications
    results/bistability_plots.png     # Visualization
"""

import numpy as np
import json
import csv
import os
import sys
import argparse
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directory to path for MorseGraph imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MorseGraph.systems import ives_model_log


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class AttractorInfo:
    """Information about a detected attractor."""
    attractor_type: str  # 'equilibrium', 'period_12', 'other', 'none'
    representative_point: Optional[List[float]]  # One point on the attractor
    orbit_points: Optional[List[List[float]]] = None  # For periodic orbits

@dataclass
class ParameterResult:
    """Results for a single parameter value."""
    c_value: float
    has_equilibrium: bool
    has_period_12: bool
    is_bistable: bool
    equilibrium_point: Optional[List[float]]
    period_12_orbit: Optional[List[List[float]]]
    notes: str = ""


# ============================================================================
# Trajectory Analysis
# ============================================================================

def simulate_trajectory(c_value: float,
                       initial_condition: np.ndarray,
                       n_steps: int = 1000,
                       transient: int = 200,
                       **model_params) -> np.ndarray:
    """
    Simulate the Ives model for given parameter c.

    Args:
        c_value: Consumption coefficient
        initial_condition: Starting point in log space
        n_steps: Total simulation steps
        transient: Steps to discard before analysis
        **model_params: Additional Ives model parameters

    Returns:
        Trajectory array of shape (n_steps, 3)
    """
    # Set default parameters (from 7_ives_model.py)
    params = {
        'r1': 3.873,
        'r2': 11.746,
        'c': c_value,
        'd': 0.5517,
        'p': 0.06659,
        'q': 0.9026,
        'offset': 0.001
    }
    params.update(model_params)

    trajectory = np.zeros((n_steps, 3))
    trajectory[0] = initial_condition

    for i in range(n_steps - 1):
        trajectory[i + 1] = ives_model_log(trajectory[i], **params)

    return trajectory


def detect_equilibrium(c_value: float,
                       ic_near_equilibrium: np.ndarray,
                       n_steps: int = 500,
                       transient: int = 200,
                       tolerance: float = 1e-5) -> Tuple[bool, Optional[np.ndarray]]:
    """
    Detect if trajectory converges to an equilibrium.

    Args:
        c_value: Parameter value
        ic_near_equilibrium: Initial condition near expected equilibrium
        n_steps: Simulation steps
        transient: Steps to wait before checking convergence
        tolerance: Convergence criterion ||x(t+1) - x(t)|| < tol

    Returns:
        (has_equilibrium, equilibrium_point)
    """
    traj = simulate_trajectory(c_value, ic_near_equilibrium, n_steps)

    # Check fixed point condition after transient
    for t in range(transient, n_steps - 1):
        diff = np.linalg.norm(traj[t + 1] - traj[t])
        if diff < tolerance:
            # Found equilibrium
            equilibrium = traj[t:].mean(axis=0)  # Average last points
            return True, equilibrium

    return False, None


def detect_period_12(c_value: float,
                    test_ics: List[np.ndarray],
                    n_steps: int = 600,
                    transient: int = 200,
                    period_tolerance: float = 1e-4,
                    equilibrium_distance: float = 1e-2) -> Tuple[bool, Optional[np.ndarray]]:
    """
    Detect period-12 periodic orbit by testing multiple initial conditions.

    Args:
        c_value: Parameter value
        test_ics: List of initial conditions to test
        n_steps: Simulation steps
        transient: Steps to wait before checking periodicity
        period_tolerance: Criterion ||x(t+12) - x(t)|| < tol
        equilibrium_distance: Must be ||x(t) - eq|| > dist (not equilibrium)

    Returns:
        (has_period_12, orbit_points) where orbit_points is array of shape (12, 3)
    """
    # We'll use a rough equilibrium estimate to exclude it
    # (actual equilibrium shifts slightly with c, but should be close)
    approx_equilibrium = np.array([0.792107, 0.209010, 0.376449])

    for ic in test_ics:
        traj = simulate_trajectory(c_value, ic, n_steps)

        # Check period-12 condition in the settled region
        for t in range(transient, n_steps - 12):
            diff_12 = np.linalg.norm(traj[t + 12] - traj[t])
            dist_eq = np.linalg.norm(traj[t] - approx_equilibrium)

            if diff_12 < period_tolerance and dist_eq > equilibrium_distance:
                # Found period-12 orbit! Extract the 12 points
                orbit_points = np.array([traj[t + k] for k in range(12)])
                return True, orbit_points

    return False, None


def analyze_parameter(c_value: float,
                     equilibrium_ic: np.ndarray,
                     test_ics: List[np.ndarray],
                     period_tolerance: float = 1e-4,
                     verbose: bool = False) -> ParameterResult:
    """
    Analyze a single parameter value for bistability.

    Args:
        c_value: Parameter to test
        equilibrium_ic: IC near equilibrium
        test_ics: ICs for periodic orbit search
        period_tolerance: Tolerance for period-12 detection
        verbose: Print diagnostic info

    Returns:
        ParameterResult with full analysis
    """
    # Detect equilibrium
    has_eq, eq_point = detect_equilibrium(c_value, equilibrium_ic)

    # Detect period-12
    has_p12, p12_orbit = detect_period_12(c_value, test_ics, period_tolerance=period_tolerance)

    # Determine bistability
    is_bistable = has_eq and has_p12

    result = ParameterResult(
        c_value=c_value,
        has_equilibrium=has_eq,
        has_period_12=has_p12,
        is_bistable=is_bistable,
        equilibrium_point=eq_point.tolist() if eq_point is not None else None,
        period_12_orbit=p12_orbit.tolist() if p12_orbit is not None else None
    )

    if verbose:
        status = "BISTABLE ✓" if is_bistable else f"Eq:{has_eq} P12:{has_p12}"
        print(f"  c={c_value:.3e}: {status}")

    return result


# ============================================================================
# Parameter Sweep
# ============================================================================

def generate_test_ics(n_ics: int = 50,
                     domain_bounds: Tuple[List, List] = ([-1, -4, -1], [2, 1, 1]),
                     seed: int = 42) -> List[np.ndarray]:
    """Generate diverse initial conditions for periodic orbit search."""
    np.random.seed(seed)
    ics = []

    for _ in range(n_ics):
        ic = np.array([
            np.random.uniform(domain_bounds[0][0], domain_bounds[1][0]),
            np.random.uniform(domain_bounds[0][1], domain_bounds[1][1]),
            np.random.uniform(domain_bounds[0][2], domain_bounds[1][2])
        ])
        ics.append(ic)

    return ics


def sweep_parameters(c_min: float,
                    c_max: float,
                    num_params: int,
                    n_ics: int = 50,
                    period_tolerance: float = 1e-4,
                    output_dir: str = "results",
                    verbose: bool = True) -> List[ParameterResult]:
    """
    Sweep parameter c and detect bistability.

    Args:
        c_min, c_max: Range of c values
        num_params: Number of parameter values to test
        n_ics: Number of initial conditions to test per parameter
        period_tolerance: Tolerance for period-12 detection
        output_dir: Directory for results
        verbose: Show progress

    Returns:
        List of ParameterResult objects
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate parameter values (log-uniform spacing)
    c_values = np.logspace(np.log10(c_min), np.log10(c_max), num_params)

    # Known equilibrium (will shift slightly with c)
    equilibrium_ic = np.array([0.792107, 0.209010, 0.376449])

    # Test ICs for periodic orbit detection
    test_ics = generate_test_ics(n_ics=n_ics)

    print(f"\n{'='*80}")
    print(f"PARAMETER SWEEP: c ∈ [{c_min:.3e}, {c_max:.3e}]")
    print(f"Testing {num_params} parameter values with {n_ics} ICs each")
    print(f"Period-12 tolerance: {period_tolerance:.1e}")
    print(f"{'='*80}\n")

    results = []
    bistable_count = 0

    iterator = tqdm(c_values, desc="Sweeping parameters") if verbose else c_values

    for c_val in iterator:
        result = analyze_parameter(c_val, equilibrium_ic, test_ics, period_tolerance=period_tolerance, verbose=False)
        results.append(result)

        if result.is_bistable:
            bistable_count += 1
            if verbose:
                tqdm.write(f"  ✓ BISTABLE at c={c_val:.6e}")

    print(f"\n{'='*80}")
    print(f"SWEEP COMPLETE: Found {bistable_count}/{num_params} bistable parameters")
    print(f"{'='*80}\n")

    return results


# ============================================================================
# Results I/O
# ============================================================================

def save_results(results: List[ParameterResult], output_dir: str):
    """Save results to JSON and CSV files."""

    # Save bistable parameters with full details to JSON
    bistable_results = [r for r in results if r.is_bistable]

    bistable_data = []
    for r in bistable_results:
        bistable_data.append({
            'c_value': r.c_value,
            'equilibrium': r.equilibrium_point,
            'period_12_orbit': r.period_12_orbit
        })

    json_path = os.path.join(output_dir, "bistable_parameters.json")
    with open(json_path, 'w') as f:
        json.dump(bistable_data, f, indent=2)

    print(f"✓ Saved {len(bistable_data)} bistable parameters to {json_path}")

    # Save all parameters to CSV
    csv_path = os.path.join(output_dir, "parameter_sweep.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['c_value', 'has_equilibrium', 'has_period_12', 'is_bistable'])

        for r in results:
            writer.writerow([r.c_value, r.has_equilibrium, r.has_period_12, r.is_bistable])

    print(f"✓ Saved full sweep data to {csv_path}")


def visualize_results(results: List[ParameterResult], output_dir: str):
    """Create visualization of parameter sweep results."""

    c_values = [r.c_value for r in results]
    has_eq = [r.has_equilibrium for r in results]
    has_p12 = [r.has_period_12 for r in results]
    is_bistable = [r.is_bistable for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Equilibrium detection
    ax = axes[0, 0]
    colors = ['green' if h else 'red' for h in has_eq]
    ax.scatter(c_values, has_eq, c=colors, alpha=0.6, s=50)
    ax.set_xscale('log')
    ax.set_xlabel('c (consumption coefficient)', fontsize=11)
    ax.set_ylabel('Has Equilibrium', fontsize=11)
    ax.set_title('Equilibrium Detection', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Panel 2: Period-12 detection
    ax = axes[0, 1]
    colors = ['blue' if h else 'red' for h in has_p12]
    ax.scatter(c_values, has_p12, c=colors, alpha=0.6, s=50)
    ax.set_xscale('log')
    ax.set_xlabel('c (consumption coefficient)', fontsize=11)
    ax.set_ylabel('Has Period-12', fontsize=11)
    ax.set_title('Period-12 Orbit Detection', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Panel 3: Bistability
    ax = axes[1, 0]
    colors = ['purple' if b else 'gray' for b in is_bistable]
    sizes = [100 if b else 30 for b in is_bistable]
    ax.scatter(c_values, is_bistable, c=colors, alpha=0.7, s=sizes)
    ax.set_xscale('log')
    ax.set_xlabel('c (consumption coefficient)', fontsize=11)
    ax.set_ylabel('Is Bistable', fontsize=11)
    ax.set_title('Bistability (Eq + Period-12)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Panel 4: Summary bar chart
    ax = axes[1, 1]
    counts = [
        sum(has_eq),
        sum(has_p12),
        sum(is_bistable)
    ]
    labels = ['Equilibrium', 'Period-12', 'Bistable']
    colors_bar = ['green', 'blue', 'purple']
    ax.bar(labels, counts, color=colors_bar, alpha=0.7)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(f'Summary (n={len(results)})', fontsize=12, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)

    # Add count labels on bars
    for i, (label, count) in enumerate(zip(labels, counts)):
        ax.text(i, count + 0.5, str(count), ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()

    plot_path = os.path.join(output_dir, "bistability_plots.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to {plot_path}")

    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Parameter sweep for Ives model bistability detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/ives_parameter.py --quick                    # Quick test (~10 params, 50 ICs)
  python tests/ives_parameter.py --refined                  # Refined sweep (200 params, 50 ICs)
  python tests/ives_parameter.py --num-params 50 --n-ics 100  # Custom settings
  python tests/ives_parameter.py --c-min 2e-7 --c-max 3.5e-7 --num-params 100
  python tests/ives_parameter.py --period-tolerance 1e-3    # Looser tolerance
        """
    )

    parser.add_argument('--quick', action='store_true',
                       help='Quick test with ~10 parameters')
    parser.add_argument('--refined', action='store_true',
                       help='Refined sweep with ~200 parameters')
    parser.add_argument('--num-params', type=int, default=None,
                       help='Custom number of parameters to test')
    parser.add_argument('--c-min', type=float, default=1.5e-7,
                       help='Minimum c value (default: 1.5e-7)')
    parser.add_argument('--c-max', type=float, default=3.67e-7,
                       help='Maximum c value (default: 3.67e-7)')
    parser.add_argument('--n-ics', type=int, default=50,
                       help='Number of initial conditions to test per parameter (default: 50)')
    parser.add_argument('--period-tolerance', type=float, default=1e-4,
                       help='Tolerance for period-12 detection (default: 1e-4)')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory (default: results)')

    args = parser.parse_args()

    # Determine number of parameters
    if args.quick:
        num_params = 10
    elif args.refined:
        num_params = 200
    elif args.num_params is not None:
        num_params = args.num_params
    else:
        num_params = 10  # Default to quick

    # Run sweep
    results = sweep_parameters(
        c_min=args.c_min,
        c_max=args.c_max,
        num_params=num_params,
        n_ics=args.n_ics,
        period_tolerance=args.period_tolerance,
        output_dir=args.output_dir,
        verbose=True
    )

    # Save results
    save_results(results, args.output_dir)

    # Visualize
    visualize_results(results, args.output_dir)

    # Print bistable parameters
    bistable = [r for r in results if r.is_bistable]
    if bistable:
        print(f"\n{'='*80}")
        print(f"BISTABLE PARAMETERS FOUND: {len(bistable)}")
        print(f"{'='*80}")
        for r in bistable:
            print(f"\nc = {r.c_value:.8e}")
            print(f"  Equilibrium: {r.equilibrium_point}")
            print(f"  Period-12 orbit (12 points):")
            if r.period_12_orbit:
                for i, pt in enumerate(r.period_12_orbit):
                    print(f"    Point {i+1}: [{pt[0]:7.4f}, {pt[1]:7.4f}, {pt[2]:7.4f}]")
    else:
        print("\nNo bistable parameters found in this range.")
        print("Try:")
        print("  - Expanding the search range")
        print("  - Increasing num_params for finer resolution")
        print("  - Adjusting tolerance parameters in detect_period_12()")


if __name__ == "__main__":
    main()
