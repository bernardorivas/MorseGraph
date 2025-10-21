#!/usr/bin/env python3
"""
Find the stable period-12 periodic orbit in the Ives model.

Based on insights from Ives et al. (2008) Nature paper:
- The periodic orbit goes very close to 0 in linear space
- The floor parameter c prevents extinction and causes bounce-back
- Should use 500 iterations with transient analysis

Strategy:
1. Sample many ICs focusing on the "floor region" where populations are small
2. Run 500 iterations, analyze last 100 points for periodicity
3. Test stability by perturbing found orbits
4. Compare with existing period-7 orbit
"""

import numpy as np
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MorseGraph.systems import ives_model_log


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class PeriodicOrbit:
    """Information about a detected periodic orbit."""
    period: int
    orbit_points: np.ndarray  # Shape (period, 3)
    is_stable: bool
    ic_that_found_it: np.ndarray
    num_ics_converging: int = 1

    def to_dict(self):
        return {
            'period': self.period,
            'orbit_points': self.orbit_points.tolist(),
            'is_stable': self.is_stable,
            'ic_that_found_it': self.ic_that_found_it.tolist(),
            'num_ics_converging': self.num_ics_converging
        }


# ============================================================================
# Ives Model Parameters
# ============================================================================

# Default parameters from the Nature paper
DEFAULT_PARAMS = {
    'r1': 3.873,
    'r2': 11.746,
    'c': 10**-6.435,  # ~3.67e-7
    'd': 0.5517,
    'p': 0.06659,
    # 'q': 0.9026,  # Original value from paper
    'q': 0.738,  # Testing for bistability with period-12
    'offset': 0.001
}


# ============================================================================
# Initial Condition Sampling
# ============================================================================

def generate_strategic_ics(
    n_floor_region: int = 300,
    n_broad: int = 200,
    c_value: float = 10**-6.435,
    offset: float = 0.001,
    seed: int = 42
) -> List[np.ndarray]:
    """
    Generate initial conditions with strategic sampling.

    Two strategies:
    1. Focus on "floor region" where populations are small (near c)
    2. Broad sampling across entire domain for completeness

    Args:
        n_floor_region: Number of ICs near the floor
        n_broad: Number of ICs broadly distributed
        c_value: The floor parameter value
        offset: Offset for log transformation
        seed: Random seed

    Returns:
        List of initial conditions in log10 space
    """
    np.random.seed(seed)
    ics = []

    # Strategy 1: Floor region sampling
    # In linear space, c acts as floor, so sample from c to ~10*c
    # In log10 space: log10(c + offset) to log10(10*c + offset)
    floor_log = np.log10(c_value + offset)
    low_pop_log = np.log10(10 * c_value + offset)

    print(f"Floor region sampling:")
    print(f"  log10(c + offset) = {floor_log:.3f}")
    print(f"  log10(10*c + offset) = {low_pop_log:.3f}")
    print(f"  Generating {n_floor_region} ICs in this region...")

    for _ in range(n_floor_region):
        # Sample each component independently
        # Midge: can be anywhere (they drive the dynamics)
        # Algae: focus on floor region
        # Detritus: focus on floor region
        ic = np.array([
            np.random.uniform(-1, 1),  # Midge: broad range
            np.random.uniform(floor_log, low_pop_log),  # Algae: near floor
            np.random.uniform(floor_log, low_pop_log),  # Detritus: near floor
        ])
        ics.append(ic)

    # Strategy 2: Broad sampling
    # Cover the entire domain to catch any other basins
    domain_lower = np.array([-1, -4, -1])
    domain_upper = np.array([2, 1, 1])

    print(f"\nBroad domain sampling:")
    print(f"  Domain: {domain_lower} to {domain_upper}")
    print(f"  Generating {n_broad} ICs across domain...")

    for _ in range(n_broad):
        ic = np.random.uniform(domain_lower, domain_upper)
        ics.append(ic)

    print(f"\nTotal ICs generated: {len(ics)}")
    return ics


# ============================================================================
# Trajectory Simulation and Period Detection
# ============================================================================

def simulate_trajectory(
    ic: np.ndarray,
    n_steps: int = 500,
    params: dict = None
) -> np.ndarray:
    """
    Simulate the Ives model trajectory.

    Args:
        ic: Initial condition in log10 space
        n_steps: Number of iterations
        params: Model parameters (uses defaults if None)

    Returns:
        Trajectory array of shape (n_steps, 3)
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()

    trajectory = np.zeros((n_steps, 3))
    trajectory[0] = ic

    for i in range(n_steps - 1):
        trajectory[i + 1] = ives_model_log(trajectory[i], **params)

        # Check for divergence
        if np.any(np.isnan(trajectory[i + 1])) or np.any(np.isinf(trajectory[i + 1])):
            # Trajectory diverged, return what we have
            return trajectory[:i+1]

    return trajectory


def detect_period(
    trajectory: np.ndarray,
    transient: int = 400,
    max_period: int = 20,
    tolerance: float = 1e-4
) -> Tuple[Optional[int], Optional[np.ndarray]]:
    """
    Detect periodic behavior in trajectory.

    Analyzes the settled region (after transient) for periodicity.

    Args:
        trajectory: Trajectory array (n_steps, 3)
        transient: Number of initial steps to ignore
        max_period: Maximum period to check
        tolerance: Tolerance for period detection ||x(t+p) - x(t)|| < tol

    Returns:
        (period, orbit_points) or (None, None) if no period found
        orbit_points is array of shape (period, 3) containing one full cycle
    """
    n_steps = len(trajectory)

    if n_steps < transient + max_period:
        return None, None

    # Check each possible period
    for period in range(1, max_period + 1):
        # Need enough points to verify periodicity
        if n_steps < transient + 2 * period:
            continue

        # Check if x(t+period) ≈ x(t) for all t in settled region
        is_periodic = True
        for t in range(transient, n_steps - period):
            diff = np.linalg.norm(trajectory[t + period] - trajectory[t])
            if diff > tolerance:
                is_periodic = False
                break

        if is_periodic:
            # Found periodic orbit! Extract the points
            orbit_points = np.array([trajectory[transient + k] for k in range(period)])
            return period, orbit_points

    return None, None


# ============================================================================
# Stability Testing
# ============================================================================

def test_orbit_stability(
    orbit: PeriodicOrbit,
    n_perturbations: int = 20,
    perturbation_size: float = 1e-3,
    n_steps: int = 200,
    tolerance: float = 1e-3,
    params: dict = None
) -> bool:
    """
    Test if a periodic orbit is stable.

    Strategy:
    1. Take a point on the orbit
    2. Perturb it slightly in random directions
    3. Iterate and see if it returns to the orbit
    4. If most perturbations return → stable, otherwise → unstable

    Args:
        orbit: The PeriodicOrbit to test
        n_perturbations: Number of perturbations to test
        perturbation_size: Size of perturbation in log space
        n_steps: Steps to iterate perturbed point
        tolerance: Distance to orbit to be considered "returned"
        params: Model parameters

    Returns:
        True if stable, False if unstable
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()

    # Take the first point of the orbit as reference
    ref_point = orbit.orbit_points[0]

    converged_count = 0

    for _ in range(n_perturbations):
        # Random perturbation
        perturbation = np.random.randn(3) * perturbation_size
        perturbed_point = ref_point + perturbation

        # Iterate
        traj = simulate_trajectory(perturbed_point, n_steps=n_steps, params=params)

        # Check if final state is close to any point on the orbit
        final_point = traj[-1]

        min_dist = min([
            np.linalg.norm(final_point - orbit_point)
            for orbit_point in orbit.orbit_points
        ])

        if min_dist < tolerance:
            converged_count += 1

    # Consider stable if > 70% of perturbations return
    stability_ratio = converged_count / n_perturbations
    return stability_ratio > 0.7


# ============================================================================
# Main Analysis
# ============================================================================

def find_periodic_orbits(
    params: dict = None,
    n_floor_ics: int = 300,
    n_broad_ics: int = 200,
    n_steps: int = 500,
    transient: int = 400,
    max_period: int = 20,
    period_tolerance: float = 1e-4,
    test_stability: bool = True,
    verbose: bool = True
) -> Dict[int, List[PeriodicOrbit]]:
    """
    Main function to find and classify periodic orbits.

    Args:
        params: Ives model parameters
        n_floor_ics: Number of ICs near floor region
        n_broad_ics: Number of ICs across domain
        n_steps: Simulation steps
        transient: Transient to discard
        max_period: Maximum period to detect
        period_tolerance: Tolerance for period detection
        test_stability: Whether to test stability (slow)
        verbose: Print progress

    Returns:
        Dictionary mapping period -> list of PeriodicOrbit objects
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()

    if verbose:
        print("="*80)
        print("FINDING STABLE PERIOD-12 ORBIT IN IVES MODEL")
        print("="*80)
        print(f"\nParameters:")
        for key, val in params.items():
            if key == 'c':
                print(f"  {key} = {val:.6e}")
            else:
                print(f"  {key} = {val}")

    # Generate initial conditions
    ics = generate_strategic_ics(
        n_floor_region=n_floor_ics,
        n_broad=n_broad_ics,
        c_value=params['c'],
        offset=params['offset']
    )

    # Track found orbits by period
    orbits_by_period: Dict[int, List[PeriodicOrbit]] = {}

    if verbose:
        print(f"\n{'='*80}")
        print(f"SIMULATING {len(ics)} TRAJECTORIES")
        print(f"{'='*80}")
        print(f"  Steps per trajectory: {n_steps}")
        print(f"  Transient: {transient}")
        print(f"  Analysis window: {n_steps - transient} points")

    # Simulate all ICs
    equilibrium_count = 0
    periodic_count = 0
    other_count = 0

    for i, ic in enumerate(ics):
        if verbose and (i+1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(ics)}")

        # Simulate
        traj = simulate_trajectory(ic, n_steps=n_steps, params=params)

        if len(traj) < n_steps:
            # Diverged
            other_count += 1
            continue

        # Detect period
        period, orbit_points = detect_period(
            traj,
            transient=transient,
            max_period=max_period,
            tolerance=period_tolerance
        )

        if period is None:
            other_count += 1
            continue

        if period == 1:
            equilibrium_count += 1
            continue

        # Found periodic orbit!
        periodic_count += 1

        # Check if we've already found this orbit
        found_duplicate = False
        if period in orbits_by_period:
            for existing_orbit in orbits_by_period[period]:
                # Check if first point is close to any point in existing orbit
                distances = [
                    np.linalg.norm(orbit_points[0] - existing_pt)
                    for existing_pt in existing_orbit.orbit_points
                ]
                if min(distances) < 0.1:  # Same orbit
                    existing_orbit.num_ics_converging += 1
                    found_duplicate = True
                    break

        if not found_duplicate:
            # New orbit found!
            new_orbit = PeriodicOrbit(
                period=period,
                orbit_points=orbit_points,
                is_stable=False,  # Will test later
                ic_that_found_it=ic,
                num_ics_converging=1
            )

            if period not in orbits_by_period:
                orbits_by_period[period] = []
            orbits_by_period[period].append(new_orbit)

            if verbose:
                print(f"\n  ✓ NEW PERIOD-{period} ORBIT FOUND!")
                print(f"    IC: {ic}")
                print(f"    First point: {orbit_points[0]}")

    # Summary
    if verbose:
        print(f"\n{'='*80}")
        print("TRAJECTORY CLASSIFICATION")
        print(f"{'='*80}")
        print(f"  Equilibrium (period-1): {equilibrium_count}")
        print(f"  Periodic (period ≥ 2):  {periodic_count}")
        print(f"  Other/diverged:         {other_count}")
        print(f"  Total:                  {len(ics)}")

    # Test stability
    if test_stability and verbose:
        print(f"\n{'='*80}")
        print("STABILITY TESTING")
        print(f"{'='*80}")

    for period in sorted(orbits_by_period.keys()):
        for orbit in orbits_by_period[period]:
            if test_stability:
                if verbose:
                    print(f"\n  Testing period-{period} orbit...")
                orbit.is_stable = test_orbit_stability(orbit, params=params)
                if verbose:
                    stability_str = "STABLE ✓" if orbit.is_stable else "UNSTABLE ✗"
                    print(f"    → {stability_str}")
                    print(f"    Found by {orbit.num_ics_converging} ICs")

    return orbits_by_period


# ============================================================================
# Visualization
# ============================================================================

def visualize_results(
    orbits_by_period: Dict[int, List[PeriodicOrbit]],
    output_dir: str = "results",
    params: dict = None
):
    """Create visualization of found periodic orbits."""

    os.makedirs(output_dir, exist_ok=True)

    # Prepare data
    periods = []
    num_orbits = []
    num_stable = []

    for period in sorted(orbits_by_period.keys()):
        periods.append(period)
        orbits = orbits_by_period[period]
        num_orbits.append(len(orbits))
        num_stable.append(sum(1 for o in orbits if o.is_stable))

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel 1: Bar chart of periods found
    ax = axes[0, 0]
    x = np.arange(len(periods))
    width = 0.35
    ax.bar(x - width/2, num_orbits, width, label='Total', alpha=0.7)
    ax.bar(x + width/2, num_stable, width, label='Stable', alpha=0.7)
    ax.set_xlabel('Period', fontsize=12)
    ax.set_ylabel('Number of orbits', fontsize=12)
    ax.set_title('Periodic Orbits Found', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{p}' for p in periods])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: 3D scatter of orbit points (period ≥ 7)
    ax = axes[0, 1]
    ax.remove()
    ax = fig.add_subplot(2, 2, 2, projection='3d')

    colors = plt.cm.tab10(np.linspace(0, 1, len(periods)))

    for i, period in enumerate(periods):
        if period >= 7:  # Only show higher periods
            for orbit in orbits_by_period[period]:
                points = orbit.orbit_points
                marker = 'o' if orbit.is_stable else 'x'
                ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                          c=[colors[i]], marker=marker, s=100,
                          label=f'Period-{period} {"(stable)" if orbit.is_stable else "(unstable)"}',
                          alpha=0.7)

    ax.set_xlabel('log₁₀(Midge)', fontsize=10)
    ax.set_ylabel('log₁₀(Algae)', fontsize=10)
    ax.set_zlabel('log₁₀(Detritus)', fontsize=10)
    ax.set_title('3D Orbit Locations', fontsize=12, fontweight='bold')

    # Avoid duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=8)

    # Panel 3: Time series of period-12 (if found)
    ax = axes[1, 0]
    if 12 in orbits_by_period:
        orbit_12 = orbits_by_period[12][0]  # First period-12 orbit
        points = orbit_12.orbit_points

        t = np.arange(len(points))
        ax.plot(t, points[:, 0], 'o-', label='Midge', alpha=0.7)
        ax.plot(t, points[:, 1], 's-', label='Algae', alpha=0.7)
        ax.plot(t, points[:, 2], '^-', label='Detritus', alpha=0.7)
        ax.set_xlabel('Step in cycle', fontsize=12)
        ax.set_ylabel('log₁₀(Population)', fontsize=12)
        stability = "STABLE" if orbit_12.is_stable else "UNSTABLE"
        ax.set_title(f'Period-12 Orbit ({stability})', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Period-12 orbit NOT FOUND',
               ha='center', va='center', fontsize=14, fontweight='bold')
        ax.set_title('Period-12 Orbit', fontsize=12, fontweight='bold')

    # Panel 4: Basin of attraction summary
    ax = axes[1, 1]

    # Count ICs converging to each attractor type
    eq_count = 0  # Would need to track this separately
    data = []
    labels_list = []

    for period in sorted(orbits_by_period.keys()):
        for j, orbit in enumerate(orbits_by_period[period]):
            label = f"Period-{period}"
            if orbit.is_stable:
                label += " (stable)"
            if len(orbits_by_period[period]) > 1:
                label += f" #{j+1}"
            data.append(orbit.num_ics_converging)
            labels_list.append(label)

    if data:
        colors_bar = ['green' if 'stable' in label else 'orange' for label in labels_list]
        ax.barh(range(len(data)), data, color=colors_bar, alpha=0.7)
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels(labels_list, fontsize=9)
        ax.set_xlabel('Number of ICs converging', fontsize=12)
        ax.set_title('Basin Sizes', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'period12_search_results.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to {output_path}")
    plt.close()


# ============================================================================
# Results I/O
# ============================================================================

def save_results(
    orbits_by_period: Dict[int, List[PeriodicOrbit]],
    output_dir: str = "results"
):
    """Save results to JSON."""

    os.makedirs(output_dir, exist_ok=True)

    # Convert to serializable format
    results = {}
    for period, orbits in orbits_by_period.items():
        results[f"period_{period}"] = [orbit.to_dict() for orbit in orbits]

    output_path = os.path.join(output_dir, 'periodic_orbits_found.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ Saved results to {output_path}")

    # Also save summary
    summary_path = os.path.join(output_dir, 'periodic_orbits_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("PERIODIC ORBIT SEARCH RESULTS\n")
        f.write("="*80 + "\n\n")

        for period in sorted(orbits_by_period.keys()):
            f.write(f"\nPERIOD-{period} ORBITS:\n")
            f.write("-" * 40 + "\n")

            for i, orbit in enumerate(orbits_by_period[period]):
                stability = "STABLE" if orbit.is_stable else "UNSTABLE"
                f.write(f"\nOrbit #{i+1}: {stability}\n")
                f.write(f"  Found by {orbit.num_ics_converging} ICs\n")
                f.write(f"  Example IC: {orbit.ic_that_found_it}\n")
                f.write(f"  Orbit points:\n")
                for j, pt in enumerate(orbit.orbit_points):
                    f.write(f"    {j+1}: [{pt[0]:.6f}, {pt[1]:.6f}, {pt[2]:.6f}]\n")

    print(f"✓ Saved summary to {summary_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    """Main entry point."""

    print("\n" + "="*80)
    print("SEARCH FOR STABLE PERIOD-12 ORBIT IN IVES MODEL")
    print("Based on Ives et al. (2008) Nature paper")
    print("="*80 + "\n")

    # Run the search
    orbits_by_period = find_periodic_orbits(
        n_floor_ics=300,
        n_broad_ics=200,
        n_steps=500,
        transient=400,
        max_period=20,
        period_tolerance=1e-4,
        test_stability=True,
        verbose=True
    )

    # Print summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}\n")

    if not orbits_by_period:
        print("No periodic orbits found!")
    else:
        for period in sorted(orbits_by_period.keys()):
            orbits = orbits_by_period[period]
            stable_count = sum(1 for o in orbits if o.is_stable)
            print(f"\nPeriod-{period}:")
            print(f"  Total orbits found: {len(orbits)}")
            print(f"  Stable orbits: {stable_count}")
            print(f"  Unstable orbits: {len(orbits) - stable_count}")

            for i, orbit in enumerate(orbits):
                stability = "STABLE ✓" if orbit.is_stable else "UNSTABLE ✗"
                print(f"    Orbit #{i+1}: {stability}, found by {orbit.num_ics_converging} ICs")

    # Check for period-12 specifically
    if 12 in orbits_by_period:
        print(f"\n{'='*80}")
        print("✓ PERIOD-12 ORBIT FOUND!")
        print(f"{'='*80}")

        for i, orbit in enumerate(orbits_by_period[12]):
            stability = "STABLE" if orbit.is_stable else "UNSTABLE"
            print(f"\nOrbit #{i+1}: {stability}")
            print(f"  Found by {orbit.num_ics_converging} ICs")
            print(f"  Orbit points:")
            for j, pt in enumerate(orbit.orbit_points):
                print(f"    {j+1}: [{pt[0]:.6f}, {pt[1]:.6f}, {pt[2]:.6f}]")
    else:
        print(f"\n{'='*80}")
        print("✗ PERIOD-12 ORBIT NOT FOUND")
        print(f"{'='*80}")
        print("\nSuggestions:")
        print("  - Increase number of ICs (n_floor_ics, n_broad_ics)")
        print("  - Adjust tolerance (period_tolerance)")
        print("  - Check parameter values (c, d, etc.)")
        print("  - Increase n_steps for better convergence")

    # Save results
    save_results(orbits_by_period, output_dir="results")

    # Visualize
    visualize_results(orbits_by_period, output_dir="results")

    print(f"\n{'='*80}")
    print("DONE!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
