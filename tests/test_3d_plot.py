#!/usr/bin/env python3
"""
Test script to visualize 3D barycenters from different viewing angles.

Generates a grid of subplots showing the same data from various perspectives
to help choose the best visualization angle.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

# Path to barycenter data
BARYCENTER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'examples', 'leslie_map_3d', 'leslie_map_3d_results', 'barycenters.npz'
)

def plot_barycenters_multiple_views(barycenter_path, output_path=None):
    """
    Create a grid of 3D plots from different viewing angles.

    Args:
        barycenter_path: Path to barycenters.npz file
        output_path: Optional path to save the figure
    """
    # Load barycenter data
    try:
        barycenters_data = np.load(barycenter_path, allow_pickle=True)
    except FileNotFoundError:
        print(f"Error: Barycenter file not found at {barycenter_path}")
        return

    # Count Morse sets
    num_morse_sets = len(barycenters_data.files)
    if num_morse_sets == 0:
        print("No Morse sets found in barycenter data")
        return

    print(f"Found {num_morse_sets} Morse sets")

    # Define viewing angles (elevation, azimuth)
    # Format: (elev, azim, description)
    views = [
        (30, 45, 'Default (30°, 45°)'),
        (20, 45, 'Low (20°, 45°)'),
        (40, 45, 'High (40°, 45°)'),
        (30, 30, 'Front-Right (30°, 30°)'),
        (30, 60, 'Side-Right (30°, 60°)'),
        (30, 90, 'Side (30°, 90°)'),
        (30, 135, 'Back-Left (30°, 135°)'),
        (30, 180, 'Back (30°, 180°)'),
        (10, 45, 'Very Low (10°, 45°)'),
        (50, 45, 'Very High (50°, 45°)'),
        (30, 0, 'Front (30°, 0°)'),
        (45, 225, 'Back-Right (45°, 225°)')
    ]

    # Create figure with subplots
    fig = plt.figure(figsize=(24, 18))

    # Generate colormap
    colors = matplotlib.cm.cool(np.linspace(0, 1, num_morse_sets))
    sorted_keys = sorted(barycenters_data.files, key=lambda k: int(k.split('_')[-1]))

    for idx, (elev, azim, description) in enumerate(views, start=1):
        ax = fig.add_subplot(3, 4, idx, projection='3d')

        # Plot each Morse set
        for key in sorted_keys:
            morse_set_index = int(key.split('_')[-1])
            points_3d = barycenters_data[key]

            if points_3d.size > 0:
                color = colors[morse_set_index]
                ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
                          c=[color], marker='s', s=1, alpha=0.6,
                          label=f'MS {morse_set_index}')

        # Set view angle
        ax.view_init(elev=elev, azim=azim)

        # Labels and title
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title(description, fontsize=10)

        # Only show legend on first plot to avoid clutter
        if idx == 1:
            ax.legend(loc='upper right', fontsize=6, markerscale=2)

    plt.suptitle('Barycenter Visualizations from Different Viewing Angles',
                 fontsize=16, y=0.995)
    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_recommended_views(barycenter_path, output_path=None):
    """
    Create a smaller figure with 4 recommended viewing angles.

    Args:
        barycenter_path: Path to barycenters.npz file
        output_path: Optional path to save the figure
    """
    # Load barycenter data
    try:
        barycenters_data = np.load(barycenter_path, allow_pickle=True)
    except FileNotFoundError:
        print(f"Error: Barycenter file not found at {barycenter_path}")
        return

    num_morse_sets = len(barycenters_data.files)
    if num_morse_sets == 0:
        print("No Morse sets found in barycenter data")
        return

    # Four recommended views
    views = [
        (30, 45, 'Default View'),
        (20, 30, 'Low Front-Right'),
        (40, 60, 'High Side-Right'),
        (30, 135, 'Back-Left View')
    ]

    fig = plt.figure(figsize=(16, 12))
    colors = matplotlib.cm.cool(np.linspace(0, 1, num_morse_sets))
    sorted_keys = sorted(barycenters_data.files, key=lambda k: int(k.split('_')[-1]))

    for idx, (elev, azim, description) in enumerate(views, start=1):
        ax = fig.add_subplot(2, 2, idx, projection='3d')

        for key in sorted_keys:
            morse_set_index = int(key.split('_')[-1])
            points_3d = barycenters_data[key]

            if points_3d.size > 0:
                color = colors[morse_set_index]
                ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
                          c=[color], marker='s', s=2, alpha=0.7,
                          label=f'MS {morse_set_index}')

        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title(f'{description}\n(elev={elev}°, azim={azim}°)', fontsize=11)

        if idx == 1:
            ax.legend(loc='upper right', fontsize=8, markerscale=3)

    plt.suptitle('Recommended Viewing Angles for Barycenter Plots',
                 fontsize=14, y=0.995)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved recommended views to: {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    """Main entry point."""
    print("3D Barycenter Plot View Explorer")
    print("=" * 50)
    print(f"Loading data from: {BARYCENTER_PATH}")
    print()

    # Generate output directory
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'tests', 'test_outputs'
    )
    os.makedirs(output_dir, exist_ok=True)

    # Generate full grid of views
    print("Generating 12-view comparison...")
    full_output = os.path.join(output_dir, 'barycenters_12_views.png')
    plot_barycenters_multiple_views(BARYCENTER_PATH, full_output)

    print()

    # Generate recommended views
    print("Generating 4 recommended views...")
    recommended_output = os.path.join(output_dir, 'barycenters_recommended_views.png')
    plot_recommended_views(BARYCENTER_PATH, recommended_output)

    print()
    print("=" * 50)
    print("View generation complete!")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
