"""
Plotting and Visualization Utilities for Ives Model

This module contains all visualization functions for:
- Morse graphs
- Morse sets (2D and 3D)
- Barycenters
- Latent space transformations
- Preimage classifications
- Training history
"""

import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import io
from matplotlib.patches import Rectangle
import CMGDB


# ============================================================================
# Basic Morse Graph Plotting
# ============================================================================

def plot_morse_graph_to_ax(morse_graph, ax, title='', cmap=plt.cm.cool):
    """
    Plot a Morse graph to a matplotlib axis.

    Args:
        morse_graph: CMGDB MorseGraph object
        ax: Matplotlib axis
        title: Plot title
        cmap: Color map for the graph
    """
    if morse_graph is None or morse_graph.num_vertices() == 0:
        ax.text(0.5, 0.5, 'Not computed', ha='center', va='center')
        ax.set_title(title)
        ax.axis('off')
        return
    
    # Render as PNG
    gv_source = CMGDB.PlotMorseGraph(morse_graph, cmap=cmap)
    img_data = gv_source.pipe(format='png')
    img = plt.imread(io.BytesIO(img_data))
    
    # Display image while preserving aspect ratio
    ax.imshow(img, aspect='equal', interpolation='bilinear')
    ax.set_title(title)
    ax.axis('off')


def plot_morse_graph(morse_graph, output_path, title='Morse Graph', cmap=plt.cm.cool):
    """
    Plot and save a Morse graph to a file.

    Args:
        morse_graph: CMGDB MorseGraph object
        output_path: Path to save the figure
        title: Plot title
        cmap: Color map
    """
    _, ax = plt.subplots(figsize=(8, 8))
    plot_morse_graph_to_ax(morse_graph, ax, title=title, cmap=cmap)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# 3D Visualization
# ============================================================================

def plot_morse_sets_3d_cmgdb(morse_graph, ax, lower_bounds, upper_bounds, data=None, cmap=plt.cm.cool):
    """
    Plot 3D Morse sets with optional data overlay.

    Args:
        morse_graph: CMGDB MorseGraph object
        ax: Matplotlib 3D axis
        lower_bounds: Lower bounds [x_min, y_min, z_min]
        upper_bounds: Upper bounds [x_max, y_max, z_max]
        data: Optional data points to overlay (N x 3 array)
        cmap: Color map
    """
    colors = cmap(np.linspace(0, 1, morse_graph.num_vertices()))

    # Calculate appropriate marker size based on box sizes
    all_boxes = []
    for morse_idx in range(morse_graph.num_vertices()):
        boxes = morse_graph.morse_set_boxes(morse_idx)
        if boxes:
            all_boxes.extend(boxes)

    if all_boxes:
        # Compute average box size across all dimensions
        box_sizes = []
        for b in all_boxes:
            dim = len(b) // 2
            size = np.mean([b[d + dim] - b[d] for d in range(dim)])
            box_sizes.append(size)
        avg_box_size = np.mean(box_sizes)

        # Calculate marker size: scale by domain size, with min/max bounds
        domain_size = np.mean([upper_bounds[d] - lower_bounds[d] for d in range(len(lower_bounds))])
        marker_size = max(10, min(200, (avg_box_size / domain_size) * 5000))
    else:
        marker_size = 20

    for morse_idx in range(morse_graph.num_vertices()):
        boxes = morse_graph.morse_set_boxes(morse_idx)
        if boxes:
            centers = np.array([[(b[d] + b[d+len(b)//2]) / 2.0 for d in range(len(b)//2)] for b in boxes])
            ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
                      c=[colors[morse_idx]], s=marker_size, alpha=0.4, marker='s')

    # Plot data points if provided
    if data is not None and len(data) > 0:
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='black', s=1, alpha=0.3, label='Data')

    ax.set_xlabel('log(midge)')
    ax.set_ylabel('log(algae)')
    ax.set_zlabel('log(detritus)')
    # ax.set_xlim(lower_bounds[0], upper_bounds[0])
    # ax.set_ylim(lower_bounds[1], upper_bounds[1])
    # ax.set_zlim(lower_bounds[2], upper_bounds[2])
    ax.view_init(elev=20, azim=45)


def plot_barycenters_3d(morse_graph, domain_bounds, output_path, title='Barycenters of Morse Sets',
                         data_overlay=None, cmap=plt.cm.cool):
    """
    Compute and plot 3D barycenters of Morse sets.

    Args:
        morse_graph: CMGDB MorseGraph object
        domain_bounds: [[x_min, y_min, z_min], [x_max, y_max, z_max]]
        output_path: Path to save the figure
        title: Plot title
        data_overlay: Optional data to overlay (N x 3 array)
        cmap: Color map
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    num_morse_sets = morse_graph.num_vertices()

    # Plot data overlay first (behind everything else) if provided
    if data_overlay is not None and len(data_overlay) > 0:
        ax.scatter(data_overlay[:, 0], data_overlay[:, 1], data_overlay[:, 2],
                  c='black', s=0.5, alpha=0.05, zorder=1)

    if num_morse_sets == 0:
        ax.text2D(0.5, 0.5, 'No Morse sets', ha='center', va='center', transform=ax.transAxes)
    else:
        colors = cmap(np.linspace(0, 1, num_morse_sets))

        for i in range(num_morse_sets):
            morse_set_boxes = morse_graph.morse_set_boxes(i)
            if morse_set_boxes:
                dim = len(morse_set_boxes[0]) // 2
                barycenters = []
                for box in morse_set_boxes:
                    barycenter = np.array([(box[j] + box[j + dim]) / 2.0 for j in range(dim)])
                    barycenters.append(barycenter)

                data = np.array(barycenters)
                ax.scatter(data[:, 0], data[:, 1], data[:, 2],
                          c=[colors[i]], marker='s', s=10, label=f'Morse Set {i}', zorder=2)

    ax.set_xlabel('log(midge)')
    ax.set_ylabel('log(algae)')
    ax.set_zlabel('log(detritus)')
    ax.set_title(title)

    if domain_bounds is not None:
        ax.set_xlim(domain_bounds[0][0], domain_bounds[1][0])
        ax.set_ylim(domain_bounds[0][1], domain_bounds[1][1])
        ax.set_zlim(domain_bounds[0][2], domain_bounds[1][2])

    ax.view_init(elev=30, azim=45)

    if 0 < num_morse_sets <= 10:
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# 2D Visualization
# ============================================================================

def plot_morse_sets_2d_cmgdb(morse_graph, ax, cmap=plt.cm.cool, data=None, data_color='black'):
    """
    Plot 2D Morse sets as rectangles.

    Args:
        morse_graph: CMGDB MorseGraph object
        ax: Matplotlib axis
        cmap: Color map
        data: Optional data points to overlay
        data_color: Color for data points
    """
    if morse_graph is None:
        return

    num_morse_sets = morse_graph.num_vertices()
    if num_morse_sets == 0:
        return

    colors = cmap(np.linspace(0, 1, num_morse_sets))

    for morse_idx in range(num_morse_sets):
        boxes = morse_graph.morse_set_boxes(morse_idx)
        if boxes:
            for b in boxes:
                rect = Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1],
                               facecolor=colors[morse_idx],
                               edgecolor='none')
                ax.add_patch(rect)

    # Plot data points if provided
    if data is not None and len(data) > 0:
        ax.scatter(data[:, 0], data[:, 1], c=data_color, s=1, alpha=0.3, label='Data')


def plot_data_boxes(ax, data, latent_bounds, subdivision_depth=10):
    """
    Plot boxes that contain data points as white square markers.

    Args:
        ax: Matplotlib axis
        data: Data points (N x 2 array)
        latent_bounds: [[x_min, y_min], [x_max, y_max]]
        subdivision_depth: Grid subdivision depth
    """
    if data is None or len(data) == 0:
        return

    # Compute box size based on subdivision depth
    lower = np.array(latent_bounds[0])
    upper = np.array(latent_bounds[1])
    n_boxes = 2 ** subdivision_depth
    box_width = (upper - lower) / n_boxes

    # Find unique boxes containing data
    data_boxes = set()
    for point in data:
        # Find which box this point belongs to
        box_idx = np.floor((point - lower) / box_width).astype(int)
        # Clamp to valid range
        box_idx = np.clip(box_idx, 0, n_boxes - 1)
        data_boxes.add(tuple(box_idx))

    if not data_boxes:
        return

    # Convert to centers for scatter plot
    centers = []
    for box_idx in data_boxes:
        box_center = lower + (np.array(box_idx) + 0.5) * box_width
        centers.append(box_center)
    centers = np.array(centers)

    # Calculate exact marker size to match box size in the figure
    ref_point = centers[0]
    corner1 = ax.transData.transform([ref_point[0], ref_point[1]])
    corner2 = ax.transData.transform([ref_point[0] + box_width[0], ref_point[1] + box_width[1]])
    box_size_points = np.mean(np.abs(corner2 - corner1))
    marker_size = box_size_points ** 2
    marker_size = max(1, min(1000, marker_size))

    # Plot as scatter with square markers
    ax.scatter(centers[:, 0], centers[:, 1],
              s=9*marker_size, marker='s',
              facecolor='white', edgecolor='none',
              linewidth=0.5, alpha=0.6, zorder=1) # marker_size * 9 for white box of 3*size(grid_box)


def plot_latent_morse_sets_restricted_cmgdb(morse_graph, ax, latent_bounds, data=None,
                                            subdivision_depth=10, cmap=plt.cm.cool, data_color='black'):
    """
    Plot latent Morse sets with grey background and white data boxes.

    Args:
        morse_graph: CMGDB MorseGraph object
        ax: Matplotlib axis
        latent_bounds: [[x_min, y_min], [x_max, y_max]]
        data: Optional data points
        subdivision_depth: Grid subdivision depth
        cmap: Color map
        data_color: Color for data points
    """
    # Plot grey background
    grey_rect = Rectangle((latent_bounds[0][0], latent_bounds[0][1]),
                           latent_bounds[1][0] - latent_bounds[0][0],
                           latent_bounds[1][1] - latent_bounds[0][1],
                           facecolor='#e0e0e0', edgecolor='none')
    ax.add_patch(grey_rect)

    # Plot white boxes containing data
    plot_data_boxes(ax, data, latent_bounds, subdivision_depth=subdivision_depth)

    # Plot morse sets on top
    plot_morse_sets_2d_cmgdb(morse_graph, ax, data=data, cmap=cmap, data_color=data_color)

    ax.set_xlim(latent_bounds[0][0], latent_bounds[1][0])
    ax.set_ylim(latent_bounds[0][1], latent_bounds[1][1])


# ============================================================================
# Training History
# ============================================================================

def plot_training_losses(train_losses, val_losses, output_path):
    """
    Plot training and validation losses.

    Args:
        train_losses: Dict with keys: 'reconstruction', 'dynamics_recon', 'dynamics_consistency', 'total'
        val_losses: Dict with same keys
        output_path: Path to save the figure
    """
    _, axes = plt.subplots(2, 2, figsize=(14, 10))

    def set_dynamic_ylim(ax, train_loss_history, val_loss_history):
        num_epochs = len(train_loss_history)
        start_epoch = int(num_epochs * 0.2) if num_epochs > 10 else 0
        final_losses = train_loss_history[start_epoch:] + val_loss_history[start_epoch:]
        if final_losses:
            upper_limit = max(final_losses) * 1.2
            ax.set_ylim(bottom=0, top=upper_limit + 1e-6)
        else:
            ax.set_ylim(bottom=0)

    # Total loss
    axes[0, 0].plot(train_losses['total'], label='Train', linewidth=2)
    axes[0, 0].plot(val_losses['total'], label='Validation', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    set_dynamic_ylim(axes[0, 0], train_losses['total'], val_losses['total'])

    # Reconstruction loss
    axes[0, 1].plot(train_losses['reconstruction'], label='Train', linewidth=2)
    axes[0, 1].plot(val_losses['reconstruction'], label='Validation', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Reconstruction: ||D(E(x)) - x||²')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    set_dynamic_ylim(axes[0, 1], train_losses['reconstruction'], val_losses['reconstruction'])

    # Dynamics reconstruction loss
    axes[1, 0].plot(train_losses['dynamics_recon'], label='Train', linewidth=2)
    axes[1, 0].plot(val_losses['dynamics_recon'], label='Validation', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Dynamics Recon: ||D(G(E(x))) - f(x)||²')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    set_dynamic_ylim(axes[1, 0], train_losses['dynamics_recon'], val_losses['dynamics_recon'])

    # Dynamics consistency loss
    axes[1, 1].plot(train_losses['dynamics_consistency'], label='Train', linewidth=2)
    axes[1, 1].plot(val_losses['dynamics_consistency'], label='Validation', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Dynamics Cons: ||G(E(x)) - E(f(x))||²')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    set_dynamic_ylim(axes[1, 1], train_losses['dynamics_consistency'], val_losses['dynamics_consistency'])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# Utility Functions
# ============================================================================

def compute_data_bounds(data, padding_factor=0.05):
    """
    Compute the actual bounding box of the data with optional padding.

    Args:
        data: Array of shape (N, D) containing data points
        padding_factor: Fraction of range to add as padding (default 5%)

    Returns:
        [lower_bounds, upper_bounds] where each is a list of length D
    """
    if data is None or len(data) == 0:
        return None

    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)
    data_range = data_max - data_min

    # Add padding
    lower_bounds = (data_min - padding_factor * data_range).tolist()
    upper_bounds = (data_max + padding_factor * data_range).tolist()

    return [lower_bounds, upper_bounds]


# ============================================================================
# Helper Functions for Advanced Visualizations
# ============================================================================

def classify_points_to_morse_sets(points, morse_graph, latent_bounds):
    """
    Classify which Morse set each point belongs to based on spatial containment.

    Args:
        points: Array of shape (N, D) containing points in latent space
        morse_graph: CMGDB MorseGraph object
        latent_bounds: [lower, upper] bounds for latent space

    Returns:
        numpy array of shape (N,) with morse set index for each point (-1 if not in any set)
    """
    if morse_graph is None or points is None or len(points) == 0:
        return np.full(len(points), -1)

    n_points = len(points)
    labels = np.full(n_points, -1, dtype=int)

    # Get all morse sets (vertices of the morse graph)
    num_morse_sets = morse_graph.num_vertices()

    for morse_set_idx in range(num_morse_sets):
        # Get boxes in this morse set using the correct CMGDB API
        boxes = morse_graph.morse_set_boxes(morse_set_idx)

        # Check each point against each box
        for box in boxes:
            # box format: [x_min, y_min, ..., x_max, y_max, ...] for D-dimensional space
            # Check if points are inside this box
            inside = np.ones(n_points, dtype=bool)
            dim = len(latent_bounds[0])

            for d in range(dim):
                inside &= (points[:, d] >= box[d]) & (points[:, d] <= box[d + dim])

            # Assign points to this morse set (first match wins)
            labels[inside & (labels == -1)] = morse_set_idx

    return labels


def get_morse_set_colors(num_sets, cmap=plt.cm.cool):
    """
    Get consistent colors for morse sets.

    Args:
        num_sets: Number of morse sets
        cmap: Matplotlib colormap to use

    Returns:
        List of colors (one per morse set)
    """
    if num_sets == 0:
        return []

    # Generate colors evenly spaced across colormap
    colors = [cmap(i / max(1, num_sets - 1)) for i in range(num_sets)]
    return colors


def create_latent_grid(latent_bounds, grid_size=70):
    """
    Create a uniform grid in latent space for visualization.

    Args:
        latent_bounds: [lower, upper] bounds for latent space
        grid_size: Number of points per dimension (default 70)

    Returns:
        Array of shape (grid_size^D, D) containing grid points
    """
    lower, upper = latent_bounds
    dim = len(lower)

    # Create 1D grids for each dimension
    grids_1d = [np.linspace(lower[d], upper[d], grid_size) for d in range(dim)]

    # Create meshgrid
    grid_points = np.meshgrid(*grids_1d, indexing='ij')

    # Flatten and stack
    grid_flat = np.column_stack([g.flatten() for g in grid_points])

    return grid_flat


# ============================================================================
# Advanced Visualization Functions
# ============================================================================

def plot_latent_transformation_analysis(x_data, encoder, decoder, device, latent_bounds, output_path, grid_size=70):
    """
    Generate Figure 1: Latent Transformation Analysis (3x2 grid).

    Top row (3D plots):
    - Original Data x
    - Decoded Grid D(grid)
    - Reconstructed Data D(E(x))

    Bottom row (2D plots):
    - Encoded Data E(x)
    - Latent Grid
    - Re-encoded Grid E(D(grid))

    Args:
        x_data: Original 3D data array of shape (N, 3)
        encoder: Trained encoder model
        decoder: Trained decoder model
        device: torch device
        latent_bounds: [lower, upper] bounds for latent space
        output_path: Path to save figure
        grid_size: Grid resolution (default 70)
    """
    import torch
    from mpl_toolkits.mplot3d import Axes3D

    # Use all data points (no downsampling)
    x_vis = x_data

    # Create latent grid
    grid_latent = create_latent_grid(latent_bounds, grid_size=grid_size)

    # Encode data
    with torch.no_grad():
        x_tensor = torch.FloatTensor(x_vis).to(device)
        z_encoded = encoder(x_tensor).cpu().numpy()

        # Decode grid
        grid_tensor = torch.FloatTensor(grid_latent).to(device)
        grid_decoded_3d = decoder(grid_tensor).cpu().numpy()

        # Reconstruct data (E -> D)
        x_reconstructed = decoder(encoder(x_tensor)).cpu().numpy()

        # Re-encode decoded grid (D -> E)
        grid_reencoded = encoder(decoder(grid_tensor)).cpu().numpy()

    # Create figure
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('Autoencoder Transformation Analysis', fontsize=16, fontweight='bold')

    # Top row: 3D plots
    # Column 1: Original Data x
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.scatter(x_vis[:, 0], x_vis[:, 1], x_vis[:, 2], c='black', s=1, alpha=0.5)
    ax1.set_title(f'Original Data x\n({len(x_vis)} points)', fontsize=12)
    ax1.set_xlabel('log(midge)'); ax1.set_ylabel('log(algae)'); ax1.set_zlabel('log(detritus)')

    # Column 2: Decoded Grid D(grid)
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    ax2.scatter(grid_decoded_3d[:, 0], grid_decoded_3d[:, 1], grid_decoded_3d[:, 2], c='blue', s=1, alpha=0.3)
    ax2.set_title(f'Decoded Grid D(grid)\n({grid_size}x{grid_size} = {len(grid_latent)} points)', fontsize=12)
    ax2.set_xlabel('log(midge)'); ax2.set_ylabel('log(algae)'); ax2.set_zlabel('log(detritus)')

    # Column 3: Reconstructed Data D(E(x))
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    ax3.scatter(x_reconstructed[:, 0], x_reconstructed[:, 1], x_reconstructed[:, 2], c='red', s=1, alpha=0.5)
    ax3.set_title(f'Reconstructed Data D(E(x))\n({len(x_vis)} points)', fontsize=12)
    ax3.set_xlabel('log(midge)'); ax3.set_ylabel('log(algae)'); ax3.set_zlabel('log(detritus)')

    # Bottom row: 2D plots
    # Column 1: Encoded Data E(x)
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.scatter(z_encoded[:, 0], z_encoded[:, 1], c='black', s=1, alpha=0.5)
    ax4.set_title(f'Encoded Data E(x)\n({len(x_vis)} points)', fontsize=12)
    ax4.set_xlabel('Latent 0'); ax4.set_ylabel('Latent 1')
    ax4.set_aspect('equal')

    # Column 2: Latent Grid
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.scatter(grid_latent[:, 0], grid_latent[:, 1], c='blue', s=1, alpha=0.3)
    ax5.set_title(f'Latent Grid\n({grid_size}x{grid_size} = {len(grid_latent)} points)', fontsize=12)
    ax5.set_xlabel('Latent 0'); ax5.set_ylabel('Latent 1')
    ax5.set_aspect('equal')

    # Column 3: Re-encoded Grid E(D(grid))
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.scatter(grid_reencoded[:, 0], grid_reencoded[:, 1], c='red', s=1, alpha=0.5)
    ax6.set_title(f'Re-encoded Grid E(D(grid))\n({len(grid_latent)} points)', fontsize=12)
    ax6.set_xlabel('Latent 0'); ax6.set_ylabel('Latent 1')
    ax6.set_aspect('equal')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_morse_graph_comparison(
        morse_graph_img_path,
        barycenters,
        latent_mg_full,
        latent_mg_train,
        latent_mg_val,
        latent_mg_large,
        encoder,
        device,
        x_full,
        z_train,
        z_val,
        latent_bounds,
        domain_bounds,
        output_dir):
    """
    Generate Figure 2: Morse Graph Comparison (multiple variants).

    Creates 5 variant figures showing:
    - Top row: 3D MG, Latent full MG, Latent train MG, Latent val MG, Latent large MG
    - Bottom row: 3D barycenters, Full 2D overlay, Train 2D overlay, Val 2D overlay, Large 2D overlay

    Args:
        morse_graph: Precomputed CMGDB morse graph
        barycenters: Numpy file with barycenter data
        latent_mg_full: CMGDB morse graph for full latent dynamics
        latent_mg_train: CMGDB morse graph for train-restricted latent dynamics
        latent_mg_val: CMGDB morse graph for val-restricted latent dynamics
        latent_mg_large: CMGDB morse graph for large sample with neighbors
        encoder: Trained encoder model
        device: torch device
        x_full: Full 3D data array
        z_train: Train latent data
        z_val: Val latent data
        latent_bounds: [lower, upper] bounds for latent space
        domain_bounds: [lower, upper] bounds for 3D space
        output_dir: Directory to save figures
    """
    import torch
    from mpl_toolkits.mplot3d import Axes3D

    # Use viridis colormap for latent plots (to distinguish from cool in 3D)
    latent_cmap = plt.cm.viridis

    # Encode barycenters if available
    z_barycenters = None
    barycenters_3d_list = []
    if barycenters is not None:
        # Handle both old format ('barycenters' key) and new format ('morse_set_N' keys)
        if 'barycenters' in barycenters:
            barycenters_3d_list = barycenters['barycenters']
        else:
            # New format: morse_set_0, morse_set_1, etc.
            morse_set_keys = sorted([k for k in barycenters.keys() if k.startswith('morse_set_')],
                                   key=lambda x: int(x.split('_')[-1]))
            barycenters_3d_list = [barycenters[k] for k in morse_set_keys]

        # Encode ALL barycenters for each Morse set
        if barycenters_3d_list:
            with torch.no_grad():
                z_barycenters = []
                for bary_set in barycenters_3d_list:
                    if len(bary_set) > 0:
                        # Encode all barycenters in this Morse set
                        bary_tensor = torch.FloatTensor(bary_set).to(device)
                        z_bary_set = encoder(bary_tensor).cpu().numpy()
                        z_barycenters.append(z_bary_set)
                    else:
                        z_barycenters.append(np.array([]))

    # Classify points to morse sets for coloring
    labels_full = classify_points_to_morse_sets(z_train, latent_mg_full, latent_bounds) if latent_mg_full else None
    labels_train = classify_points_to_morse_sets(z_train, latent_mg_train, latent_bounds) if latent_mg_train else None
    labels_val = classify_points_to_morse_sets(z_val, latent_mg_val, latent_bounds) if latent_mg_val else None
    labels_large = classify_points_to_morse_sets(z_train, latent_mg_large, latent_bounds) if latent_mg_large else None

    # Get colors for morse sets using viridis
    num_sets_full = latent_mg_full.num_vertices() if latent_mg_full else 0
    colors_full = get_morse_set_colors(num_sets_full, cmap=latent_cmap)

    # Get colors for barycenters using cool colormap (to match morse graph)
    barycenter_colors = get_morse_set_colors(len(barycenters_3d_list), cmap=plt.cm.cool) if barycenters_3d_list else []

    # Use all data (no downsampling)
    z_train_sample = z_train
    z_val_sample = z_val

    # Generate variants
    variant_configs = {
        '': {'grey_bg': False, 'show_all_data': True, 'show_morse_boxes': True, 'show_colored_morse_data': True, 'data_color': 'black', 'show_barycenters': True},
        '_grey': {'grey_bg': False, 'show_all_data': True, 'show_morse_boxes': True, 'show_colored_morse_data': True, 'data_color': 'grey', 'show_barycenters': True},
        '_clean': {'grey_bg': False, 'show_all_data': False, 'show_morse_boxes': True, 'show_colored_morse_data': True, 'data_color': 'black', 'show_barycenters': False},
        '_minimal': {'grey_bg': False, 'show_all_data': False, 'show_morse_boxes': True, 'show_colored_morse_data': False, 'data_color': 'black', 'show_barycenters': False},
        '_no_overlay': {'grey_bg': False, 'show_all_data': True, 'show_morse_boxes': False, 'show_colored_morse_data': False, 'data_color': 'black', 'show_barycenters': False},
    }

    for variant, config in variant_configs.items():
        fig = plt.figure(figsize=(30, 12))

        # Top row: Morse graphs (uses cool colormap from pre-rendered image)
        ax1 = fig.add_subplot(2, 5, 1)
        try:
            img = plt.imread(morse_graph_img_path)
            ax1.imshow(img)
            ax1.set_title('Morse Graph (Pre-computed)')
        except FileNotFoundError:
            ax1.text(0.5, 0.5, 'Image not found', ha='center', va='center')
            ax1.set_title('Morse Graph (Pre-computed)')
        ax1.axis('off')

        # Latent full (use viridis)
        ax2 = fig.add_subplot(2, 5, 2)
        if latent_mg_full:
            plot_morse_graph_to_ax(latent_mg_full, ax2, title='Latent dynamics - MG', cmap=latent_cmap)
        else:
            ax2.text(0.5, 0.5, 'No Morse graph', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Latent dynamics - MG')
            ax2.axis('off')

        # Latent train (use viridis)
        ax3 = fig.add_subplot(2, 5, 3)
        if latent_mg_train:
            plot_morse_graph_to_ax(latent_mg_train, ax3, title='Train-restricted MG', cmap=latent_cmap)
        else:
            ax3.text(0.5, 0.5, 'No Morse graph', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Train-restricted MG')
            ax3.axis('off')

        # Latent val (use viridis)
        ax4 = fig.add_subplot(2, 5, 4)
        if latent_mg_val:
            plot_morse_graph_to_ax(latent_mg_val, ax4, title='Val-restricted MG', cmap=latent_cmap)
        else:
            ax4.text(0.5, 0.5, 'No Morse graph', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Val-restricted MG')
            ax4.axis('off')

        # Latent large (use viridis)
        ax5 = fig.add_subplot(2, 5, 5)
        if latent_mg_large:
            plot_morse_graph_to_ax(latent_mg_large, ax5, title='Large (+neighbors) MG', cmap=latent_cmap)
        else:
            ax5.text(0.5, 0.5, 'No Morse graph', ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Large (+neighbors) MG')
            ax5.axis('off')

        # Bottom row: 3D barycenters + 2D overlays
        ax6 = fig.add_subplot(2, 5, 6, projection='3d')
        if barycenters_3d_list:
            # Use cool colormap for 3D (matches pre-rendered Morse graph)
            gt_colors = get_morse_set_colors(len(barycenters_3d_list), cmap=plt.cm.cool)
            for i, bary_set in enumerate(barycenters_3d_list):
                if len(bary_set) > 0:
                    # Plot all barycenters in this Morse set
                    color = gt_colors[i] if i < len(gt_colors) else 'gray'
                    ax6.scatter(bary_set[:, 0], bary_set[:, 1], bary_set[:, 2], c=[color], s=1, marker='s', alpha=0.6)
            ax6.set_title('Scatterplot of barycenters', fontsize=10)
            ax6.set_xlabel('log(midge)'); ax6.set_ylabel('log(algae)'); ax6.set_zlabel('log(detritus)')
        else:
            ax6.text(0.5, 0.5, 0.5, 'No barycenters', ha='center', va='center')
            ax6.set_title('Scatterplot of barycenters')

        # Full: 2D overlay (use viridis colormap for latent)
        # No grey background or data boxes for full latent dynamics
        ax7 = fig.add_subplot(2, 5, 7)
        _plot_2d_latent_overlay(ax7, z_train_sample, labels_full, colors_full, z_barycenters, barycenter_colors,
                                latent_mg_full, latent_bounds, config, 'MorseSets+E(barycenter)+E(data)',
                                cmap=latent_cmap, show_grey_bg=False, show_data_boxes=False)

        # Train: 2D overlay (use viridis)
        ax8 = fig.add_subplot(2, 5, 8)
        _plot_2d_latent_overlay(ax8, z_train_sample, labels_train, colors_full, z_barycenters, barycenter_colors,
                                latent_mg_train, latent_bounds, config, 'Train-restricted',
                                cmap=latent_cmap, show_grey_bg=config['grey_bg'])

        # Val: 2D overlay (use viridis)
        ax9 = fig.add_subplot(2, 5, 9)
        _plot_2d_latent_overlay(ax9, z_val_sample, labels_val, colors_full, z_barycenters, barycenter_colors,
                                latent_mg_val, latent_bounds, config, 'Val-restricted',
                                cmap=latent_cmap, show_grey_bg=config['grey_bg'])

        # Large: 2D overlay (use viridis)
        ax10 = fig.add_subplot(2, 5, 10)
        _plot_2d_latent_overlay(ax10, z_train_sample, labels_large, colors_full, z_barycenters, barycenter_colors,
                                latent_mg_large, latent_bounds, config, 'Large (+neighbors)',
                                cmap=latent_cmap, show_grey_bg=config['grey_bg'])

        plt.tight_layout()
        output_path = os.path.join(output_dir, f'morse_graph_comparison{variant}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


def _plot_2d_latent_overlay(ax, z_data, labels, colors, z_barycenters, barycenter_colors, morse_graph, latent_bounds, config, title, cmap=plt.cm.viridis, show_grey_bg=True, show_data_boxes=True):
    """Helper function to plot 2D latent space with various overlays.
    
    Args:
        show_grey_bg: If True, show grey background rectangle (default True)
        show_data_boxes: If True, show white boxes around data locations (default True)
    """
    from matplotlib.patches import Rectangle

    # Conditionally show grey background + white data boxes
    if show_grey_bg:
        # Plot grey background rectangle
        grey_rect = Rectangle((latent_bounds[0][0], latent_bounds[0][1]),
                               latent_bounds[1][0] - latent_bounds[0][0],
                               latent_bounds[1][1] - latent_bounds[0][1],
                               facecolor='#e0e0e0', edgecolor='none', zorder=0)
        ax.add_patch(grey_rect)

    # Plot white boxes showing where data exists
    if show_data_boxes and z_data is not None and len(z_data) > 0:
        plot_data_boxes(ax, z_data, latent_bounds, subdivision_depth=20)

    # Plot Morse sets as colored rectangles (zorder=2)
    if config.get('show_morse_boxes', False):
        if morse_graph is not None and morse_graph.num_vertices() > 0:
            num_morse_sets = morse_graph.num_vertices()
            morse_colors = cmap(np.linspace(0, 1, num_morse_sets))

            for morse_idx in range(num_morse_sets):
                boxes = morse_graph.morse_set_boxes(morse_idx)
                if boxes:
                    for b in boxes:
                        rect = Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1],
                                       facecolor=morse_colors[morse_idx],
                                       edgecolor='none', alpha=0.6, zorder=2)
                        ax.add_patch(rect)


    # All data (black scatter) - zorder=4 to be on top of morse sets
    if config.get('show_all_data', False):
        ax.scatter(z_data[:, 0], z_data[:, 1], c=config.get('data_color', 'black'), s=1, alpha=0.5, zorder=4)

    # Colored morse sets - plot data points colored by their morse set (zorder=5 on top of everything)
    if config.get('show_colored_morse_data', False):
        if labels is not None and len(colors) > 0:
            for i, color in enumerate(colors):
                mask = labels == i
                if np.any(mask):
                    ax.scatter(z_data[mask, 0], z_data[mask, 1], c=[color], s=5, alpha=0.8, zorder=5)

    # E(barycenters) as colored squares - use MG colors (cool colormap) - zorder=6 to be on top of everything
    if z_barycenters is not None and config.get('show_barycenters', False):
        for i, z_bary_set in enumerate(z_barycenters):
            if len(z_bary_set) > 0:
                color = barycenter_colors[i] if i < len(barycenter_colors) else 'gray'
                ax.scatter(z_bary_set[:, 0], z_bary_set[:, 1], c=[color], s=1, marker='s', alpha=0.6, zorder=6)

    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Latent 0'); ax.set_ylabel('Latent 1')
    ax.set_aspect('equal')
    if latent_bounds:
        ax.set_xlim(latent_bounds[0][0], latent_bounds[1][0])
        ax.set_ylim(latent_bounds[0][1], latent_bounds[1][1])


def plot_preimage_classification(
        latent_mg_full,
        latent_mg_train,
        latent_mg_val,
        latent_mg_large,
        encoder,
        decoder,
        device,
        x_full,
        z_train,
        z_val,
        latent_bounds,
        domain_bounds,
        output_path,
        num_samples=10000):
    """
    Generate Figure 3: Preimage Classification (3x4 grid).

    Rows:
    - Top: 3D preimages E^-1(Morse Sets) for Full/Train/Val/Large
    - Middle: Morse Graphs for Full/Train/Val/Large
    - Bottom: 2D Latent Morse Sets + Samples for Full/Train/Val/Large

    Args:
        latent_mg_full: CMGDB morse graph for full latent dynamics
        latent_mg_train: CMGDB morse graph for train-restricted latent dynamics
        latent_mg_val: CMGDB morse graph for val-restricted latent dynamics
        latent_mg_large: CMGDB morse graph for large sample with neighbors
        encoder: Trained encoder model
        decoder: Trained decoder model
        device: torch device
        x_full: Full 3D data array
        z_train: Train latent data
        z_val: Val latent data
        latent_bounds: [lower, upper] bounds for latent space
        domain_bounds: [lower, upper] bounds for 3D space
        output_path: Path to save figure
        num_samples: Number of samples to generate for preimage visualization
    """
    import torch
    from mpl_toolkits.mplot3d import Axes3D

    # Classify points
    labels_full = classify_points_to_morse_sets(z_train, latent_mg_full, latent_bounds) if latent_mg_full else None
    labels_train = classify_points_to_morse_sets(z_train, latent_mg_train, latent_bounds) if latent_mg_train else None
    labels_val = classify_points_to_morse_sets(z_val, latent_mg_val, latent_bounds) if latent_mg_val else None
    labels_large = classify_points_to_morse_sets(z_train, latent_mg_large, latent_bounds) if latent_mg_large else None

    # Get colors
    num_sets_full = latent_mg_full.num_vertices() if latent_mg_full else 0
    colors = get_morse_set_colors(num_sets_full)

    # Sample latent points uniformly
    z_samples = create_latent_grid(latent_bounds, grid_size=int(np.sqrt(num_samples)))

    # Decode samples to 3D
    with torch.no_grad():
        z_tensor = torch.FloatTensor(z_samples).to(device)
        x_decoded = decoder(z_tensor).cpu().numpy()

    # Classify samples
    labels_samples_full = classify_points_to_morse_sets(z_samples, latent_mg_full, latent_bounds) if latent_mg_full else None

    # Create figure
    fig = plt.figure(figsize=(26, 15))
    fig.suptitle('Preimage Classification: E^-1(Morse Sets) in 3D Space', fontsize=16, fontweight='bold')

    # Column 1: Full
    # Top: 3D preimages
    ax1 = fig.add_subplot(3, 4, 1, projection='3d')
    # Plot original data first (black, behind everything)
    if x_full is not None and len(x_full) > 0:
        ax1.scatter(x_full[:, 0], x_full[:, 1], x_full[:, 2], c='black', s=0.5, alpha=0.1, zorder=1)
    if labels_samples_full is not None:
        for i, color in enumerate(colors):
            mask = labels_samples_full == i
            if np.any(mask):
                ax1.scatter(x_decoded[mask, 0], x_decoded[mask, 1], x_decoded[mask, 2], c=[color], s=1, alpha=0.5, zorder=2)
        ax1.set_title('Full: 3D Preimages', fontsize=10)
    else:
        ax1.text(0.5, 0.5, 0.5, 'No Morse sets', ha='center', va='center')
        ax1.set_title('Full: 3D Preimages')
    ax1.set_xlabel('log(midge)'); ax1.set_ylabel('log(algae)'); ax1.set_zlabel('log(detritus)')

    # Middle: Morse graph
    ax5 = fig.add_subplot(3, 4, 5)
    if latent_mg_full:
        plot_morse_graph_to_ax(latent_mg_full, ax5, title='Full: Morse Graph')
    else:
        ax5.text(0.5, 0.5, 'No Morse graph', ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Full: Morse Graph')
        ax5.axis('off')

    # Bottom: Latent morse sets + samples
    ax9 = fig.add_subplot(3, 4, 9)
    if labels_full is not None:
        # Plot all samples in grey
        ax9.scatter(z_train[:, 0], z_train[:, 1], c='lightgrey', s=1, alpha=0.3)
        # Plot classified samples with colors
        for i, color in enumerate(colors):
            mask = labels_full == i
            if np.any(mask):
                ax9.scatter(z_train[mask, 0], z_train[mask, 1], c=[color], s=5, alpha=0.8)
        ax9.set_title('Full: Latent Morse Sets + Samples', fontsize=10)
    else:
        ax9.text(0.5, 0.5, 'No Morse sets', ha='center', va='center', transform=ax9.transAxes)
        ax9.set_title('Full: Latent Morse Sets + Samples')
    ax9.set_xlabel('Latent 0'); ax9.set_ylabel('Latent 1')
    ax9.set_aspect('equal')

    # Column 2: Train
    # Top: 3D preimages
    ax2 = fig.add_subplot(3, 4, 2, projection='3d')
    # Plot original data first (black, behind everything)
    if x_full is not None and len(x_full) > 0:
        ax2.scatter(x_full[:, 0], x_full[:, 1], x_full[:, 2], c='black', s=0.5, alpha=0.1, zorder=1)
    labels_samples_train = classify_points_to_morse_sets(z_samples, latent_mg_train, latent_bounds) if latent_mg_train else None
    if labels_samples_train is not None:
        # Check if there are any classified points
        if np.any(labels_samples_train >= 0):
            for i, color in enumerate(colors):
                mask = labels_samples_train == i
                if np.any(mask):
                    ax2.scatter(x_decoded[mask, 0], x_decoded[mask, 1], x_decoded[mask, 2], c=[color], s=1, alpha=0.5, zorder=2)
            ax2.set_title('Train: 3D Preimages', fontsize=10)
        else:
            ax2.text(0.5, 0.5, 0.5, 'No Morse sets', ha='center', va='center')
            ax2.set_title('Train: 3D Preimages')
    else:
        ax2.text(0.5, 0.5, 0.5, 'No Morse sets', ha='center', va='center')
        ax2.set_title('Train: 3D Preimages')
    ax2.set_xlabel('log(midge)'); ax2.set_ylabel('log(algae)'); ax2.set_zlabel('log(detritus)')

    # Middle: Morse graph
    ax6 = fig.add_subplot(3, 4, 6)
    if latent_mg_train:
        plot_morse_graph_to_ax(latent_mg_train, ax6, title='Train: Morse Graph')
    else:
        ax6.text(0.5, 0.5, 'No Morse graph', ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Train: Morse Graph')
        ax6.axis('off')

    # Bottom: Latent morse sets + samples
    ax10 = fig.add_subplot(3, 4, 10)
    if labels_train is not None and np.any(labels_train >= 0):
        ax10.scatter(z_train[:, 0], z_train[:, 1], c='lightgrey', s=1, alpha=0.3)
        for i, color in enumerate(colors):
            mask = labels_train == i
            if np.any(mask):
                ax10.scatter(z_train[mask, 0], z_train[mask, 1], c=[color], s=5, alpha=0.8)
        ax10.set_title('Train: Latent Morse Sets + Samples', fontsize=10)
    else:
        ax10.text(0.5, 0.5, 'No Morse sets', ha='center', va='center', transform=ax10.transAxes)
        ax10.set_title('Train: Latent Morse Sets + Samples')
    ax10.set_xlabel('Latent 0'); ax10.set_ylabel('Latent 1')
    ax10.set_aspect('equal')

    # Column 3: Val
    # Top: 3D preimages
    ax3 = fig.add_subplot(3, 4, 3, projection='3d')
    # Plot original data first (black, behind everything)
    if x_full is not None and len(x_full) > 0:
        ax3.scatter(x_full[:, 0], x_full[:, 1], x_full[:, 2], c='black', s=0.5, alpha=0.1, zorder=1)
    labels_samples_val = classify_points_to_morse_sets(z_samples, latent_mg_val, latent_bounds) if latent_mg_val else None
    if labels_samples_val is not None:
        if np.any(labels_samples_val >= 0):
            for i, color in enumerate(colors):
                mask = labels_samples_val == i
                if np.any(mask):
                    ax3.scatter(x_decoded[mask, 0], x_decoded[mask, 1], x_decoded[mask, 2], c=[color], s=1, alpha=0.5, zorder=2)
            ax3.set_title('Val: 3D Preimages', fontsize=10)
        else:
            ax3.text(0.5, 0.5, 0.5, 'No Morse sets', ha='center', va='center')
            ax3.set_title('Val: 3D Preimages')
    else:
        ax3.text(0.5, 0.5, 0.5, 'No Morse sets', ha='center', va='center')
        ax3.set_title('Val: 3D Preimages')
    ax3.set_xlabel('log(midge)'); ax3.set_ylabel('log(algae)'); ax3.set_zlabel('log(detritus)')

    # Middle: Morse graph
    ax7 = fig.add_subplot(3, 4, 7)
    if latent_mg_val:
        plot_morse_graph_to_ax(latent_mg_val, ax7, title='Val: Morse Graph')
    else:
        ax7.text(0.5, 0.5, 'No Morse graph', ha='center', va='center', transform=ax7.transAxes)
        ax7.set_title('Val: Morse Graph')
        ax7.axis('off')

    # Bottom: Latent morse sets + samples
    ax11 = fig.add_subplot(3, 4, 11)
    if labels_val is not None and np.any(labels_val >= 0):
        ax11.scatter(z_val[:, 0], z_val[:, 1], c='lightgrey', s=1, alpha=0.3)
        for i, color in enumerate(colors):
            mask = labels_val == i
            if np.any(mask):
                ax11.scatter(z_val[mask, 0], z_val[mask, 1], c=[color], s=5, alpha=0.8)
        ax11.set_title('Val: Latent Morse Sets + Samples', fontsize=10)
    else:
        ax11.text(0.5, 0.5, 'No Morse sets', ha='center', va='center', transform=ax11.transAxes)
        ax11.set_title('Val: Latent Morse Sets + Samples')
    ax11.set_xlabel('Latent 0'); ax11.set_ylabel('Latent 1')
    ax11.set_aspect('equal')

    # Column 4: Large
    # Top: 3D preimages
    ax4 = fig.add_subplot(3, 4, 4, projection='3d')
    # Plot original data first (black, behind everything)
    if x_full is not None and len(x_full) > 0:
        ax4.scatter(x_full[:, 0], x_full[:, 1], x_full[:, 2], c='black', s=0.5, alpha=0.1, zorder=1)
    labels_samples_large = classify_points_to_morse_sets(z_samples, latent_mg_large, latent_bounds) if latent_mg_large else None
    if labels_samples_large is not None:
        if np.any(labels_samples_large >= 0):
            for i, color in enumerate(colors):
                mask = labels_samples_large == i
                if np.any(mask):
                    ax4.scatter(x_decoded[mask, 0], x_decoded[mask, 1], x_decoded[mask, 2], c=[color], s=1, alpha=0.5, zorder=2)
            ax4.set_title('Large: 3D Preimages', fontsize=10)
        else:
            ax4.text(0.5, 0.5, 0.5, 'No Morse sets', ha='center', va='center')
            ax4.set_title('Large: 3D Preimages')
    else:
        ax4.text(0.5, 0.5, 0.5, 'No Morse sets', ha='center', va='center')
        ax4.set_title('Large: 3D Preimages')
    ax4.set_xlabel('log(midge)'); ax4.set_ylabel('log(algae)'); ax4.set_zlabel('log(detritus)')

    # Middle: Morse graph
    ax8 = fig.add_subplot(3, 4, 8)
    if latent_mg_large:
        plot_morse_graph_to_ax(latent_mg_large, ax8, title='Large: Morse Graph')
    else:
        ax8.text(0.5, 0.5, 'No Morse graph', ha='center', va='center', transform=ax8.transAxes)
        ax8.set_title('Large: Morse Graph')
        ax8.axis('off')

    # Bottom: Latent morse sets + samples
    ax12 = fig.add_subplot(3, 4, 12)
    if labels_large is not None and np.any(labels_large >= 0):
        ax12.scatter(z_train[:, 0], z_train[:, 1], c='lightgrey', s=1, alpha=0.3)
        for i, color in enumerate(colors):
            mask = labels_large == i
            if np.any(mask):
                ax12.scatter(z_train[mask, 0], z_train[mask, 1], c=[color], s=5, alpha=0.8)
        ax12.set_title('Large: Latent Morse Sets + Samples', fontsize=10)
    else:
        ax12.text(0.5, 0.5, 'No Morse sets', ha='center', va='center', transform=ax12.transAxes)
        ax12.set_title('Large: Latent Morse Sets + Samples')
    ax12.set_xlabel('Latent 0'); ax12.set_ylabel('Latent 1')
    ax12.set_aspect('equal')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
