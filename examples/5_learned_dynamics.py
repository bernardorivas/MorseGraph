#!/usr/bin/env python3
"""
MorseGraph Example 5: Learned Dynamics

This script shows how to compute a Morse Graph from learned neural network models.

The LearnedMapDynamics class uses PyTorch models (encoder, decoder, latent map)
to define dynamics in a learned latent space.

As an example, we train models on Hénon map data and analyze
the learned representation, showcasing the full ML pipeline.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# ML dependencies - check availability
try:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    ML_AVAILABLE = True
except ImportError:
    print("PyTorch not available. This example requires PyTorch for ML functionality.")
    ML_AVAILABLE = False

# Import MorseGraph components
from MorseGraph.grids import UniformGrid
from MorseGraph.core import Model
from MorseGraph.analysis import compute_morse_graph
from MorseGraph.plot import plot_morse_sets

if ML_AVAILABLE:
    from MorseGraph.models import Encoder, Decoder, LatentDynamics
    from MorseGraph.training import Training
    from MorseGraph.dynamics import LearnedMapDynamics
    from MorseGraph.learning.latent_dynamics import MLPDynamics

# Set up output directory for figures
output_dir = os.path.dirname(os.path.abspath(__file__))
figures_dir = output_dir

def henon_map(x, a=1.4, b=0.3):
    """Standard Henon map, vectorized."""
    x_next = 1 - a * x[:, 0]**2 + x[:, 1]
    y_next = b * x[:, 0]
    return np.column_stack([x_next, y_next])

def generate_training_data(num_points=10000, domain_bounds=None):
    """Generate training data from Henon map trajectories."""
    if domain_bounds is None:
        domain_bounds = [[-1.5, -0.4], [1.5, 0.4]]
    
    # Generate random initial conditions
    x_min, y_min = domain_bounds[0]
    x_max, y_max = domain_bounds[1]
    
    x_t = np.random.uniform(x_min, x_max, (num_points, 2))
    
    # Apply Henon map to get next states
    x_t_plus_1 = henon_map(x_t)
    
    # Filter to keep only points that stay within domain
    valid_mask = (
        (x_t_plus_1[:, 0] >= x_min) & (x_t_plus_1[:, 0] <= x_max) &
        (x_t_plus_1[:, 1] >= y_min) & (x_t_plus_1[:, 1] <= y_max)
    )
    
    x_t = x_t[valid_mask]
    x_t_plus_1 = x_t_plus_1[valid_mask]
    
    return x_t, x_t_plus_1

def main():
    if not ML_AVAILABLE:
        print("Machine Learning dependencies not available. Please install PyTorch to run this example.")
        return
        
    print("MorseGraph Example 5: Learned Dynamics")
    print("=====================================")
    
    # 1. Generate Training Data
    print("\n1. Generating training data from Henon map...")
    
    domain_bounds = [[-1.5, -0.4], [1.5, 0.4]]
    x_t, x_t_plus_1 = generate_training_data(num_points=10000, domain_bounds=domain_bounds)
    
    print(f"Generated {len(x_t)} valid trajectory points")
    print(f"Data shapes: x_t {x_t.shape}, x_t+1 {x_t_plus_1.shape}")
    
    # Visualize training data
    plt.figure(figsize=(10, 6))
    plt.scatter(x_t[:, 0], x_t[:, 1], s=1, alpha=0.5, c='blue', label='Current States')
    plt.scatter(x_t_plus_1[:, 0], x_t_plus_1[:, 1], s=1, alpha=0.5, c='red', label='Next States')
    plt.title("Training Data: Henon Map Trajectories")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save training data plot
    data_plot_path = os.path.join(figures_dir, "5_learned_dynamics_data.png")
    plt.savefig(data_plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved training data plot to: {data_plot_path}")
    
    # 2. Prepare Data for Training
    print("\n2. Preparing data for neural network training...")
    
    # Split into train/validation
    split_idx = int(0.8 * len(x_t))
    x_train, x_val = x_t[:split_idx], x_t[split_idx:]
    y_train, y_val = x_t_plus_1[:split_idx], x_t_plus_1[split_idx:]
    
    # Convert to PyTorch tensors
    train_dataset = TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(x_val), torch.FloatTensor(y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"Training set: {len(x_train)} samples")
    print(f"Validation set: {len(x_val)} samples")
    
    # 3. Define and Train Models
    print("\n3. Training neural network models...")
    
    # Model architecture parameters
    input_dim = 2  # 2D state space
    latent_dim = 2  # 2D latent space (same as input for visualization)
    hidden_dim = 64
    
    # Create models
    encoder = Encoder(input_dim=input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim)
    decoder = Decoder(latent_dim=latent_dim, output_dim=input_dim, hidden_dim=hidden_dim)
    latent_dynamics = LatentDynamics(latent_dim=latent_dim, hidden_dim=hidden_dim)
    
    # Create training manager
    trainer = Training(encoder, decoder, latent_dynamics, learning_rate=0.001)
    
    # Train the models
    print("Training models for 50 epochs...")
    trainer.train(
        train_loader=train_loader,
        epochs=50,
        val_loader=val_loader,
        dynamics_weight=0.5  # Balance reconstruction and dynamics losses
    )
    
    print("Training completed!")
    
    # 4. Create Learned Dynamics for Morse Graph Analysis
    print("\n4. Setting up learned dynamics for Morse graph analysis...")
    
    # Create a learned dynamics wrapper using the trained models
    # We'll use the MLPDynamics wrapper with the trained latent dynamics
    
    # Extract trained latent dynamics as a compatible model
    class TrainedLatentDynamics:
        def __init__(self, trained_model):
            self.model = trained_model
            self.model.eval()
            
        def predict(self, Z):
            with torch.no_grad():
                Z_tensor = torch.FloatTensor(Z)
                Z_next = self.model(Z_tensor)
                return Z_next.numpy()
    
    trained_dynamics = TrainedLatentDynamics(latent_dynamics)
    learned_dynamics = LearnedMapDynamics(trained_dynamics, bloating=0.05)
    
    # 5. Compute Morse Graph in Latent Space
    print("\n5. Computing Morse graph using learned dynamics...")
    
    # Create grid in latent space (same domain since latent_dim = input_dim)
    domain = np.array(domain_bounds)
    divisions = np.array([24, 24])  # Reasonable resolution
    grid = UniformGrid(bounds=domain, divisions=divisions)
    
    # Create model with learned dynamics
    model = Model(grid, learned_dynamics)
    
    # Compute BoxMap and Morse decomposition
    box_map = model.compute_box_map()
    morse_graph = compute_morse_graph(box_map)
    
    print(f"Learned dynamics BoxMap: {len(box_map.nodes())} nodes, {len(box_map.edges())} edges")
    print(f"Learned dynamics Morse graph: {len(morse_graph.nodes())} non-trivial Morse sets")
    
    # 6. Visualize Results
    print("\n6. Creating visualizations...")
    
    # Plot learned dynamics Morse sets
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left: Original training data
    ax1.scatter(x_t[:, 0], x_t[:, 1], s=2, alpha=0.3, c='gray', label='Training Data')
    ax1.set_title("Original Henon Map Data")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Right: Learned dynamics Morse sets
    plot_morse_sets(grid, morse_graph, ax=ax2)
    ax2.scatter(x_t[::10, 0], x_t[::10, 1], s=1, alpha=0.3, c='gray', label='Training Data')
    ax2.set_title("Morse Sets from Learned Dynamics")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.legend()
    
    plt.tight_layout()
    
    # Save results plot
    results_plot_path = os.path.join(figures_dir, "5_learned_dynamics_results.png")
    plt.savefig(results_plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved results plot to: {results_plot_path}")
    
    # 7. Model Evaluation
    print("\n7. Evaluating learned model...")
    
    # Test reconstruction quality
    with torch.no_grad():
        sample_data = torch.FloatTensor(x_val[:100])
        encoded = encoder(sample_data)
        reconstructed = decoder(encoded)
        
        reconstruction_error = torch.mean((sample_data - reconstructed) ** 2).item()
        print(f"Mean reconstruction error: {reconstruction_error:.6f}")
    
    # Test dynamics prediction
    with torch.no_grad():
        sample_x = torch.FloatTensor(x_val[:100])
        sample_y_true = torch.FloatTensor(y_val[:100])
        
        encoded_x = encoder(sample_x)
        predicted_encoded_y = latent_dynamics(encoded_x)
        predicted_y = decoder(predicted_encoded_y)
        
        dynamics_error = torch.mean((sample_y_true - predicted_y) ** 2).item()
        print(f"Mean dynamics prediction error: {dynamics_error:.6f}")
    
    # 8. Analysis
    print("\n8. Analysis:")
    
    # Count Morse sets
    attractors = [node for node in morse_graph.nodes() if morse_graph.out_degree(node) == 0]
    sources = [node for node in morse_graph.nodes() if morse_graph.in_degree(node) == 0]
    
    print(f"Attractors discovered: {len(attractors)}")
    print(f"Sources discovered: {len(sources)}")
    print(f"Total Morse sets: {len(morse_graph.nodes())}")
    
    print(f"\nThe learned dynamics captured the essential")
    print(f"structure of the Henon map for Morse graph computation.")
    
    print("\nExample completed.")

if __name__ == "__main__":
    main()