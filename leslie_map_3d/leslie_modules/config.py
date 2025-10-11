"""
Unified Configuration for Leslie Map 3D Experiments

This module centralizes all configuration parameters for:
- Ground truth Morse graph computation
- Learning experiments (autoencoder + latent dynamics)
- Seed validation
- Output directory organization
"""

import os


class Config:
    """Unified configuration for all Leslie Map 3D experiments."""

    # ========================================================================
    # Leslie Map Parameters
    # ========================================================================
    THETA_1 = 28.9
    THETA_2 = 29.8
    THETA_3 = 22.0
    SURVIVAL_1 = 0.7
    SURVIVAL_2 = 0.7

    # ========================================================================
    # Domain Bounds
    # ========================================================================
    LOWER_BOUNDS = [-0.01, -0.01, -0.01]
    UPPER_BOUNDS = [90.0, 70.0, 70.0]
    DOMAIN_BOUNDS = [LOWER_BOUNDS, UPPER_BOUNDS]  # Alternative format

    # ========================================================================
    # Ground Truth CMGDB Parameters (for 1_compute_ground_truth.py)
    # ========================================================================
    SUBDIV_MIN = 30
    SUBDIV_MAX = 42
    SUBDIV_INIT = 0
    SUBDIV_LIMIT = 10000
    PADDING = True

    # ========================================================================
    # Data Generation (for Learning Experiments)
    # ========================================================================
    N_TRAJECTORIES = 5000
    N_POINTS = 20
    SKIP_INITIAL = 0
    RANDOM_SEED = None  # Use None for random, or set integer for reproducibility

    # ========================================================================
    # Model Architecture
    # ========================================================================
    INPUT_DIM = 3
    LATENT_DIM = 2
    HIDDEN_DIM = 32
    NUM_LAYERS = 3
    OUTPUT_ACTIVATION = None  # Options: None, "tanh", "sigmoid"

    # Per-network activation overrides (None means use OUTPUT_ACTIVATION)
    ENCODER_ACTIVATION = None # MORALS uses tanh
    DECODER_ACTIVATION = None # MORALS uses sigmoid
    LATENT_DYNAMICS_ACTIVATION = None # MORALS uses tanh

    # ========================================================================
    # Training Parameters (Learning)
    # ========================================================================
    NUM_EPOCHS = 1500
    BATCH_SIZE = 1024
    LEARNING_RATE = 0.001
    EARLY_STOPPING_PATIENCE = 50
    MIN_DELTA = 1e-5

    # Loss weights
    W_RECON = 1.0       # ||D(E(x)) - x||² (reconstruction)
    W_DYN_RECON = 1.0   # ||D(G(E(x))) - f(x)||² (dynamics reconstruction)
    W_DYN_CONS = 1.0    # ||G(E(x)) - E(f(x))||² (dynamics consistency)

    # ========================================================================
    # MorseGraph Parameters
    # ========================================================================

    # For learned latent system (2D)
    LATENT_SUBDIV_MIN = 20 # 22
    LATENT_SUBDIV_MAX = 28 # 28
    LATENT_SUBDIV_INIT = 0
    LATENT_SUBDIV_LIMIT = 10000
    LATENT_PADDING = True

    # Latent bounds
    LATENT_BOUNDS_PADDING = 1.01  # Bounding box padding around E(data)

    # ========================================================================
    # Output Directory
    # ========================================================================
    @staticmethod
    def get_base_dir():
        """Get the base directory (examples/leslie_map_3d/)."""
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    @staticmethod
    def get_experiments_dir():
        """Get the main experiments output directory."""
        return os.path.join(Config.get_base_dir(), "experiments")

    @staticmethod
    def get_ground_truth_dir():
        """Get the ground truth results directory."""
        return os.path.join(Config.get_experiments_dir(), "ground_truth")

    @staticmethod
    def get_learning_dir():
        """Get the learning experiments directory."""
        return os.path.join(Config.get_experiments_dir(), "learning")

    @staticmethod
    def get_single_runs_dir():
        """Get the single learning runs directory."""
        return os.path.join(Config.get_learning_dir(), "single")

    @staticmethod
    def get_sweeps_dir():
        """Get the parameter sweeps directory."""
        return os.path.join(Config.get_learning_dir(), "sweeps")

    @staticmethod
    def get_validation_dir():
        """Get the seed validation directory."""
        return os.path.join(Config.get_experiments_dir(), "validation")


