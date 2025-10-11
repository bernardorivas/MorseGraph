"""
This module has parameters for:
- 3D MorseGraph computation
- Learning experiments (autoencoder + latent dynamics)
- Output directory organization
"""

import os


class Config:
    """Configuration file for IvesModel MorseGraph Analysis"""

    # ========================================================================
    # Ives Model Parameters
    # ========================================================================
    R1 = 3.873          # Midge reproduction rate
    R2 = 11.746         # Algae growth rate
    C = 10**-6.435      # Constant input of algae and detritus
    D = 0.5517          # Detritus decay rate
    P = 0.06659         # Relative palatability of detritus
    Q = 0.9026 # 0.902           # Exponent in midge consumption # 
    LOG_OFFSET = 0.001  # Offset added before log transform to avoid log(0)

    # ========================================================================
    # Domain Bounds (log-scale)
    # ========================================================================
    # Working in log coordinates for midge, algae, and detritus abundances
    # LOWER_BOUNDS = [-15.0, -15.0, -15.0]
    # UPPER_BOUNDS = [7.0, 7.0, 7.0]
    LOWER_BOUNDS = [-1,-4,-1] # bounds for morse sets
    UPPER_BOUNDS = [2,1,1] # bounds for morse sets
    DOMAIN_BOUNDS = [LOWER_BOUNDS, UPPER_BOUNDS]  # Alternative format

    # ========================================================================
    # CMGDB Parameters (for 1_compute_morse_graph.py)
    # ========================================================================
    SUBDIV_MIN = 36
    SUBDIV_MAX = 50
    SUBDIV_INIT = 24
    SUBDIV_LIMIT = 1000000
    PADDING = True # True

    # ========================================================================
    # Data Generation (for Learning)
    # ========================================================================
    N_TRAJECTORIES = 5000
    N_POINTS = 20
    SKIP_INITIAL = 0
    RANDOM_SEED = 42 # None  # Use None for random

    # ========================================================================
    # Model Architecture
    # ========================================================================
    INPUT_DIM = 3
    LATENT_DIM = 2 
    HIDDEN_DIM = 32 
    NUM_LAYERS = 3 # MORALS uses 1
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
        """Get the base directory (ives_model/)."""
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    @staticmethod
    def get_experiments_dir():
        """Get the main experiments output directory."""
        return os.path.join(Config.get_base_dir(), "experiments")

    @staticmethod
    def get_morse_graph_dir():
        """Get pre-computed morse graph directory."""
        return os.path.join(Config.get_experiments_dir(), "morse_graph")

    @staticmethod
    def get_learning_dir():
        """Get the learning experiments directory."""
        return os.path.join(Config.get_experiments_dir(), "learning")

    @staticmethod
    def get_single_runs_dir():
        """Get the single runs directory."""
        return os.path.join(Config.get_learning_dir(), "single")

    @staticmethod
    def get_multiple_runs_dir():
        """Get the multiple runs directory."""
        return os.path.join(Config.get_learning_dir(), "multiple")

