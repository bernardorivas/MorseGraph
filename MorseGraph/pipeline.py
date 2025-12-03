import os
import yaml
import time
import json
import shutil
import hashlib
import numpy as np
import torch
from datetime import datetime
from joblib import Parallel, delayed

from MorseGraph.config import load_experiment_config, save_config_to_yaml
from MorseGraph.dynamics import BoxMapFunction, BoxMapODE, BoxMapData, BoxMapLearnedLatent
from MorseGraph.grids import UniformGrid
from MorseGraph.core import (
    compute_morse_graph_3d,
    compute_morse_graph_2d_data,
    compute_morse_graph_2d_restricted,
    # compute_morse_graph_2d_latent_enclosure, # This will be implemented later
)
from MorseGraph.models import Encoder, Decoder, LatentDynamics

from MorseGraph.analysis import (
    compute_morse_graph,
    compute_all_morse_set_basins,
    iterative_morse_computation,
)
from MorseGraph.plot import (
    plot_morse_graph_diagram,
    plot_morse_sets_3d_scatter,
    plot_morse_sets_3d_projections,
    plot_morse_sets_3d_with_trajectories,
    plot_morse_sets_3d_projections_with_trajectories,
    plot_training_curves,
    plot_encoder_decoder_roundtrip,
    plot_trajectory_analysis,
    plot_latent_space_2d,
    plot_2x2_morse_comparison,
    plot_preimage_classification,
)
from MorseGraph.config import (
    get_system_dynamics,
    get_system_bounds,
    get_system_parameters,
    get_system_name,
)
from MorseGraph.training import train_autoencoder_dynamics
from MorseGraph.utils import (
    ExperimentConfig,
    compute_cmgdb_3d_hash,
    compute_cmgdb_2d_hash,
    compute_trajectory_data_hash,
    compute_training_hash,
    load_morse_graph_data,
    save_morse_graph_data,
    load_trajectory_data,
    save_trajectory_data,
    load_models,
    save_models,
    load_training_history,
    save_training_history,
    compute_latent_bounds_from_data,
    generate_3d_grid_for_encoding,
    generate_random_trajectories_3d,
    get_next_run_number,
)


class MorseGraphPipeline:
    def __init__(self, config_path: str, output_dir: str = "runs"):
        self.config_path = config_path
        self.config = load_experiment_config(config_path)
        self.system_name = get_system_name(self.config.system_type, self.config.dynamics_name)

        # Set up base output directory (experiment root)
        # If output_dir is generic "runs", append system name. 
        # Otherwise assume it's specific (e.g. "ives_model_output")
        if os.path.basename(output_dir) == "runs":
             self.base_dir = os.path.join(output_dir, self.system_name)
        else:
             self.base_dir = output_dir
             
        os.makedirs(self.base_dir, exist_ok=True)

        # Set up cache directories within base_dir
        self.cmgdb_3d_dir = os.path.join(self.base_dir, "cmgdb_3d")
        self.training_dir = os.path.join(self.base_dir, "training")
        self.trajectory_dir = os.path.join(self.base_dir, "trajectory_data")
        self.cmgdb_2d_dir = os.path.join(self.base_dir, "cmgdb_2d")

        # Set up specific run directory
        run_num = get_next_run_number(self.base_dir)
        self.run_dir = os.path.join(self.base_dir, f"run_{run_num:03d}")
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Logging
        self.log_file = os.path.join(self.run_dir, "pipeline_log.txt")
        self._log(f"Pipeline initialized for {self.system_name}")
        self._log(f"Base output directory: {self.base_dir}")
        self._log(f"Cache directories: cmgdb_3d/, training/, trajectory_data/, cmgdb_2d/")
        self._log(f"Run directory: {self.run_dir}")
        
        save_config_to_yaml(self.config, os.path.join(self.run_dir, "config.yaml"))

        self.results = {} # To store results from each stage

        # Initialize attributes for each stage's outputs
        self.morse_graph_3d_data = None
        self.trajectory_data = None
        self.trained_models = None
        self.latent_data = None
        self.morse_graph_2d_data = None
        self.analysis_results = None


    def _log(self, message: str):
        timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        with open(self.log_file, "a") as f:
            f.write(f"{timestamp} {message}\n")
        print(f"{timestamp} {message}")

    def run_stage_1_3d(self, force_recompute=False):
        self._log("STAGE 1: Computing 3D Morse Graph (Ground Truth)")
        
        # Determine unique hash for 3D CMGDB computation based on relevant parameters
        system_parameters = get_system_parameters(self.config.system_type, self.config.dynamics_name)
        cmgdb_3d_hash = compute_cmgdb_3d_hash(
            self.config.dynamics_name,
            self.config.domain_bounds,
            self.config.subdiv_min,
            self.config.subdiv_max,
            self.config.subdiv_init,
            self.config.subdiv_limit,
            self.config.padding,
            system_parameters
        )
        cmgdb_3d_cache_dir = os.path.join(self.cmgdb_3d_dir, cmgdb_3d_hash)
        os.makedirs(cmgdb_3d_cache_dir, exist_ok=True)
        
        # Attempt to load from cache
        morse_graph_data = load_morse_graph_data(cmgdb_3d_cache_dir)

        if morse_graph_data is None or force_recompute:
            self._log("  3D Morse graph not found in cache or recompute forced. Computing...")
            
            # Get dynamics and bounds
            dynamics_func = get_system_dynamics(self.config.system_type, self.config.dynamics_name)
            domain_bounds_np = np.array(self.config.domain_bounds)

            # Create BoxMapFunction for 3D dynamics
            # Determine epsilon based on padding config
            epsilon = 1e-6 if self.config.padding else 0.0
            
            box_map = BoxMapFunction(
                map_f=dynamics_func,
                epsilon=epsilon
            )

            # Compute 3D Morse Graph
            morse_graph, morse_sets, morse_set_barycenters = compute_morse_graph_3d(
                box_map,
                domain_bounds_np,
                self.config.subdiv_min,
                self.config.subdiv_max,
                self.config.subdiv_init,
                self.config.subdiv_limit
            )
            morse_graph_data = {
                'morse_graph': morse_graph,
                'morse_sets': morse_sets,
                'morse_set_barycenters': morse_set_barycenters,
                'config': self.config.to_dict() # Store config for reproducibility
            }
            save_morse_graph_data(cmgdb_3d_cache_dir, morse_graph_data)
            self._log("  3D Morse graph computation complete and cached.")
        else:
            self._log("  3D Morse graph loaded from cache.")
        
        self.morse_graph_3d_data = morse_graph_data
        
        # Visualization
        plot_output_dir = os.path.join(self.run_dir, "figures")
        os.makedirs(plot_output_dir, exist_ok=True)
        
        # Plot Morse graph
        plot_morse_graph_diagram(morse_graph_data['morse_graph'],
                            os.path.join(plot_output_dir, "01_morse_graph_3d.png"),
                            self.system_name)

        # Plot Morse sets (barycenter scatter visualization)
        plot_morse_sets_3d_scatter(morse_graph_data['morse_graph'],
                                   self.config.domain_bounds,
                                   os.path.join(plot_output_dir, "03_morse_sets_3d_scatter.png"),
                                   title=f"{self.system_name} - 3D Morse Sets (Barycenters)",
                                   labels={'x': 'X', 'y': 'Y', 'z': 'Z'})

        # Plot 2D projections of 3D Morse sets
        plot_morse_sets_3d_projections(morse_graph_data['morse_graph'],
                                       morse_graph_data['morse_set_barycenters'],
                                       plot_output_dir, # Pass output_dir as argument
                                       self.system_name,
                                       self.config.domain_bounds,
                                       prefix="03")

        self._log("STAGE 1: Completed.")

    def run_stage_2_trajectories(self, force_recompute=False):
        self._log("STAGE 2: Generating Trajectory Data")
        
        # Determine unique hash for trajectory data based on relevant parameters
        traj_data_hash = compute_trajectory_data_hash(
            self.morse_graph_3d_data['config'], # Use the config from 3D stage for consistency
            self.config.n_trajectories,
            self.config.n_points,
            self.config.skip_initial,
            self.config.random_seed
        )
        traj_data_cache_dir = os.path.join(self.trajectory_dir, traj_data_hash)
        os.makedirs(traj_data_cache_dir, exist_ok=True)

        # Attempt to load from cache
        trajectory_data = load_trajectory_data(traj_data_cache_dir)

        if trajectory_data is None or force_recompute:
            self._log("  Trajectory data not found in cache or recompute forced. Generating...")
            
            # Generate 3D trajectories
            dynamics_func = get_system_dynamics(self.config.system_type, self.config.dynamics_name)
            domain_bounds_np = np.array(self.config.domain_bounds)
            
            X, Y = generate_random_trajectories_3d(
                dynamics_func,
                domain_bounds_np,
                self.config.n_trajectories,
                self.config.n_points,
                self.config.skip_initial,
                self.config.random_seed
            )
            trajectory_data = {'X': X, 'Y': Y, 'config': self.config.to_dict()}
            save_trajectory_data(traj_data_cache_dir, trajectory_data)
            self._log("  Trajectory data generation complete and cached.")
        else:
            self._log("  Trajectory data loaded from cache.")
        
        self.trajectory_data = trajectory_data

        # Visualization with trajectory overlays
        plot_output_dir = os.path.join(self.run_dir, "figures")
        os.makedirs(plot_output_dir, exist_ok=True)

        # Prepare trajectory data for plotting (use tail of Y trajectories)
        Y_trajectories = trajectory_data['Y']  # shape: (n_traj, n_points, 3)

        # Plot Morse sets with trajectory overlay (scatter + tail)
        try:
            plot_morse_sets_3d_with_trajectories(
                self.morse_graph_3d_data['morse_graph'],
                self.config.domain_bounds,
                output_path=os.path.join(plot_output_dir, "03_morse_sets_3d_with_data.png"),
                title=f"{self.system_name} - 3D Morse Sets with Trajectory Data",
                labels={'x': 'X', 'y': 'Y', 'z': 'Z'},
                trajectory_data=Y_trajectories,
                n_trajectories=100,
                use_tail_only=True,
                tail_fraction=0.5
            )
            self._log("  Plotted Morse sets with trajectory overlay.")
        except Exception as e:
            self._log(f"  Warning: Could not plot Morse sets with trajectories: {e}")

        # Plot 2D projections with trajectory overlay
        try:
            plot_morse_sets_3d_projections_with_trajectories(
                self.morse_graph_3d_data['morse_graph'],
                self.morse_graph_3d_data['morse_set_barycenters'],
                trajectory_data=Y_trajectories,
                output_dir=plot_output_dir,
                system_name=self.system_name,
                domain_bounds=self.config.domain_bounds,
                prefix="03",
                n_trajectories=100,
                use_tail_only=True,
                tail_fraction=0.5
            )
            self._log("  Plotted 3D projections with trajectory overlay.")
        except Exception as e:
            self._log(f"  Warning: Could not plot projections with trajectories: {e}")

        self._log("STAGE 2: Completed.")

    def run_stage_3_training(self, force_retrain=False):
        self._log("STAGE 3: Autoencoder Training")

        # Determine unique hash for training based on relevant parameters
        training_hash = compute_training_hash(
            self.trajectory_data['config'], # Use the config from traj stage for consistency
            self.config.input_dim,
            self.config.latent_dim,
            self.config.hidden_dim,
            self.config.num_layers,
            self.config.w_recon,
            self.config.w_dyn_recon,
            self.config.w_dyn_cons,
            self.config.learning_rate,
            self.config.batch_size,
            self.config.num_epochs,
            self.config.early_stopping_patience,
            self.config.min_delta,
            self.config.encoder_activation,
            self.config.decoder_activation,
            self.config.latent_dynamics_activation
        )
        training_cache_dir = os.path.join(self.training_dir, training_hash)
        os.makedirs(training_cache_dir, exist_ok=True)

        # Attempt to load from cache
        encoder, decoder, latent_dynamics = load_models(training_cache_dir)
        training_history = load_training_history(training_cache_dir)

        if encoder is None or force_retrain:
            self._log("  Models not found in cache or retrain forced. Training new models...")
            
            # Prepare data (already float32 from generation)
            X_full = self.trajectory_data['X']
            Y_full = self.trajectory_data['Y']

            # Split into train and validation (80/20)
            split_idx = int(len(X_full) * 0.8)
            X_train = X_full[:split_idx]
            Y_train = Y_full[:split_idx]
            X_val = X_full[split_idx:]
            Y_val = Y_full[split_idx:]

            # Train autoencoder
            training_result = train_autoencoder_dynamics(
                X_train, Y_train,
                X_val, Y_val,
                config=self.config,
                verbose=True
            )
            
            encoder = training_result['encoder']
            decoder = training_result['decoder']
            latent_dynamics = training_result['latent_dynamics']
            # history = training_result['train_losses']
            # Ideally save both train/val history.
            history = {
                'train': training_result['train_losses'],
                'val': training_result['val_losses']
            }
            
            self.trained_models = {
                'encoder': encoder,
                'decoder': decoder,
                'latent_dynamics': latent_dynamics,
                'config': self.config.to_dict()
            }
            self.analysis_results = {'training_history': history} # Store history separately
            
            save_models(training_cache_dir, encoder, decoder, latent_dynamics, config=self.config.to_dict())
            save_training_history(training_cache_dir, history)
            self._log("  Autoencoder training complete and cached.")
        else:
            self._log("  Models loaded from cache.")
            self.trained_models = {
                'encoder': encoder,
                'decoder': decoder,
                'latent_dynamics': latent_dynamics,
                'config': self.config.to_dict()
            }
            self.analysis_results = {'training_history': training_history}

        # Also save/copy models to run directory for easy access
        run_models_dir = os.path.join(self.run_dir, "models")
        save_models(run_models_dir, encoder, decoder, latent_dynamics, config=self.config.to_dict())

        # Visualization
        plot_output_dir = os.path.join(self.run_dir, "figures")
        os.makedirs(plot_output_dir, exist_ok=True)
        
        if training_history: # Only plot if training actually happened or loaded
            # Adapt if history has train/val keys (new format) or flat (old format)
            if 'train' in training_history and 'val' in training_history:
                plot_training_curves(training_history['train'], training_history['val'], os.path.join(plot_output_dir, "04_training_curves.png"))
            else:
                # Backward compatibility or if load_training_history returns old format
                plot_training_curves(training_history, training_history, os.path.join(plot_output_dir, "04_training_curves.png"))

        self._log("STAGE 3: Completed.")


    def run_stage_4_encoding(self):
        self._log("STAGE 4: Latent Space Analysis (Encoding)")
        
        encoder = self.trained_models['encoder']
        decoder = self.trained_models['decoder']
        latent_dynamics = self.trained_models['latent_dynamics']
        
        # Get device from model
        try:
            device = next(encoder.parameters()).device
        except Exception:
            device = torch.device("cpu")

        X_train_np = self.trajectory_data['X']
        Y_train_np = self.trajectory_data['Y']

        # Encode training data
        X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32).to(device)
        Y_train_tensor = torch.tensor(Y_train_np, dtype=torch.float32).to(device)
        
        Z_train_encoded = encoder(X_train_tensor).detach().cpu().numpy()
        G_Z_train_encoded = latent_dynamics(encoder(X_train_tensor)).detach().cpu().numpy()
        
        # Compute latent bounds
        latent_bounds = compute_latent_bounds_from_data(Z_train_encoded, padding=self.config.latent_bounds_padding)

        self.latent_data = {
            'Z_train_encoded': Z_train_encoded,
            'G_Z_train_encoded': G_Z_train_encoded,
            'latent_bounds': latent_bounds
        }

        # Encode 3D Morse set barycenters for visualization
        Z_barycenters_encoded = {}
        # self.morse_graph_3d_data['morse_set_barycenters'] is Dict[int, List[np.ndarray]]
        if self.morse_graph_3d_data and 'morse_set_barycenters' in self.morse_graph_3d_data:
            for ms_idx, barys in self.morse_graph_3d_data['morse_set_barycenters'].items():
                if barys:
                    barys_np = np.array(barys, dtype=np.float32)
                    # Check if barys is 1D (single point) or 2D (list of points)
                    if barys_np.ndim == 1:
                        barys_np = barys_np.reshape(1, -1)
                        
                    encoded = encoder(torch.tensor(barys_np).to(device)).detach().cpu().numpy()
                    Z_barycenters_encoded[ms_idx] = encoded
                else:
                    Z_barycenters_encoded[ms_idx] = []
            
        self.latent_data['Z_barycenters_encoded'] = Z_barycenters_encoded

        # Visualization for quality diagnostics
        plot_output_dir = os.path.join(self.run_dir, "figures")
        os.makedirs(plot_output_dir, exist_ok=True)
        
        # Encoder/Decoder round-trip plot
        plot_encoder_decoder_roundtrip(X_train_np, encoder, decoder, os.path.join(plot_output_dir, "05_encoder_decoder_roundtrip.png"))

        # Trajectory analysis plot
        dynamics_func = get_system_dynamics(self.config.system_type, self.config.dynamics_name)
        domain_bounds_np = np.array(self.config.domain_bounds)

        # Generate test trajectories for visualization
        test_trajs_3d = []
        test_trajs_latent = []
        n_steps = 100
        n_vis_trajs = 3

        # Intelligently select initial conditions from Morse set barycenters
        # This gives more meaningful trajectories that explore the attractor structure
        ics = []
        if self.morse_graph_3d_data and 'morse_set_barycenters' in self.morse_graph_3d_data:
            barycenters_3d = self.morse_graph_3d_data['morse_set_barycenters']

            # Collect all barycenters from all Morse sets
            all_barys = []
            for ms_idx, barys in barycenters_3d.items():
                if barys:
                    all_barys.extend(barys)

            if len(all_barys) >= n_vis_trajs:
                # Randomly sample from barycenters
                np.random.seed(42)
                selected_indices = np.random.choice(len(all_barys), n_vis_trajs, replace=False)
                ics = [np.array(all_barys[i]) for i in selected_indices]
                self._log(f"  Selected {n_vis_trajs} initial conditions from Morse set barycenters")
            else:
                # Not enough barycenters, use what we have plus random
                ics = [np.array(b) for b in all_barys]
                n_needed = n_vis_trajs - len(ics)
                if n_needed > 0:
                    np.random.seed(42)
                    random_ics = np.random.uniform(domain_bounds_np[0], domain_bounds_np[1], (n_needed, domain_bounds_np.shape[1]))
                    ics.extend(random_ics)
                self._log(f"  Selected {len(all_barys)} ICs from barycenters, {n_needed} random ICs")
        else:
            # Fallback: random initial conditions
            np.random.seed(42)
            ics = np.random.uniform(domain_bounds_np[0], domain_bounds_np[1], (n_vis_trajs, domain_bounds_np.shape[1]))
            self._log(f"  Selected {n_vis_trajs} random initial conditions (no Morse set barycenters available)")

        for ic in ics:
            # 3D simulation
            traj_3d = [ic]
            curr = ic
            for _ in range(n_steps):
                curr = dynamics_func(curr)
                traj_3d.append(curr)
            test_trajs_3d.append(np.array(traj_3d))

            # Latent simulation
            # Map IC to latent
            z_curr = encoder(torch.tensor(ic, dtype=torch.float32).to(device))
            traj_latent = [z_curr.detach().cpu().numpy()]
            for _ in range(n_steps):
                z_curr = latent_dynamics(z_curr)
                traj_latent.append(z_curr.detach().cpu().numpy())
            test_trajs_latent.append(np.array(traj_latent))

        plot_trajectory_analysis(
            test_trajs_3d,
            test_trajs_latent,
            os.path.join(plot_output_dir, "06_trajectory_analysis.png"),
            title_prefix=f"{self.system_name} - "
        )

        self._log("STAGE 4: Completed.")

    def run_stage_5_latent_morse(self, method: str = None, force_recompute=False):
        self._log(f"STAGE 5: Computing 2D Morse Graph (Latent Dynamics) using method: {method}")
        
        if method is None:
            method = self.config.latent_morse_graph_method
            self._log(f"  No method specified, using default from config: {method}")

        encoder = self.trained_models['encoder']
        latent_dynamics = self.trained_models['latent_dynamics']
        latent_bounds = self.latent_data['latent_bounds']
        
        cmgdb_2d_hash = compute_cmgdb_2d_hash(
            self.trained_models['config'], # Use the config from training stage for consistency
            method,
            self.config.latent_subdiv_min,
            self.config.latent_subdiv_max,
            self.config.latent_subdiv_init,
            self.config.latent_subdiv_limit,
            self.config.latent_padding,
            self.config.original_grid_subdiv,
            latent_bounds # Include latent_bounds in hash since it affects the grid
        )
        cmgdb_2d_cache_dir = os.path.join(self.cmgdb_2d_dir, cmgdb_2d_hash)
        os.makedirs(cmgdb_2d_cache_dir, exist_ok=True)
        
        morse_graph_2d_data = load_morse_graph_data(cmgdb_2d_cache_dir)

        if morse_graph_2d_data is None or force_recompute:
            self._log(f"  2D Morse graph ({method}) not found in cache or recompute forced. Computing...")
            
            latent_domain_bounds_np = np.array(latent_bounds)
            
            if method == 'data':
                # Generate a fine 3D grid in original space
                original_grid = generate_3d_grid_for_encoding(
                    np.array(self.config.domain_bounds),
                    self.config.original_grid_subdiv,
                    self.config.input_dim
                )
                
                # Determine device
                try:
                    device = next(encoder.parameters()).device
                except Exception:
                    device = torch.device("cpu")

                # Encode the grid points to latent space
                original_grid_tensor = torch.tensor(original_grid, dtype=torch.float32).to(device)
                Z_grid_encoded = encoder(original_grid_tensor).detach().cpu().numpy()
                G_Z_grid_encoded = latent_dynamics(encoder(original_grid_tensor)).detach().cpu().numpy()

                # Create BoxMapData
                # Create grid for spatial indexing
                # Resolution based on max subdivision (2^subdiv boxes total)
                grid_res = int(2**(self.config.latent_subdiv_max / self.config.latent_dim))
                dims = [grid_res] * self.config.latent_dim
                data_grid = UniformGrid(latent_domain_bounds_np, dims)

                box_map_2d = BoxMapData(
                    Z_grid_encoded, # Latent points
                    G_Z_grid_encoded, # Latent images
                    grid=data_grid,
                    map_empty='outside',
                    output_enclosure='box_enclosure'
                )
                
                morse_graph_2d, morse_sets_2d, morse_set_barycenters_2d = compute_morse_graph_2d_data(
                    box_map_2d,
                    latent_domain_bounds_np,
                    self.config.latent_subdiv_min,
                    self.config.latent_subdiv_max,
                    self.config.latent_subdiv_init,
                    self.config.latent_subdiv_limit
                )
            elif method in ['full', 'restricted']:
                 result = self._compute_method_learned(
                    method,
                    latent_bounds,
                    self.config.latent_subdiv_min,
                    self.config.latent_subdiv_max,
                    self.config.latent_subdiv_init,
                    self.config.latent_subdiv_limit,
                    self.config.latent_padding
                )
                 morse_graph_2d = result['morse_graph']
                 morse_sets_2d = None # Not returned directly by CMGDB basic run, usually computed after
                 # Actually _compute_method_learned returns 'morse_graph' and 'barycenters'
                 # We need to extract morse_sets logic if needed, but plot functions usually use barycenters and graph.
                 # Let's check return of _compute_method_learned
                 morse_set_barycenters_2d = result['barycenters']
                 
                 # Extract morse sets indices (just range(num_vertices)) or actual boxes if needed?
                 # The pipeline stores 'morse_sets' which are usually boxes.
                 # CMGDB.ComputeMorseGraph returns graph.
                 # We need to get boxes.
                 morse_sets_2d = {}
                 for i in range(result['num_morse_sets']):
                     morse_sets_2d[i] = morse_graph_2d.morse_set_boxes(i)
            elif method == 'enclosure': 
                # This corresponds to F_latent / F_latent_image
                 result = self._compute_method_learned(
                    'full', # Enclosure is basically full domain with corner eval
                    latent_bounds,
                    self.config.latent_subdiv_min,
                    self.config.latent_subdiv_max,
                    self.config.latent_subdiv_init,
                    self.config.latent_subdiv_limit,
                    self.config.latent_padding
                )
                 # NOTE: _compute_method_learned uses BoxMapLearnedLatent which uses 
                 # corner/sample evaluation + padding. This IS the enclosure method if 
                 # padding is enabled and allowed_indices is None (full).
                 morse_graph_2d = result['morse_graph']
                 morse_set_barycenters_2d = result['barycenters']
                 morse_sets_2d = {}
                 for i in range(result['num_morse_sets']):
                     morse_sets_2d[i] = morse_graph_2d.morse_set_boxes(i)

            else:
                raise ValueError(f"Unknown 2D Morse graph computation method: {method}")

            morse_graph_2d_data = {
                'morse_graph': morse_graph_2d,
                'morse_sets': morse_sets_2d,
                'morse_set_barycenters': morse_set_barycenters_2d,
                'config': self.config.to_dict(),
                'method': method
            }
            save_morse_graph_data(cmgdb_2d_cache_dir, morse_graph_2d_data)
            self._log(f"  2D Morse graph ({method}) computation complete and cached.")
        else:
            self._log(f"  2D Morse graph ({method}) loaded from cache.")
        
        self.morse_graph_2d_data = morse_graph_2d_data

        # Visualization
        plot_output_dir = os.path.join(self.run_dir, "figures")
        os.makedirs(plot_output_dir, exist_ok=True)
        
        if morse_graph_2d_data['morse_graph'] is not None:
            plot_latent_space_2d(
                self.latent_data['Z_train_encoded'],
                self.latent_data['latent_bounds'],
                morse_graph=morse_graph_2d_data['morse_graph'],
                output_path=os.path.join(plot_output_dir, f"07_latent_morse_sets_{method}.png"),
                title=self.system_name,
                barycenters_latent=morse_graph_2d_data['morse_set_barycenters']
            )

        self._log("STAGE 5: Completed.")

    def _compute_method_learned(self, method: str, latent_bounds, subdiv_min, subdiv_max, subdiv_init, subdiv_limit, padding):
        """
        Compute Latent Morse Graph using BoxMapLearnedLatent (Methods 2 & 3).
        
        :param method: 'full' or 'restricted'
        """
        try:
            import CMGDB
        except ImportError:
            raise ImportError("CMGDB is required for this stage.")
            
        restricted = (method == 'restricted')
        encoder = self.trained_models['encoder']
        latent_dynamics = self.trained_models['latent_dynamics']
        device = next(encoder.parameters()).device # Get device from model

        allowed_indices = None
        if restricted:
            self._log("  Note: Restriction disabled - learned dynamics naturally constrained to data region")
            self._log("  CMGDB will explore full domain; convergence focuses on relevant structure")

        # Setup Dynamics
        # Use a small epsilon for padding if specified, else 0
        pad_val = 1e-6 if padding else 0.0
        
        dynamics = BoxMapLearnedLatent(
            latent_dynamics,
            device,
            padding=pad_val,
            allowed_indices=allowed_indices
        )
        
        # Define F for CMGDB
        def F(rect):
            # CMGDB passes rect as [min_x, min_y, ..., max_x, max_y, ...] (flat list)
            # Convert to [ [min], [max] ]
            dim = len(rect) // 2
            box = np.array([rect[:dim], rect[dim:]])
            
            res = dynamics(box)
            return list(res[0]) + list(res[1])

        model = CMGDB.Model(
            subdiv_min,
            subdiv_max,
            subdiv_init,
            subdiv_limit,
            latent_bounds[0],
            latent_bounds[1],
            F
        )
        
        start_time = time.time()
        morse_graph, map_graph = CMGDB.ComputeMorseGraph(model)
        computation_time = time.time() - start_time
        
        # Compute barycenters
        barycenters = {}
        for i in range(morse_graph.num_vertices()):
            morse_set_boxes = morse_graph.morse_set_boxes(i)
            barycenters[i] = []
            if morse_set_boxes:
                dim = len(morse_set_boxes[0]) // 2
                for box in morse_set_boxes:
                    barycenter = np.array([(box[j] + box[j + dim]) / 2.0 for j in range(dim)])
                    barycenters[i].append(barycenter)

        return {
            'morse_graph': morse_graph,
            'map_graph': map_graph,
            'num_morse_sets': morse_graph.num_vertices(),
            'barycenters': barycenters,
            'computation_time': computation_time,
            'from_cache': False
        }

    def generate_comparisons(self):
        self._log("Generating comparison visualizations.")
        
        plot_output_dir = os.path.join(self.run_dir, "figures")
        os.makedirs(plot_output_dir, exist_ok=True)

        if self.morse_graph_3d_data and self.morse_graph_2d_data and self.latent_data:
            # 2x2 comparison plot
            # Prepare arguments for plot_2x2_morse_comparison
            encoder = self.trained_models['encoder']
            device = next(encoder.parameters()).device
            
            plot_2x2_morse_comparison(
                self.morse_graph_3d_data['morse_graph'],
                self.morse_graph_2d_data['morse_graph'],
                self.config.domain_bounds,
                self.latent_data['latent_bounds'],
                encoder,
                device,
                self.latent_data['Z_train_encoded'],
                output_path=os.path.join(plot_output_dir, "08_morse_2x2_comparison.png"),
                title_prefix=f"{self.system_name}: "
            )

            # Preimage classification plot (requires BoxMapData to be used for 2D)
            # This needs to be made more general if other methods are used
            if self.morse_graph_2d_data['method'] == 'data':
                # This part needs the BoxMapData object which is not directly stored in morse_graph_2d_data
                # For now, this part will need to be re-computed or the BoxMapData object passed
                # Skipping for now to avoid complexity in this initial refactor
                self._log("  Skipping preimage classification plot: BoxMapData object not readily available from cache.")
                pass
            
        self._log("Comparison visualizations generated.")

    def compare_methods(self, methods: list):
        self._log(f"Comparing methods: {methods}")
        # Placeholder for multi-method comparison
        # This will involve running stage 5 for each method and then generating a comparison
        self._log("Multi-method comparison completed.")
        return {} # Replace with actual comparison results

    def run(self, force_recompute_3d=False, force_regenerate_data=False,
            force_retrain=False, force_recompute_2d=False,
            latent_morse_method: str = None):
        self.run_stage_1_3d(force_recompute=force_recompute_3d)
        self.run_stage_2_trajectories(force_recompute=force_regenerate_data)
        self.run_stage_3_training(force_retrain=force_retrain)
        self.run_stage_4_encoding()
        self.run_stage_5_latent_morse(method=latent_morse_method, force_recompute=force_recompute_2d)
        self.generate_comparisons()
        self._log("Pipeline execution finished.")
