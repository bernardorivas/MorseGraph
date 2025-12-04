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
    _run_cmgdb_compute,
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
    PLOT_PREFIX_3D_SETS, # Added this import
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

    @property
    def plot_output_dir(self):
        """Ensures the plotting output directory exists and returns its path."""
        path = os.path.join(self.run_dir, "figures")
        os.makedirs(path, exist_ok=True)
        return path

    def _get_or_compute_cached_data(self, cache_base_dir: str, hash_value: str, load_func, compute_func, save_func, force_recompute: bool, log_message_prefix: str):
        cache_dir = os.path.join(cache_base_dir, hash_value)
        os.makedirs(cache_dir, exist_ok=True)
        
        data = load_func(cache_dir)

        if data is None or force_recompute:
            self._log(f"  {log_message_prefix} not found in cache or recompute forced. Computing...")
            data = compute_func()
            save_func(cache_dir, data)
            self._log(f"  {log_message_prefix} computation complete and cached.")
        else:
            self._log(f"  {log_message_prefix} loaded from cache.")
        return data

    def _log(self, message: str):
        timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        with open(self.log_file, "a") as f:
            f.write(f"{timestamp} {message}\n")
        print(f"{timestamp} {message}")

    def _plot_stage_1_results(self, morse_graph_data):
        plot_output_dir = self.plot_output_dir
        
        plot_morse_graph_diagram(morse_graph_data['morse_graph'],
                            os.path.join(plot_output_dir, "01_morse_graph_3d.png"),
                            self.system_name)

        plot_morse_sets_3d_scatter(morse_graph_data['morse_graph'],
                                   self.config.domain_bounds,
                                   os.path.join(plot_output_dir, "03_morse_sets_3d_scatter.png"),
                                   title=f"{self.system_name} - 3D Morse Sets (Barycenters)",
                                   labels={'x': 'X', 'y': 'Y', 'z': 'Z'})

        plot_morse_sets_3d_projections(morse_graph_data['morse_graph'],
                                       morse_graph_data['morse_set_barycenters'],
                                       plot_output_dir,
                                       self.system_name,
                                       self.config.domain_bounds,
                                       prefix=PLOT_PREFIX_3D_SETS)

    def run_stage_1_3d(self, force_recompute=False):
        self._log("STAGE 1: Computing 3D Morse Graph (Ground Truth)")
        
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

        def compute_3d_morse_graph_data():
            dynamics_func = get_system_dynamics(self.config.system_type, self.config.dynamics_name)
            domain_bounds_np = np.array(self.config.domain_bounds)
            epsilon = 1e-6 if self.config.padding else 0.0
            
            box_map = BoxMapFunction(map_f=dynamics_func, epsilon=epsilon)

            morse_graph, morse_sets, morse_set_barycenters, _ = compute_morse_graph_3d(
                box_map,
                domain_bounds_np,
                self.config.subdiv_min,
                self.config.subdiv_max,
                self.config.subdiv_init,
                self.config.subdiv_limit
            )
            return {
                'morse_graph': morse_graph,
                'morse_sets': morse_sets,
                'morse_set_barycenters': morse_set_barycenters,
                'config': self.config.to_dict()
            }

        self.morse_graph_3d_data = self._get_or_compute_cached_data(
            self.cmgdb_3d_dir,
            cmgdb_3d_hash,
            load_morse_graph_data,
            compute_3d_morse_graph_data,
            save_morse_graph_data,
            force_recompute,
            "3D Morse graph"
        )
        
        # Visualization
        self._plot_stage_1_results(self.morse_graph_3d_data)

        self._log("STAGE 1: Completed.")

    def _plot_stage_2_results(self, trajectory_data):
        plot_output_dir = self.plot_output_dir
        Y_trajectories = trajectory_data['Y']

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

        try:
            plot_morse_sets_3d_projections_with_trajectories(
                self.morse_graph_3d_data['morse_graph'],
                self.morse_graph_3d_data['morse_set_barycenters'],
                trajectory_data=Y_trajectories,
                output_dir=plot_output_dir,
                system_name=self.system_name,
                domain_bounds=self.config.domain_bounds,
                prefix=PLOT_PREFIX_3D_SETS,
                n_trajectories=100,
                use_tail_only=True,
                tail_fraction=0.5
            )
            self._log("  Plotted 3D projections with trajectory overlay.")
        except Exception as e:
            self._log(f"  Warning: Could not plot projections with trajectories: {e}")

    def run_stage_2_trajectories(self, force_recompute=False):
        self._log("STAGE 2: Generating Trajectory Data")
        
        traj_data_hash = compute_trajectory_data_hash(
            self.morse_graph_3d_data['config'],
            self.config.n_trajectories,
            self.config.n_points,
            self.config.skip_initial,
            self.config.random_seed
        )

        def generate_trajectory_data_func():
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
            return {'X': X, 'Y': Y, 'config': self.config.to_dict()}

        self.trajectory_data = self._get_or_compute_cached_data(
            self.trajectory_dir,
            traj_data_hash,
            load_trajectory_data,
            generate_trajectory_data_func,
            save_trajectory_data,
            force_recompute,
            "Trajectory data"
        )
        
        # Visualization with trajectory overlays
        self._plot_stage_2_results(self.trajectory_data)

        self._log("STAGE 2: Completed.")

    def _plot_stage_3_results(self, training_history):
        plot_output_dir = self.plot_output_dir
        
        if training_history: # Only plot if training actually happened or loaded
            # Adapt if history has train/val keys (new format) or flat (old format)
            if 'train' in training_history and 'val' in training_history:
                plot_training_curves(training_history['train'], training_history['val'], os.path.join(plot_output_dir, "04_training_curves.png"))
            else:
                # Backward compatibility or if load_training_history returns old format
                plot_training_curves(training_history, training_history, os.path.join(plot_output_dir, "04_training_curves.png"))

    def run_stage_3_training(self, force_retrain=False):
        self._log("STAGE 3: Autoencoder Training")

        training_hash = compute_training_hash(
            self.trajectory_data['config'],
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

        def load_models_wrapper(cache_dir):
            encoder, decoder, latent_dynamics = load_models(cache_dir)
            history = load_training_history(cache_dir)
            if encoder and decoder and latent_dynamics and history:
                return {'encoder': encoder, 'decoder': decoder, 'latent_dynamics': latent_dynamics, 'history': history}
            return None

        def compute_models_func():
            X_full = self.trajectory_data['X']
            Y_full = self.trajectory_data['Y']
            split_idx = int(len(X_full) * 0.8)
            X_train = X_full[:split_idx]
            Y_train = Y_full[:split_idx]
            X_val = X_full[split_idx:]
            Y_val = Y_full[split_idx:]

            training_result = train_autoencoder_dynamics(
                X_train, Y_train,
                X_val, Y_val,
                config=self.config,
                verbose=True
            )
            
            encoder = training_result['encoder']
            decoder = training_result['decoder']
            latent_dynamics = training_result['latent_dynamics']
            history = {
                'train': training_result['train_losses'],
                'val': training_result['val_losses']
            }
            return {'encoder': encoder, 'decoder': decoder, 'latent_dynamics': latent_dynamics, 'history': history}

        def save_models_wrapper(cache_dir, data):
            save_models(cache_dir, data['encoder'], data['decoder'], data['latent_dynamics'], config=self.config.to_dict())
            save_training_history(cache_dir, data['history'])

        cached_training_data = self._get_or_compute_cached_data(
            self.training_dir,
            training_hash,
            load_models_wrapper,
            compute_models_func,
            save_models_wrapper,
            force_retrain,
            "Autoencoder models and training history"
        )
        
        self.trained_models = {
            'encoder': cached_training_data['encoder'],
            'decoder': cached_training_data['decoder'],
            'latent_dynamics': cached_training_data['latent_dynamics'],
            'config': self.config.to_dict()
        }
        self.analysis_results = {'training_history': cached_training_data['history']}

        # Also save/copy models to run directory for easy access
        run_models_dir = os.path.join(self.run_dir, "models")
        save_models(run_models_dir, self.trained_models['encoder'], self.trained_models['decoder'], self.trained_models['latent_dynamics'], config=self.config.to_dict())

        # Visualization
        self._plot_stage_3_results(cached_training_data['history'])

        self._log("STAGE 3: Completed.")


    def _plot_stage_4_results(self, X_train_np, encoder, decoder, latent_dynamics, morse_graph_3d_data, latent_data):
        plot_output_dir = self.plot_output_dir
        
        # Get device from model
        try:
            device = next(encoder.parameters()).device
        except Exception:
            device = torch.device("cpu")

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
        if morse_graph_3d_data and 'morse_set_barycenters' in morse_graph_3d_data:
            barycenters_3d = morse_graph_3d_data['morse_set_barycenters']

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
        self._plot_stage_4_results(
            X_train_np,
            encoder,
            decoder,
            latent_dynamics,
            self.morse_graph_3d_data,
            self.latent_data
        )

        self._log("STAGE 4: Completed.")

    def _plot_stage_5_results(self, morse_graph_2d_data, latent_data, method):
        plot_output_dir = self.plot_output_dir
        
        if morse_graph_2d_data['morse_graph'] is not None:
            plot_latent_space_2d(
                latent_data['Z_train_encoded'],
                latent_data['latent_bounds'],
                morse_graph=morse_graph_2d_data['morse_graph'],
                output_path=os.path.join(plot_output_dir, f"07_latent_morse_sets_{method}.png"),
                title=self.system_name,
                barycenters_latent=morse_graph_2d_data['morse_set_barycenters']
            )

    def run_stage_5_latent_morse(self, method: str = None, force_recompute=False):
        self._log(f"STAGE 5: Computing 2D Morse Graph (Latent Dynamics) using method: {method}")
        
        if method is None:
            method = self.config.latent_morse_graph_method
            self._log(f"  No method specified, using default from config: {method}")

        encoder = self.trained_models['encoder']
        latent_dynamics = self.trained_models['latent_dynamics']
        latent_bounds = self.latent_data['latent_bounds']
        
        cmgdb_2d_hash = compute_cmgdb_2d_hash(
            self.trained_models['config'],
            method,
            self.config.latent_subdiv_min,
            self.config.latent_subdiv_max,
            self.config.latent_subdiv_init,
            self.config.latent_subdiv_limit,
            self.config.latent_padding,
            self.config.original_grid_subdiv,
            latent_bounds
        )

        def compute_2d_morse_graph_data_func():
            latent_domain_bounds_np = np.array(latent_bounds)
            morse_graph_2d = None
            morse_sets_2d = None
            morse_set_barycenters_2d = None

            if method == 'data':
                original_grid = generate_3d_grid_for_encoding(
                    np.array(self.config.domain_bounds),
                    self.config.original_grid_subdiv,
                    self.config.input_dim
                )
                
                try:
                    device = next(encoder.parameters()).device
                except Exception:
                    device = torch.device("cpu")

                original_grid_tensor = torch.tensor(original_grid, dtype=torch.float32).to(device)
                Z_grid_encoded = encoder(original_grid_tensor).detach().cpu().numpy()
                G_Z_grid_encoded = latent_dynamics(encoder(original_grid_tensor)).detach().cpu().numpy()

                grid_res = int(2**(self.config.latent_subdiv_max / self.config.latent_dim))
                dims = [grid_res] * self.config.latent_dim
                data_grid = UniformGrid(latent_domain_bounds_np, dims)

                box_map_2d = BoxMapData(
                    Z_grid_encoded,
                    G_Z_grid_encoded,
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
            elif method in ['full', 'restricted', 'enclosure']: # 'enclosure' is handled as 'full' in _compute_method_learned
                 result = self._compute_method_learned(
                    method if method != 'enclosure' else 'full', # Use 'full' for enclosure method in _compute_method_learned
                    latent_bounds,
                    self.config.latent_subdiv_min,
                    self.config.latent_subdiv_max,
                    self.config.latent_subdiv_init,
                    self.config.latent_subdiv_limit,
                    self.config.latent_padding
                )
                 morse_graph_2d = result['morse_graph']
                 morse_set_barycenters_2d = result['barycenters']
                 morse_sets_2d = {}
                 for i in range(result['num_morse_sets']):
                     morse_sets_2d[i] = morse_graph_2d.morse_set_boxes(i)
            else:
                raise ValueError(f"Unknown 2D Morse graph computation method: {method}")

            return {
                'morse_graph': morse_graph_2d,
                'morse_sets': morse_sets_2d,
                'morse_set_barycenters': morse_set_barycenters_2d,
                'config': self.config.to_dict(),
                'method': method
            }

        self.morse_graph_2d_data = self._get_or_compute_cached_data(
            self.cmgdb_2d_dir,
            cmgdb_2d_hash,
            load_morse_graph_data,
            compute_2d_morse_graph_data_func,
            save_morse_graph_data,
            force_recompute,
            f"2D Morse graph ({method})"
        )

        # Visualization
        self._plot_stage_5_results(self.morse_graph_2d_data, self.latent_data, method)

        self._log("STAGE 5: Completed.")

    def _compute_restricted_allowed_indices(self, latent_bounds, subdiv_max):
        """
        Computes the allowed_indices for the 'restricted' method based on training data and dilation.
        """
        z_train = self.latent_data['Z_train_encoded']
        latent_dim = z_train.shape[1]

        # Create temporary grid at subdiv_max resolution
        dims = [2**subdiv_max] * latent_dim
        temp_grid = UniformGrid(np.array([latent_bounds[0], latent_bounds[1]]), dims)

        # Map training data to box indices at this resolution
        cell_size = (np.array(latent_bounds[1]) - np.array(latent_bounds[0])) / np.array(dims)
        indices_vec = np.floor((z_train - np.array(latent_bounds[0])) / cell_size).astype(int)
        indices_vec = np.clip(indices_vec, 0, np.array(dims) - 1)
        flat_indices = np.ravel_multi_index(indices_vec.T, dims)
        active_set = set(flat_indices)

        # Dilate by radius=1 (Moore/King neighborhood)
        active_array = np.array(list(active_set))
        dilated_array = temp_grid.dilate_indices(active_array, radius=1)
        allowed_indices = set(dilated_array)

        self._log(f"  Restricted domain: {len(active_set)} data boxes -> {len(allowed_indices)} allowed boxes")
        return allowed_indices

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
            allowed_indices = self._compute_restricted_allowed_indices(latent_bounds, subdiv_max)

        # Setup Dynamics
        # Use a small epsilon for padding if specified, else 0
        pad_val = 1e-6 if padding else 0.0
        
        dynamics = BoxMapLearnedLatent(
            latent_dynamics,
            device,
            padding=pad_val,
            allowed_indices=allowed_indices
        )
        
        # Use shared helper from core.py
        morse_graph, morse_sets, barycenters, map_graph = _run_cmgdb_compute(
            dynamics,
            [latent_bounds[0], latent_bounds[1]],
            subdiv_min,
            subdiv_max,
            subdiv_init,
            subdiv_limit,
            verbose=True
        )

        return {
            'morse_graph': morse_graph,
            'map_graph': map_graph,
            'num_morse_sets': morse_graph.num_vertices(),
            'barycenters': barycenters,
            'computation_time': 0.0, # Placeholder or measure if needed
            'from_cache': False
        }

    def _plot_comparisons(self):
        plot_output_dir = self.plot_output_dir

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

    def generate_comparisons(self):
        self._log("Generating comparison visualizations.")
        
        self._plot_comparisons()
            
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
