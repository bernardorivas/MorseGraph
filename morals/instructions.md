# MORALS: Morse Graph-aided discovery of Regions of Attraction in a learned Latent Space - Complete Reference

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Python API Reference](#python-api-reference)
4. [Core Components](#core-components)
5. [Systems](#systems)
6. [Training Pipeline](#training-pipeline)
7. [Morse Graph Computation](#morse-graph-computation)
8. [File Structure](#file-structure)
9. [Usage Patterns](#usage-patterns)

---

## Overview

**MORALS** (Morse Graph-aided discovery of Regions of Attraction in a learned Latent Space) is a Python package that combines autoencoding neural networks with Morse Graphs to analyze high-dimensional robot controllers and dynamical systems. It projects dynamics into a learned latent space and constructs Morse Graphs representing bistability (desired vs. undesired behavior).

### Key Concept

1. **Latent Space Learning**: Uses autoencoders to learn a low-dimensional representation of high-dimensional system states
2. **Latent Dynamics Learning**: Learns the dynamics in the latent space using neural networks
3. **Morse Graph Construction**: Constructs Morse Graphs in the latent space to identify regions of attraction
4. **Bistability Detection**: Identifies when controllers result in desired vs. undesired behaviors

### Mathematical Foundation

- **Autoencoder**: Encoder $E: X \to Z$ and Decoder $D: Z \to X$ where $X$ is high-dimensional state space and $Z$ is low-dimensional latent space
- **Latent Dynamics**: Function $f: Z \to Z$ learned from trajectory data
- **Morse Graph**: Represents the global dynamics structure in latent space, identifying attractors and their basins
- **Regions of Attraction (RoA)**: Basins of attraction for desired behaviors

### Key Capabilities

- Learn low-dimensional representations of high-dimensional systems
- Learn dynamics in latent space
- Compute Morse Graphs for latent dynamics
- Identify regions of attraction
- Support for various robot systems (pendulum, cartpole, humanoid, etc.)
- Contrastive learning for separating success/failure behaviors

---

## Architecture

MORALS consists of several main components:

### 1. **Neural Network Models** (`models.py`)
- Encoder: Maps high-dimensional states to latent space
- Decoder: Maps latent states back to high-dimensional space
- LatentDynamics: Learns dynamics in latent space

### 2. **Training System** (`training.py`)
- Multi-stage training with different loss weights
- Autoencoder reconstruction loss
- Dynamics prediction loss
- Contrastive loss for success/failure separation

### 3. **Data Utilities** (`data_utils.py`)
- DynamicsDataset: Trajectory pairs for dynamics learning
- LabelsDataset: Success/failure labels for contrastive learning
- TrajectoryDataset: Full trajectories for visualization

### 4. **Dynamics Utilities** (`dynamics_utils.py`)
- DynamicsUtils: Wrapper for encoder/decoder/dynamics models
- EnsembleDynamics: Ensemble of multiple dynamics models

### 5. **Grid Utilities** (`grid.py`)
- Grid: Uniform grid for latent space discretization
- Coordinate transformations for CMGDB compatibility

### 6. **Morse Graph Utilities** (`mg_utils.py`)
- MorseGraphOutputProcessor: Processes CMGDB output files

### 7. **Systems** (`systems/`)
- BaseSystem: Abstract base class for dynamical systems
- Various system implementations (pendulum, cartpole, etc.)

---

## Python API Reference

### Neural Network Models

#### `MORALS.models.Encoder`

**Purpose**: Encodes high-dimensional states to latent space.

**Constructor**:
```python
Encoder(config)
```

**Parameters**:
- `config` (dict): Configuration dictionary with:
  - `high_dims` (int): Input dimension
  - `low_dims` (int): Latent dimension
  - `num_layers` (int): Number of hidden layers (default: 2)
  - `hidden_shape` (int): Hidden layer size (default: 32)

**Architecture**:
- Input: `high_dims` → Hidden layers (ReLU) → Output: `low_dims` (Tanh)

**Methods**:
- `forward(x)`: Forward pass through encoder

**Example**:
```python
from MORALS.models import Encoder
config = {'high_dims': 4, 'low_dims': 2, 'num_layers': 2, 'hidden_shape': 32}
encoder = Encoder(config)
z = encoder(x)  # x: [batch, 4], z: [batch, 2]
```

#### `MORALS.models.Decoder`

**Purpose**: Decodes latent states back to high-dimensional space.

**Constructor**:
```python
Decoder(config)
```

**Parameters**: Same as Encoder

**Architecture**:
- Input: `low_dims` → Hidden layers (ReLU) → Output: `high_dims` (Sigmoid)

**Methods**:
- `forward(z)`: Forward pass through decoder

**Example**:
```python
from MORALS.models import Decoder
decoder = Decoder(config)
x_recon = decoder(z)  # z: [batch, 2], x_recon: [batch, 4]
```

#### `MORALS.models.LatentDynamics`

**Purpose**: Learns dynamics in latent space.

**Constructor**:
```python
LatentDynamics(config)
```

**Parameters**:
- `config` (dict): Configuration dictionary with:
  - `low_dims` (int): Latent dimension
  - `num_layers` (int): Number of hidden layers (default: 2)
  - `hidden_shape` (int): Hidden layer size (default: 32)

**Architecture**:
- Input: `low_dims` → Hidden layers (ReLU) → Output: `low_dims` (Tanh)

**Methods**:
- `forward(z)`: Predicts next latent state

**Example**:
```python
from MORALS.models import LatentDynamics
dynamics = LatentDynamics(config)
z_next = dynamics(z)  # z: [batch, 2], z_next: [batch, 2]
```

### Training System

#### `MORALS.training.Training`

**Purpose**: Manages training of encoder, decoder, and dynamics models.

**Constructor**:
```python
Training(config, loaders, verbose)
```

**Parameters**:
- `config` (dict): Configuration dictionary
- `loaders` (dict): Data loaders with keys:
  - `'train_dynamics'`: Training dynamics DataLoader
  - `'test_dynamics'`: Test dynamics DataLoader
  - `'train_labels'`: Training labels DataLoader
  - `'test_labels'`: Test labels DataLoader
- `verbose` (bool): Print training progress

**Methods**:
- `train(epochs, patience, weight)`: Train models
  - `epochs` (int): Maximum number of epochs
  - `patience` (int): Early stopping patience
  - `weight` (list): Loss weights `[w_ae1, w_ae2, w_dyn, w_contrastive]`
- `save_models()`: Save encoder, decoder, dynamics models
- `save_logs(suffix)`: Save training logs
- `reset_losses()`: Reset loss history

**Loss Components**:
- `loss_ae1`: Reconstruction loss for $x_t$
- `loss_ae2`: Reconstruction loss for $x_{t+\tau}$ via dynamics
- `loss_dyn`: Latent dynamics prediction loss
- `loss_contrastive`: Contrastive loss separating success/failure

**Example**:
```python
from MORALS.training import Training
trainer = Training(config, loaders, verbose=True)
trainer.train(epochs=1500, patience=50, weight=[1, 1, 1, 0])
trainer.save_models()
```

#### `MORALS.training.TrainingConfig`

**Purpose**: Parses training weight configuration string.

**Constructor**:
```python
TrainingConfig(weights_str)
```

**Parameters**:
- `weights_str` (str): String like `"1x1x1x1x_0x0x1x0x"` where each segment has 4 weights

**Example**:
```python
from MORALS.training import TrainingConfig
config = TrainingConfig("1x1x1x1x_0x0x1x0x")
# config[0] = [1, 1, 1, 1]  # First training stage
# config[1] = [0, 0, 1, 0]  # Second training stage
```

#### `MORALS.training.LabelsLoss`

**Purpose**: Contrastive loss for separating success/failure encodings.

**Methods**:
- `forward(x, y)`: Compute contrastive loss
  - Uses sigmoid of negative distance: $\sigma(-100 \cdot ||x - y||_2)$

### Data Utilities

#### `MORALS.data_utils.DynamicsDataset`

**Purpose**: Dataset for learning dynamics from trajectory pairs.

**Constructor**:
```python
DynamicsDataset(config)
```

**Parameters**:
- `config` (dict): Configuration with:
  - `data_dir` (str): Directory containing trajectory files
  - `step` (int): Time step for dynamics (default: 1)
  - `subsample` (int): Subsampling rate (default: 1)
  - `system` (str): System name
  - `high_dims` (int): State dimension
  - `model_dir` (str): Directory to save normalization parameters

**Features**:
- Loads trajectories from CSV files
- Creates pairs $(x_t, x_{t+\tau})$
- Normalizes data to [0, 1]
- Saves normalization parameters (`X_min.txt`, `X_max.txt`)

**Methods**:
- `__len__()`: Number of pairs
- `__getitem__(idx)`: Get pair `(x_t, x_{t+\tau})`

**Example**:
```python
from MORALS.data_utils import DynamicsDataset
config = {
    'data_dir': 'data/pendulum/',
    'step': 1,
    'subsample': 1,
    'system': 'pendulum',
    'high_dims': 4,
    'model_dir': 'output/models/'
}
dataset = DynamicsDataset(config)
```

#### `MORALS.data_utils.LabelsDataset`

**Purpose**: Dataset for contrastive learning with success/failure labels.

**Constructor**:
```python
LabelsDataset(config)
```

**Parameters**:
- `config` (dict): Configuration with:
  - `data_dir` (str): Directory containing trajectory files
  - `labels_fname` (str): CSV file with labels (filename, label)
  - `system` (str): System name
  - `high_dims` (int): State dimension
  - `model_dir` (str): Directory with normalization parameters

**Features**:
- Loads final states from trajectories
- Pairs success and failure states for contrastive learning
- Generates cartesian product of success/failure indices

**Methods**:
- `__len__()`: Number of pairs
- `__getitem__(idx)`: Get pair indices
- `collate_fn(batch)`: Custom collate function for DataLoader

**Example**:
```python
from MORALS.data_utils import LabelsDataset
config = {
    'data_dir': 'data/pendulum/',
    'labels_fname': 'data/pendulum_success.txt',
    'system': 'pendulum',
    'high_dims': 4,
    'model_dir': 'output/models/'
}
dataset = LabelsDataset(config)
```

#### `MORALS.data_utils.TrajectoryDataset`

**Purpose**: Dataset for loading full trajectories (useful for visualization).

**Constructor**:
```python
TrajectoryDataset(config)
```

**Parameters**:
- `config` (dict): Configuration with:
  - `data_dir` (str): Directory containing trajectory files
  - `subsample` (int): Subsampling rate
  - `system` (str): System name
  - `labels_fname` (str, optional): Labels file

**Methods**:
- `__len__()`: Number of trajectories
- `__getitem__(idx)`: Get trajectory array
- `get_label(index)`: Get label for trajectory
- `get_successful_initial_conditions()`: Get successful initial states
- `get_unsuccessful_initial_conditions()`: Get unsuccessful initial states
- `get_successful_final_conditions()`: Get successful final states
- `get_unsuccessful_final_conditions()`: Get unsuccessful final states

### Dynamics Utilities

#### `MORALS.dynamics_utils.DynamicsUtils`

**Purpose**: Wrapper for encoder, decoder, and dynamics models with utility functions.

**Constructor**:
```python
DynamicsUtils(config)
```

**Parameters**:
- `config` (dict): Configuration with:
  - `system` (str): System name
  - `model_dir` (str): Directory with saved models
  - `use_limits` (bool): Whether to use custom limits (default: False)

**Methods**:
- `f(z)`: Apply latent dynamics
  - Input: Latent state `z` (numpy array)
  - Output: Next latent state (numpy array)
- `encode(x, normalize=True)`: Encode high-dimensional state
  - Input: Raw state `x` (numpy array)
  - Output: Latent state `z` (numpy array)
  - `normalize`: Whether to normalize input (default: True)
- `decode(z)`: Decode latent state
  - Input: Latent state `z` (numpy array)
  - Output: High-dimensional state (numpy array)

**Example**:
```python
from MORALS.dynamics_utils import DynamicsUtils
dyn_utils = DynamicsUtils(config)
z = dyn_utils.encode(x)  # Encode state
z_next = dyn_utils.f(z)  # Predict next latent state
x_next = dyn_utils.decode(z_next)  # Decode back
```

#### `MORALS.dynamics_utils.EnsembleDynamics`

**Purpose**: Ensemble of multiple dynamics models for uncertainty estimation.

**Constructor**:
```python
EnsembleDynamics(configs)
```

**Parameters**:
- `configs` (list): List of configuration dictionaries

**Methods**:
- `f(z)`: Returns `(mean, std)` of predictions
- `encode(x)`: Returns `(mean, std)` of encodings
- `decode(x)`: Not implemented

**Example**:
```python
from MORALS.dynamics_utils import EnsembleDynamics
configs = [config1, config2, config3]
ensemble = EnsembleDynamics(configs)
mean_z, std_z = ensemble.f(z)
```

### Grid Utilities

#### `MORALS.grid.Grid`

**Purpose**: Uniform grid for latent space discretization (compatible with CMGDB).

**Constructor**:
```python
Grid(lower_bounds, upper_bounds, subdivision, base_name="")
```

**Parameters**:
- `lower_bounds` (list[float]): Lower bounds per dimension
- `upper_bounds` (list[float]): Upper bounds per dimension
- `subdivision` (int): Total subdivision level
- `base_name` (str): Base name for files (default: "")

**Properties**:
- `dim`: Dimension of space
- `subdiv`: Subdivision per dimension (distributed evenly)
- `size_of_box`: Size of each box per dimension

**Methods**:
- `point2cell(point)`: Convert point to cell index
- `point2cell_coord(point)`: Convert point to cell coordinates
- `point2indexCMGDB(point)`: Convert point to CMGDB index
- `coordinates2index(coordinate)`: Convert coordinates to index
- `get_id_vertex(position, dim, subdivision)`: Get vertex ID from position
- `position_at_grid(id, dim, subdivision)`: Get position from vertex ID
- `grid_vertex2vertex(face, position)`: Convert grid vertex to geometric vertex
- `vertex2grid_vertex(vertex)`: Convert geometric vertex to grid vertex
- `uniform_sample()`: Generate uniform samples on grid
- `valid_grid(data, transform, neighbors)`: Create valid cell list from data
- `id2image(data)`: Map cell IDs to images (for box map)
- `write_map_grid(f, base_name, write_regions)`: Write map evaluation to CSV
- `load_map_grid(file_name)`: Load map from CSV
- `image_of_vertex_from_loaded_map(map, vertex)`: Get image from loaded map
- `neighbors(cell_index)`: Get neighbors of a cell

**Example**:
```python
from MORALS.grid import Grid
grid = Grid([-1, -1], [1, 1], subdivision=14)
cell = grid.point2cell([0.5, 0.3])
samples = grid.uniform_sample()
```

### Morse Graph Utilities

#### `MORALS.mg_utils.MorseGraphOutputProcessor`

**Purpose**: Processes CMGDB output files to extract Morse graph information.

**Constructor**:
```python
MorseGraphOutputProcessor(config)
```

**Parameters**:
- `config` (dict): Configuration with:
  - `output_dir` (str): Directory with CMGDB output files
  - `low_dims` (int): Latent space dimension

**Expected Files**:
- `MG_RoA_.csv`: Morse graph regions of attraction
- `MG_attractors.txt`: Attractor information
- `MG`: Morse graph file (Graphviz format)

**Methods**:
- `get_num_attractors()`: Get number of attractors
- `get_corner_points_of_attractor(id)`: Get corner points of attractor
- `get_corner_points_of_morse_node(id)`: Get corner points of Morse node
- `which_morse_node(point)`: Determine which Morse node contains point

**Example**:
```python
from MORALS.mg_utils import MorseGraphOutputProcessor
processor = MorseGraphOutputProcessor(config)
num_attractors = processor.get_num_attractors()
node = processor.which_morse_node(point)
```

### Systems

#### `MORALS.systems.utils.get_system`

**Purpose**: Factory function to get system instance.

**Function**:
```python
get_system(name, dims=10, **kwargs)
```

**Parameters**:
- `name` (str): System name
- `dims` (int): Dimension (for some systems)
- `**kwargs`: Additional system-specific arguments

**Supported Systems**:
- `"pendulum"`: Simple pendulum
- `"ndpendulum"`: N-link pendulum
- `"cartpole"`: Cart-pole system
- `"bistable"`: Bistable system
- `"N_CML"`: Coupled map lattice
- `"leslie_map"`: Leslie map (2D)
- `"leslie_map_3d"`: Leslie map (3D)
- `"humanoid"`: Humanoid robot
- `"trifinger"`: Trifinger robot
- `"bistable_rot"`: Rotating bistable system
- `"unifinger"`: Unifinger robot
- `"pendulum3links"`: 3-link pendulum

**Example**:
```python
from MORALS.systems.utils import get_system
system = get_system("pendulum")
```

#### `MORALS.systems.system.BaseSystem`

**Purpose**: Abstract base class for dynamical systems.

**Methods**:
- `f(s)`: System dynamics (default: identity)
- `sample_state(num_pts)`: Sample states from true bounds
- `get_bounds()`: Get bounds in embedded space
- `get_true_bounds()`: Get bounds in parameter space
- `dimension()`: Get dimension of parameter space
- `transform(s)`: Transform from parameter to embedded space
- `inverse_transform(s)`: Transform from embedded to parameter space

**Subclasses**: All system implementations inherit from `BaseSystem`

---

## Core Components

### Configuration Dictionary

The configuration dictionary is central to MORALS. It contains:

**Required Keys**:
- `system` (str): System name
- `high_dims` (int): High-dimensional state space dimension
- `low_dims` (int): Latent space dimension
- `data_dir` (str): Directory with trajectory data
- `model_dir` (str): Directory for saved models
- `log_dir` (str): Directory for training logs
- `output_dir` (str): Directory for output files

**Training Keys**:
- `epochs` (int): Maximum training epochs
- `patience` (int): Early stopping patience
- `batch_size` (int): Batch size
- `learning_rate` (float): Learning rate
- `seed` (int): Random seed
- `experiment` (str): Training weight configuration string
- `num_layers` (int): Number of hidden layers
- `hidden_shape` (int): Hidden layer size

**Data Keys**:
- `step` (int): Time step for dynamics
- `subsample` (int): Subsampling rate
- `labels_fname` (str, optional): Labels file path
- `use_limits` (bool): Use custom limits

**Example Config**:
```python
config = {
    'system': 'pendulum',
    'high_dims': 4,
    'low_dims': 2,
    'step': 1,
    'subsample': 1,
    'epochs': 1500,
    'patience': 50,
    'batch_size': 1024,
    'learning_rate': 0.001,
    'seed': 0,
    'experiment': '1x1x1x1x',
    'num_layers': 2,
    'data_dir': 'data/pendulum/',
    'labels_fname': 'data/pendulum_success.txt',
    'model_dir': 'output/models/',
    'log_dir': 'output/logs/',
    'output_dir': 'output/',
}
```

### Training Pipeline

The training pipeline consists of multiple stages:

1. **Data Loading**:
   - Load trajectories from CSV files
   - Create dynamics pairs $(x_t, x_{t+\tau})$
   - Load success/failure labels (if available)

2. **Model Initialization**:
   - Create encoder, decoder, dynamics models
   - Move to appropriate device (CPU/CUDA/MPS)

3. **Multi-Stage Training**:
   - Stage 1: Train autoencoder (`weight=[1, 0, 0, 0]`)
   - Stage 2: Train dynamics (`weight=[0, 1, 1, 0]`)
   - Stage 3: Fine-tune with contrastive loss (`weight=[0, 0, 1, 1]`)

4. **Model Saving**:
   - Save encoder, decoder, dynamics models
   - Save normalization parameters
   - Save training logs

### Morse Graph Computation Pipeline

1. **Load Trained Models**:
   - Load encoder, decoder, dynamics from `model_dir`

2. **Create Latent Space Grid**:
   - Define bounds (typically [-1, 1] per dimension)
   - Create uniform grid with specified subdivision

3. **Validate Grid**:
   - Sample points in original space
   - Encode to latent space
   - Determine valid grid cells

4. **Define Box Map**:
   - Create function that evaluates dynamics on grid boxes
   - Use Lipschitz constant for rigorous bounds

5. **Run CMGDB**:
   - Call CMGDB to compute Morse graph
   - Save results to output directory

6. **Process Results**:
   - Extract attractors
   - Compute regions of attraction (if requested)
   - Visualize results

---

## Systems

### Available Systems

#### Pendulum (`pendulum.py`)
- **Dimension**: 2D parameter space, 4D embedded space
- **Transform**: Converts angle/angular velocity to Cartesian coordinates
- **Bounds**: Angle $[-\pi, \pi]$, Angular velocity $[-2\pi, 2\pi]$

#### N-Link Pendulum (`ndpendulum.py`)
- **Dimension**: Configurable N-link pendulum
- **Transform**: Converts joint angles to Cartesian coordinates

#### Cart-Pole (`cartpole.py`)
- **Dimension**: 4D state space
- **Transform**: Standard cart-pole coordinates

#### Bistable System (`bistable.py`)
- **Dimension**: 2D
- **Features**: Two stable equilibria

#### Leslie Map (`leslie_map.py`, `leslie_map_3d.py`)
- **Dimension**: 2D or 3D
- **Features**: Population dynamics model

#### Humanoid (`humanoid.py`)
- **Dimension**: High-dimensional (data-driven bounds)

#### Trifinger (`trifinger.py`)
- **Dimension**: High-dimensional robot system

#### Unifinger (`unifinger.py`)
- **Dimension**: High-dimensional robot system

#### 3-Link Pendulum (`pendulum3links.py`)
- **Dimension**: 3-link pendulum system

#### Coupled Map Lattice (`N_CML.py`)
- **Dimension**: Configurable coupled map lattice

### Creating Custom Systems

To create a custom system:

1. **Create System Class**:
```python
from MORALS.systems.system import BaseSystem
import numpy as np

class MySystem(BaseSystem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "my_system"
        self.state_bounds = np.array([[...], [...]])  # Embedded space bounds
    
    def get_true_bounds(self):
        return np.array([[...], [...]])  # Parameter space bounds
    
    def transform(self, s):
        # Transform from parameter to embedded space
        return transformed_state
    
    def inverse_transform(self, s):
        # Transform from embedded to parameter space
        return parameter_state
```

2. **Register in `utils.py`**:
```python
from MORALS.systems.mysystem import MySystem

def get_system(name, dims=10, **kwargs):
    # ... existing systems ...
    elif name == "my_system":
        system = MySystem(**kwargs)
    # ...
```

---

## Training Pipeline

### Step 1: Prepare Data

1. **Trajectory Files**: Place CSV files in `data_dir/`
   - Each file: One trajectory, states as rows, dimensions as columns

2. **Labels File** (optional): Create `labels_fname` CSV
   - Format: `filename,label` where label is 0 (failure) or 1 (success)

### Step 2: Create Config File

Create a config file (e.g., `config/my_system.txt`):
```python
{
    'system': 'pendulum',
    'high_dims': 4,
    'low_dims': 2,
    'step': 1,
    'subsample': 1,
    'epochs': 1500,
    'patience': 50,
    'batch_size': 1024,
    'learning_rate': 0.001,
    'seed': 0,
    'experiment': '1x1x1x1x_0x0x1x0x',
    'num_layers': 2,
    'data_dir': 'data/pendulum/',
    'labels_fname': 'data/pendulum_success.txt',
    'model_dir': 'output/pendulum/models/',
    'log_dir': 'output/pendulum/logs/',
    'output_dir': 'output/pendulum/',
}
```

### Step 3: Train Models

```bash
python train.py --config my_system.txt --verbose
```

**Training Stages**:
- Stage 1 (`1x1x1x1x`): Train autoencoder
- Stage 2 (`0x0x1x0x`): Train dynamics only

### Step 4: Compute Morse Graph

```bash
python get_MG_RoA.py --config my_system.txt --name_out my_system --RoA --sub 16
```

**Arguments**:
- `--config`: Config file name
- `--name_out`: Output name
- `--RoA`: Compute regions of attraction
- `--sub`: Subdivision level
- `--validation_type`: `'uniform'`, `'random'`, or `'trajectory'`
- `--Lips`: Increase Lipschitz constant by percentage

---

## Morse Graph Computation

### Workflow

1. **Load Models**:
   ```python
   dyn_utils = DynamicsUtils(config)
   ```

2. **Create Grid**:
   ```python
   from MORALS.grid import Grid
   grid = Grid([-1]*dim, [1]*dim, subdivision=16)
   ```

3. **Sample Validation Points**:
   ```python
   # Option 1: Uniform sampling
   grid_orig = Grid(lower_bounds_orig, upper_bounds_orig, sub+2)
   samples_orig = grid_orig.uniform_sample()
   
   # Option 2: Random sampling
   samples_orig = system.sample_state(2**(sub+2))
   
   # Option 3: From trajectories
   dataset = DynamicsDataset(config)
   samples_orig = dataset.Xt.numpy()
   ```

4. **Encode to Latent Space**:
   ```python
   samples_latent = dyn_utils.encode(samples_orig)
   ```

5. **Create Valid Grid**:
   ```python
   valid_grid = grid.valid_grid(samples_latent)
   ```

6. **Define Box Map**:
   ```python
   def g(X):
       return dyn_tools.iterate(dyn_utils.f, X, n=steps)
   
   K = [1.1] * dim_latent  # Lipschitz constant
   
   def F(rect):
       return MG_util.BoxMapK_valid(g, rect, K, valid_grid, grid.point2cell)
   ```

7. **Run CMGDB**:
   ```python
   morse_graph, map_graph = MG_util.run_CMGDB(
       subdiv_min, subdiv_max, lower_bounds, upper_bounds,
       phase_periodic, F, base_name, subdiv_init
   )
   ```

### Output Files

- `MG`: Morse graph (Graphviz format)
- `MG.csv`: Morse graph data
- `MG_RoA_.csv`: Regions of attraction data
- `MG_RoA_.png`: Visualization of regions of attraction
- `MG_attractors.txt`: Attractor information

---

## File Structure

```
morals/
├── README.md                    # Package overview
├── instructions.md              # This file
├── setup.py                     # Package setup
├── pyproject.toml              # Package metadata
├── LICENSE                     # License file
│
├── MORALS/                     # Main package
│   ├── __init__.py            # Package initialization
│   ├── models.py              # Neural network models
│   ├── training.py            # Training system
│   ├── data_utils.py          # Dataset classes
│   ├── dynamics_utils.py       # Dynamics utilities
│   ├── grid.py                # Grid utilities
│   ├── mg_utils.py            # Morse graph utilities
│   │
│   └── systems/               # System implementations
│       ├── __init__.py
│       ├── system.py          # BaseSystem class
│       ├── utils.py           # System factory
│       ├── pendulum.py
│       ├── ndpendulum.py
│       ├── cartpole.py
│       ├── bistable.py
│       ├── leslie_map.py
│       ├── leslie_map_3d.py
│       ├── humanoid.py
│       ├── trifinger.py
│       ├── unifinger.py
│       ├── pendulum3links.py
│       ├── bistable_rot.py
│       └── N_CML.py
│
└── examples/                   # Example scripts
    ├── train.py               # Training script
    ├── get_MG_RoA.py          # Morse graph computation script
    ├── get_data.py            # Data generation script
    ├── GettingStarted.ipynb   # Getting started notebook
    ├── leslie_map_3d_example.ipynb
    │
    ├── config/                # Configuration files
    │   ├── pendulum_lqr.txt
    │   ├── bistable.txt
    │   ├── leslie_map.txt
    │   └── ...
    │
    ├── data/                  # Data directory
    │   └── ...
    │
    └── output/                # Output directory
        └── ...
```

---

## Usage Patterns

### Complete Workflow Example

```python
# 1. Prepare data (trajectories in CSV files)
# data/pendulum/traj_001.csv, traj_002.csv, ...

# 2. Create config file
config = {
    'system': 'pendulum',
    'high_dims': 4,
    'low_dims': 2,
    'step': 1,
    'subsample': 1,
    'epochs': 1500,
    'patience': 50,
    'batch_size': 1024,
    'learning_rate': 0.001,
    'seed': 0,
    'experiment': '1x1x1x1x_0x0x1x0x',
    'num_layers': 2,
    'data_dir': 'data/pendulum/',
    'labels_fname': 'data/pendulum_success.txt',
    'model_dir': 'output/pendulum/models/',
    'log_dir': 'output/pendulum/logs/',
    'output_dir': 'output/pendulum/',
}

# 3. Train models
from MORALS.training import Training, TrainingConfig
from MORALS.data_utils import DynamicsDataset, LabelsDataset
from torch.utils.data import DataLoader

# Load data
dynamics_dataset = DynamicsDataset(config)
labels_dataset = LabelsDataset(config)

# Create loaders
loaders = {
    'train_dynamics': DataLoader(...),
    'test_dynamics': DataLoader(...),
    'train_labels': DataLoader(...),
    'test_labels': DataLoader(...),
}

# Train
trainer = Training(config, loaders, verbose=True)
experiment = TrainingConfig(config['experiment'])
for weights in experiment:
    trainer.train(epochs=1500, patience=50, weight=weights)
trainer.save_models()

# 4. Compute Morse graph
from MORALS.dynamics_utils import DynamicsUtils
from MORALS.grid import Grid
import dytop.CMGDB_util as CMGDB_util

dyn_utils = DynamicsUtils(config)
MG_util = CMGDB_util.CMGDB_util()

grid = Grid([-1, -1], [1, 1], subdivision=16)
samples_latent = dyn_utils.encode(samples_orig)
valid_grid = grid.valid_grid(samples_latent)

def F(rect):
    return MG_util.BoxMapK_valid(g, rect, K, valid_grid, grid.point2cell)

morse_graph, map_graph = MG_util.run_CMGDB(
    subdiv_min=16, subdiv_max=16,
    lower_bounds=[-1, -1], upper_bounds=[1, 1],
    phase_periodic=[False, False],
    F=F, base_name="MG", subdiv_init=16
)

# 5. Process results
from MORALS.mg_utils import MorseGraphOutputProcessor
processor = MorseGraphOutputProcessor(config)
num_attractors = processor.get_num_attractors()
node = processor.which_morse_node(point)
```

### Using Trained Models

```python
from MORALS.dynamics_utils import DynamicsUtils

# Load models
dyn_utils = DynamicsUtils(config)

# Encode state
x = np.array([0.1, 0.2, 0.3, 0.4])  # High-dimensional state
z = dyn_utils.encode(x)

# Predict next latent state
z_next = dyn_utils.f(z)

# Decode back
x_next = dyn_utils.decode(z_next)
```

### Custom System Example

```python
from MORALS.systems.system import BaseSystem
import numpy as np

class MyCustomSystem(BaseSystem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "my_custom_system"
        # Define bounds in embedded space
        self.state_bounds = np.array([
            [-10, 10],  # x
            [-10, 10],  # y
            [-5, 5]     # z
        ])
    
    def get_true_bounds(self):
        # Define bounds in parameter space
        return np.array([
            [-np.pi, np.pi],  # angle
            [-2, 2]           # velocity
        ])
    
    def transform(self, s):
        # Transform from parameter space to embedded space
        angle, vel = s[:, 0], s[:, 1]
        x = np.cos(angle) * vel
        y = np.sin(angle) * vel
        z = vel**2
        return np.array([x, y, z]).T
    
    def inverse_transform(self, s):
        # Transform from embedded space to parameter space
        x, y, z = s[:, 0], s[:, 1], s[:, 2]
        angle = np.arctan2(y, x)
        vel = np.sqrt(z)
        return np.array([angle, vel]).T
```

---

## Dependencies

### Required
- `numpy`: Numerical arrays
- `scipy`: Scientific computing
- `matplotlib`: Visualization
- `scikit-learn`: Machine learning utilities
- `pandas`: Data manipulation
- `seaborn`: Statistical visualization
- `tqdm`: Progress bars
- `torch`: PyTorch for neural networks
- `torchvision`: PyTorch vision utilities
- `CMGDB`: Conley Morse Graph Database
- `dytop`: Dynamic topology utilities

### Optional
- CUDA/MPS: GPU acceleration for PyTorch

---

## Key Design Decisions

### Latent Space Bounds
- Default: `[-1, 1]` per dimension (from Tanh activation)
- Can be adjusted based on data distribution

### Normalization
- Data normalized to `[0, 1]` using min/max from training data
- Normalization parameters saved in `X_min.txt`, `X_max.txt`

### Training Strategy
- Multi-stage training with different loss weights
- Early stopping based on validation loss
- Learning rate scheduling

### Grid Validation
- Only cells containing data are considered valid
- Neighbors can be included for robustness

### Lipschitz Constant
- Used for rigorous box map bounds
- Can be increased for safety margin (`--Lips` argument)

---

## Limitations

1. **Latent Space Dimension**: Must be low-dimensional (typically 2-3D) for CMGDB
2. **Data Requirements**: Requires trajectory data with consistent format
3. **System Bounds**: Some systems require data-driven bounds
4. **Training Time**: Can be time-consuming for large datasets
5. **Memory**: Large grids can be memory-intensive

---

## Best Practices

1. **Data Preparation**:
   - Ensure trajectories are consistent length
   - Use appropriate subsampling
   - Label success/failure trajectories

2. **Model Architecture**:
   - Start with small hidden layers
   - Adjust based on data complexity
   - Use appropriate latent dimension (2-3D)

3. **Training**:
   - Use multi-stage training
   - Monitor loss curves
   - Use early stopping

4. **Morse Graph Computation**:
   - Start with low subdivision
   - Increase gradually
   - Validate grid cells properly

5. **System Design**:
   - Implement `transform` and `inverse_transform` correctly
   - Define bounds appropriately
   - Test with simple examples first

---

## Troubleshooting

### Training Issues

**Problem**: Models not learning
- **Solution**: Check data normalization, adjust learning rate, verify data format

**Problem**: Collapse (encoder maps everything to same point)
- **Solution**: Use collapse detection, adjust architecture, check data diversity

**Problem**: High reconstruction error
- **Solution**: Increase model capacity, check data quality, adjust training weights

### Morse Graph Issues

**Problem**: No valid grid cells
- **Solution**: Check latent space bounds, verify encoding, adjust validation sampling

**Problem**: Too many/few Morse sets
- **Solution**: Adjust subdivision level, check dynamics model, verify Lipschitz constant

**Problem**: Out of memory
- **Solution**: Reduce subdivision, use trajectory-based validation, reduce batch size

---

## References

- **Paper**: "MORALS: Analysis of High-Dimensional Robot Controllers via Topological Tools in a Latent Space" (ICRA 2024)
- **Repository**: https://github.com/Ewerton-Vieira/MORALS
- **CMGDB**: Conley Morse Graph Database documentation
- **dytop**: Dynamic topology utilities

---

## Version Information

- **Current Version**: 0.1.4 (from setup.py)
- **Python**: Python 3.x
- **PyTorch**: Compatible with recent versions
- **License**: See LICENSE file

---

## Contact and Support

- **Authors**: Ewerton R. Vieira, Aravind Sivaramakrishnan, Sumanth Tangirala, Edgar Granados, Konstantin Mischaikow, Kostas E. Bekris
- **Repository**: https://github.com/Ewerton-Vieira/MORALS
- **Paper**: ICRA 2024 (Best Paper Award in Automation finalist)

---

*This document serves as a comprehensive reference for the MORALS package. For specific usage examples, see the `examples/` directory and notebooks.*

