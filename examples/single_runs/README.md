# Single Runs Directory

This directory contains individual CMGDB experiment runs with auto-incrementing numbering.

## Directory Structure

Each run is stored in a `run_XXX` directory with the following structure:

```
single_runs/
├── run_001/
│   ├── settings.json          # Complete configuration for reproducibility
│   ├── training_data/
│   │   ├── trajectory_data.npz   # Original Leslie map trajectories
│   │   └── latent_data.npz       # Encoded latent space data
│   ├── models/
│   │   ├── encoder.pt
│   │   ├── decoder.pt
│   │   └── latent_dynamics.pt
│   └── results/
│       ├── training_history.json     # Epoch-by-epoch losses
│       ├── training_losses.png       # Loss curves
│       └── morse_graph_comparison.png
├── run_002/
│   └── ...
└── run_003/
    └── ...
```

## settings.json

The `settings.json` file contains everything needed to reproduce the run:

- **run_info**: Run name, directory, timestamp
- **configuration**: All config parameters (data generation, training, Morse graph computation)
- **model_architecture**: Layer sizes, parameter counts, activation functions
- **training_info**: Epochs, batch size, learning rate, training time, device, loss weights
- **data_info**: Number of trajectories, domain bounds, random seed, train/val split
- **results**: Final errors, Morse set counts, training metrics

## Usage

### Auto-numbered run (recommended)
```bash
python cmgdb_single_run.py
```
This will create `run_001`, then `run_002`, etc.

### Named run
```bash
python cmgdb_single_run.py --name my_experiment
```
This will create `run_my_experiment`.

### Custom parameters
```bash
python cmgdb_single_run.py --hidden-dim 64 --num-epochs 500 --num-layers 4
```

### Custom output directory
```bash
python cmgdb_single_run.py --output-dir /path/to/experiments
```

## Reproducing a Run

To reproduce a run from `settings.json`:

```python
import json
from cmgdb_single_run import run_single_experiment_cmgdb

# Load settings
with open('single_runs/run_001/settings.json', 'r') as f:
    settings = json.load(f)

# Extract config
config = settings['configuration']

# Run experiment with same settings
result = run_single_experiment_cmgdb(
    run_name='001_reproduction',
    output_dir='reproductions',
    config_overrides=config,
    verbose=True
)
```
