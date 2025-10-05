# Tests Directory

This directory contains tests and demonstrations for the MorseGraph repository's multi-package structure.

## Repository Structure

This repository contains **four separate packages**:

```
MorseGraph/
├── MorseGraph/              → morsegraph package (modern Python)
├── cmgdb/                   → CMGDB package (legacy C++ bindings)
├── hybrid_boxmap/           → hybrid-dynamics package
├── morals/                  → MORALS package (ML + Morse graphs)
└── tests/                   → Shared tests/demos
```

## Installation

Each package needs to be installed separately. From the repository root:

```bash
# Install main package (always do this first)
pip install -e .

# Install legacy packages (as needed)
pip install -e ./cmgdb          # Requires cmake
pip install -e ./hybrid_boxmap
pip install -e ./morals         # Depends on CMGDB
```

### Installing CMGDB (C++ Package)

CMGDB has C++ code and requires cmake:

```bash
# macOS
brew install cmake
pip install -e ./cmgdb

# Ubuntu/Debian
sudo apt-get install cmake
pip install -e ./cmgdb

# Windows
# Download cmake from https://cmake.org/download/
pip install -e ./cmgdb
```

## Package Names vs. Import Names

| Directory         | Package Name       | Import As             |
|-------------------|--------------------|-----------------------|
| `MorseGraph/`     | `morsegraph`       | `from MorseGraph import ...` |
| `cmgdb/`          | `CMGDB`            | `import CMGDB` |
| `hybrid_boxmap/`  | `hybrid-dynamics`  | `from hybrid_dynamics import ...` |
| `morals/`         | `MORALS`           | `import MORALS` |

## Test Files

### `test_multi_package_imports.py`

Demonstrates importing and using all four packages. This is a **diagnostic tool** that shows which packages are installed and working.

**Run it:**
```bash
python tests/test_multi_package_imports.py
```

**What it does:**
- Tests each package import separately
- Shows clear error messages if packages aren't installed
- Demonstrates basic usage of each package
- Tests cross-package compatibility (using multiple packages together)

**Expected output:**
```
morsegraph           ✓ PASS   (if installed)
CMGDB                ✗ FAIL   (if not installed)
hybrid_dynamics      ✗ FAIL   (if not installed)
MORALS               ✗ FAIL   (if not installed)
cross_package        ✗ FAIL   (if dependencies missing)
```

### `test_cmgdb_ode_examples.py`

Python script version of the CMGDB notebook examples (`ODE_Computations_Examples.ipynb`).

**Run it:**
```bash
# Quick demo (Example 1 only, ~45 seconds)
python tests/test_cmgdb_ode_examples.py

# All examples (WARNING: ~20+ minutes total)
python tests/test_cmgdb_ode_examples.py --all
```

**Requires:** CMGDB installed (`pip install -e ./cmgdb`)

**Examples included:**
1. **Competition Model** - 2D ecological competition system (~45s)
2. **Van der Pol** - Classic oscillator (~4 minutes)
3. **Radial System** - Polar coordinate system (coarse: ~19s, fine: ~2 min)
4. **Lorenz** - 3D chaotic attractor (~14 minutes) ⚠️

**What it demonstrates:**
- Converting Jupyter notebooks to executable Python scripts
- Using CMGDB for ODE-based Morse graph computation
- Time-tau maps with scipy's `solve_ivp`
- Adaptive grid refinement
- Morse graph visualization

## Using Multiple Packages Together

Here's how you can import from different packages in your examples:

```python
# Modern MorseGraph package
from MorseGraph import Model, UniformGrid, BoxMapData

# Legacy CMGDB package (C++ bindings)
import CMGDB

# Hybrid dynamics package
from hybrid_dynamics import HybridSystem, Grid
from hybrid_dynamics.examples.bouncing_ball import BouncingBall

# MORALS package (ML + Morse graphs)
import MORALS
from MORALS.training import train_latent_dynamics
```

### Example: Hybrid Dynamics → MorseGraph Analysis

```python
from hybrid_dynamics.examples.bouncing_ball import BouncingBall
from MorseGraph import UniformGrid, BoxMapData
import numpy as np

# Generate data with hybrid_dynamics
ball = BouncingBall(max_jumps=5)
states, next_states = [], []

for _ in range(100):
    initial = [np.random.uniform(0.5, 3), np.random.uniform(-2, 2)]
    traj = ball.system.simulate(initial, tau=0.5)
    if len(traj.times) > 1:
        states.append(initial)
        next_states.append(traj.states[-1])

# Analyze with MorseGraph
dynamics = BoxMapData(
    states=np.array(states),
    next_states=np.array(next_states),
    epsilon=0.1
)

grid = UniformGrid(
    bounds=ball.domain_bounds,
    divisions=np.array([20, 20])
)

# Now use Model for Morse graph computation
```

## Development Workflow

1. **Start with MorseGraph** - Install the main package first
2. **Add legacy packages as needed** - Install CMGDB, hybrid_dynamics, or MORALS only if you need their specific features
3. **Use tests to verify** - Run `test_multi_package_imports.py` to check what's working
4. **Reference examples** - Look at `test_cmgdb_ode_examples.py` for converting notebooks to scripts

## Common Issues

### "No module named 'CMGDB'"
- Solution: `pip install -e ./cmgdb` (requires cmake)

### "No module named 'hybrid_dynamics'"
- Solution: `pip install -e ./hybrid_boxmap`

### CMGDB installation fails with "cmake not found"
- Solution: Install cmake first (see Installation section above)

### Import works but tests fail
- Check that you're using the correct API (see test files for examples)
- Verify package is installed in development mode (`pip install -e .`)

## Writing Your Own Examples

When creating examples that use multiple packages:

1. **Place them in `examples/`** - Keep tests directory for testing/demos only
2. **Import by package name** - Use the correct import names from the table above
3. **Document dependencies** - List which packages your example requires
4. **Test imports first** - Use try/except to provide helpful error messages

```python
# examples/my_example.py
try:
    from MorseGraph import Model
    import CMGDB
    from hybrid_dynamics import HybridSystem
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install -e . && pip install -e ./cmgdb")
    exit(1)

# Your code here...
```

## Questions?

- **Installation issues?** See CLAUDE.md in each package directory
- **API questions?** Check test files for working examples
- **Want to contribute?** Tests are executable examples - PR welcome!
