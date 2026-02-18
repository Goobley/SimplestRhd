# Experiment Templates

This directory contains experiment templates for SimplestRhd.

## Structure

Each experiment subdirectory contains:

- **`setup.py`**: Experiment configuration and problem setup
  - `config`: Dictionary with all parameters for this experiment
  - Initial condition functions (e.g., `sod_ics()`)
  - Boundary condition functions (e.g., `sod_bcs()`)
  - Any problem-specific utilities

- **`run.py`**: Executable script to run the experiment
  - Imports setup from `setup.py`
  - Constructs the grid and state
  - Runs the simulation
  - Generates plots

## Available Experiments

### `sod_shock/`
The Sod shock tube test case.

```bash
cd sod_shock
python run.py
```

## Creating Your Own Experiment

1. **Create a new directory**:
   ```bash
   mkdir experiments/my_test_case
   ```

2. **Create `setup.py`** with your configuration:
   ```python
   from simplestrhd import prim_to_cons, SYMMETRIC_BC, ...

   config = {
       "max_time": 0.5,
       "output_cadence": 0.1,
       "max_cfl": 0.1,
       "gamma": 1.4,
       "num_grid_points": 256,
       "x_min": 0.0,
       "x_max": 1.0,
   }

   def my_ics(x, gamma):
       # Set up your initial conditions
       w = np.stack([rho, v, p, spec_e_ion])
       return prim_to_cons(w, gamma=gamma)

   def my_bcs():
       return [SYMMETRIC_BC, SYMMETRIC_BC]  # or your BCs
   ```

3. **Create `run.py`** to execute the simulation:
   ```python
   from setup import my_ics, my_bcs, config
   from simplestrhd import run_sim, cons_to_prim

   # Build your state dictionary
   state = {
       "xcc": grid,
       "dx": dx,
       "Q": my_ics(grid, gamma=config["gamma"]),
       "fixed_bcs": None,
       "user_bcs": None,
       "sources": [],
   }

   # Run simulation
   snaps = run_sim(state, my_bcs(), **config, gamma=config["gamma"])

   # Process results...
   ```

## Tips

- Each experiment is independent and can be run in its own directory
- You can run multiple experiments in parallel without interference
- Keep all configuration in `setup.py` for easy experimentation
- Use relative imports to access utilities from the package
