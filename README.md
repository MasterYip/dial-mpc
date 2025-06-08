# DIAL-MPC JAX

Differentiable Algorithms for Learning Model Predictive Control (DIAL-MPC) implemented in JAX with Brax environments.

## Overview

This package provides JAX-based implementations of trajectory optimization using diffusion-based model predictive control. It includes both the original JAX/JIT-optimized implementation and a test implementation using PyTorch-based trajectory gradient sampling.

## Features

- **JAX-based trajectory optimization** with JIT compilation for high performance
- **Brax environment integration** for physics simulation
- **Diffusion-based trajectory planning** with sampling-based optimization
- **Multiple update methods**: MPPI, WBFO (Weighted Basis Function Optimization), AVWBFO
- **Spline interpolation** for smooth trajectory generation
- **Visualization tools** with interactive web interface
- **Test implementation** using PyTorch trajectory gradient sampling

## Installation

1. Install the required dependencies:
```bash
pip install jax jaxlib brax
pip install torch  # For test implementation
pip install matplotlib tqdm pyyaml flask emoji art scienceplots
```

2. Install JAX-specific dependencies:
```bash
pip install jax-cosmo  # For spline interpolation
```

## Usage

### Original JAX Implementation

Run the main DIAL-MPC implementation:

```bash
# Using an example configuration
python -m dial_mpc.core.dial_core --example unitree_h1_walk

# Using a custom configuration file
python -m dial_mpc.core.dial_core --config path/to/config.yaml

# List available examples
python -m dial_mpc.core.dial_core --list-examples
```

### Test Implementation with PyTorch Trajectory Gradient Sampling

Run the test implementation that uses PyTorch-based trajectory optimization:

```bash
# Using an example configuration
python -m dial_mpc.core.dial_core_test --example unitree_h1_walk
python -m dial_mpc.core.dial_core_test --example unitree_go2_trot

# Using a custom configuration file
python -m dial_mpc.core.dial_core_test --config path/to/config.yaml

# List available examples
python -m dial_mpc.core.dial_core_test --list-examples
```

## Implementation Differences

### Original Implementation (`dial_core.py`)
- **JAX/JIT optimized**: Uses JAX's just-in-time compilation for maximum performance
- **Vectorized operations**: Batch processing of trajectories using JAX's vmap
- **Spline interpolation**: Uses JAX-cosmo for smooth trajectory interpolation
- **Memory efficient**: Optimized for GPU/TPU acceleration

### Test Implementation (`dial_core_test.py`)
- **PyTorch backend**: Uses the trajectory gradient sampling module from `legged_gym_cmp`
- **No JIT compilation**: Easier debugging and development
- **Flexible optimization**: Supports MPPI, WBFO, and AVWBFO update methods
- **Modular design**: Leverages existing PyTorch trajectory optimization infrastructure
- **No RL warmstart**: Simplified version without reinforcement learning initialization

## Configuration

The configuration uses the `DialConfig` dataclass with the following key parameters:

```python
@dataclass
class DialConfig:
    # Experiment settings
    seed: int = 0
    output_dir: str = "output"
    n_steps: int = 100
    
    # Environment
    env_name: str = "unitree_h1_walk"
    
    # Trajectory optimization
    Nsample: int = 2048      # Number of trajectory samples
    Hsample: int = 16        # Horizon length in samples
    Hnode: int = 4           # Number of control nodes
    Ndiffuse: int = 2        # Diffusion steps per iteration
    Ndiffuse_init: int = 10  # Initial diffusion steps
    temp_sample: float = 0.06 # Sampling temperature
    
    # Noise scheduling
    horizon_diffuse_factor: float = 0.9
    traj_diffuse_factor: float = 0.5
    
    # Optimization method
    update_method: str = "mppi"  # "mppi", "wbfo", or "avwbfo"
```

## Key Components

### MBDPI Class (Original)
- Implements Model-Based Diffusion Policy Iteration
- JAX-optimized trajectory optimization
- Spline-based trajectory interpolation
- Vectorized rollout evaluation

### MBDPITest Class (Test Implementation)
- Implements the same logic using PyTorch trajectory gradient sampling
- Single-environment trajectory optimization
- Integration with existing WBFO infrastructure
- Simplified debugging and development

### Trajectory Optimization Methods

1. **MPPI (Model Predictive Path Integral)**
   - Standard sampling-based optimization
   - Softmax weighting of trajectory samples
   - Temperature-controlled exploration

2. **WBFO (Weighted Basis Function Optimization)**
   - Advanced weighting scheme using basis functions
   - Spline-based trajectory representation
   - Improved convergence properties

3. **AVWBFO (Action-Value Weighted Basis Function Optimization)**
   - Extension of WBFO with discounted rewards
   - Temporal credit assignment
   - Enhanced performance for long-horizon tasks

## Output

Both implementations generate:

- **Trajectory plots**: Reward evolution and optimization progress
- **Interactive visualization**: Web-based 3D simulation viewer
- **State data**: Saved trajectory states and predictions
- **Performance metrics**: Timing and convergence statistics

Output files include:
- `reward_comparison.pdf/png`: Comparison of predicted vs actual rewards
- `reward_components.pdf/png`: Individual reward component analysis
- `brax_visualization.html`: Interactive 3D visualization
- `states.npy`: Saved trajectory data

## Dependencies

### Core Dependencies
- `jax>=0.4.0`: Numerical computing with automatic differentiation
- `jaxlib>=0.4.0`: JAX linear algebra library
- `brax>=0.9.0`: Physics simulation environment
- `torch>=1.13.0`: PyTorch (for test implementation)

### Visualization and Utilities
- `matplotlib`: Plotting and visualization
- `tqdm`: Progress bars
- `pyyaml`: Configuration file parsing
- `flask`: Web server for interactive visualization
- `emoji`: Terminal output enhancement
- `art`: ASCII art headers

### Optional Dependencies
- `jax-cosmo`: Spline interpolation (for original implementation)
- `scienceplots`: Enhanced matplotlib styling

## Architecture

```
dial_mpc_jax/
├── dial_mpc/
│   ├── core/
│   │   ├── dial_core.py          # Original JAX implementation
│   │   ├── dial_core_test.py     # Test PyTorch implementation
│   │   └── dial_config.py        # Configuration dataclass
│   ├── envs/                     # Environment configurations
│   ├── examples/                 # Example configurations
│   └── utils/                    # Utility functions
└── README.md                     # This file
```

## Performance Considerations

### Original Implementation
- **High throughput**: JAX JIT compilation provides 10-100x speedup
- **GPU/TPU ready**: Optimized for accelerator hardware
- **Memory efficient**: Vectorized operations minimize memory allocation
- **Scalable**: Handles large batch sizes and long horizons

### Test Implementation
- **Development friendly**: No compilation overhead, easier debugging
- **Flexible**: Easy to modify and extend trajectory optimization
- **Educational**: Clear separation of components for learning
- **Compatible**: Uses existing PyTorch ecosystem tools

## Contributing

When extending this codebase:

1. **For performance-critical code**: Use the original JAX implementation
2. **For development and experimentation**: Use the test PyTorch implementation
3. **Maintain compatibility**: Both implementations should produce similar results
4. **Add tests**: Verify that modifications work with both implementations

## Citation

If you use this implementation in your research, please cite the original DIAL-MPC paper and acknowledge the JAX/Brax ecosystem.

## License

This project follows the same license as the parent repository.