from dial_mpc.core.dial_config import DialConfig
from dial_mpc.examples import examples
from dial_mpc.utils.io_utils import get_example_path, load_dataclass_from_dict
import dial_mpc.envs as dial_envs
import brax.envs as brax_envs
from brax.io import html
import functools
import jax
from jax import numpy as jnp
import emoji
import art
import scienceplots
import os
import time
from dataclasses import dataclass
import importlib
import sys

import yaml
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
matplotlib.use('Agg')  # Use non-interactive backend
plt.rcParams['text.usetex'] = False  # Disable LaTeX
plt.rcParams['font.family'] = 'DejaVu Sans'  # Use a standard font

# PyTorch imports for trajectory optimization
import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union

# Import trajectory gradient sampling module
sys.path.append('/home/user/CodeSpace/Python/PredictiveDiffusionPlanner_Dev/')

from traj_sampling.traj_grad_sampling import TrajGradSampling, TrajGradSamplingCfg


class JAXSplineTrajGradSampling(TrajGradSampling):
    """Extended TrajGradSampling class that uses JAX spline interpolation.

    This class inherits from TrajGradSampling and overrides the conversion methods
    to use JAX spline interpolation instead of PyTorch linear interpolation.
    """

    def __init__(self, cfg, device, num_envs, num_actions, dt, main_env_indices, args: DialConfig):
        """Initialize with JAX spline interpolation capabilities.

        Args:
            cfg: Configuration object for trajectory optimization
            device: Device for computations
            num_envs: Total number of environments
            num_actions: Number of action dimensions
            dt: Environment timestep
            main_env_indices: Indices of main environments
            args: DialConfig containing horizon parameters
        """
        super().__init__(cfg, device, num_envs, num_actions, dt, main_env_indices)

        self.args = args
        self.nu = num_actions

        # Initialize JAX spline interpolation functions
        self._init_jax_spline_functions()

        print(f"JAXSplineTrajGradSampling initialized with JAX spline interpolation")

    def _init_jax_spline_functions(self):
        """Initialize JAX spline interpolation functions like in the original dial_core.py"""
        from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline

        # Initialize time steps for interpolation (same as original MBDPI)
        self.ctrl_dt = 0.02
        self.step_us = jnp.linspace(0, self.ctrl_dt * self.args.Hsample, self.args.Hsample + 1)
        self.step_nodes = jnp.linspace(0, self.ctrl_dt * self.args.Hsample, self.args.Hnode + 1)
        self.node_dt = self.ctrl_dt * (self.args.Hsample) / (self.args.Hnode)

        # Create JAX spline interpolation functions (same as original MBDPI)
        @functools.partial(jax.jit, static_argnums=(0,))
        def jax_node2u(self, nodes):
            spline = InterpolatedUnivariateSpline(self.step_nodes, nodes, k=2)
            us = spline(self.step_us)
            return us

        @functools.partial(jax.jit, static_argnums=(0,))
        def jax_u2node(self, us):
            spline = InterpolatedUnivariateSpline(self.step_us, us, k=2)
            nodes = spline(self.step_nodes)
            return nodes

        # Bind the methods to self
        self.jax_node2u = jax_node2u.__get__(self, type(self))
        self.jax_u2node = jax_u2node.__get__(self, type(self))

        # Create vectorized versions (same as original MBDPI)
        self.jax_node2u_vmap = jax.jit(jax.vmap(self.jax_node2u, in_axes=(1), out_axes=(1)))
        self.jax_u2node_vmap = jax.jit(jax.vmap(self.jax_u2node, in_axes=(1), out_axes=(1)))
        self.jax_node2u_vvmap = jax.jit(jax.vmap(self.jax_node2u_vmap, in_axes=(0)))
        self.jax_u2node_vvmap = jax.jit(jax.vmap(self.jax_u2node_vmap, in_axes=(0)))

    def jax_to_torch(self, jax_array):
        """Convert JAX array to PyTorch tensor."""
        if isinstance(jax_array, torch.Tensor):
            # Already a PyTorch tensor, just move to correct device
            return jax_array.to(self.device)
        return torch.from_numpy(np.array(jax_array)).to(self.device)

    def torch_to_jax(self, torch_tensor):
        """Convert PyTorch tensor to JAX array."""
        if not isinstance(torch_tensor, torch.Tensor):
            # Already a JAX array or numpy array
            return jnp.array(torch_tensor)
        return jnp.array(torch_tensor.cpu().numpy())

    def node2u(self, nodes: torch.Tensor) -> torch.Tensor:
        """Convert control nodes to dense control sequence using JAX spline interpolation.

        Args:
            nodes: Control nodes as PyTorch tensor [Hnode+1, action_dim]

        Returns:
            Dense control sequence as PyTorch tensor [Hsample+1, action_dim]
        """
        # Convert to JAX, apply spline interpolation, convert back to PyTorch
        nodes_jax = self.torch_to_jax(nodes)
        us_jax = self.jax_node2u_vmap(nodes_jax)
        return self.jax_to_torch(us_jax)

    def u2node(self, us: torch.Tensor) -> torch.Tensor:
        """Convert dense control sequence to control nodes using JAX spline interpolation.

        Args:
            us: Dense control sequence as PyTorch tensor [Hsample+1, action_dim]

        Returns:
            Control nodes as PyTorch tensor [Hnode+1, action_dim]
        """
        # Convert to JAX, apply spline interpolation, convert back to PyTorch
        us_jax = self.torch_to_jax(us)
        nodes_jax = self.jax_u2node_vmap(us_jax)
        return self.jax_to_torch(nodes_jax)

    def node2u_batch(self, nodes_batch: torch.Tensor) -> torch.Tensor:
        """Convert batch of control nodes to dense control sequences using JAX spline interpolation.

        Args:
            nodes_batch: Batch of control nodes [batch_size, Hnode+1, action_dim]

        Returns:
            Batch of dense control sequences [batch_size, Hsample+1, action_dim]
        """
        # Convert to JAX, apply batch spline interpolation, convert back to PyTorch
        nodes_batch_jax = self.torch_to_jax(nodes_batch)
        us_batch_jax = self.jax_node2u_vvmap(nodes_batch_jax)
        return self.jax_to_torch(us_batch_jax)

    def u2node_batch(self, us_batch: torch.Tensor) -> torch.Tensor:
        """Convert batch of dense control sequences to control nodes using JAX spline interpolation.

        Args:
            us_batch: Batch of dense control sequences [batch_size, Hsample+1, action_dim]

        Returns:
            Batch of control nodes [batch_size, Hnode+1, action_dim]
        """
        # Convert to JAX, apply batch spline interpolation, convert back to PyTorch
        us_batch_jax = self.torch_to_jax(us_batch)
        nodes_batch_jax = self.jax_u2node_vvmap(us_batch_jax)
        return self.jax_to_torch(nodes_batch_jax)

    def shift_nodetraj_batch(self, trajs: torch.Tensor, n_steps: int = 1) -> torch.Tensor:
        """Shift multiple trajectories by n time steps using JAX spline interpolation.

        Args:
            trajs: Trajectories to shift [batch_size, length, action_dim]
            n_steps: Number of steps to shift by

        Returns:
            Shifted trajectories [batch_size, length, action_dim]
        """
        # Convert to dense control sequences using JAX spline interpolation
        u_batch = self.node2u_batch(trajs)

        # Convert to JAX for shifting operations
        u_batch_jax = self.torch_to_jax(u_batch)

        # Shift all dense controls by n steps using JAX operations
        u_batch_jax = jnp.roll(u_batch_jax, -n_steps, axis=1)

        # Fill the last n_steps controls with zeros
        u_batch_jax = u_batch_jax.at[:, -n_steps:, :].set(0.0)

        # Convert back to PyTorch and then to nodes using JAX spline interpolation
        u_batch_torch = self.jax_to_torch(u_batch_jax)
        shifted = self.u2node_batch(u_batch_torch)

        return shifted


def rollout_us(step_env, state, us):
    """JAX-based rollout function for single trajectory."""
    def step(state, u):
        state = step_env(state, u)
        return state, (state.reward, state.pipeline_state, state.metrics if hasattr(state, 'metrics') else {})

    _, (rews, pipline_states, metrics_seq) = jax.lax.scan(step, state, us)
    return rews, pipline_states, metrics_seq


class MBDPITest:
    """Test implementation of MBDPI using PyTorch trajectory gradient sampling.

    This class maintains the same interface as the original MBDPI but uses
    PyTorch-based trajectory optimization instead of JAX for easier debugging
    and development.
    """

    def __init__(self, args: DialConfig, env):
        self.args = args
        self.env = env
        self.nu = env.action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Add option to choose interpolation method
        self.use_jax_spline = getattr(args, 'use_jax_spline', True)  # Default to JAX spline

        # Create trajectory gradient sampling configuration
        self.traj_cfg = TrajGradSamplingCfg()
        self.traj_cfg.trajectory_opt.horizon_samples = args.Hsample
        self.traj_cfg.trajectory_opt.horizon_nodes = args.Hnode
        self.traj_cfg.trajectory_opt.num_samples = args.Nsample
        self.traj_cfg.trajectory_opt.num_diffuse_steps = args.Ndiffuse
        self.traj_cfg.trajectory_opt.num_diffuse_steps_init = args.Ndiffuse_init
        self.traj_cfg.trajectory_opt.temp_sample = args.temp_sample
        self.traj_cfg.trajectory_opt.horizon_diffuse_factor = args.horizon_diffuse_factor
        self.traj_cfg.trajectory_opt.traj_diffuse_factor = args.traj_diffuse_factor
        self.traj_cfg.trajectory_opt.update_method = args.update_method
        self.traj_cfg.env.num_actions = self.nu

        # Initialize trajectory gradient sampling module
        # We use a single main environment for testing
        num_envs = 1
        main_env_indices = [0]
        dt = 0.02  # Control timestep

        # Choose the appropriate TrajGradSampling class based on interpolation method
        if self.use_jax_spline:
            self.traj_sampler = JAXSplineTrajGradSampling(
                cfg=self.traj_cfg,
                device=self.device,
                num_envs=num_envs,
                num_actions=self.nu,
                dt=dt,
                main_env_indices=main_env_indices,
                args=args
            )
            traj_grad_sampler = "JAXSplineTrajGradSampling"
        else:
            self.traj_sampler = TrajGradSampling(
                cfg=self.traj_cfg,
                device=self.device,
                num_envs=num_envs,
                num_actions=self.nu,
                dt=dt,
                main_env_indices=main_env_indices
            )
            traj_grad_sampler = "TrajGradSampling"

        # Apply JAX JIT compilation to environment rollout functions (like in original dial_core.py)
        self.rollout_us = jax.jit(functools.partial(rollout_us, self.env.step))
        self.rollout_us_vmap = jax.jit(jax.vmap(self.rollout_us, in_axes=(None, 0)))

        print(f"MBDPITest initialized with PyTorch trajectory optimization")
        print(f"Device: {self.device}")
        print(f"Traj Grad Sampler: {traj_grad_sampler}")
        print(f"Horizon samples: {args.Hsample}, Horizon nodes: {args.Hnode}")
        print(f"Num samples: {args.Nsample}, Update method: {args.update_method}")
        print(f"JAX JIT compiled rollout functions for improved performance")

    def jax_to_torch(self, jax_array):
        """Convert JAX array to PyTorch tensor."""
        if isinstance(jax_array, torch.Tensor):
            return jax_array.to(self.device)
        return torch.from_numpy(np.array(jax_array)).to(self.device)

    def torch_to_jax(self, torch_tensor):
        """Convert PyTorch tensor to JAX array."""
        if not isinstance(torch_tensor, torch.Tensor):
            return jnp.array(torch_tensor)
        return jnp.array(torch_tensor.cpu().numpy())

    def create_rollout_callback(self, state):
        """Create a rollout callback function for trajectory optimization.

        Args:
            state: Current JAX environment state

        Returns:
            Callback function that can evaluate trajectory batches
        """
        def rollout_callback(us_batch_torch):
            """Evaluate a batch of control sequences efficiently using JAX vectorization.

            Args:
                us_batch_torch: Batch of control sequences [batch_size, horizon, action_dim]

            Returns:
                Batch of trajectory rewards [batch_size, horizon]
            """
            # Convert entire batch to JAX at once (more efficient)
            us_batch_jax = self.torch_to_jax(us_batch_torch)

            # Use JAX vectorized rollout for efficient batch evaluation
            # This is much faster than looping over individual trajectories
            rews_batch, _, _ = self.rollout_us_vmap(state, us_batch_jax)

            # Convert results back to PyTorch
            rewards_batch = self.jax_to_torch(rews_batch)
            return rewards_batch

        return rollout_callback

    def reverse_once(self, state, Ybar_i_jax, noise_scale_jax):
        """Perform one reverse diffusion step using PyTorch trajectory optimization.

        Args:
            state: JAX environment state
            Ybar_i_jax: Current mean trajectory in JAX format
            noise_scale_jax: Noise scale in JAX format

        Returns:
            Updated trajectory and info dictionary
        """
        # Convert JAX inputs to PyTorch
        Ybar_i_torch = self.jax_to_torch(Ybar_i_jax).unsqueeze(0)  # Add batch dimension
        noise_scale_torch = self.jax_to_torch(noise_scale_jax)

        # Create rollout callback
        rollout_callback = self.create_rollout_callback(state)

        # Use trajectory gradient sampling to optimize
        updated_traj_torch = self.traj_sampler.eval_all_traj_grad(
            mean_trajs=Ybar_i_torch,
            rollout_callback=rollout_callback,
            noise_scale=noise_scale_torch,
            n_samples=self.args.Nsample
        )

        # Convert back to JAX
        Ybar_updated = self.torch_to_jax(updated_traj_torch.squeeze(0))

        # Evaluate the updated trajectory to get additional info
        us_updated = self.traj_sampler.node2u(updated_traj_torch.squeeze(0))
        us_updated_jax = self.torch_to_jax(us_updated)
        rews, pipeline_states, metrics_seq = self.rollout_us(state, us_updated_jax)

        # Extract state information for visualization
        qbar = pipeline_states.q
        qdbar = pipeline_states.qd
        xbar = pipeline_states.x.pos

        # Process reward components from metrics if available
        mean_reward_components = {}
        if isinstance(metrics_seq, dict) and len(metrics_seq) > 0:
            reward_keys = [k for k in metrics_seq.keys() if k.startswith('reward_')]
            for key in reward_keys:
                component_values = metrics_seq[key]
                mean_reward_components[key] = component_values[0]  # Get first timestep

        # Create info dictionary
        info = {
            "rews": rews,
            "qbar": qbar,
            "qdbar": qdbar,
            "xbar": xbar,
            "new_noise_scale": noise_scale_jax,  # Keep original noise scale
            "mean_traj_reward": rews.mean(),
            "reward_components": mean_reward_components,
        }
        return Ybar_updated, info

    def reverse(self, state, YN, rng=None):
        """Run the full reverse diffusion process using PyTorch trajectory optimization.

        Args:
            state: JAX environment state
            YN: Initial trajectory (typically zeros)
            rng: Random number generator (unused in PyTorch version)

        Returns:
            Optimized trajectory
        """
        Yi = YN

        # Use PyTorch trajectory optimization for the full reverse process
        Yi_torch = self.jax_to_torch(Yi).unsqueeze(0)  # Add batch dimension
        rollout_callback = self.create_rollout_callback(state)

        # Perform trajectory optimization with the specified number of diffusion steps
        n_diffuse = self.args.Ndiffuse
        self.traj_sampler.optimize_all_trajectories(
            rollout_callback=rollout_callback,
            n_diffuse=n_diffuse,
            initial=False
        )

        # Get the optimized trajectory
        optimized_traj = self.traj_sampler.node_trajectories[0]  # Get first (and only) trajectory
        Yi_optimized = self.torch_to_jax(optimized_traj)

        return Yi_optimized

    def shift(self, Y):
        """Shift trajectory by one timestep.

        Args:
            Y: Trajectory to shift

        Returns:
            Shifted trajectory
        """
        # Convert to PyTorch, shift using the sampler, and convert back
        Y_torch = self.jax_to_torch(Y)
        Y_shifted_torch = self.traj_sampler.shift_nodetraj_batch(
            Y_torch.unsqueeze(0), n_steps=1
        ).squeeze(0)
        return self.torch_to_jax(Y_shifted_torch)

    def shift_Y_from_u(self, u, n_step):
        """Shift trajectory from control sequence.

        Args:
            u: Control sequence
            n_step: Number of steps to shift

        Returns:
            Shifted trajectory in node representation
        """
        # Convert to PyTorch
        u_torch = self.jax_to_torch(u)

        # Shift the control sequence
        u_shifted = torch.roll(u_torch, -n_step, dims=0)
        u_shifted[-n_step:] = 0.0  # Zero out the last n_step controls

        # Convert to node representation
        Y_shifted_torch = self.traj_sampler.u2node(u_shifted)

        return self.torch_to_jax(Y_shifted_torch)


def main():
    """Main function implementing the same logic as the original JAX version."""

    def reverse_scan(state_Y0, factor):
        """Scan function for diffusion steps."""
        state, Y0 = state_Y0
        Y0, info = mbdpi.reverse_once(state, Y0, factor)
        return (state, Y0), info

    art.tprint("LeCAR @ CMU\nDIAL-MPC", font="big", chr_ignore=True)
    parser = argparse.ArgumentParser()
    config_or_example = parser.add_mutually_exclusive_group(required=True)
    config_or_example.add_argument("--config", type=str, default=None)
    config_or_example.add_argument("--example", type=str, default=None)
    config_or_example.add_argument("--list-examples", action="store_true")
    parser.add_argument(
        "--custom-env",
        type=str,
        default=None,
        help="Custom environment to import dynamically",
    )
    args = parser.parse_args()

    if args.list_examples:
        print("Examples:")
        for example in examples:
            print(f"  {example}")
        return

    if args.custom_env is not None:
        sys.path.append(os.getcwd())
        importlib.import_module(args.custom_env)

    if args.example is not None:
        config_dict = yaml.safe_load(open(get_example_path(args.example + ".yaml")))
    else:
        config_dict = yaml.safe_load(open(args.config))

    dial_config = load_dataclass_from_dict(DialConfig, config_dict)

    # Find env config
    env_config_type = dial_envs.get_config(dial_config.env_name)
    env_config = load_dataclass_from_dict(
        env_config_type, config_dict, convert_list_to_array=True
    )

    print(emoji.emojize(":rocket:") + "Creating environment")
    env = brax_envs.get_environment(dial_config.env_name, config=env_config)
    reset_env = jax.jit(env.reset)
    step_env = jax.jit(env.step)

    # Create MBDPITest instead of MBDPI
    mbdpi = MBDPITest(dial_config, env)

    rng = jax.random.PRNGKey(seed=dial_config.seed)
    rng, rng_reset = jax.random.split(rng)
    state_init = reset_env(rng_reset)

    YN = jnp.zeros([dial_config.Hnode + 1, mbdpi.nu])
    Y0 = YN

    Nstep = dial_config.n_steps
    rews = []
    rews_plan = []
    mean_traj_rewards = []  # Store mean trajectory rewards for plotting
    reward_components_history = {}  # Store history of all reward components
    rollout = []
    state = state_init
    us = []
    infos = []

    with tqdm(range(Nstep), desc="Rollout") as pbar:
        for t in pbar:
            # Forward single step
            state = step_env(state, Y0[0])
            rollout.append(state.pipeline_state)
            rews.append(state.reward)
            us.append(Y0[0])

            # Update Y0
            Y0 = mbdpi.shift(Y0)

            n_diffuse = dial_config.Ndiffuse
            if t == 0:
                n_diffuse = dial_config.Ndiffuse_init
                print("Performing initial optimization with PyTorch trajectory sampling")

            t0 = time.time()

            # Create noise factors for diffusion steps
            traj_diffuse_factors = []
            for i in range(n_diffuse):
                factor = mbdpi.traj_sampler.sigma_control * (dial_config.traj_diffuse_factor ** i)
                traj_diffuse_factors.append(factor)

            # Perform diffusion steps
            current_state = (state, Y0)
            for i, factor in enumerate(traj_diffuse_factors):
                current_state, info = reverse_scan(current_state, factor)

            # Extract the final optimized trajectory
            _, Y0 = current_state

            # Collect mean trajectory reward for plotting
            final_mean_traj_reward = float(info["mean_traj_reward"])
            mean_traj_rewards.append(final_mean_traj_reward)

            # Collect reward components for plotting
            final_reward_components = info["reward_components"]
            for component_name, component_value in final_reward_components.items():
                if component_name not in reward_components_history:
                    reward_components_history[component_name] = []
                reward_components_history[component_name].append(float(component_value))

            rews_plan.append(info["rews"].mean())
            infos.append(info)
            freq = 1 / (time.time() - t0)
            pbar.set_postfix({"rew": f"{state.reward:.2e}", "freq": f"{freq:.2f}"})

    rew = jnp.array(rews).mean()
    print(f"mean reward = {rew:.2e}")

    # Create result dir if not exist
    if not os.path.exists(dial_config.output_dir):
        os.makedirs(dial_config.output_dir)

    # Plot mean trajectory rewards
    plt.rcParams['text.usetex'] = False
    plt.figure(figsize=(10, 6))
    plt.plot(mean_traj_rewards, 'b-', linewidth=2, label='Mean Trajectory Reward')
    plt.plot(rews, 'r--', alpha=0.7, label='Actual Environment Reward')
    plt.xlabel('Time Step')
    plt.ylabel('Reward')
    plt.title('Mean Trajectory Rewards vs Actual Environment Rewards (PyTorch Test)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(dial_config.output_dir, "reward_comparison.pdf"))
    plt.savefig(os.path.join(dial_config.output_dir, "reward_comparison.png"), dpi=300)
    print(f"Reward plot saved to {dial_config.output_dir}/reward_comparison.pdf")

    # Plot individual reward components if available
    if reward_components_history:
        # Create a subplot for each reward component
        n_components = len(reward_components_history)
        if n_components > 0:
            n_cols = min(3, n_components)
            n_rows = (n_components + n_cols - 1) // n_cols

            plt.figure(figsize=(15, 5 * n_rows))
            for i, (component_name, values) in enumerate(reward_components_history.items()):
                plt.subplot(n_rows, n_cols, i + 1)
                plt.plot(values, 'g-', linewidth=2)
                plt.xlabel('Time Step')
                plt.ylabel('Reward Value')
                plt.title(f'{component_name.replace("reward_", "").title()} Reward')
                plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(dial_config.output_dir, "reward_components.pdf"))
            plt.savefig(os.path.join(dial_config.output_dir, "reward_components.png"), dpi=300)
            print(f"Reward components plot saved to {dial_config.output_dir}/reward_components.pdf")

            # Also create a single plot with all components
            plt.figure(figsize=(12, 8))
            # Use a safe colormap that definitely exists
            colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(reward_components_history)))
            for i, (component_name, values) in enumerate(reward_components_history.items()):
                clean_name = component_name.replace("reward_", "").title()
                plt.plot(values, color=colors[i], linewidth=2, label=clean_name, alpha=0.8)

            plt.xlabel('Time Step')
            plt.ylabel('Reward Value')
            plt.title('All Mean Trajectory Reward Components Over Time (PyTorch Test)')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(dial_config.output_dir, "all_reward_components.pdf"))
            plt.savefig(os.path.join(dial_config.output_dir, "all_reward_components.png"), dpi=300)
            print(f"All reward components plot saved to {dial_config.output_dir}/all_reward_components.pdf")

    # Also plot just the mean trajectory rewards
    plt.figure(figsize=(10, 6))
    plt.plot(mean_traj_rewards, 'b-', linewidth=2)
    plt.xlabel('Time Step')
    plt.ylabel('Mean Trajectory Reward')
    plt.title('Mean Trajectory Rewards During Optimization (PyTorch Test)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(dial_config.output_dir, "mean_traj_rewards.pdf"))
    plt.savefig(os.path.join(dial_config.output_dir, "mean_traj_rewards.png"), dpi=300)
    print(f"Mean trajectory rewards plot saved to {dial_config.output_dir}/mean_traj_rewards.pdf")

    # Host webpage with flask
    print("Processing rollout for visualization")
    import flask

    app = flask.Flask(__name__)

    # Get environment system and timestep properly with better fallback handling
    try:
        if hasattr(env, 'sys'):
            env_sys = env.sys
        elif hasattr(env, '_env') and hasattr(env._env, 'sys'):
            env_sys = env._env.sys
        else:
            # Try to get from the underlying environment
            env_sys = getattr(env, 'sys', None)
            if env_sys is None:
                print("Warning: Could not find environment system, using basic visualization")
                env_sys = None
    except:
        env_sys = None

    # Get timestep with fallback
    try:
        if hasattr(env, 'dt'):
            env_dt = env.dt
        else:
            env_dt = 0.02  # Standard 50Hz control rate
    except:
        env_dt = 0.02

    if env_sys is not None:
        webpage = html.render(
            env_sys.tree_replace({"opt.timestep": env_dt}), rollout, 1080, True
        )
    else:
        # Create a simple fallback webpage
        webpage = "<html><body><h1>Rollout completed but visualization not available</h1></body></html>"

    # Save the html file
    with open(
        os.path.join(dial_config.output_dir, "brax_visualization.html"),
        "w",
    ) as f:
        f.write(webpage)

    # Save the rollout
    data = []
    xdata = []
    for i in range(len(rollout)):
        pipeline_state = rollout[i]
        data.append(
            jnp.concatenate(
                [
                    jnp.array([i]),
                    pipeline_state.qpos,
                    pipeline_state.qvel,
                    pipeline_state.ctrl,
                ]
            )
        )
        if i < len(infos):
            xdata.append(infos[i]["xbar"][-1])
        else:
            # Fallback for missing xdata
            xdata.append(jnp.zeros((3,)))

    data = jnp.array(data)
    xdata = jnp.array(xdata)
    jnp.save(os.path.join(dial_config.output_dir, "states"), data)
    jnp.save(os.path.join(dial_config.output_dir, "predictions"), xdata)

    print(f"Visualization and data saved to {dial_config.output_dir}/")
    print("PyTorch trajectory optimization test completed successfully!")

    @app.route("/")
    def index():
        return webpage

    app.run(port=5000)


if __name__ == "__main__":
    main()
