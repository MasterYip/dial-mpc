import os
import sys
import time
import yaml
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch
import emoji
import art

# Add the legged_gym_cmp path to import the trajectory gradient sampling module
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../legged_gym_cmp"))

from legged_gym.utils.traj_grad_sampling import TrajGradSampling, TrajGradSamplingCfg

from dial_mpc.core.dial_config import DialConfig
from dial_mpc.examples import examples
from dial_mpc.utils.io_utils import get_example_path, load_dataclass_from_dict
import dial_mpc.envs as dial_envs
import brax.envs as brax_envs
from brax.io import html
from jax import numpy as jnp
import jax
import functools

matplotlib.use('Agg')  # Use non-interactive backend
plt.rcParams['text.usetex'] = False  # Disable LaTeX
plt.rcParams['font.family'] = 'DejaVu Sans'  # Use a standard font

# Set matplotlib to not use LaTeX and use a safe style
try:
    plt.style.use("science")
except:
    # Fallback to default style if science plots style fails
    plt.style.use("default")


def rollout_us(step_env, state, us):
    """Rollout control sequence using JAX environment."""
    def step(state, u):
        state = step_env(state, u)
        return state, (state.reward, state.pipeline_state, state.metrics if hasattr(state, 'metrics') else {})

    _, (rews, pipeline_states, metrics_seq) = jax.lax.scan(step, state, us)
    return rews, pipeline_states, metrics_seq


class MBDPITest:
    """Test implementation of MBDPI using PyTorch trajectory gradient sampling."""
    
    def __init__(self, args: DialConfig, env):
        self.args = args
        self.env = env
        self.nu = env.action_size
        
        # Create trajectory gradient sampling configuration
        self.traj_cfg = TrajGradSamplingCfg()
        
        # Map DIAL config to trajectory optimization config
        self.traj_cfg.trajectory_opt.enable_traj_opt = True
        self.traj_cfg.trajectory_opt.horizon_samples = args.Hsample
        self.traj_cfg.trajectory_opt.horizon_nodes = args.Hnode
        self.traj_cfg.trajectory_opt.num_samples = args.Nsample
        self.traj_cfg.trajectory_opt.num_diffuse_steps = args.Ndiffuse
        self.traj_cfg.trajectory_opt.num_diffuse_steps_init = args.Ndiffuse_init
        self.traj_cfg.trajectory_opt.temp_sample = args.temp_sample
        self.traj_cfg.trajectory_opt.horizon_diffuse_factor = args.horizon_diffuse_factor
        self.traj_cfg.trajectory_opt.traj_diffuse_factor = args.traj_diffuse_factor
        self.traj_cfg.trajectory_opt.update_method = args.update_method
        self.traj_cfg.trajectory_opt.interp_method = "linear"
        self.traj_cfg.trajectory_opt.compute_predictions = True
        
        # Disable RL warmstart as requested
        self.traj_cfg.rl_warmstart.enable = False
        
        # Initialize trajectory gradient sampling
        # Use CPU/CUDA device based on availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # For this test, we'll use a single environment (no batch)
        self.num_envs = 1
        self.main_env_indices = [0]
        
        # Time step (should match environment dt)
        self.ctrl_dt = 0.02
        
        # Initialize trajectory gradient sampling module
        self.traj_sampler = TrajGradSampling(
            cfg=self.traj_cfg,
            device=self.device,
            num_envs=self.num_envs,
            num_actions=self.nu,
            dt=self.ctrl_dt,
            main_env_indices=self.main_env_indices
        )
        
        # JAX rollout functions (keep original for environment interaction)
        self.rollout_us = jax.jit(functools.partial(rollout_us, self.env.step))
        self.rollout_us_vmap = jax.jit(jax.vmap(self.rollout_us, in_axes=(None, 0)))
        
        print(f"Trajectory optimization initialized with {args.Hsample} horizon samples, {args.Hnode} nodes")
        print(f"Using {args.Nsample} samples for optimization")

    def rollout_callback(self, us_batch_torch):
        """Callback function for trajectory rollout evaluation.
        
        Args:
            us_batch_torch: Batch of control sequences as PyTorch tensor [batch_size, horizon+1, action_dim]
            
        Returns:
            rewards_batch: Batch of trajectory rewards as PyTorch tensor [batch_size, horizon+1]
        """
        # Convert PyTorch tensor to JAX array
        us_batch_jax = jnp.array(us_batch_torch.detach().cpu().numpy())
        
        # Since we're using a single environment, we need to evaluate trajectories sequentially
        batch_size = us_batch_jax.shape[0]
        rewards_list = []
        
        # Get current state for rollout (use the stored state)
        current_state = self.current_state
        
        for i in range(batch_size):
            us_i = us_batch_jax[i]
            rews_i, _, _ = self.rollout_us(current_state, us_i)
            rewards_list.append(rews_i)
        
        # Stack rewards and convert back to PyTorch
        rewards_batch_jax = jnp.stack(rewards_list, axis=0)
        rewards_batch_torch = torch.from_numpy(np.array(rewards_batch_jax)).to(self.device)
        
        return rewards_batch_torch

    def optimize_trajectory(self, state, current_traj, initial=False):
        """Optimize trajectory using trajectory gradient sampling.
        
        Args:
            state: Current environment state
            current_traj: Current trajectory as JAX array [horizon_nodes+1, action_dim]
            initial: Whether this is initial optimization
            
        Returns:
            optimized_traj: Optimized trajectory as JAX array [horizon_nodes+1, action_dim]
            info: Information dictionary
        """
        # Store current state for rollout callback
        self.current_state = state
        
        # Convert JAX trajectory to PyTorch and store in trajectory sampler
        current_traj_torch = torch.from_numpy(np.array(current_traj)).to(self.device)
        self.traj_sampler.node_trajectories[0] = current_traj_torch
        
        # Perform trajectory optimization
        n_diffuse = self.args.Ndiffuse_init if initial else self.args.Ndiffuse
        
        mean_traj_rewards = []
        reward_components_history = {}
        
        # Get current trajectories for optimization (single environment)
        curr_trajs = self.traj_sampler.node_trajectories[0:1]  # [1, horizon_nodes+1, action_dim]
        
        # Perform diffusion steps
        for i in range(n_diffuse):
            # Calculate noise scale for this diffusion step
            noise_scale = self.traj_sampler.sigma_control * (self.traj_sampler.traj_diffuse_factor ** i)
            
            # Perform gradient evaluation and update
            curr_trajs = self.traj_sampler.eval_all_traj_grad(
                curr_trajs, 
                self.rollout_callback, 
                noise_scale
            )
            
            # Evaluate mean trajectory for monitoring
            mean_traj_torch = curr_trajs[0]
            mean_us_torch = self.traj_sampler.node2u(mean_traj_torch)
            mean_rewards = self.rollout_callback(mean_us_torch.unsqueeze(0))
            mean_traj_reward = float(mean_rewards[0].mean())
            mean_traj_rewards.append(mean_traj_reward)
        
        # Update stored trajectory
        self.traj_sampler.node_trajectories[0] = curr_trajs[0]
        
        # Convert back to JAX for return
        optimized_traj_jax = jnp.array(curr_trajs[0].detach().cpu().numpy())
        
        # Create info dictionary
        info = {
            "mean_traj_reward": mean_traj_rewards[-1] if mean_traj_rewards else 0.0,
            "reward_components": reward_components_history,
            "optimization_steps": n_diffuse
        }
        
        return optimized_traj_jax, info

    def shift_trajectory(self, Y):
        """Shift trajectory by one time step."""
        # Convert to PyTorch
        Y_torch = torch.from_numpy(np.array(Y)).to(self.device)
        
        # Convert to dense control sequence
        u_torch = self.traj_sampler.node2u(Y_torch)
        
        # Shift by one step
        u_shifted = torch.roll(u_torch, -1, dims=0)
        u_shifted[-1] = torch.zeros_like(u_shifted[-1])
        
        # Convert back to nodes
        Y_shifted = self.traj_sampler.u2node(u_shifted)
        
        # Convert back to JAX
        return jnp.array(Y_shifted.detach().cpu().numpy())


def main():
    """Main function for DIAL-MPC test with trajectory gradient sampling."""
    
    art.tprint("LeCAR @ CMU\nDIAL-MPC-TEST", font="big", chr_ignore=True)
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
        import importlib
        importlib.import_module(args.custom_env)

    if args.example is not None:
        config_dict = yaml.safe_load(open(get_example_path(args.example + ".yaml")))
    else:
        config_dict = yaml.safe_load(open(args.config))

    dial_config = load_dataclass_from_dict(DialConfig, config_dict)
    rng = jax.random.PRNGKey(seed=dial_config.seed)

    # Find env config
    env_config_type = dial_envs.get_config(dial_config.env_name)
    env_config = load_dataclass_from_dict(
        env_config_type, config_dict, convert_list_to_array=True
    )

    print(emoji.emojize(":rocket:") + "Creating environment")
    env = brax_envs.get_environment(dial_config.env_name, config=env_config)
    reset_env = jax.jit(env.reset)
    step_env = jax.jit(env.step)
    
    # Initialize test MBDPI
    mbdpi_test = MBDPITest(dial_config, env)

    rng, rng_reset = jax.random.split(rng)
    state_init = reset_env(rng_reset)

    # Initialize trajectory
    Y0 = jnp.zeros([dial_config.Hnode + 1, mbdpi_test.nu])

    Nstep = dial_config.n_steps
    rews = []
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

            # Shift trajectory
            Y0 = mbdpi_test.shift_trajectory(Y0)

            # Optimize trajectory
            initial = (t == 0)
            if initial:
                print("Performing initial trajectory optimization (no JIT compilation needed)")

            t0 = time.time()
            Y0, info = mbdpi_test.optimize_trajectory(state, Y0, initial=initial)

            # Collect mean trajectory reward for plotting
            mean_traj_rewards.append(info["mean_traj_reward"])

            # Collect reward components for plotting
            for component_name, component_value in info["reward_components"].items():
                if component_name not in reward_components_history:
                    reward_components_history[component_name] = []
                reward_components_history[component_name].append(component_value)

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
    plt.plot(mean_traj_rewards, 'b-', linewidth=2, label='Mean Trajectory Reward (Test)')
    plt.plot(rews, 'r--', alpha=0.7, label='Actual Environment Reward')
    plt.xlabel('Time Step')
    plt.ylabel('Reward')
    plt.title('Mean Trajectory Rewards vs Actual Environment Rewards (Test Implementation)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(dial_config.output_dir, "test_reward_comparison.pdf"))
    plt.savefig(os.path.join(dial_config.output_dir, "test_reward_comparison.png"), dpi=300)
    print(f"Test reward plot saved to {dial_config.output_dir}/test_reward_comparison.pdf")

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
                plt.title(f'{component_name.replace("reward_", "").title()} Reward (Test)')
                plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(dial_config.output_dir, "test_reward_components.pdf"))
            plt.savefig(os.path.join(dial_config.output_dir, "test_reward_components.png"), dpi=300)
            print(f"Test reward components plot saved to {dial_config.output_dir}/test_reward_components.pdf")

    # Also plot just the mean trajectory rewards
    plt.figure(figsize=(10, 6))
    plt.plot(mean_traj_rewards, 'b-', linewidth=2)
    plt.xlabel('Time Step')
    plt.ylabel('Mean Trajectory Reward')
    plt.title('Mean Trajectory Rewards During Optimization (Test Implementation)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(dial_config.output_dir, "test_mean_traj_rewards.pdf"))
    plt.savefig(os.path.join(dial_config.output_dir, "test_mean_traj_rewards.png"), dpi=300)
    print(f"Test mean trajectory rewards plot saved to {dial_config.output_dir}/test_mean_traj_rewards.pdf")

    # Host webpage with flask (same as original)
    print("Processing rollout for visualization")
    import flask

    app = flask.Flask(__name__)
    
    # Try to access env attributes safely
    try:
        webpage = html.render(
            env.sys.tree_replace({"opt.timestep": env.dt}), rollout, 1080, True
        )
    except AttributeError:
        # Fallback if env doesn't have sys or dt attributes
        try:
            webpage = html.render(env, rollout, 1080, True)
        except:
            print("Warning: Could not generate HTML visualization")
            webpage = "<html><body>Visualization not available</body></html>"

    # Save the html file
    with open(
        os.path.join(dial_config.output_dir, "test_brax_visualization.html"),
        "w",
    ) as f:
        f.write(webpage)

    # Save the rollout
    data = []
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
    data = jnp.array(data)
    jnp.save(os.path.join(dial_config.output_dir, "test_states"), data)

    print(f"Test visualization and data saved to {dial_config.output_dir}/")

    @app.route("/")
    def index():
        return webpage

    print("Starting Flask server on port 5001 (test implementation)")
    app.run(port=5001)


if __name__ == "__main__":
    main()