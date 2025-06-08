from dial_mpc.core.dial_config import DialConfig
from dial_mpc.examples import examples
from dial_mpc.utils.io_utils import get_example_path, load_dataclass_from_dict
import dial_mpc.envs as dial_envs
import brax.envs as brax_envs
from brax.io import html
from jax import debug
import functools
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline
from jax import numpy as jnp
import jax
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
matplotlib.use('Agg')  # Use non-interactive backend
plt.rcParams['text.usetex'] = False  # Disable LaTeX
plt.rcParams['font.family'] = 'DejaVu Sans'  # Use a standard font


# Set matplotlib to not use LaTeX and use a safe style
try:
    plt.style.use("science")
except:
    # Fallback to default style if science plots style fails
    plt.style.use("default")

# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags


def rollout_us(step_env, state, us):
    def step(state, u):
        state = step_env(state, u)
        return state, (state.reward, state.pipeline_state, state.metrics if hasattr(state, 'metrics') else {})

    _, (rews, pipline_states, metrics_seq) = jax.lax.scan(step, state, us)
    return rews, pipline_states, metrics_seq

# def scan(f, init, xs, length=None):
#   if xs is None:
#     xs = [None] * length
#   carry = init
#   ys = []
#   for x in xs:
#     carry, y = f(carry, x)
#     ys.append(y)
#   return carry, np.stack(ys)


@jax.jit
def softmax_update(weights, Y0s, sigma, mu_0t):
    mu_0tm1 = jnp.einsum("n,nij->ij", weights, Y0s)
    return mu_0tm1, sigma


class MBDPI:
    def __init__(self, args: DialConfig, env):
        self.args = args
        self.env = env
        self.nu = env.action_size

        self.update_fn = {
            "mppi": softmax_update,
        }[args.update_method]

        sigma0 = 1e-2
        sigma1 = 1.0
        A = sigma0
        B = jnp.log(sigma1 / sigma0) / args.Ndiffuse
        self.sigmas = A * jnp.exp(B * jnp.arange(args.Ndiffuse))
        self.sigma_control = (
            args.horizon_diffuse_factor ** jnp.arange(args.Hnode + 1)[::-1]
        )
        print(f"sigma_control = {self.sigma_control}")
        print(f"sigmas = {self.sigmas}")

        # node to u
        self.ctrl_dt = 0.02
        self.step_us = jnp.linspace(0, self.ctrl_dt * args.Hsample, args.Hsample + 1)
        self.step_nodes = jnp.linspace(0, self.ctrl_dt * args.Hsample, args.Hnode + 1)
        self.node_dt = self.ctrl_dt * (args.Hsample) / (args.Hnode)

        # setup function
        self.rollout_us = jax.jit(functools.partial(rollout_us, self.env.step))
        self.rollout_us_vmap = jax.jit(jax.vmap(self.rollout_us, in_axes=(None, 0)))
        self.node2u_vmap = jax.jit(
            jax.vmap(self.node2u, in_axes=(1), out_axes=(1))
        )  # process (horizon, node)
        self.u2node_vmap = jax.jit(jax.vmap(self.u2node, in_axes=(1), out_axes=(1)))
        self.node2u_vvmap = jax.jit(
            jax.vmap(self.node2u_vmap, in_axes=(0))
        )  # process (batch, horizon, node)
        self.u2node_vvmap = jax.jit(jax.vmap(self.u2node_vmap, in_axes=(0)))

    @functools.partial(jax.jit, static_argnums=(0,))
    def node2u(self, nodes):
        spline = InterpolatedUnivariateSpline(self.step_nodes, nodes, k=2)
        us = spline(self.step_us)
        return us

    @functools.partial(jax.jit, static_argnums=(0,))
    def u2node(self, us):
        spline = InterpolatedUnivariateSpline(self.step_us, us, k=2)
        nodes = spline(self.step_nodes)
        return nodes

    @functools.partial(jax.jit, static_argnums=(0,))
    def reverse_once(self, state, rng, Ybar_i, noise_scale):
        # sample from q_i
        rng, Y0s_rng = jax.random.split(rng)
        eps_Y = jax.random.normal(
            Y0s_rng, (self.args.Nsample, self.args.Hnode + 1, self.nu)
        )
        Y0s = eps_Y * noise_scale[None, :, None] + Ybar_i
        # we can't change the first control
        Y0s = Y0s.at[:, 0].set(Ybar_i[0, :])
        # append Y0s with Ybar_i to also evaluate Ybar_i
        Y0s = jnp.concatenate([Y0s, Ybar_i[None]], axis=0)
        Y0s = jnp.clip(Y0s, -1.0, 1.0)
        # convert Y0s to us
        us = self.node2u_vvmap(Y0s)

        # esitimate mu_0tm1
        rewss, pipeline_statess, metrics_seq = self.rollout_us_vmap(state, us)
        rew_Ybar_i = rewss[-1].mean()
        qss = pipeline_statess.q
        qdss = pipeline_statess.qd
        xss = pipeline_statess.x.pos
        rews = rewss.mean(axis=-1)
        logp0 = (rews - rew_Ybar_i) / rews.std(axis=-1) / self.args.temp_sample

        weights = jax.nn.softmax(logp0)
        Ybar, new_noise_scale = self.update_fn(weights, Y0s, noise_scale, Ybar_i)

        # NOTE: update only with reward
        Ybar = jnp.einsum("n,nij->ij", weights, Y0s)
        qbar = jnp.einsum("n,nij->ij", weights, qss)
        qdbar = jnp.einsum("n,nij->ij", weights, qdss)
        xbar = jnp.einsum("n,nijk->ijk", weights, xss)

        # Process reward sub-terms from metrics if available
        mean_reward_components = {}
        # metrics_seq is a dict where each key maps to arrays of shape (n_trajectories, n_timesteps)

        if isinstance(metrics_seq, dict) and len(metrics_seq) > 0:
            # Get reward component keys
            reward_keys = [k for k in metrics_seq.keys() if k.startswith('reward_')]
            for key in reward_keys:
                # metrics_seq[key] has shape (n_trajectories, n_timesteps)
                component_values = metrics_seq[key]
                # Compute weighted mean across trajectories, then mean across timesteps
                weighted_component = jnp.einsum("n,nt->t", weights, component_values).mean()
                mean_reward_components[key] = weighted_component


        info = {
            "rews": rews,
            "qbar": qbar,
            "qdbar": qdbar,
            "xbar": xbar,
            "new_noise_scale": new_noise_scale,
            "mean_traj_reward": rews.mean(),
            "reward_components": mean_reward_components,  # Store reward sub-terms
        }

        return rng, Ybar, info

    def reverse(self, state, YN, rng):
        Yi = YN
        with tqdm(range(self.args.Ndiffuse - 1, 0, -1), desc="Diffusing") as pbar:
            for i in pbar:
                t0 = time.time()
                rng, Yi, rews = self.reverse_once(
                    state, rng, Yi, self.sigmas[i] * jnp.ones(self.args.Hnode + 1)
                )
                Yi.block_until_ready()
                freq = 1 / (time.time() - t0)
                pbar.set_postfix({"rew": f"{rews['mean_traj_reward']:.2e}", "freq": f"{freq:.2f}"})
        return Yi

    @functools.partial(jax.jit, static_argnums=(0,))
    def shift(self, Y):
        u = self.node2u_vmap(Y)
        u = jnp.roll(u, -1, axis=0)
        u = u.at[-1].set(jnp.zeros(self.nu))
        Y = self.u2node_vmap(u)
        return Y

    def shift_Y_from_u(self, u, n_step):
        u = jnp.roll(u, -n_step, axis=0)
        u = u.at[-n_step:].set(jnp.zeros_like(u[-n_step:]))
        Y = self.u2node_vmap(u)
        return Y


def main():

    def reverse_scan(rng_Y0_state, factor):
        rng, Y0, state = rng_Y0_state
        rng, Y0, info = mbdpi.reverse_once(state, rng, Y0, factor)
        return (rng, Y0, state), info

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
    rng = jax.random.PRNGKey(seed=dial_config.seed)

    # find env config
    env_config_type = dial_envs.get_config(dial_config.env_name)
    env_config = load_dataclass_from_dict(
        env_config_type, config_dict, convert_list_to_array=True
    )

    print(emoji.emojize(":rocket:") + "Creating environment")
    env = brax_envs.get_environment(dial_config.env_name, config=env_config)
    reset_env = jax.jit(env.reset)
    step_env = jax.jit(env.step)
    mbdpi = MBDPI(dial_config, env)

    rng, rng_reset = jax.random.split(rng)
    state_init = reset_env(rng_reset)

    YN = jnp.zeros([dial_config.Hnode + 1, mbdpi.nu])

    rng_exp, rng = jax.random.split(rng)
    # Y0 = mbdpi.reverse(state_init, YN, rng_exp)
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
            # forward single step
            state = step_env(state, Y0[0])
            rollout.append(state.pipeline_state)
            rews.append(state.reward)
            us.append(Y0[0])

            # update Y0
            Y0 = mbdpi.shift(Y0)

            n_diffuse = dial_config.Ndiffuse
            if t == 0:
                n_diffuse = dial_config.Ndiffuse_init
                print("Performing JIT on DIAL-MPC")

            t0 = time.time()
            traj_diffuse_factors = (
                mbdpi.sigma_control * dial_config.traj_diffuse_factor ** (jnp.arange(n_diffuse))[:, None]
            )
            (rng, Y0, _), info = jax.lax.scan(
                reverse_scan, (rng, Y0, state), traj_diffuse_factors
            )

            # Collect mean trajectory reward for plotting
            final_mean_traj_reward = float(info["mean_traj_reward"][-1])
            mean_traj_rewards.append(final_mean_traj_reward)

            # Collect reward components for plotting
            final_reward_components = info["reward_components"]
            for component_name, component_value in final_reward_components.items():
                if component_name not in reward_components_history:
                    reward_components_history[component_name] = []
                reward_components_history[component_name].append(component_value[-1])

            rews_plan.append(info["rews"][-1].mean())
            infos.append(info)
            freq = 1 / (time.time() - t0)
            pbar.set_postfix({"rew": f"{state.reward:.2e}", "freq": f"{freq:.2f}"})

    rew = jnp.array(rews).mean()
    print(f"mean reward = {rew:.2e}")

    # create result dir if not exist
    if not os.path.exists(dial_config.output_dir):
        os.makedirs(dial_config.output_dir)

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Plot mean trajectory rewards
    plt.rcParams['text.usetex'] = False
    plt.figure(figsize=(10, 6))
    plt.plot(mean_traj_rewards, 'b-', linewidth=2, label='Mean Trajectory Reward')
    plt.plot(rews, 'r--', alpha=0.7, label='Actual Environment Reward')
    plt.xlabel('Time Step')
    plt.ylabel('Reward')
    plt.title('Mean Trajectory Rewards vs Actual Environment Rewards')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(dial_config.output_dir, f"{timestamp}_reward_comparison.pdf"))
    plt.savefig(os.path.join(dial_config.output_dir, f"{timestamp}_reward_comparison.png"), dpi=300)
    print(f"Reward plot saved to {dial_config.output_dir}/{timestamp}_reward_comparison.pdf")

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
            plt.savefig(os.path.join(dial_config.output_dir, f"{timestamp}_reward_components.pdf"))
            plt.savefig(os.path.join(dial_config.output_dir, f"{timestamp}_reward_components.png"), dpi=300)
            print(f"Reward components plot saved to {dial_config.output_dir}/{timestamp}_reward_components.pdf")

            # Also create a single plot with all components
            plt.figure(figsize=(12, 8))
            colors = plt.cm.tab10(jnp.linspace(0, 1, len(reward_components_history)))
            for i, (component_name, values) in enumerate(reward_components_history.items()):
                clean_name = component_name.replace("reward_", "").title()
                plt.plot(values, color=colors[i], linewidth=2, label=clean_name, alpha=0.8)

            plt.xlabel('Time Step')
            plt.ylabel('Reward Value')
            plt.title('All Mean Trajectory Reward Components Over Time')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(dial_config.output_dir, f"{timestamp}_all_reward_components.pdf"))
            plt.savefig(os.path.join(dial_config.output_dir, f"{timestamp}_all_reward_components.png"), dpi=300)
            print(f"All reward components plot saved to {dial_config.output_dir}/{timestamp}_all_reward_components.pdf")

    # Also plot just the mean trajectory rewards
    plt.figure(figsize=(10, 6))
    plt.plot(mean_traj_rewards, 'b-', linewidth=2)
    plt.xlabel('Time Step')
    plt.ylabel('Mean Trajectory Reward')
    plt.title('Mean Trajectory Rewards During Optimization')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(dial_config.output_dir, f"{timestamp}_mean_traj_rewards.pdf"))
    plt.savefig(os.path.join(dial_config.output_dir, f"{timestamp}_mean_traj_rewards.png"), dpi=300)
    print(f"Mean trajectory rewards plot saved to {dial_config.output_dir}/{timestamp}_mean_traj_rewards.pdf")

    # host webpage with flask
    print("Processing rollout for visualization")
    import flask

    app = flask.Flask(__name__)
    webpage = html.render(
        env.sys.tree_replace({"opt.timestep": env.dt}), rollout, 1080, True
    )

    # save the html file
    with open(
        os.path.join(dial_config.output_dir, f"{timestamp}_brax_visualization.html"),
        "w",
    ) as f:
        f.write(webpage)

    # save the rollout
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
        xdata.append(infos[i]["xbar"][-1])
    data = jnp.array(data)
    xdata = jnp.array(xdata)
    jnp.save(os.path.join(dial_config.output_dir, f"{timestamp}_states"), data)
    jnp.save(os.path.join(dial_config.output_dir, f"{timestamp}_predictions"), xdata)

    # Save reward data
    jnp.save(os.path.join(dial_config.output_dir, f"{timestamp}_mean_traj_rewards"), jnp.array(mean_traj_rewards))
    jnp.save(os.path.join(dial_config.output_dir, f"{timestamp}_env_rewards"), jnp.array(rews))

    # Save reward components data
    if reward_components_history:
        for component_name, values in reward_components_history.items():
            jnp.save(os.path.join(dial_config.output_dir, f"{timestamp}_{component_name}"), jnp.array(values))

    print(f"Reward data saved to {dial_config.output_dir}/{timestamp}_*_rewards.npy")

    @app.route("/")
    def index():
        return webpage

    app.run(port=5000)


if __name__ == "__main__":
    main()
