# The MBDPI Class: Core of DIAL-MPC

The Model-Based Diffusion Policy Iteration (MBDPI) class is the central component of the DIAL-MPC algorithm. This class handles the diffusion process, trajectory optimization, and conversion between different control representations.

## Core Functionality

The MBDPI class provides several key functionalities:

1. **Control Representation Conversion**:
   - Converts between node-based representation (`Y`) and control signals (`u`)
   - Uses spline interpolation for smooth transitions

2. **Diffusion Process**:
   - Implements reverse diffusion to iteratively refine control trajectories
   - Uses a carefully scheduled noise profile across the diffusion process

3. **Trajectory Optimization**:
   - Generates and evaluates multiple candidate trajectories
   - Selects optimal controls using softmax-weighted averaging based on rewards

## Key Methods

### Reverse Process

The diffusion process works backward from noise to structured control:

```python
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
            pbar.set_postfix({"rew": f"{rews.mean():.2e}", "freq": f"{freq:.2f}"})
    return Yi
```

Each step of the reverse process (`reverse_once`) samples multiple trajectory candidates and combines them based on their expected rewards.

### Trajectory Shifting

As the robot moves forward in time, the control trajectory is shifted and updated:

```python
@functools.partial(jax.jit, static_argnums=(0,))
def shift(self, Y):
    u = self.node2u_vmap(Y)
    u = jnp.roll(u, -1, axis=0)
    u = u.at[-1].set(jnp.zeros(self.nu))
    Y = self.u2node_vmap(u)
    return Y
```

This approach implements the receding horizon aspect of model predictive control.

## Optimization Strategy

The MBDPI class uses a unique optimization approach:

1. **Sampling-Based Optimization**:
   - Generates multiple trajectory samples with controlled noise
   - Evaluates each trajectory using the environment's reward function
   - Combines trajectories using a temperature-controlled softmax over rewards

2. **Multi-Resolution Control**:
   - Uses a reduced number of control nodes to improve efficiency
   - Interpolates between nodes to generate smooth control signals

3. **Iterative Refinement**:
   - Repeatedly applies diffusion steps to refine the trajectory
   - More diffusion steps are used during initialization for better compilation

## Integration with JAX

The class leverages JAX's capabilities for high-performance computing:

- JIT compilation for efficient execution
- Vectorized operations (vmap) for parallel trajectory evaluation
- Functional programming patterns for clean, maintainable code

This design allows DIAL-MPC to efficiently optimize complex control problems in real-time.

# Control Parameterization in DIAL-MPC

The DIAL-MPC (Diffusion-Based Iterative Linear Model Predictive Control) implementation uses a sophisticated control trajectory parameterization system. Let's examine how control signals are represented and converted between different formats:

## Nodes and Control Signals

The system maintains two key representations:

1. **Node representation (`Y`)**: A reduced set of control points that define the trajectory
2. **Control signal representation (`u`)**: The actual control inputs sent to the environment

## Conversion Between Representations

The MBDPI (Model-Based Diffusion Policy Iteration) class handles the conversion between these representations:

```python
# Setup timing parameters
self.ctrl_dt = 0.02
self.step_us = jnp.linspace(0, self.ctrl_dt * args.Hsample, args.Hsample + 1)
self.step_nodes = jnp.linspace(0, self.ctrl_dt * args.Hsample, args.Hnode + 1)
self.node_dt = self.ctrl_dt * (args.Hsample) / (args.Hnode)
```

Where:
- `Hsample`: Number of control steps in the horizon
- `Hnode`: Number of node points used to represent the trajectory
- `ctrl_dt`: Time step for control signals

### Conversion Functions

The system uses spline interpolation to convert between node and control representations:

```python
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
```

These functions:
- Use quadratic (k=2) spline interpolation
- Allow for smooth transitions between control points
- Enable working with lower-dimensional node representation during optimization

## Vectorized Operations

The code uses JAX's vectorization capabilities to efficiently process multiple trajectories:

```python
# Vectorized versions of conversion functions
self.node2u_vmap = jax.jit(
    jax.vmap(self.node2u, in_axes=(1), out_axes=(1))
)  # process (horizon, node)
self.u2node_vmap = jax.jit(jax.vmap(self.u2node, in_axes=(1), out_axes=(1)))
self.node2u_vvmap = jax.jit(
    jax.vmap(self.node2u_vmap, in_axes=(0))
)  # process (batch, horizon, node)
self.u2node_vvmap = jax.jit(jax.vmap(self.u2node_vmap, in_axes=(0)))
```

## Noise Schedule for Diffusion

The control optimization uses a diffusion process with carefully scheduled noise levels:

```python
sigma0 = 1e-2
sigma1 = 1.0
A = sigma0
B = jnp.log(sigma1 / sigma0) / args.Ndiffuse
self.sigmas = A * jnp.exp(B * jnp.arange(args.Ndiffuse))
self.sigma_control = (
    args.horizon_diffuse_factor ** jnp.arange(args.Hnode + 1)[::-1]
)
```

This creates:
1. An exponentially increasing noise schedule from `sigma0` to `sigma1`
2. A time-dependent noise factor that varies along the horizon (decreasing as we move further into the future)

These parameters allow the algorithm to manage the exploration-exploitation tradeoff during trajectory optimization.

# Batch Environment Rollouts in DIAL-MPC

DIAL-MPC leverages JAX's vectorization capabilities to efficiently evaluate multiple trajectories in parallel within the Brax physics engine. This approach is fundamental to the diffusion-based optimization method, which requires evaluating many candidate trajectories.

## Core Rollout Function

The batch rollout functionality is implemented through two key functions:

1. **Base Rollout Function**:
```python
def rollout_us(step_env, state, us):
    def step(state, u):
        state = step_env(state, u)
        return state, (state.reward, state.pipeline_state)

    _, (rews, pipline_states) = jax.lax.scan(step, state, us)
    return rews, pipline_states
```

This function takes a single environment state and a sequence of control inputs, then simulates the trajectory by:
- Applying each control sequentially using JAX's `lax.scan` operation
- Collecting rewards and pipeline states at each step
- Returning the entire sequence of rewards and states

2. **Vectorized Rollout**:
```python
self.rollout_us = jax.jit(functools.partial(rollout_us, self.env.step))
self.rollout_us_vmap = jax.jit(jax.vmap(self.rollout_us, in_axes=(None, 0)))
```

The function is vectorized using JAX's `vmap` operation to process multiple trajectories in parallel. The `in_axes=(None, 0)` parameter indicates that:
- The environment state is shared across all trajectories (not batched)
- The control inputs are batched along the first dimension

## Batch Trajectory Evaluation

The batched evaluation is used in the `reverse_once` method of the MBDPI class:

```python
# Convert control nodes to control signals
us = self.node2u_vvmap(Y0s)

# Batch evaluate all trajectories
rewss, pipeline_statess = self.rollout_us_vmap(state, us)
```

This approach:
1. Generates `Nsample` trajectories by adding noise to the current trajectory
2. Converts all trajectories to control signals using vectorized spline interpolation
3. Evaluates all trajectories in parallel using the `rollout_us_vmap` function
4. Computes statistics across all trajectories to update the control plan

## Key Advantages of This Approach

1. **Parallelization**: JAX's vectorization transforms the operations to run in parallel on GPU/TPU
2. **JIT Compilation**: All functions are JIT-compiled for high performance
3. **Single-Program Multiple-Data (SPMD)**: The same control evaluation is applied across many different inputs
4. **Memory Efficiency**: Leverages JAX's efficient memory management for large batches

This design allows DIAL-MPC to efficiently sample and evaluate hundreds of potential trajectories in milliseconds, which is crucial for the diffusion-based optimization approach that requires many samples for effective trajectory planning.

# Environment Vectorization in DIAL-MPC

DIAL-MPC leverages Brax's physics engine and JAX's functional programming paradigm to create efficient, vectorized robotic environments. Using the UnitreeGo2Env as an example, we can see how the environment is structured to enable parallel simulation and efficient control.

## Environment Architecture

The environment hierarchy follows a well-structured inheritance pattern:

1. **PipelineEnv (Brax)**: The base class from Brax that handles physics simulation
2. **BaseEnv**: A DIAL-MPC specific base class that adds common functionality
3. **UnitreeGo2Env**: The specific implementation for the Go2 quadruped robot

This layered approach allows for code reuse while maintaining environment-specific behaviors.

## Key Components of UnitreeGo2Env

The UnitreeGo2Env initializes several important components:

```python
def __init__(self, config: UnitreeGo2EnvConfig):
    super().__init__(config)
    
    # Robot configurations
    self._foot_radius = 0.0175
    
    # Gait configurations
    self._gait = config.gait
    self._gait_phase = {
        "stand": jnp.zeros(4),
        "walk": jnp.array([0.0, 0.5, 0.75, 0.25]),
        "trot": jnp.array([0.0, 0.5, 0.5, 0.0]),
        "canter": jnp.array([0.0, 0.33, 0.33, 0.66]),
        "gallop": jnp.array([0.0, 0.05, 0.4, 0.35]),
    }
    self._gait_params = {
        # ratio, cadence, amplitude
        "stand": jnp.array([1.0, 1.0, 0.0]),
        "walk": jnp.array([0.75, 1.0, 0.08]),
        "trot": jnp.array([0.45, 2, 0.08]),
        "canter": jnp.array([0.4, 4, 0.06]),
        "gallop": jnp.array([0.3, 3.5, 0.10]),
    }
    
    # Robot model references
    self._torso_idx = mujoco.mj_name2id(
        self.sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, "base"
    )
    
    # Initial state and constraints
    self._init_q = jnp.array(self.sys.mj_model.keyframe("home").qpos)
    self._default_pose = self.sys.mj_model.keyframe("home").qpos[7:]
    self.joint_range = jnp.array([...])  # Defines range of motion for each joint
```

## Vectorizable Operations

The environment is designed with JAX's functional programming model in mind:

1. **Pure Functions**: Methods like `step`, `reset`, and `_get_obs` are pure functions that take a state and return a new state without side effects
2. **jit Compilation**: Functions are decorated with `@jax.jit` or `@functools.partial(jax.jit, static_argnums=(0,))` for faster execution
3. **Array Manipulation**: All operations use JAX NumPy (`jnp`) rather than regular NumPy for GPU/TPU acceleration

## Integration with Brax Physics

The environment interacts with Brax's physics engine through:

```python
def make_system(self, config: UnitreeGo2EnvConfig) -> System:
    model_path = get_model_path("unitree_go2", "mjx_scene_force.xml")
    sys = mjcf.load(model_path)
    sys = sys.tree_replace({"opt.timestep": config.timestep})
    return sys
```

This creates a physics system based on an MJCF (MuJoCo) model file, which is then used by Brax for simulation.

## State and Action Processing

Actions undergo a transformation process to map from normalized control inputs to robot-specific joint angles or torques:

```python
def act2joint(self, act: jax.Array) -> jax.Array:
    act_normalized = (act * self._config.action_scale + 1.0) / 2.0
    joint_targets = self.joint_range[:, 0] + act_normalized * (
        self.joint_range[:, 1] - self.joint_range[:, 0]
    )
    joint_targets = jnp.clip(
        joint_targets,
        self.physical_joint_range[:, 0],
        self.physical_joint_range[:, 1],
    )
    return joint_targets
```

## Reward Computation

The reward function combines multiple objectives to guide the robot behavior:

```python
# Simplified reward computation from UnitreeGo2Env
reward = (
    reward_gaits * 0.1
    + reward_air_time * 0.0
    + reward_pos * 0.0
    + reward_upright * 0.5
    + reward_yaw * 0.3
    + reward_vel * 1.0
    + reward_ang_vel * 1.0
    + reward_height * 1.0
    + reward_energy * 0.00
    + reward_alive * 0.0
)
```

This weighted sum approach enables multi-objective optimization with configurable priorities.

## Batch Simulation with JAX

The environment itself doesn't directly implement batching, but is designed to be compatible with JAX's vectorization. In DIAL-MPC, this happens through:

```python
# In MBDPI class
self.rollout_us = jax.jit(functools.partial(rollout_us, self.env.step))
self.rollout_us_vmap = jax.jit(jax.vmap(self.rollout_us, in_axes=(None, 0)))
```

This vectorizes the environment's `step` function, allowing multiple control trajectories to be evaluated in parallel from the same initial state.

## Benefits of Environment Vectorization

1. **Performance**: Simulation of multiple trajectories in parallel on GPU/TPU
2. **Determinism**: Pure functional approach ensures reproducible results
3. **Efficiency**: JAX's compilation optimizes the execution for specific hardware
4. **Scalability**: Batch size can be adjusted based on available compute resources

The environment design in DIAL-MPC demonstrates how modern robotics simulation can leverage JAX's functional programming model and hardware acceleration to enable efficient trajectory optimization through massively parallel simulation.

# Physics System and State Separation in Brax

A key insight into how DIAL-MPC efficiently handles batch environment simulation lies in understanding Brax's architecture and JAX's functional programming model. Let's examine why environments can support batch rollout simply by applying `vmap` to `env.step` without needing to copy the physics system data.

## System vs. State Separation

In Brax and DIAL-MPC, there's a crucial separation between:

1. **Physics System (`sys`)**: Contains static parameters like the robot model, joint limits, and collision properties
2. **Environment State (`state`)**: Contains dynamic variables like positions, velocities, and rewards

Looking at `UnitreeGo2Env.make_system()`:

```python
def make_system(self, config: UnitreeGo2EnvConfig) -> System:
    model_path = get_model_path("unitree_go2", "mjx_scene_force.xml")
    sys = mjcf.load(model_path)
    sys = sys.tree_replace({"opt.timestep": config.timestep})
    return sys
```

This method creates the physics system once during initialization. The system is then used as a reference (essentially as a constant) during simulation.

## Functional Design Pattern

Brax environments follow a pure functional design. The `step` method in `UnitreeGo2Env`:

```python
def step(self, state: State, action: jax.Array) -> State:
    # ...
    pipeline_state = self.pipeline_step(state.pipeline_state, ctrl)
    # ...
    state = state.replace(
        pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
    )
    return state
```

This method:
1. Takes a state and action as input
2. Computes a new state without modifying any environment variables
3. Returns the new state

The `self` reference contains the physics system, but it's used as a read-only reference.

## Why Data Copying Isn't Needed

When JAX applies `vmap` to `rollout_us`:

```python
self.rollout_us_vmap = jax.jit(jax.vmap(self.rollout_us, in_axes=(None, 0)))
```

JAX can efficiently vectorize without copying the physics system because:

1. **Immutable Data Structures**: JAX uses immutable arrays, so there's no risk of one batch element modifying data used by another
2. **Read-Only System References**: The physics system is referenced but never modified during simulation
3. **Pure Functions**: All computations create new outputs rather than modifying inputs

## Under the Hood: JAX's Implementation

What's happening under the hood:

```
┌─────────────────────────────────────────┐
│              Physics System             │
│  (Robot model, joint limits, collision) │
└────────────────────┬────────────────────┘
                     │ Reference (not copied)
                     ▼
┌─────────────────────────────────────────┐
│           Environment Step fn           │
│     (Takes state and action as input)   │
└────────────────────┬────────────────────┘
                     │ vmapped
                     ▼
┌─────────────────────────────────────────┐
│         Parallel Computation            │
│  State₁  State₂  State₃  ...  Stateₙ   │
│  Action₁ Action₂ Action₃ ... Actionₙ   │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│         New States & Rewards            │
└─────────────────────────────────────────┘
```

During compilation, JAX's tracing mechanism recognizes that the physics system parameters are constants with respect to the batch elements, so it optimizes the computation to avoid redundant operations.

## Benefits of This Approach

1. **Memory Efficiency**: Only one copy of the potentially large physics system exists
2. **Computational Efficiency**: Hardware accelerators (GPUs/TPUs) can process many trajectories in parallel
3. **Programming Simplicity**: Environment authors don't need to manually manage batching
4. **Optimization Opportunities**: JAX can apply optimizations like common subexpression elimination across batch elements

This architecture explains why DIAL-MPC can efficiently evaluate hundreds of trajectory candidates in parallel without facing memory bottlenecks from duplicating the physics system for each sample.

# Understanding DIAL-MPC's Main Logic

The `main()` function in `dial_mpc/core/dial_core.py` implements the core execution flow of the DIAL-MPC (Diffusion-Based Iterative Linear Model Predictive Control) algorithm. Here's a breakdown of the main logic:

## Initialization Phase

1. **Argument Parsing**: The function starts by parsing command line arguments to determine whether to:
   - Use a configuration file (`--config`)
   - Use a predefined example (`--example`)
   - List available examples (`--list-examples`)
   - Import a custom environment (`--custom-env`)

2. **Environment Setup**:
   - Loads configuration from either a specified file or a predefined example
   - Creates the environment based on the configured parameters
   - Initializes the MBDPI (Model-Based Diffusion Policy Iteration) controller
   - Resets the environment to get the initial state

3. **Control Initialization**:
   - Initializes a control trajectory `Y0` (initially zeros)
   - Sets up variables to store rewards, environment states, and control inputs

## Simulation Loop

The main simulation runs for `n_steps` iterations with the following steps in each iteration:

1. **Forward Step**: 
   - Applies the first control from the trajectory to the environment
   - Records the resulting state, reward, and control

2. **Trajectory Update**:
   - Shifts the control trajectory (removing the first control that was just applied)
   - Runs multiple iterations of the diffusion process to refine the trajectory

3. **Diffusion Process**:
   - For a specified number of diffusion steps, applies the `reverse_scan` function
   - Each step generates multiple trajectory samples and uses weighted averaging to update the trajectory
   - The weights are determined by the reward obtained from each sampled trajectory
   - The first step uses more diffusion iterations (`Ndiffuse_init`) for better initial JIT compilation

## Visualization and Data Collection

After completing the simulation:

1. **Results Processing**:
   - Reports the mean reward
   - Creates a 3D visualization of the rollout using Brax's HTML renderer
   - Saves the visualization to an HTML file
   - Saves the state trajectory and prediction data as NumPy arrays

2. **Web Server**:
   - Starts a Flask web server to view the visualization in a browser

The code effectively implements a model predictive control approach where a diffusion-based trajectory optimizer repeatedly plans and refines control inputs over a receding horizon.