# so100 Tower Environment
This covers the SO100 Tower environment, providing a high-level overview as to how the simulations initialisation, observations/action space, and reward functions for robotic cube-stacking are setup. The goal is to train a manipulation policy using demonstrations and SAC.
This covers the SO100 Tower environment, providing a high-level overview as to how the simulations initialisation, observations/action space, and reward functions for robotic cube-stacking are setup. The goal is to train a manipulation policy using demonstrations and SAC.

## Initialisation
The environment is initialised in `so100_tower.py` by creating the simulated world, robot, and scene assets. It inherits from the `EnvironmentGpu` class, defined in `environment.py`, which implements common methods related to initialisation, rendering, simulation stepping, and resetting.
The environment is initialised in `so100_tower.py` by creating the simulated world, robot, and scene assets. It inherits from the `EnvironmentGpu` class, defined in `environment.py`, which implements common methods related to initialisation, rendering, simulation stepping, and resetting.
The class constructor calls `super.__init__()` which creates a `vlearn.Gym` instance, setting global paramaters.

The `create_envs()` method creates the so100_tower environment definition, including the robot, boxes and cameras. It imports the environment assets, defines robot articulations, adds the rigid bodies, configures the RGB cameras for segmentation and passes the environment definition handle to `super().create_envs()`.
The `create_envs()` method creates the so100_tower environment definition, including the robot, boxes and cameras. It imports the environment assets, defines robot articulations, adds the rigid bodies, configures the RGB cameras for segmentation and passes the environment definition handle to `super().create_envs()`.

The method `allocate_buffers()` creates buffers for the robots state (joint positions), kinematics, box state, camera, force sensors, rewards and termination conditions.
The method `allocate_buffers()` creates buffers for the robots state (joint positions), kinematics, box state, camera, force sensors, rewards and termination conditions.
The following buffers are allocated by `Environment.allocate_buffers()`:
- self.obs_buf: observations
- self.rew_buf: rewards
- self.act_buf: actions
- self.term_buf: flags indicating that environment has terminated
- self.trunc_buf: flags indicating that environment has reached the maximum episode length
- self.progress_buf: counter for number of steps taken by environment (current episode length)


### Observation and Action Spaces
The environment provides observations and actions that correspond directly to the robot arm's degrees of freedom (DOFs) and its control limits, which we define as follows in `init_obs_and_act_spaces()`:
```
self.single_observation_space = Box(
    low=np.array([np.finfo('f').min] * self.num_obs, dtype=np.float32),
    high=np.array([np.finfo('f').max] * self.num_obs, dtype=np.float32),
    dtype=np.float32)
```
The observation space is 12-dimensional (6 current + 6 previous joint positions).

```
self.single_action_space = Box(
    low=np.array([-d for d in per_joint_max_deltas], dtype=np.float32),
    high=np.array(per_joint_max_deltas, dtype=np.float32),
    dtype=np.float32)
```
The action space is a 6-dimensional continuous vector, controlling joint positions for each motor in radians. The vlearn.spaces.Box class is defined in the module vlearn.spaces.

## Reset
All Vlearn environments are vectorized, so environments auto-reset when they reach a terminal state (termination) or reach the maximum episode length (truncation). The flags in the reset buffer are set and `reset_idx()` is called in `post_physics_step()`. During reset, the robot arm is returned to a home configuration, all joint velocities are zeroed, and internal buffers, tracking state, rewards, and flags are cleared.The `compute_observations()` method updates the buffer of the states and the observation buffer.

At the start of each episode (or upon termination), the environment resets the robot joints return to home configuration, box positions are randomised: a random number (1–4) of stacked boxes plus one target box and the camera poses are randomized.

## Reward
The function `compute_rewards_jit()` computes a shaped, multi-stage reward encouraging successful stacking. As an example, the pickup reward function `pickup_reward_jit()` calculates a reward based on how effectively the agent grasps the object.
```
    # Grab force feedback reward
    jaw_force = jaw_cube_force_sensor_val[:, 0]
    ...
    ...
    enough_force = jaw_force < -2.0
```
It measures the force applied by the gripper’s jaws on the object and converts it into a reward using an exponential shaping function. The reward is scaled by a weight (`w_force`) to encourage the agent to apply sufficient force for grasping. A threshold check (`enough_force`) marks when the gripper has successfully exerted enough pressure to hold the object.

These rewards use the mathematical form: $R = {w} \cdot {e^{{-k} \cdot {error}}}$
where $w$ is the weight which is the maximum reward for the term, $error$ being the magnitude of deviation and $k$ ($k = \frac{ln(2)}{tolerance}$) is the shaping constant. This sets how quickly the exponential reward decays, so that the reward drops to half its maximum when the error equals the specified tolerance. The exponential shape ensures a smooth feedback.
A masked tensor is applied to these rewards so rewards are only applied when certain conditions are met, then these rewards are placed into the reward buffer.