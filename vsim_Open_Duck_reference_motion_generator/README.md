# Usage

## 1. Tune and visualize a single trajectory
- Configure `open_duck_reference_motion_generator/robots/open_duck_mini_v2/placo_defaults.json` file parameters.
- Run `uv run open_duck_reference_motion_generator/gait_playground.py --duck "open_duck_mini_v2"` to start the Placo Gait Engine server and click `Run`.
- If you don't see the duck in the environment, try rebooting your computer.
- NOTE: take a note of the `period` parameter, which will be printed to the terminal, for instance: `##### period: 0.72 #####`. The parameters `double_support_ratio` and `single_support_duration` control the duration of a single gait cycle.
- Move to step 2. when you're happy with the gait and recorded the `period` parameter for yourself.

## 2. Export one or many trajectories
- Find the `do_sample()` function in `open_duck_reference_motion_generator/gait_playground_offline.py`. This is the only function you need to change here.
- The parameter ranges are for linear velocity x (forward velocity) and angular velocity z (yaw rate).
- You can return an arbitrary array of samples if you don't want to use the default sampling. For instance, return a single sample for straight backward walking (negative lin vel + zero ang vel): `scaled_samples = np.array([[-0.1, 0.0]])`.
- Adjust the number of samples and parameter ranges as needed.
- Run `uv run open_duck_reference_motion_generator/gait_playground_offline.py` to generate many trajectories.
- Check `gait_samples.png` to see the sampled linear and angular velocity parameters.
- The gait data you need for training is now saved to `all_episodes.json`. This is the only file you need for training. You will need to copy it over to `vlearn`.
- Copy the `all_episodes.json` to `vlearn/train/envs/data`.

## Original open source repo
For anything further, refer to https://github.com/apirrone/Open_Duck_reference_motion_generator.