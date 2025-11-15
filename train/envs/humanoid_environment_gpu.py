import torch
import numpy as np
from vlearn.spaces import Box

import vlearn as v

import sys
import os

if __name__ == "__main__":
    from environment import EnvironmentGpu
    from common import create_plane, reset_noise_helper
    from humanoid_environment_common import (
        create_envs_helper,
        store_initial_conditions_helper,
        compute_observations_helper,
        compute_reward_termination_truncation_helper,
        allocate_gpu_buffers)
else:
    from .environment import EnvironmentGpu
    from .common import create_plane, reset_noise_helper
    from .humanoid_environment_common import (
        create_envs_helper,
        store_initial_conditions_helper,
        compute_observations_helper,
        compute_reward_termination_truncation_helper,
        allocate_gpu_buffers)


class HumanoidEnvironmentGpu(EnvironmentGpu):

    def __init__(self,
                 num_envs: int,
                 device: torch.device,
                 rendering: bool = False,
                 enable_scene_query: bool = False,
                 max_episode_length: int = 1000,
                 gravity: v.Vec3 = v.Vec3(0, -9.81, 0),
                 timestep: float = 0.01667,
                 frame_skip: int = 1,
                 spacing: float = 2,
                 forward_reward_weight: float = 1.25,
                 ctrl_cost_weight: float = 0.1,
                 healthy_reward: float = 5.0,
                 healthy_y_range: tuple[float, float] = (1.0, 2.0),
                 reset_noise_scale: float = 1,
                 use_contact_forces: bool = True,
                 contact_cost_weight: float = 5e-7,
                 contact_cost_range: tuple[float, float] = (-torch.inf, 10.0),
                 initial_is_paused: bool = False,
                 send_interrupt: bool = False,
                 print_hash: bool = False,
                 max_contact_pairs_per_env: int = 128,
                 with_window: bool = True,
                 ):

        self.num_dofs = 21

        super().__init__(
            num_envs,
            device,
            rendering,
            enable_scene_query,
            max_episode_length,
            timestep,
            frame_skip,
            spacing,
            gravity,
            self.num_dofs,
            initial_is_paused=initial_is_paused,
            send_interrupt=send_interrupt,
            print_hash=print_hash,
            max_contact_pairs_per_env=max_contact_pairs_per_env,
            with_window=with_window)

        # Store environment-specific configuration
        self.forward_reward_weight = forward_reward_weight
        self.ctrl_cost_weight = ctrl_cost_weight
        self.healthy_reward = healthy_reward
        self.healthy_y_range = healthy_y_range
        self.reset_noise_scale = reset_noise_scale
        self.use_contact_forces = use_contact_forces
        self.contact_cost_weight = contact_cost_weight
        self.contact_cost_range = contact_cost_range

        # Hardcode these for now
        self.num_motors = 21
        self.num_sensors = 2
        self.num_links = 13

        # Number of observations
        num_obs = 0
        num_obs += 1                    # root y-position
        num_obs += 4                    # root rotation
        num_obs += self.num_dofs        # joint positions
        num_obs += 6                    # root velocity
        num_obs += self.num_dofs        # joint velocities
        num_obs += self.num_links * 6   # link velocities
        num_obs += self.num_motors      # motor forces
        if self.use_contact_forces:     # sensor forces
            num_obs += self.num_sensors * 6

        # Initialise observation space
        self.single_observation_space = Box(
            low=np.array([np.finfo('f').min] * num_obs, dtype=np.float32),
            high=np.array([np.finfo('f').max] * num_obs, dtype=np.float32),
            dtype=np.float32)

        # Create environment
        self.create_envs()

        # Initialise action space
        low = []
        high = []
        for i in range(self.num_motors):
            motor_def = self.art_def.get_motor_def(i)
            low.append(motor_def.low_limit)
            high.append(motor_def.high_limit)
        low = np.array(low, dtype=np.float32)
        high = np.array(high, dtype=np.float32)

        self.single_action_space = Box(low=low, high=high, dtype=np.float32)
        assert self.num_motors == self.art_def.get_num_motor_defs()

        # Store DOF data
        self.store_initial_conditions()

        # Allocate buffers
        self.allocate_buffers()

        # Create plane
        create_plane(self.gym)

        # Finalize gym
        self.gym.gym_finalize()

    def create_envs(self):

        self.env_def_handle, self.art_def, self.arti_handle = create_envs_helper(self.gym)

        self.gym.get_environment_def(self.env_def_handle).finalize()

        super().create_envs(self.env_def_handle)

    def store_initial_conditions(self):

        self.dof_pos_init, self.root_trans_init, self.root_vel_init = store_initial_conditions_helper(
            self.art_def, self.device)

    def allocate_buffers(self):

        super().allocate_buffers()

        allocate_gpu_buffers(self)

        self.old_root_pos_buf = torch.zeros(
            (self.num_envs, 7), device=self.device, dtype=torch.float32)

        # Link velocities buffer
        self.link_vel_buf = torch.zeros((self.num_envs, self.num_links, 6),
                                        device=self.device, dtype=torch.float32)

        # Get link velocities GPU command
        get_link_velocities_command = self.env_group.create_link_velocity_command(
            v.wrap_gpu_buffer(self.link_vel_buf), self.arti_handle, (0, self.num_links))

        self.gpu_get_link_velocities_command_array = self.gym.create_link_velocity_command_gpu_array([
                                                                                                     get_link_velocities_command])

        self.sensor_force_buf = None
        if self.use_contact_forces:

            # Sensor forces buffer
            self.sensor_force_buf = torch.zeros((self.num_envs, self.num_sensors * 6),
                                                device=self.device, dtype=torch.float32)

            env_def = self.gym.get_environment_def(self.env_def_handle)
            articulation = env_def.get_articulation(self.arti_handle)

            self.force_sensor_handles = []
            self.force_sensor_buffers = []
            self.force_sensor_cmds = []
            for i in range(self.num_sensors):
                self.force_sensor_handles.append(articulation.get_force_sensor_handle(i))
                self.force_sensor_buffers.append(torch.zeros((self.num_envs, 6), device=self.device,
                                                             dtype=torch.float32))
                self.force_sensor_cmds.append(self.env_group.create_force_sensor_command(
                    v.wrap_gpu_buffer(self.force_sensor_buffers[-1]), self.force_sensor_handles[-1]))

            self.gpu_get_sensor_forces_command_array = \
                self.gym.create_force_sensor_command_gpu_array(self.force_sensor_cmds)

    def reset_idx(self):

        # Set kinematic state
        self.set_dof_pos_buf[:] = self.gpu_init_dof_pos + \
            reset_noise_helper(self.gpu_init_dof_pos, self.reset_noise_scale, 0.4, 0.2)
        self.set_dof_vel_buf[:] = reset_noise_helper(
            self.gpu_init_dof_vel, self.reset_noise_scale, 0.2, 0.1)

        self.gym.set_articulation_kinematic_states(self.gpu_set_kinematic_state_command_array)

        # Reset remaining buffers
        self.old_root_pos_buf[:] = torch.where(
            self.reset_buf.view(-1, 1), self.gpu_init_root_transforms, self.old_root_pos_buf)

        self.act_buf[:] = torch.where(self.reset_buf.view(-1, 1), 0, self.act_buf)

        self.progress_buf[:] = torch.where(self.reset_buf, 0, self.progress_buf)

    def reset(self):

        super().reset()

        self.compute_observations(self.act_buf)

        return self.obs_buf.clone(), {}

    def pre_physics_step(self, actions):

        assert isinstance(actions, torch.Tensor)

        # Save old root transforms for reward calculation
        self.old_root_pos_buf[:] = self.root_pos_buf

        self.act_buf[:] = actions

        self.gym.set_motor_forces(self.gpu_set_motor_data_array)

    def post_physics_step(self):
        self.progress_buf += 1

        self.reset_buf[:] = torch.logical_or(self.term_buf, self.trunc_buf)
        self.reset_idx()

        self.compute_observations(self.act_buf)
        self.compute_reward_termination_truncation(self.act_buf)

    def compute_observations(self, actions):

        self.gym.get_articulation_kinematic_states(self.gpu_get_kinematic_state_command_array)

        self.gym.get_link_velocities(self.gpu_get_link_velocities_command_array)

        if self.use_contact_forces:
            self.gym.get_sensor_forces(self.gpu_get_sensor_forces_command_array)

            # Conversion operation for compatibility with handle-based sensor API
            self.sensor_force_buf[:] = torch.cat(self.force_sensor_buffers, dim=-1)

        self.obs_buf[:] = compute_observations_helper(
            self.root_pos_buf,
            self.root_vel_buf,
            self.get_dof_pos_buf,
            self.get_dof_vel_buf,
            self.link_vel_buf,
            actions,
            self.sensor_force_buf)

    def compute_reward_termination_truncation(self, actions):

        self.rew_buf, self.term_buf, self.trunc_buf = compute_reward_termination_truncation_helper(
            actions, self.obs_buf, self.root_pos_buf,
            self.old_root_pos_buf, self.progress_buf, self.healthy_y_range,
            self.healthy_reward, self.dt, self.forward_reward_weight,
            self.ctrl_cost_weight, self.max_episode_length,
            self.contact_cost_weight, self.contact_cost_range,
            self.sensor_force_buf)


if __name__ == "__main__":
    from time import sleep
    from sys import argv
    from vlearn.utils import get_VL_VISUAL_TESTS

    rendering = True
    with_window = get_VL_VISUAL_TESTS()

    num_iter = np.inf
    if len(argv) >= 2:
        num_iter = int(argv[1])

    num_envs = 2
    max_episode_length = 1000

    device = torch.device("cuda:0")

    envs = HumanoidEnvironmentGpu(num_envs, device, rendering=rendering,
                                  max_episode_length=max_episode_length, with_window=with_window)

    obs, _ = envs.reset()

    gym = v.get_gym()
    render = gym.get_render()

    if render is not None:
        # Reset interface
        reset_box = v.UserCheckbox("Reset", False)
        render.register_menu_item(reset_box)

        sliders = []

        for i in range(envs.art_def.get_num_motor_defs()):
            name = envs.art_def.get_motor_def_name(i)
            motor_def = envs.art_def.get_motor_def(i)
            low = motor_def.low_limit
            high = motor_def.high_limit

            sliders.append(v.UserSlider(name, low, high, 0))

            render.register_menu_item(sliders[-1])

        def control_by_menu():

            if reset_box.get_value():
                obs, _ = envs.reset()
                print_obs(obs)

            return torch.tensor([slider.get_value() for slider in sliders])

        control_fn = control_by_menu

    else:
        def control_fn(n=num_envs, m=envs.art_def.get_num_motor_defs()): return torch.zeros((n, m))

    def print_obs(obs):
        # print("obs: {}".format(obs))
        # print("obs[:, 0]: {}".format(obs[:, 0])) # root pos
        # print("obs[:, 1:5]: {}".format(obs[:, 1:5])) # root rot
        # print("obs[:, 5:26]: {}".format(obs[:, 5:26])) # dof pos
        # print("obs[:, 26:29]: {}".format(obs[:, 26:29])) # root linear vel
        # print("obs[:, 29:32]: {}".format(obs[:, 29:32])) # root angular vel
        # print("obs[:, 32:53]: {}".format(obs[:, 32:53])) # dof vel
        # print("obs[:, 53:92]: {}".format(obs[:, 53:92])) # link velocity
        # print("obs[:, 92:131]: {}".format(obs[:, 92:131])) # link angular velocity
        # print("obs[:, 131:152]: {}".format(obs[:, 131:152])) # motor forces
        # print("obs[:, 152:164]: {}".format(obs[:, 152:164])) # sensor forces
        pass

    print_obs(obs)

    # Environment loop
    if render is not None:
        render.capped_step = True
    finished = False
    idx = 0
    while not finished:

        actions = control_fn()

        # print("action: {}".format(action))

        obs, reward, terminated, truncated, info = envs.step(actions)

        # print("reward: {}".format(reward))
        print_obs(obs)

        # print(obs, reward, terminated, truncated, info)

        finished = envs.render_finished

        idx += 1
        if idx >= num_iter:
            finished = True
