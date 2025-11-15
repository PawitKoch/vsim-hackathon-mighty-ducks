import numpy as np
import torch
from vlearn.spaces import Box
from typing import Union

import vlearn as v

if __name__ == "__main__":
    from environment import EnvironmentGpu
    from cartpole_environment_common import (
        compute_observations_helper,
        compute_reward_termination_truncation_helper,
        create_envs_helper,
        reset_idx_helper)
else:
    from .environment import EnvironmentGpu
    from .cartpole_environment_common import (
        compute_observations_helper,
        compute_reward_termination_truncation_helper,
        create_envs_helper,
        reset_idx_helper)


class CartpoleEnvironmentGpu(EnvironmentGpu):

    def __init__(self,
                 num_envs: int,
                 device: Union[str, torch.device],
                 rendering: bool = False,
                 enable_scene_query: bool = False,
                 max_episode_steps: int = 1000,
                 cart_bounds: tuple[float, float] = [-3.5, 3.5],
                 pole_bounds: tuple[float, float] = [-0.25, 0.25],
                 force_bounds: tuple[float, float] = [-10, 10],
                 reset_noise_scale: float = 1,
                 spacing: float = 10,
                 gravity: v.Vec3 = v.Vec3(0, -9.81, 0),
                 timestep: float = 0.01667,
                 frame_skip: int = 1,
                 initial_is_paused: bool = False,
                 send_interrupt: bool = False,
                 max_contact_pairs_per_env: int = 1,
                 with_window: bool = True,
                 ):

        num_dofs = 2
        assert (max_episode_steps % frame_skip == 0)
        num_training_steps = max_episode_steps // frame_skip

        if isinstance(device, str):
            device = torch.device(device)

        super().__init__(
            num_envs,
            device,
            rendering,
            enable_scene_query,
            num_training_steps,
            timestep,
            frame_skip,
            spacing,
            gravity,
            num_dofs,
            initial_is_paused=initial_is_paused,
            send_interrupt=send_interrupt,
            max_contact_pairs_per_env=max_contact_pairs_per_env,
            with_window=with_window)

        self.cart_bounds = cart_bounds
        self.pole_bounds = pole_bounds
        self.force_bounds = force_bounds
        self.reset_noise_scale = reset_noise_scale

        # Initialise observation and action space
        self.single_observation_space = Box(
            low=np.array([-4, np.finfo('f').min, np.finfo('f').min, np.finfo('f').min]),
            high=np.array([4, np.finfo('f').max, np.finfo('f').max, np.finfo('f').max]),
            dtype=np.float32)
        self.single_action_space = Box(low=force_bounds[0], high=force_bounds[1], dtype=np.float32)

        # Create environments
        self.create_envs()

        # Allocate buffers
        self.allocate_buffers()

        # Finalize gym
        self.gym.gym_finalize()

    def create_envs(self):

        self.env_def_handle, self.arti_handle = create_envs_helper(self.gym)

        self.gym.get_environment_def(self.env_def_handle).finalize()

        super().create_envs(self.env_def_handle)

    def allocate_buffers(self):

        super().allocate_buffers()

        # Buffers for setting kinematic state
        self.set_dof_pos_buf = torch.zeros(
            (self.num_envs, self.num_dofs), device=self.device, dtype=torch.float32)
        self.set_dof_vel_buf = torch.zeros(
            (self.num_envs, self.num_dofs), device=self.device, dtype=torch.float32)

        # Create command for setting joint forces
        set_force_data = self.env_group.create_joint_state_command(v.wrap_gpu_buffer(self.act_buf),
                                                                   self.arti_handle, (0, 1))

        self.gpu_set_force_data_array = self.gym.create_joint_state_command_gpu_array(
            [set_force_data])

        # Create command for setting joint positions
        set_pos_data = self.env_group.create_joint_state_command(
            v.wrap_gpu_buffer(
                self.set_dof_pos_buf), self.arti_handle, masks_buffer=v.wrap_gpu_buffer(
                self.reset_buf))

        self.gpu_set_pos_data_array = self.gym.create_joint_state_command_gpu_array([set_pos_data])

        # Create command for setting joint velocities
        set_vel_data = self.env_group.create_joint_state_command(
            v.wrap_gpu_buffer(
                self.set_dof_vel_buf), self.arti_handle, masks_buffer=v.wrap_gpu_buffer(
                self.reset_buf))

        self.gpu_set_vel_data_array = self.gym.create_joint_state_command_gpu_array([set_vel_data])

        # Create command for getting joint positions
        get_pos_data = self.env_group.create_joint_state_command(
            v.wrap_gpu_buffer(self.get_dof_pos_buf), self.arti_handle)

        self.gpu_get_pos_data_array = self.gym.create_joint_state_command_gpu_array([get_pos_data])

        # Create command for getting joint velocities
        get_vel_data = self.env_group.create_joint_state_command(
            v.wrap_gpu_buffer(self.get_dof_vel_buf), self.arti_handle)

        self.gpu_get_vel_data_array = self.gym.create_joint_state_command_gpu_array([get_vel_data])

    def reset_idx(self):

        self.set_dof_pos_buf[:], self.set_dof_vel_buf[:] = reset_idx_helper(
            self.num_envs, self.num_dofs, self.reset_noise_scale,
            self.device)

        # Set kinematic state
        self.gym.set_joint_positions(self.gpu_set_pos_data_array)
        self.gym.set_joint_velocities(self.gpu_set_vel_data_array)

        self.progress_buf[:] = torch.where(self.reset_buf, 0, self.progress_buf)

    def reset(self):

        super().reset()

        # Return observations
        self.compute_observations()

        return self.obs_buf.clone(), {}

    def pre_physics_step(self, actions):
        self.act_buf[:] = actions
        self.gym.set_joint_forces(self.gpu_set_force_data_array)

    def post_physics_step(self):
        self.progress_buf += 1

        self.reset_buf[:] = torch.logical_or(self.term_buf, self.trunc_buf)
        self.reset_idx()

        self.compute_observations()
        self.compute_reward_termination_truncation()

    def compute_observations(self):

        self.gym.get_joint_positions(self.gpu_get_pos_data_array)
        self.gym.get_joint_velocities(self.gpu_get_vel_data_array)

        self.obs_buf[:] = compute_observations_helper(self.get_dof_pos_buf, self.get_dof_vel_buf)

    def compute_reward_termination_truncation(self):

        self.rew_buf[:], self.term_buf[:], self.trunc_buf[:] = compute_reward_termination_truncation_helper(
            self.obs_buf, self.progress_buf, self.cart_bounds, self.pole_bounds, self.max_episode_length)


if __name__ == "__main__":
    from time import sleep
    from sys import argv
    from vlearn.utils import get_VL_VISUAL_TESTS

    rendering = True
    with_window = get_VL_VISUAL_TESTS()

    num_iter = np.inf
    if len(argv) >= 2:
        num_iter = int(argv[1])

    num_envs = 8
    # num_envs = 1

    assert torch.cuda.is_available()
    device = torch.device("cuda:0")

    envs = CartpoleEnvironmentGpu(num_envs, device=device, rendering=rendering,
                                  with_window=with_window)
    obs, _ = envs.reset()

    sleep(1)

    if rendering:
        gym = v.get_gym()
        render = gym.get_render()
        mode = 'manual'
        print(
            """Controls:
 F: -10
 G:  -5
 H:   5
 J:  10
""")

    else:
        mode = 'random'

    if rendering:
        render.capped_step = True
    finished = False
    idx = 0
    while not finished:

        if mode == 'manual':
            action = 0
            if render.is_key_down('f'):
                action = -10
            if render.is_key_down('g'):
                action = -5
            if render.is_key_down('h'):
                action = 5
            if render.is_key_down('j'):
                action = 10

            actions = torch.tensor([action])

        elif mode == 'random':
            actions = torch.rand((num_envs, 1)) * 20 - 10

        # print("actions: {}".format(actions))

        obs, rew, reset, timeout, _ = envs.step(actions)
        # print(obs, rew, reset, timeout)

        finished = envs.render_finished

        idx += 1
        if idx >= num_iter:
            finished = True
