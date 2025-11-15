import numpy as np
import torch
import matplotlib.pyplot as plt

from vlearn.spaces import Box

from sys import argv
from os import path

from collections import deque

import vlearn as v

if __name__ == "__main__":
    from environment import EnvironmentGpu
    from cartpole_environment_common import create_envs_helper, reset_idx_helper
else:
    from .environment import EnvironmentGpu
    from .cartpole_environment_common import create_envs_helper, reset_idx_helper


class CartpoleEnvironmentVision(EnvironmentGpu):

    def __init__(self,
                 num_envs: int,
                 device: torch.device,
                 rendering: bool = False,
                 max_episode_steps: int = 500,
                 cart_bounds: tuple[float, float] = [-3.5, 3.5],
                 pole_bounds: tuple[float, float] = [-0.25 * torch.pi, 0.25 * torch.pi],
                 initial_pole_bounds: tuple[float, float] = [-0.05 * torch.pi, 0.05 * torch.pi],
                 action_clip: float = 1.0,
                 obs_clip: float = 5.0,
                 action_scale: float = 100.0,
                 rew_scale_alive: float = 1.0,
                 rew_scale_terminated: float = -2.0,
                 rew_scale_pole_pos: float = -1.0,
                 rew_scale_cart_vel: float = -0.01,
                 rew_scale_pole_vel: float = -0.005,
                 spacing: float = 10,
                 gravity: v.Vec3 = v.Vec3(0, -9.81, 0),
                 timestep: float = 0.01667,
                 resolution_x: int = None,
                 resolution_y: int = None,
                 frame_stack: int = 2,
                 frame_skip: int = 1,
                 initial_is_paused: bool = False,
                 send_interrupt: bool = False,
                 max_contact_pairs_per_env: int = 1,
                 with_window: bool = True,
                 use_rgb: bool = False,
                 use_uint: bool = False,
                 ):

        num_dofs = 2

        super().__init__(
            num_envs,
            device,
            rendering,
            True,
            max_episode_steps,
            timestep,
            frame_skip,
            spacing,
            gravity,
            num_dofs,
            initial_is_paused=initial_is_paused,
            send_interrupt=send_interrupt,
            max_contact_pairs_per_env=max_contact_pairs_per_env,
            with_window=with_window)

        self.gym.set_num_solver_iterations(1)

        self.cart_bounds = cart_bounds
        self.pole_bounds = pole_bounds
        self.initial_pole_bounds = initial_pole_bounds
        self.action_scale = action_scale
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y
        self.frame_stack = frame_stack
        self.action_clip = action_clip
        self.obs_clip = obs_clip
        self.use_rgb = use_rgb
        self.use_uint = use_uint

        if self.use_uint:
            assert self.use_rgb, "Only RGB camera is compatible with unsigned integer output"

        self.rew_scale_alive = rew_scale_alive
        self.rew_scale_terminated = rew_scale_terminated
        self.rew_scale_pole_pos = rew_scale_pole_pos
        self.rew_scale_cart_vel = rew_scale_cart_vel
        self.rew_scale_pole_vel = rew_scale_pole_vel

        # Create environments
        self.create_envs()

        if self.use_rgb:
            self.num_channels = 3
        else:
            self.num_channels = 1

        # Initialise observation and action space
        self.single_observation_space = Box(
            low=np.full((self.resolution_y, self.resolution_x, self.num_channels *
                         self.frame_stack), -self.obs_clip),
            high=np.full((self.resolution_y, self.resolution_x, self.num_channels *
                          self.frame_stack), self.obs_clip), dtype=np.float32)
        self.single_action_space = Box(low=-self.action_clip, high=self.action_clip,
                                       dtype=np.float32)

        # Allocate buffers
        self.allocate_buffers()

        # Finalize gym
        self.gym.gym_finalize()

        if rendering:
            self.filename_prefix = path.dirname(path.abspath(argv[0]))
            self.filename_prefix += "/cartpole_environment_vision"

            self.env_list = v.UserCombo("Environment to save", ["Env {}".format(idx) for idx
                                                                in range(self.num_envs)], 0)

            self.image_box = v.UserCheckbox("Save image", False)

            self.gym_render.register_menu_item(self.env_list)
            self.gym_render.register_menu_item(self.image_box)

            self.first_image = True

            self.gym_render.reset_camera(v.Vec3(-6.0, 2.0, 100.0), v.Vec3(1.0, 0.0, 0.0))

    def render_callback(self):
        if self.image_box.get_value():
            self.save_image(self.env_list.get_current_index())
            self.image_box.set_value(False)

    def save_image(self, env_idx):

        if self.use_uint:
            dtype = torch.uint8
        else:
            dtype = torch.float32

        obs = self.obs_buf.reshape(self.num_envs, self.resolution_y, self.resolution_x,
                                   self.num_channels, self.frame_stack).to(dtype)

        for i in range(self.frame_stack):

            plt.imshow(obs[env_idx, :, :, :, i].cpu().numpy().reshape(
                (self.resolution_y, self.resolution_x, self.num_channels)), vmin=0, vmax=self.obs_clip)
            plt.gca().invert_yaxis()

            if self.first_image:
                if not self.use_rgb:
                    plt.colorbar()
                self.first_image = False

            filename = self.filename_prefix + "{}.png".format(i)

            plt.savefig(filename, bbox_inches='tight', pad_inches=0)

            print("Frame {} saved in {}".format(i, filename))

    def create_envs(self):

        if self.use_rgb:
            cartpole_asset_vision = "assets/vsim/cartpole_rgb/cartpole.vsim"
        else:
            cartpole_asset_vision = "assets/vsim/cartpole_depth.vsim"
        self.env_def_handle, self.arti_handle = create_envs_helper(
            self.gym, vision=True, cartpole_asset_vision=cartpole_asset_vision)

        # Get camera def
        self.env_def = self.gym.get_environment_def(self.env_def_handle)
        art_def_handle = self.env_def.get_articulation_def_handle_by_name("cartpole")
        art_def = self.env_def.get_articulation_def(art_def_handle)

        if self.use_rgb:
            camera_def = art_def.get_rgb_camera_def(0)
        else:
            camera_def = art_def.get_depth_camera_def(0)

        if self.resolution_x is not None:
            camera_def.resolution_x = self.resolution_x

        if self.resolution_y is not None:
            camera_def.resolution_y = self.resolution_y

        self.resolution_x = camera_def.resolution_x
        self.resolution_y = camera_def.resolution_y
        self.far_clip = camera_def.far_clip
        self.num_pixels = self.resolution_x * self.resolution_y

        # For debugging
        if self.rendering:
            if self.use_rgb:
                camera = art_def.get_rgb_camera(0)
            else:
                camera = art_def.get_depth_camera(0)
            camera.render_relative_transform = v.Transform(v.Quat(v.Vec3(0, 1, 0), torch.pi / 2),
                                                           v.Vec3(1.5, 2, 0))
            camera.render_width = 2
            camera.render_height = 0.5

            if not self.use_rgb:
                camera.render_min_depth = 0
                camera.render_max_depth = self.far_clip

        self.gym.get_environment_def(self.env_def_handle).finalize()

        super().create_envs(self.env_def_handle)

    def allocate_buffers(self):

        super().allocate_buffers()

        # Buffers for setting kinematic state
        self.set_dof_pos_buf = torch.zeros((self.num_envs, self.num_dofs), device=self.device,
                                           dtype=torch.float32)
        self.set_dof_vel_buf = torch.zeros((self.num_envs, self.num_dofs), device=self.device,
                                           dtype=torch.float32)

        # Create command for setting joint forces
        set_force_cmd = self.env_group.create_joint_state_command(v.wrap_gpu_buffer(self.act_buf),
                                                                  self.arti_handle, (0, 1))

        self.gpu_set_force_command_array = self.gym.create_joint_state_command_gpu_array(
            [set_force_cmd])

        # Create command for setting joint positions
        set_pos_cmd = self.env_group.create_joint_state_command(
            v.wrap_gpu_buffer(self.set_dof_pos_buf), self.arti_handle,
            masks_buffer=v.wrap_gpu_buffer(self.reset_buf))

        self.gpu_set_pos_command_array = self.gym.create_joint_state_command_gpu_array([
                                                                                       set_pos_cmd])

        # Create command for setting joint velocities
        set_vel_cmd = self.env_group.create_joint_state_command(
            v.wrap_gpu_buffer(self.set_dof_vel_buf), self.arti_handle,
            masks_buffer=v.wrap_gpu_buffer(self.reset_buf))

        self.gpu_set_vel_command_array = self.gym.create_joint_state_command_gpu_array([
                                                                                       set_vel_cmd])

        # Create command for getting joint positions
        get_pos_cmd = self.env_group.create_joint_state_command(
            v.wrap_gpu_buffer(self.get_dof_pos_buf), self.arti_handle)

        self.gpu_get_pos_command_array = self.gym.create_joint_state_command_gpu_array([
                                                                                       get_pos_cmd])

        # Create command for getting joint velocities
        get_vel_cmd = self.env_group.create_joint_state_command(
            v.wrap_gpu_buffer(self.get_dof_vel_buf), self.arti_handle)

        self.gpu_get_vel_command_array = self.gym.create_joint_state_command_gpu_array([
                                                                                       get_vel_cmd])

        # Create command for getting camera images
        articulation = self.env_def.get_articulation_by_name("cartpole")
        if self.use_rgb:
            camera_handle = articulation.get_rgb_camera_handle(0)
            num_output_channels = 4
        else:
            camera_handle = articulation.get_depth_camera_handle(0)
            num_output_channels = 1

        if self.use_uint:
            dtype = torch.uint8
        else:
            dtype = torch.float32

        self.image_buffer = torch.zeros((self.num_envs, self.resolution_y, self.resolution_x,
                                         num_output_channels), device=self.device, dtype=dtype)

        if self.use_rgb:
            get_rgb_cmd = self.env_group.create_rgb_camera_command(
                v.wrap_gpu_buffer(self.image_buffer), camera_handle)

            self.gpu_get_rgb_command_array = self.gym.create_rgb_camera_command_gpu_array(
                [get_rgb_cmd])
        else:
            get_depth_cmd = self.env_group.create_depth_camera_command(
                v.wrap_gpu_buffer(self.image_buffer), camera_handle)
            self.gpu_get_depth_command_array = self.gym.create_depth_camera_command_gpu_array(
                [get_depth_cmd])

        # Set up frame stack
        self.obs_history = deque(maxlen=self.frame_stack)
        for _ in range(self.frame_stack):
            self.obs_history.append(
                torch.zeros(
                    (self.num_envs,
                     self.resolution_y,
                     self.resolution_x,
                     self.num_channels),
                    dtype=dtype,
                    device=self.device))

    def reset_idx(self):

        self.set_dof_pos_buf[:,
                             1] = self.initial_pole_bounds[0] + torch.rand(self.num_envs,
                                                                           device=self.device,
                                                                           dtype=torch.float32) * (self.initial_pole_bounds[1] - self.initial_pole_bounds[0])

        # Set kinematic state
        self.gym.set_joint_positions(self.gpu_set_pos_command_array)
        self.gym.set_joint_velocities(self.gpu_set_vel_command_array)

        self.progress_buf[:] = torch.where(self.reset_buf, 0, self.progress_buf)

        # Reset history
        for i in range(self.frame_stack):
            self.obs_history[i][:] = torch.where(self.reset_buf.view(-1, 1, 1, 1), 0,
                                                 self.obs_history[i])

    def reset(self):

        super().reset()

        # Return observations
        self.compute_observations()

        return self.obs_buf.clone(), self.info

    def pre_physics_step(self, actions):

        self.reset_buf[:] = torch.logical_or(self.term_buf, self.trunc_buf)
        self.reset_idx()

        self.act_buf[:] = self.action_scale * torch.clamp(actions, - self.action_clip,
                                                          self.action_clip)

        self.gym.set_joint_forces(self.gpu_set_force_command_array)

    def post_physics_step(self):
        self.progress_buf += 1

        self.gym.update_scene_dependent_components()

        self.compute_observations()
        self.compute_reward_termination_truncation()

    def compute_observations(self):

        if self.use_rgb:
            self.gym.get_rgb_camera_images(self.gpu_get_rgb_command_array)

            self.obs_history.append(self.image_buffer[:, :, :, :self.num_channels].clone())
        else:
            self.gym.get_depth_camera_images(self.gpu_get_depth_command_array)
            self.image_buffer[:] = torch.where(self.image_buffer >= self.far_clip, 0,
                                               self.image_buffer)
            self.obs_history.append(self.image_buffer.clone())

        obs = torch.clamp(torch.cat([torch.unsqueeze(self.obs_history[i], 4)
                          for i in range(self.frame_stack)], dim=-1), -self.obs_clip, self.obs_clip)

        self.obs_buf[:] = obs.reshape(self.num_envs, self.resolution_y,
                                      self.resolution_x, self.num_channels * self.frame_stack)

    def compute_reward_termination_truncation(self):

        self.gym.get_joint_positions(self.gpu_get_pos_command_array)
        self.gym.get_joint_velocities(self.gpu_get_vel_command_array)

        cart_pos = self.get_dof_pos_buf[:, 0]
        pole_angle = self.get_dof_pos_buf[:, 1]
        cart_vel = self.get_dof_vel_buf[:, 0]
        pole_vel = self.get_dof_vel_buf[:, 1]

        # Check if Out Of Bounds (OOB)
        cart_OOB = torch.logical_or(cart_pos < self.cart_bounds[0], self.cart_bounds[1] < cart_pos)
        pole_OOB = torch.logical_or(pole_angle < self.pole_bounds[0], self.pole_bounds[1] <
                                    pole_angle)
        self.term_buf[:] = torch.logical_or(cart_OOB, pole_OOB)

        # Reward is 1 if within bounds, else zero
        alive_rew = self.rew_scale_alive * (1.0 - self.term_buf.float())
        terminated_rew = self.rew_scale_terminated * self.term_buf.float()
        pole_pos_rew = self.rew_scale_pole_pos * torch.square(cart_pos)
        cart_vel_rew = self.rew_scale_cart_vel * torch.abs(cart_vel)
        pole_vel_rew = self.rew_scale_pole_vel * torch.abs(pole_vel)

        self.rew_buf[:] = alive_rew + terminated_rew + pole_pos_rew + cart_vel_rew + pole_vel_rew

        # Truncate if max episode length is reached
        self.trunc_buf[:] = self.progress_buf >= self.max_episode_length


if __name__ == "__main__":
    from time import sleep
    from vlearn.utils import get_VL_VISUAL_TESTS

    rendering = True
    with_window = get_VL_VISUAL_TESTS()

    num_iter = np.inf
    if len(argv) >= 2:
        try:
            num_iter = int(argv[1])
        except ValueError:
            num_iter = float(argv[1])

    use_rgb = False
    if len(argv) >= 3:
        if argv[2].lower() == "rgb":
            use_rgb = True

    # num_envs = 4096
    # num_envs = 1
    num_envs = 64

    resolution_x = None
    resolution_y = None
    frame_pause = 1 / 20

    # resolution_x = 80
    # resolution_y = 80
    # frame_pause = 0

    initial_is_paused = False

    assert torch.cuda.is_available()
    device = torch.device("cuda:0")

    envs = CartpoleEnvironmentVision(
        num_envs,
        device=device,
        rendering=rendering,
        resolution_x=resolution_x,
        resolution_y=resolution_y,
        initial_is_paused=initial_is_paused,
        with_window=with_window,
        use_rgb=use_rgb)
    obs, _ = envs.reset()

    if rendering:
        gym = v.get_gym()
        render = gym.get_render()
        mode = 'manual'
        print(
            """Controls:
 F: -100
 G:  -50
 H:   50
 J:  100
""")

        render.capped_step = True
        render.reset_camera(v.Vec3(3, 2, 0), v.Vec3(-1, 0, 0))

    else:
        mode = 'random'

    finished = False
    idx = 0
    while not finished:

        if mode == 'manual':
            action = 0
            if render.is_key_down('f'):
                action = -1
            if render.is_key_down('g'):
                action = -0.5
            if render.is_key_down('h'):
                action = 0.5
            if render.is_key_down('j'):
                action = 1

            actions = torch.tensor([action])

        elif mode == 'random':
            actions = torch.rand((num_envs, 1)) * 20 - 10

        # print("actions: {}".format(actions))

        obs, rew, reset, timeout, _ = envs.step(actions)
        # print(obs, rew, reset, timeout)

        finished = envs.render_finished

        sleep(frame_pause)

        idx += 1
        if idx >= num_iter:
            finished = True
