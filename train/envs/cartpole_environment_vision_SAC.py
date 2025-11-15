import numpy as np
import torch
import matplotlib.pyplot as plt

from vlearn.spaces import Box

import sys, os
from sys import argv
from os import path

import vlearn as v

from vlearn.rl_algos.sac.vsim_training_data import pre_proc_hwc_to_chw, save_image_from_data

if __name__ == "__main__":
    from environment import EnvironmentGpu
    from cartpole_environment_common import create_envs_helper
else:
    from .environment import EnvironmentGpu
    from .cartpole_environment_common import create_envs_helper


class CartpoleEnvironmentVisionSAC(EnvironmentGpu):

    def __init__(self,
                 num_envs: int,
                 device: torch.device,
                 rendering: bool = False,
                 max_episode_length: int = 1000,
                 cartpole_asset_vision: str = "assets/vsim/cartpole_rgb/cartpole.vsim",
                 rgb_cameras: list[str] = [],
                 depth_cameras: list[str] = [],
                 cart_bounds: tuple[float, float] = [-3.5, 3.5],
                 pole_bounds: tuple[float, float] = [-0.25 * torch.pi, 0.25 * torch.pi],
                 initial_pole_bounds: tuple[float, float] = [-0.05 * torch.pi, 0.05 * torch.pi],
                 reward_pole_bounds: tuple[float, float] = [-0.24 * torch.pi, 0.24 * torch.pi],
                 force_bounds: tuple[float, float] = [-10, 10],
                 spacing: float = 10,
                 gravity: v.Vec3 = v.Vec3(0, -9.81, 0),
                 timestep: float = 0.01667,
                 frame_skip: int = 4,
                 initial_is_paused: bool = False,
                 send_interrupt: bool = False,
                 max_contact_pairs_per_env: int = 1,
                 with_window: bool = True
                 ):

        num_dofs = 2
        assert (max_episode_length % frame_skip == 0)
        num_training_steps = max_episode_length // frame_skip

        if isinstance(device, str):
            device = torch.device(device)

        super().__init__(
            num_envs,
            device,
            rendering,
            True,
            num_training_steps,
            timestep,
            frame_skip,
            spacing,
            gravity,
            num_dofs,
            initial_is_paused=initial_is_paused,
            send_interrupt=send_interrupt,
            max_contact_pairs_per_env=max_contact_pairs_per_env,
            with_window=with_window,
            update_scene_dependent_components_in_step=True)

        self.cartpole_asset_vision = cartpole_asset_vision
        self.rgb_cameras = rgb_cameras
        self.depth_cameras = depth_cameras
        self.num_cameras = len(rgb_cameras) + len(depth_cameras)
        self.cart_bounds = cart_bounds
        self.pole_bounds = pole_bounds
        self.initial_pole_bounds = initial_pole_bounds
        self.reward_pole_bounds = reward_pole_bounds
        self.force_bounds = force_bounds

        # Initialise observation and action space
        self.num_obs = 2  # 2 dof velocities + 2 dof pos (privileged)
        self.priv_obs_handle = 2  # start of privileged obs

        self.single_observation_space = Box(
            low=np.array([np.finfo('f').min] * self.num_obs),
            high=np.array([np.finfo('f').max] * self.num_obs),
            dtype=np.float32)

        self.single_action_space = Box(low=force_bounds[0], high=force_bounds[1], dtype=np.float32)

        # Create environments
        self.create_envs()

        # Allocate buffers
        self.allocate_buffers()

        # Finalize gym
        self.gym.set_num_solver_iterations(1)
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

    def create_envs(self):
        self.env_def_handle, self.arti_handle = create_envs_helper(
            self.gym, vision=True, cartpole_asset_vision=self.cartpole_asset_vision)

        # Get camera def
        self.env_def = self.gym.get_environment_def(self.env_def_handle)
        art_def_handle = self.env_def.get_articulation_def_handle_by_name("cartpole")
        art_def = self.env_def.get_articulation_def(art_def_handle)

        # Cameras
        self.res_x = []
        self.res_y = []
        self.buf_channels = []
        self.far_clips = []

        # RGB cameras
        for i in range(len(self.rgb_cameras)):
            camera_def = art_def.get_rgb_camera_def(i)

            self.res_x.append(camera_def.resolution_x)
            self.res_y.append(camera_def.resolution_y)
            self.buf_channels.append(4)
            self.far_clips.append(camera_def.far_clip)

            print('RGB camera {} resolution x {} y {}'.format(i, self.res_x[i], self.res_y[i]))

            # For debugging
            if self.rendering:
                camera = art_def.get_rgb_camera(i)

                camera.render_relative_transform = v.Transform(
                    v.Quat(v.Vec3(0, 1, 0), torch.pi / 2), v.Vec3(1.5, 2 + i * 0.5, 0))

                camera.render_width = 2
                camera.render_height = 0.5

        # Depth cameras
        for i in range(len(self.depth_cameras)):
            camera_def = art_def.get_depth_camera_def(i)

            self.res_x.append(camera_def.resolution_x)
            self.res_y.append(camera_def.resolution_y)
            self.buf_channels.append(1)
            self.far_clips.append(camera_def.far_clip)

            print('Depth camera {} resolution x {} y {}'.format(i, self.res_x[i], self.res_y[i]))

            # For debugging
            if self.rendering:
                camera = art_def.get_depth_camera(i)

                offset = len(self.rgb_cameras)
                camera.render_relative_transform = v.Transform(
                    v.Quat(v.Vec3(0, 1, 0), torch.pi / 2), v.Vec3(1.5, 2 + (i + offset) * 0.5, 0))

                camera.render_width = 2
                camera.render_height = 0.5

                camera.render_min_depth = 0
                camera.render_max_depth = camera_def.far_clip

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

        # Create command for getting camera images
        self.image_buffers = []
        self.info['image'] = []
        self.info['image_scalars'] = []
        self.info["cameras"] = self.rgb_cameras + self.depth_cameras  # ordering is important

        get_rgb_cmds = []
        for i in range(len(self.rgb_cameras)):
            camera_handle = articulation.get_rgb_camera_handle(i)

            self.image_buffers.append(
                torch.zeros((self.total_num_envs, self.res_y[i], self.res_x[i], 4),
                            dtype=torch.uint8, device=self.device)
                )

            get_rgb_cmd = self.env_group.create_rgb_camera_command(
                v.wrap_gpu_buffer(self.image_buffers[i]), camera_handle)

            get_rgb_cmds.append(get_rgb_cmd)

            self.info['image'].append(
                torch.empty((self.total_num_envs, 3, self.res_y[i], self.res_x[i]),
                            dtype=torch.uint8, device=self.device)
                )

            self.info['image_scalars'].append(1.0 / 255.0)

        self.gpu_get_rgb_command_array = self.gym.create_rgb_camera_command_gpu_array(get_rgb_cmds)

        get_depth_cmds = []
        for i in range(len(self.depth_cameras)):
            camera_handle = articulation.get_depth_camera_handle(i)

            self.image_buffers.append(
                torch.zeros((self.total_num_envs, self.res_y[i], self.res_x[i], 1),
                            dtype=torch.float32, device=self.device)
                )

            get_depth_cmd = self.env_group.create_depth_camera_command(
                v.wrap_gpu_buffer(self.image_buffers[i]), camera_handle)

            get_depth_cmds.append(get_depth_cmd)

            self.info['image'].append(
                torch.empty((self.total_num_envs, 1, self.res_y[i], self.res_x[i]),
                            dtype=torch.float32, device=self.device)
                )

            self.info['image_scalars'].append(1.0)

        self.gpu_get_depth_command_array = self.gym.create_depth_camera_command_gpu_array(
            get_depth_cmds)

        print("info['image_scalars'] ", self.info['image_scalars'])

    def reset_idx(self):
        self.set_dof_pos_buf[:,
                             1] = self.initial_pole_bounds[0] + torch.rand(self.num_envs,
                                                                           device=self.device,
                                                                           dtype=torch.float32) * (self.initial_pole_bounds[1] - self.initial_pole_bounds[0])

        # Set kinematic state
        self.gym.set_joint_positions(self.gpu_set_pos_command_array)
        self.gym.set_joint_velocities(self.gpu_set_vel_command_array)

        self.progress_buf[:] = torch.where(self.reset_buf, 0, self.progress_buf)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx()

        # Kinematics and queries
        self.gym.compute_kinematics()
        self.gym.update_scene_dependent_components()

        # Return observations
        self.compute_observations()

        return self.obs_buf.clone(), self.info

    def pre_physics_step(self, actions):
        self.act_buf[:] = actions
        self.gym.set_joint_forces(self.gpu_set_force_command_array)

    def post_physics_step(self):
        self.progress_buf[:] += 1
        self.reset_buf[:] = torch.logical_or(self.term_buf, self.trunc_buf)

        self.reset_idx()

        self.compute_observations()
        self.compute_reward_termination()
        self.trunc_buf[:] = self.progress_buf >= self.max_episode_length

    def compute_observations(self):
        self.gym.get_joint_positions(self.gpu_get_pos_command_array)
        self.gym.get_joint_velocities(self.gpu_get_vel_command_array)

        self.obs_buf[:] = self.get_dof_vel_buf
        # self.obs_buf[:] = torch.cat((self.get_dof_vel_buf,
        #                             self.get_dof_pos_buf),
        #                             dim=-1)

        if self.rgb_cameras:
            self.gym.get_rgb_camera_images(self.gpu_get_rgb_command_array)

        if self.depth_cameras:
            self.gym.get_depth_camera_images(self.gpu_get_depth_command_array)

        i = 0
        for _ in self.rgb_cameras:
            self.info['image'][i][:] = pre_proc_hwc_to_chw(self.image_buffers[i][..., :3])
            i += 1

        for _ in self.depth_cameras:
            mask = self.image_buffers[i] > self.far_clips[i]

            self.image_buffers[i][:] = torch.where(mask, -1.0, self.image_buffers[i])

            self.info['image'][i][:] = pre_proc_hwc_to_chw(self.image_buffers[i])
            i += 1

    def compute_reward_termination(self):
        cart_pos = self.get_dof_pos_buf[:, 0]
        pole_angle = self.get_dof_pos_buf[:, 1]

        # Check if Out Of Bounds (OOB)
        cart_OOB = torch.logical_or(cart_pos < self.cart_bounds[0], self.cart_bounds[1] < cart_pos)
        pole_OOB_term = torch.logical_or(
            pole_angle < self.pole_bounds[0],
            self.pole_bounds[1] < pole_angle)
        self.term_buf[:] = torch.logical_or(cart_OOB, pole_OOB_term)

        # Reward is 1 if within bounds, else zero
        pole_OOB_rew = torch.logical_or(
            pole_angle < self.reward_pole_bounds[0],
            self.reward_pole_bounds[1] < pole_angle)
        self.rew_buf[:] = torch.logical_not(torch.logical_or(cart_OOB, pole_OOB_rew))


im_dir = 'runs/images_cartpole'
os.makedirs(im_dir, exist_ok=True)

if __name__ == "__main__":
    from time import sleep
    from vlearn.utils import get_VL_VISUAL_TESTS

    with_window = get_VL_VISUAL_TESTS()
    rendering = True

    num_iter = np.inf
    if len(argv) >= 2:
        try:
            num_iter = int(argv[1])
        except ValueError:
            num_iter = float(argv[1])

    num_envs = 1

    frame_pause = 1 / 20

    assert torch.cuda.is_available()
    device = torch.device("cuda:0")

    envs = CartpoleEnvironmentVisionSAC(num_envs, device=device, rendering=rendering,
                                        initial_is_paused=True, with_window=with_window)

    obs, info = envs.reset()

    if 'image' in info:
        for im in info['image']:
            print('Images buf dims: ', im.shape)

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

        obs, rew, reset, timeout, info = envs.step(actions)
        # print(obs, rew, reset, timeout)

        for i in range(envs.num_cameras):
            image = envs.image_buffers[i][0].cpu()
            im_file = os.path.join(im_dir, f'cartpole_{i}_{idx}.png')
            save_image_from_data(im_file, image)

        finished = envs.render_finished

        sleep(frame_pause)

        idx += 1
        if idx >= num_iter:
            finished = True
