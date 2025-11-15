import torch
import numpy as np
from vlearn.spaces import Box

import vlearn as v

import sys
import os

if __name__ == "__main__":
    from environment import EnvironmentGpu
    from common import create_plane, reset_noise_helper
    from h1_environment_common import create_envs_helper, store_initial_conditions_helper, allocate_gpu_buffers
else:
    from .environment import EnvironmentGpu
    from .common import create_plane, reset_noise_helper
    from .h1_environment_common import create_envs_helper, store_initial_conditions_helper, allocate_gpu_buffers


class H1EnvironmentGpu(EnvironmentGpu):

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
                 healthy_y_range: tuple[float, float] = (0.7, 2.0),
                 reset_noise_scale: float = 1,
                 contact_cost_weight: float = 5e-7,
                 contact_cost_range: tuple[float, float] = (-torch.inf, 10.0),
                 initial_is_paused: bool = False,
                 send_interrupt: bool = False,
                 max_contact_pairs_per_env: int = 128,
                 with_window: bool = True,
                 ):

        self.num_dofs = 19

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
            max_contact_pairs_per_env=max_contact_pairs_per_env,
            with_window=with_window)

        # Store environment-specific configuration
        self.forward_reward_weight = forward_reward_weight
        self.ctrl_cost_weight = ctrl_cost_weight
        self.healthy_reward = healthy_reward
        self.healthy_y_range = healthy_y_range
        self.reset_noise_scale = reset_noise_scale
        self.contact_cost_weight = contact_cost_weight
        self.contact_cost_range = contact_cost_range

        # Hardcode these for now
        self.num_motors = 19
        self.num_sensors = 2
        self.num_links = 20

        # Number of observations
        num_obs = 0
        num_obs += 1                    # root y-position
        num_obs += 4                    # root rotation
        num_obs += self.num_dofs        # joint positions
        num_obs += 6                    # root velocity
        num_obs += self.num_dofs        # joint velocities
        num_obs += self.num_links * 6   # link velocities
        num_obs += self.num_motors      # motor forces
        num_obs += self.num_sensors * 6

        # Define observation space
        self.single_observation_space = Box(
            low=np.array([np.finfo('f').min] * num_obs, dtype=np.float32),
            high=np.array([np.finfo('f').max] * num_obs, dtype=np.float32),
            dtype=np.float32)

        # Create environment
        self.create_envs()

        # Define action space
        assert self.num_motors == self.art_def.get_num_motor_defs()

        low = []
        high = []
        for i in range(self.num_motors):
            motor_def = self.art_def.get_motor_def(i)
            low.append(motor_def.low_limit)
            high.append(motor_def.high_limit)
        low = np.array(low, dtype=np.float32)
        high = np.array(high, dtype=np.float32)

        self.single_action_space = Box(low=low, high=high, dtype=np.float32)

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
            v.wrap_gpu_buffer(self.link_vel_buf), self.arti_handle)

        self.gpu_get_link_velocities_command_array = self.gym.create_link_velocity_command_gpu_array([
                                                                                                     get_link_velocities_command])

    def reset_idx(self):

        # Set kinematic state
        self.set_dof_pos_buf[:] = self.gpu_init_dof_pos + \
            reset_noise_helper(self.gpu_init_dof_pos, self.reset_noise_scale, 0.4, 0.2)
        self.set_dof_vel_buf[:] = reset_noise_helper(
            self.gpu_init_dof_vel, self.reset_noise_scale, 0.2, 0.1)

        self.gym.set_articulation_kinematic_states(self.gpu_set_kinematic_state_command_array)

        # Reset remaining buffers
        self.old_root_pos_buf[:] = torch.where(self.reset_buf.view(-1, 1),
                                               self.gpu_init_root_transforms, self.old_root_pos_buf)

        self.act_buf[:] = torch.where(self.reset_buf.view(-1, 1), 0, self.act_buf)
        self.progress_buf[:] = torch.where(self.reset_buf, 0, self.progress_buf)

    def reset(self):

        super().reset()

        # Return observations
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
        self.gym.get_sensor_forces(self.gpu_get_sensor_forces_command_array)

        # Conversion operation for compatibility with handle-based sensor API
        self.sensor_force_buf = torch.stack(self.force_sensor_buffers, dim=1)

        torso_pos = self.root_pos_buf[:, 5].unsqueeze(-1)  # y position
        torso_rot = self.root_pos_buf[:, 0:4]  # values 0-4 of tensor

        velocity = self.root_vel_buf[:, 3:6]
        ang_velocity = self.root_vel_buf[:, 0:3]

        link_velocity = self.link_vel_buf[:, :, 3:6].reshape(self.num_envs, self.num_links * 3)
        link_ang_velocity = self.link_vel_buf[:, :, 0:3].reshape(self.num_envs, self.num_links * 3)

        # y-position (1), rotation (4), dof pos (19), linear velocity (3), angular velocity (3), dof vel (19),
        # link linear velocity (60), link angular velocity (60), sensor forces (12), actions (19)
        obs = torch.cat((torso_pos, torso_rot, self.get_dof_pos_buf, velocity,
                         ang_velocity, self.get_dof_vel_buf, link_velocity, link_ang_velocity,
                         self.sensor_force_buf.view(-1, 6 * self.num_sensors), actions), dim=-1)

        self.obs_buf[:] = obs

    def compute_reward_termination_truncation(self, actions):

        # Termination
        healthy = torch.logical_and(
            self.healthy_y_range[0] <= self.obs_buf[:, 0],
            self.obs_buf[:, 0] <= self.healthy_y_range[1])

        self.term_buf[:] = torch.logical_not(healthy)

        # Healthy reward
        healthy_reward = torch.where(healthy, self.healthy_reward, 0)

        # Forward reward
        x_new = self.root_pos_buf[:, 4]
        x_old = self.old_root_pos_buf[:, 4]

        x_vel = (x_new - x_old) / self.dt

        forward_reward = self.forward_reward_weight * x_vel

        # Action cost
        action_cost = self.ctrl_cost_weight * torch.sum(actions * actions, dim=-1)

        # Contact cost
        sensor_forces = self.sensor_force_buf.view(-1, 6 * self.num_sensors)
        contact_cost = torch.clamp(self.contact_cost_weight *
                                   torch.sum(sensor_forces * sensor_forces, dim=-1),
                                   min=self.contact_cost_range[0],
                                   max=self.contact_cost_range[1])

        # Total reward
        # print("healthy_reward: {}".format(healthy_reward))
        # print("forward_reward: {}".format(forward_reward))
        # print("action_cost: {}".format(action_cost))
        # print("contact_cost: {}".format(contact_cost))
        self.rew_buf[:] = healthy_reward + forward_reward - action_cost - contact_cost

        self.trunc_buf[:] = self.progress_buf >= self.max_episode_length


if __name__ == "__main__":
    from time import sleep
    from sys import argv
    from vlearn.utils import get_VL_VISUAL_TESTS

    rendering = True
    with_window = get_VL_VISUAL_TESTS()

    num_iter = np.inf
    if len(argv) >= 2:
        num_iter = int(argv[1])

    mode = "joint_monkey"
    # mode = "motor"

    assert mode in ["motor", "joint_monkey"]

    num_envs = 1
    max_episode_length = 1000

    device = torch.device("cuda:0")

    envs = H1EnvironmentGpu(num_envs, device, rendering=rendering,
                            max_episode_length=max_episode_length, with_window=with_window)

    obs, _ = envs.reset()

    gym = v.get_gym()
    render = gym.get_render()

    # Reset interface
    if render is not None:
        reset_box = v.UserCheckbox("Reset", False)
        render.register_menu_item(reset_box)

        # Control motor forces
        if mode == "motor":
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

        # Joint monkey
        elif mode == "joint_monkey":
            joint_pos = torch.zeros((envs.num_envs, envs.num_dofs),
                                    dtype=torch.float32, device=device)
            joint_vel = torch.zeros((envs.num_envs, envs.num_dofs),
                                    dtype=torch.float32, device=device)
            root_pos = torch.zeros((envs.num_envs, 7), dtype=torch.float32, device=device)
            root_vel = torch.zeros((envs.num_envs, 6), dtype=torch.float32, device=device)

            rot = v.shortest_rotation(v.Vec3(0, 0, 1), v.Vec3(0, 1, 0))
            pos = v.Vec3(0, 1.4, 0)
            root_pos[:, 0] = rot.x
            root_pos[:, 1] = rot.y
            root_pos[:, 2] = rot.z
            root_pos[:, 3] = rot.w
            root_pos[:, 4] = pos.x
            root_pos[:, 5] = pos.y
            root_pos[:, 6] = pos.z

            set_kine_cmd = envs.env_group.create_articulation_kinematic_state_command(
                v.wrap_gpu_buffer(joint_pos),
                v.wrap_gpu_buffer(joint_vel),
                v.wrap_gpu_buffer(root_pos),
                v.wrap_gpu_buffer(root_vel),
                envs.arti_handle,
                link_index_range=(0, 1))

            cmd_array = envs.gym.create_articulation_kinematic_state_command_gpu_array([
                                                                                       set_kine_cmd])

            def joint_monkey_pre_physics_step(envs, actions):

                assert isinstance(actions, torch.Tensor)

                # Save old root transforms for reward calculation
                envs.old_root_pos_buf[:] = envs.root_pos_buf

                root_pos[:, 4:7] = actions[:, 0:3]

                for i in range(envs.num_envs):
                    rpy = actions[i, 3:6]
                    quat = v.quat_from_rpy(v.Vec3(rpy[0], rpy[1], rpy[2]))
                    quat = rot * quat
                    root_pos[i, 0] = quat.x
                    root_pos[i, 1] = quat.y
                    root_pos[i, 2] = quat.z
                    root_pos[i, 3] = quat.w

                joint_pos[:, :] = actions[:, 6:25]

                envs.act_buf[:] = actions[:, 25:]

                envs.gym.set_articulation_kinematic_states(cmd_array)

            H1EnvironmentGpu.pre_physics_step = joint_monkey_pre_physics_step
            H1EnvironmentGpu.reset_idx = lambda self: None

            sliders = []

            # Set root position
            sliders.append(v.UserSlider("root x", -5, 5, 0))
            sliders.append(v.UserSlider("root y", -5, 5, 1.4))
            sliders.append(v.UserSlider("root z", -5, 5, 0))

            # Set root orientation
            sliders.append(v.UserSlider("root roll", -torch.pi, torch.pi, 0))
            sliders.append(v.UserSlider("root pitch", -torch.pi, torch.pi, 0))
            sliders.append(v.UserSlider("root yaw", -torch.pi, torch.pi, 0))

            # Set joint position
            for i in range(envs.art_def.get_num_joint_dof_defs()):

                dofdef = envs.art_def.get_joint_dof_def(i)
                name = envs.art_def.get_joint_dof_def_name(i)

                low, high = dofdef.get_limits()
                init = np.clip(0, low, high)

                sliders.append(v.UserSlider(name, low, high, init))

            # Set motor forces
            for i in range(envs.art_def.get_num_motor_defs()):
                name = envs.art_def.get_motor_def_name(i)
                motor_def = envs.art_def.get_motor_def(i)
                low = motor_def.low_limit
                high = motor_def.high_limit

                sliders.append(v.UserSlider("motor." + name, low, high, 0))

            for slider in sliders:
                render.register_menu_item(slider)

            def control_by_menu():

                if reset_box.get_value():
                    obs, _ = envs.reset()
                    print_obs(obs)

                return torch.tile(torch.tensor([slider.get_value()
                                  for slider in sliders]), (envs.num_envs, 1))

            control_fn = control_by_menu

    else:
        def control_fn(n=num_envs, m=envs.art_def.get_num_motor_defs()): return torch.zeros((n, m))

    # Print observations
    def print_obs(obs):
        # print("obs.shape: {}".format(obs.shape))
        # print("obs: {}".format(obs))
        # print("root pos: {}".format(obs[:, 0])) # root pos
        # print("root rot: {}".format(obs[:, 1:5])) # root rot
        # print("dof pos: {}".format(obs[:, 5:24])) # dof pos
        # print("root linear vel: {}".format(obs[:, 24:27])) # root linear vel
        # print("root ang vel: {}".format(obs[:, 27:30])) # root angular vel
        # print("dof vel: {}".format(obs[:, 30:49])) # dof vel
        # print("link vel: {}".format(obs[:, 49:109])) # link velocity
        # print("link ang vel: {}".format(obs[:, 109:169])) # link angular velocity
        # print("sensor forces: {}".format(obs[:, 169:181])) # sensor forces
        # print("motor forces: {}".format(obs[:, 181:200])) # motor forces
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
