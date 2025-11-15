import torch
import numpy as np
from vlearn.spaces import Box

import vlearn as v

import sys
import os

if __name__ == "__main__":
    from environment import EnvironmentGpu
    from vlearn.torch_utils.torch_jit_utils import *
    from common import create_plane, reset_noise_helper
    from h1_environment_common import create_envs_helper, store_initial_conditions_helper, allocate_gpu_buffers
else:
    from .environment import EnvironmentGpu
    from vlearn.torch_utils.torch_jit_utils import *
    from .common import create_plane, reset_noise_helper
    from .h1_environment_common import create_envs_helper, store_initial_conditions_helper, allocate_gpu_buffers


@torch.jit.script
def reset_potentials(targets, gpu_init_root_translation, dt: float,
                     reset_buf, prev_potentials, potentials, zero_y):
    to_target = (targets - gpu_init_root_translation) * zero_y

    new_potentials = -torch.norm(to_target, p=2, dim=-1) / dt

    prev_potentials[:] = torch.where(reset_buf, new_potentials, prev_potentials)
    potentials[:] = torch.where(reset_buf, new_potentials, potentials)


@torch.jit.script
def compute_observations_isaac_gym_helper(
        obs_buf,
        root_pos_buf,
        root_vel_buf,
        targets,
        potentials,
        inv_start_rot,
        dof_pos,
        dof_vel,
        dof_force,
        dof_limits_lower,
        dof_limits_upper,
        dof_vel_scale: float,
        sensor_force_torques,
        actions,
        dt: float,
        contact_force_scale: float,
        angular_velocity_scale: float,
        basis_vec0,
        basis_vec1,
        up_axis_idx: int,
        zero_y):

    torso_rotation = root_pos_buf[:, 0:4]
    torso_position = root_pos_buf[:, 4:7]
    ang_velocity = root_vel_buf[:, 0:3]
    velocity = root_vel_buf[:, 3:6]

    to_target = (targets - torso_position) * zero_y
    # to_target[:, 1] = 0

    prev_potentials_new = potentials.clone()
    potentials = -torch.norm(to_target, p=2, dim=-1) / dt

    torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
        torso_rotation, inv_start_rot, to_target, basis_vec0, basis_vec1, up_axis_idx)

    vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
        torso_quat, velocity, ang_velocity, targets, torso_position)

    roll = normalize_angle(roll).unsqueeze(-1)
    yaw = normalize_angle(yaw).unsqueeze(-1)
    angle_to_target = normalize_angle(angle_to_target).unsqueeze(-1)

    # this normalizes the dof positions in scaled range of [-1, +1]
    dof_pos_scaled = unscale(dof_pos, dof_limits_lower, dof_limits_upper)
    # dof_pos_scaled = (dof_pos - dof_limits_lower)/(dof_limits_upper - dof_limits_lower)

    # obs_buf shapes: 1, 3, 3, 1, 1, 1, 1, 1, num_dofs (19), num_dofs (19),
    # num_dofs (19), 12, num_acts (19)
    obs = torch.cat((torso_position[:, 1].view(-1, 1), vel_loc, angvel_loc * angular_velocity_scale,
                     yaw, roll, angle_to_target, up_proj.unsqueeze(-1), heading_proj.unsqueeze(-1),
                     dof_pos_scaled, dof_vel * dof_vel_scale, dof_force * contact_force_scale,
                     sensor_force_torques.view(-1, 12) * contact_force_scale, actions), dim=-1)

    return obs, potentials, prev_potentials_new, up_vec, heading_vec


@torch.jit.script
def compute_reward_termination_truncation_isaac_gym_helper(
        obs_buf,
        reset_buf,
        one_masks_buf,
        zero_masks_buf,
        progress_buf,
        alive_reward_buf,
        death_cost_buf,
        actions,
        up_weight: float,
        heading_weight: float,
        potentials,
        prev_potentials,
        actions_cost_scale: float,
        energy_cost_scale: float,
        joints_at_limit_cost_scale: float,
        max_motor_effort,
        motor_efforts,
        termination_height: float,
        death_cost: float,
        max_episode_length: int,
        ):

    # reward from the direction headed
    heading = obs_buf[:, 11]
    heading_weight_ratio = heading_weight / 0.8
    heading_weight_tensor = one_masks_buf * heading_weight
    heading_reward = torch.where(
        heading > 0.8,
        heading_weight_tensor,
        heading * heading_weight_ratio)

    # reward for being upright
    up_weight_tensor = one_masks_buf * up_weight
    up_reward = torch.where(obs_buf[:, 10] > 0.93, up_weight_tensor, zero_masks_buf)

    actions_cost = torch.sum(actions * actions, dim=-1)

    # energy cost reward
    motor_effort_ratio = motor_efforts / max_motor_effort
    scaled_cost = joints_at_limit_cost_scale * (torch.abs(obs_buf[:, 12:31]) - 0.98) / 0.02
    dof_at_limit_cost = torch.sum(
        (torch.abs(obs_buf[:, 12:31]) > 0.98) * scaled_cost * motor_effort_ratio.unsqueeze(0), dim=-1)

    electricity_cost = torch.sum(
        torch.abs(actions * obs_buf[:, 31:50]) * motor_effort_ratio.unsqueeze(0), dim=-1)

    # reward for duration of being alive
    # alive_reward = torch.ones_like(potentials) * 2.0
    progress_reward = potentials - prev_potentials

    total_reward = progress_reward + alive_reward_buf + up_reward + heading_reward - \
        actions_cost_scale * actions_cost - energy_cost_scale * electricity_cost - dof_at_limit_cost

    # adjust reward for fallen agents
    total_reward = torch.where(obs_buf[:, 0] < termination_height, death_cost_buf, total_reward)

    # reset agents
    reset = torch.where(obs_buf[:, 0] < termination_height, torch.ones_like(reset_buf), reset_buf)
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)

    return total_reward, reset


class H1EnvironmentIsaacGym(EnvironmentGpu):

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
                 termination_height: float = 0.7,
                 reset_noise_scale: float = 1,
                 dof_vel_scale: float = 0.1,
                 contact_force_scale: float = 0.01,
                 up_weight: float = 0.1,
                 heading_weight: float = 0.5,
                 actions_cost_scale: float = 0.01,
                 energy_cost_scale: float = 0.25,
                 angular_velocity_scale: float = 0.25,
                 joints_at_limit_cost_scale: float = 0.25,
                 death_cost: float = -2.0,
                 power_scale: float = 1.0,
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

        # Hardcode these for now
        self.num_motors = 19
        self.num_sensors = 2
        self.num_links = 20

        # Store environment-specific configuration
        self.reset_noise_scale = reset_noise_scale

        # Isaac Gym stuff
        # self.dt = self.timestep
        self.termination_height = termination_height
        self.dof_vel_scale = dof_vel_scale
        self.contact_force_scale = contact_force_scale
        self.up_weight = up_weight
        self.heading_weight = heading_weight
        self.actions_cost_scale = actions_cost_scale
        self.energy_cost_scale = energy_cost_scale
        self.angular_velocity_scale = angular_velocity_scale
        self.joints_at_limit_cost_scale = joints_at_limit_cost_scale
        self.death_cost = death_cost
        self.power_scale = power_scale

        self.targets = torch.tensor([1000, 0, 0], device=self.device,
                                    dtype=torch.float32).repeat((self.num_envs, 1))

        self.up_axis_idx = 1  # up axis is Y
        self.basis_vec0 = torch.tensor([1, 0, 0], device=self.device,
                                       dtype=torch.float32).repeat((self.num_envs, 1))
        self.basis_vec1 = torch.tensor([0, 1, 0], device=self.device,
                                       dtype=torch.float32).repeat((self.num_envs, 1))

        # Number of observations
        num_obs = 0
        num_obs += 1                    # root y-position
        num_obs += 3                    # root linear velocity
        num_obs += 3                    # root angular velocity
        num_obs += 1                    # yaw
        num_obs += 1                    # roll
        num_obs += 1                    # angle_to_target
        num_obs += 1                    # up_proj
        num_obs += 1                    # heading_proj
        num_obs += self.num_dofs        # dof positions
        num_obs += self.num_dofs        # dof velocities
        num_obs += self.num_dofs        # dof forces
        num_obs += self.num_sensors * 6  # sensor forces
        num_obs += self.num_motors      # motor forces

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
        motor_efforts = []
        for i in range(self.num_motors):
            motor_def = self.art_def.get_motor_def(i)

            low.append(motor_def.low_limit)
            high.append(motor_def.high_limit)
            motor_efforts.append(motor_def.gear_ratio)

        low = np.array(low, dtype=np.float32)
        high = np.array(high, dtype=np.float32)
        self.force_scale = torch.tensor(np.maximum(np.abs(low), np.abs(high)), device=self.device)

        self.single_action_space = Box(low=low, high=high, dtype=np.float32)

        # Save motor efforts (also Isaac Gym)
        self.motor_efforts = torch.tensor(motor_efforts, dtype=torch.float32, device=self.device)
        self.max_motor_effort = self.motor_efforts.max()

        # Store DOF data
        self.store_initial_conditions()

        # Allocate buffers
        self.allocate_buffers()

        # Create plane
        create_plane(self.gym)

        # Finalize gym
        self.gym.gym_finalize()

    def create_envs(self):

        rot = v.shortest_rotation(v.Vec3(0, 0, 1), v.Vec3(0, 1, 0))
        pos = v.Vec3(0, 1.4, 0)
        transform = v.Transform(rot, pos)

        self.env_def_handle, self.art_def, self.arti_handle = create_envs_helper(
            self.gym, transform=transform)

        self.gym.get_environment_def(self.env_def_handle).finalize()

        super().create_envs(self.env_def_handle)

    def store_initial_conditions(self):

        self.dof_pos_init, self.root_trans_init, self.root_vel_init = store_initial_conditions_helper(
            self.art_def, self.device)

        # Isaac Gym stuff
        dof_limits_lower = []
        dof_limits_upper = []
        for dofdef in self.art_def.get_joint_dof_defs():
            low, high = dofdef.get_limits()

            dof_limits_lower.append(low)
            dof_limits_upper.append(high)

        self.dof_limits_lower = torch.tensor(
            dof_limits_lower, dtype=torch.float32, device=self.device)
        self.dof_limits_upper = torch.tensor(
            dof_limits_upper, dtype=torch.float32, device=self.device)

        inv_start_rot = self.root_trans_init.q.get_conjugate()
        self.inv_start_rot = torch.tensor([inv_start_rot.x,
                                           inv_start_rot.y,
                                           inv_start_rot.z,
                                           inv_start_rot.w],
                                          device=self.device,
                                          dtype=torch.float32).repeat((self.num_envs,
                                                                       1))

    def allocate_buffers(self):

        super().allocate_buffers()

        allocate_gpu_buffers(self)

        # Joint forces buffer
        self.joint_force_buf = torch.zeros((self.num_envs, self.num_dofs), device=self.device,
                                           dtype=torch.float32)

        # Get joint force sensor GPU command
        get_joint_force_sensor_command = self.env_group.create_joint_force_sensor_command(
            v.wrap_gpu_buffer(self.joint_force_buf), self.arti_handle)

        self.gpu_get_joint_forces_command_array = self.gym.create_joint_force_sensor_command_gpu_array(
            [get_joint_force_sensor_command])

        # Isaac gym stuff
        self.gpu_zero_y = torch.empty(3, dtype=torch.float32, device=self.device)
        self.gpu_zero_y[0] = 1
        self.gpu_zero_y[1] = 0
        self.gpu_zero_y[2] = 1
        self.gpu_zero_y = torch.tile(self.gpu_zero_y, (self.num_envs, 1))

        self.gpu_init_root_translation = torch.empty(3, dtype=torch.float32, device=self.device)
        self.gpu_init_root_translation[0] = self.root_trans_init.p.x
        self.gpu_init_root_translation[1] = self.root_trans_init.p.y
        self.gpu_init_root_translation[2] = self.root_trans_init.p.z
        self.gpu_init_root_translation = torch.tile(
            self.gpu_init_root_translation, (self.num_envs, 1))

        self.one_masks_buf = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.zero_masks_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.alive_reward_buf = self.one_masks_buf * 0.5
        self.death_cost_buf = self.one_masks_buf * self.death_cost

        self.potentials = torch.tensor([-1000.0 / self.dt],
                                       device=self.device,
                                       dtype=torch.float32).repeat(self.num_envs)
        self.prev_potentials = self.potentials.clone()

    def reset_idx(self):

        # Set kinematic state
        self.set_dof_pos_buf[:] = self.gpu_init_dof_pos + \
            reset_noise_helper(self.gpu_init_dof_pos, self.reset_noise_scale, 0.4, 0.2)
        self.set_dof_vel_buf[:] = reset_noise_helper(
            self.gpu_init_dof_vel, self.reset_noise_scale, 0.2, 0.1)

        self.gym.set_articulation_kinematic_states(self.gpu_set_kinematic_state_command_array)

        reset_potentials(self.targets, self.gpu_init_root_translation, self.dt,
                         self.reset_buf, self.prev_potentials, self.potentials, self.gpu_zero_y)

        self.act_buf[:] = torch.where(self.reset_buf.view(-1, 1), 0, self.act_buf)
        self.progress_buf[:] = torch.where(self.reset_buf, 0, self.progress_buf)

        # This is necessary because the reset buffer is not the logical_or of
        # term_buf and trunc_buf, like in the new Gymnasium API, and it is not
        # fully reset in compute_reward
        self.reset_buf[:] = torch.zeros_like(self.reset_buf)

    def reset(self):

        super().reset()

        # Return observations
        self.compute_observations(self.act_buf)

        return self.obs_buf.clone(), {}

    def pre_physics_step(self, actions):

        assert isinstance(actions, torch.Tensor)

        self.reset_idx()

        self.act_buf[:] = self.power_scale * actions

        self.gym.set_motor_forces(self.gpu_set_motor_data_array)

    def post_physics_step(self):
        self.progress_buf += 1

        self.compute_observations(self.act_buf)
        self.compute_reward_termination_truncation(self.act_buf)
        self.term_buf[:] = self.reset_buf

    def compute_observations(self, actions):

        self.gym.get_articulation_kinematic_states(self.gpu_get_kinematic_state_command_array)
        self.gym.get_sensor_forces(self.gpu_get_sensor_forces_command_array)

        # Conversion operation for compatibility with handle-based sensor API
        self.sensor_force_buf = torch.stack(self.force_sensor_buffers, dim=1)

        self.gym.get_joint_sensor_forces(self.gpu_get_joint_forces_command_array)

        self.obs_buf[:], self.potentials[:], self.prev_potentials[:], _, _ = \
            compute_observations_isaac_gym_helper(
            self.obs_buf, self.root_pos_buf, self.root_vel_buf, self.targets,
            self.potentials, self.inv_start_rot, self.get_dof_pos_buf, self.get_dof_vel_buf,
            self.joint_force_buf, self.dof_limits_lower, self.dof_limits_upper,
            self.dof_vel_scale, self.sensor_force_buf, actions, self.dt,
            self.contact_force_scale, self.angular_velocity_scale,
            self.basis_vec0, self.basis_vec1, self.up_axis_idx,
            self.gpu_zero_y)

    def compute_reward_termination_truncation(self, actions):

        self.rew_buf[:], self.reset_buf[:] = compute_reward_termination_truncation_isaac_gym_helper(
            self.obs_buf,
            self.reset_buf,
            self.one_masks_buf,
            self.zero_masks_buf,
            self.progress_buf,
            self.alive_reward_buf,
            self.death_cost_buf,
            actions,
            self.up_weight,
            self.heading_weight,
            self.potentials,
            self.prev_potentials,
            self.actions_cost_scale,
            self.energy_cost_scale,
            self.joints_at_limit_cost_scale,
            self.max_motor_effort,
            self.motor_efforts,
            self.termination_height,
            self.death_cost,
            self.max_episode_length,
            )


if __name__ == "__main__":
    from time import sleep
    from sys import argv
    from vlearn.utils import get_VL_VISUAL_TESTS

    rendering = True
    with_window = get_VL_VISUAL_TESTS()

    num_iter = np.inf
    if len(argv) >= 2:
        num_iter = int(argv[1])

    num_envs = 1
    max_episode_length = 1000

    device = torch.device("cuda:0")

    envs = H1EnvironmentIsaacGym(num_envs, device, rendering=rendering,
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
