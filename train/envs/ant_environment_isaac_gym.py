import torch
import numpy as np
from vlearn.spaces import Box
from hashlib import sha256

import vlearn as v

if __name__ == "__main__":
    from environment import EnvironmentGpu
    from vlearn.torch_utils.torch_jit_utils import *
    from common import create_plane, reset_noise_helper
    from ant_environment_common import (store_initial_conditions_helper,
                                        allocate_gpu_buffers, create_envs_helper)
else:
    from .environment import EnvironmentGpu
    from vlearn.torch_utils.torch_jit_utils import *
    from .common import create_plane, reset_noise_helper
    from .ant_environment_common import (store_initial_conditions_helper,
                                         allocate_gpu_buffers, create_envs_helper)


@torch.jit.script
def compute_observations_isaac_gym_helper(
        obs_buf, root_pos_buf, root_vel_buf, targets, potentials,
        inv_start_rot, dof_pos, dof_vel,
        dof_limits_lower, dof_limits_upper, dof_vel_scale: float,
        sensor_force_torques, actions, dt: float, contact_force_scale: float,
        basis_vec0, basis_vec1, up_axis_idx: int, zero_y):

    torso_rotation = root_pos_buf[:, 0:4]
    torso_position = root_pos_buf[:, 4:7]
    ang_velocity = root_vel_buf[:, 0:3]
    velocity = root_vel_buf[:, 3:6]

    to_target = (targets - torso_position) * zero_y
    # to_target[:, 1] = 0.0

    prev_potentials_new = potentials.clone()
    potentials = -torch.norm(to_target, p=2, dim=-1) / dt

    torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
        torso_rotation, inv_start_rot, to_target, basis_vec0, basis_vec1, up_axis_idx)

    vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
        torso_quat, velocity, ang_velocity, targets, torso_position)

    dof_pos_scaled = unscale(dof_pos, dof_limits_lower, dof_limits_upper)

    # obs_buf shapes: 1, 3, 3, 1, 1, 1, 1, 1, num_dofs(8), num_dofs(8), 24, num_dofs(8)
    obs = torch.cat((torso_position[:,
                                    up_axis_idx].view(-1,
                                                      1),
                     vel_loc,
                     angvel_loc,
                     yaw.unsqueeze(-1),
                     roll.unsqueeze(-1),
                     angle_to_target.unsqueeze(-1),
                     up_proj.unsqueeze(-1),
                     heading_proj.unsqueeze(-1),
                     dof_pos_scaled,
                     dof_vel * dof_vel_scale,
                     sensor_force_torques.view(-1,
                                               24) * contact_force_scale,
                     actions),
                    dim=-1)

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
        termination_height: float,
        max_episode_length: float
        ):

    # reward from direction headed
    heading = obs_buf[:, 11]
    heading_weight_ratio = heading_weight / 0.8
    heading_weight_tensor = one_masks_buf * heading_weight
    heading_reward = torch.where(
        heading > 0.8,
        heading_weight_tensor,
        heading * heading_weight_ratio)

    # aligning up axis of ant and environment
    up_reward = torch.where(obs_buf[:, 10] > 0.93, up_weight, zero_masks_buf)

    # energy penalty for movement
    actions_cost = torch.sum(actions * actions, dim=-1)
    electricity_cost = torch.sum(torch.abs(actions * obs_buf[:, 20:28]), dim=-1)
    dof_at_limit_cost = torch.sum(obs_buf[:, 12:20] > 0.99, dim=-1)

    # reward for duration of staying alive
    # alive_reward = torch.ones_like(potentials) * 0.5
    progress_reward = potentials - prev_potentials

    total_reward = progress_reward + alive_reward_buf + up_reward + heading_reward - actions_cost_scale * \
        actions_cost - energy_cost_scale * electricity_cost - dof_at_limit_cost * joints_at_limit_cost_scale

    # adjust reward for fallen agents
    total_reward = torch.where(obs_buf[:, 0] < termination_height, death_cost_buf, total_reward)

    # reset agents
    reset = torch.where(obs_buf[:, 0] < termination_height, one_masks_buf, reset_buf)
    reset = torch.where(progress_buf >= max_episode_length - 1, one_masks_buf, reset)

    return total_reward, reset


@torch.jit.script
def reset_potentials(targets, gpu_init_root_translation, dt: float,
                     reset_buf, prev_potentials, potentials, zero_y):
    to_target = (targets - gpu_init_root_translation) * zero_y

    new_potentials = -torch.norm(to_target, p=2, dim=-1) / dt

    prev_potentials[:] = torch.where(reset_buf, new_potentials, prev_potentials)
    potentials[:] = torch.where(reset_buf, new_potentials, potentials)


class AntEnvironmentIsaacGym(EnvironmentGpu):

    def __init__(self,
                 num_envs: int,
                 device: torch.device,
                 rendering: bool = False,
                 enable_scene_query: bool = False,
                 max_episode_length: int = 1000,
                 ctrl_cost_weight: float = 0.5,
                 healthy_reward: float = 1,
                 healthy_y_range: tuple[float, float] = (0.31, 1.1),
                 reset_noise_scale: float = 1,
                 gravity: v.Vec3 = v.Vec3(0, -9.81, 0),
                 timestep: float = 0.01667,
                 frame_skip: int = 1,
                 spacing: float = 2,
                 initial_is_paused: bool = False,
                 send_interrupt: bool = False,
                 print_hash: bool = False,
                 max_contact_pairs_per_env: int = 128,
                 with_window: bool = True,
                 ):

        self.num_dofs = 8

        super().__init__(num_envs, device, rendering, enable_scene_query, max_episode_length,
                         timestep, frame_skip, spacing, gravity, self.num_dofs,
                         initial_is_paused=initial_is_paused, send_interrupt=send_interrupt,
                         print_hash=print_hash, max_contact_pairs_per_env=max_contact_pairs_per_env,
                         with_window=with_window)

        self.gym.set_num_solver_iterations(4)

        # Hardcode these for now
        self.num_motors = 8
        self.num_sensors = 4

        self.ctrl_cost_weight = ctrl_cost_weight
        self.healthy_reward = healthy_reward
        self.healthy_y_range = healthy_y_range
        self.reset_noise_scale = reset_noise_scale

        # Isaac Gym stuff
        # self.dt = self.timestep
        self.termination_height = self.healthy_y_range[0]

        # TODO make following parameters editable through command-line args
        self.dof_vel_scale = 0.2
        self.contact_force_scale = 0.1
        self.up_weight = 0.1
        self.heading_weight = 0.5
        self.actions_cost_scale = 0.005
        self.energy_cost_scale = 0.05
        self.joints_at_limit_cost_scale = 0.1
        self.death_cost = -2.0

        self.targets = torch.tensor([1000, 0, 0], device=self.device,
                                    dtype=torch.float32).repeat((self.num_envs, 1))

        self.up_axis_idx = 1  # up axis is Y
        self.basis_vec0 = torch.tensor([1, 0, 0], device=self.device,
                                       dtype=torch.float32).repeat((self.num_envs, 1))
        self.basis_vec1 = torch.tensor([0, 1, 0], device=self.device,
                                       dtype=torch.float32).repeat((self.num_envs, 1))

        # Initialise observation and action space
        self.single_observation_space = Box(
            low=np.array([np.finfo('f').min] * 60, dtype=np.float32),
            high=np.array([np.finfo('f').max] * 60, dtype=np.float32),
            dtype=np.float32)
        self.single_action_space = Box(
            low=np.array([-1] * 8, dtype=np.float32),
            high=np.array([1] * 8, dtype=np.float32),
            dtype=np.float32)

        # Create environments
        self.create_envs()

        # Store DOF data
        self.store_initial_conditions()

        # Allocate buffers
        self.allocate_buffers()

        # Create plane
        create_plane(self.gym)

        self.gym.set_num_solver_iterations(8)

        # Finalize gym
        self.gym.gym_finalize()

    def create_envs(self):

        self.env_def_handle, self.art_def_handle, self.arti_handle = create_envs_helper(self.gym)

        env_def = self.gym.get_environment_def(self.env_def_handle)
        self.art_def = env_def.get_articulation_def(self.art_def_handle)

        env_def.finalize()

        # Call super before get_articulation as articulation instances
        # get filled out only after create_environment_set
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

        # Isaac Gym stuff
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

        if self.print_hash:
            print(
                "set_dof_pos_buf:    {}".format(
                    sha256(
                        self.set_dof_pos_buf.cpu().numpy().tobytes()).hexdigest()))
            print(
                "set_dof_vel_buf:    {}".format(
                    sha256(
                        self.set_dof_vel_buf.cpu().numpy().tobytes()).hexdigest()))

        self.gym.set_articulation_kinematic_states(self.gpu_set_kinematic_state_command_array)

        reset_potentials(self.targets, self.gpu_init_root_translation, self.dt,
                         self.reset_buf, self.prev_potentials, self.potentials, self.gpu_zero_y)

        self.act_buf[:] = torch.where(self.reset_buf.view(-1, 1), 0, self.act_buf)

        self.progress_buf[:] = torch.where(self.reset_buf, 0, self.progress_buf)

        # This line is necessary because the reset buffer is not the logical_or
        # of term_buf and trunc_buf, like in the new Gymnasium API, and it is
        # not fully reset in compute_reward
        self.reset_buf[:] = torch.zeros_like(self.reset_buf)

    def reset(self):

        super().reset()

        # Return observations
        self.compute_observations(self.act_buf)

        return self.obs_buf.clone(), {}

    def pre_physics_step(self, actions):

        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, dtype=torch.float32, device=self.device)

        self.act_buf[:] = actions

        self.reset_idx()

        self.gym.set_motor_forces(self.gpu_set_motor_data_array)

    def post_physics_step(self):
        self.progress_buf += 1

        self.compute_observations(self.act_buf)
        self.compute_reward_termination_truncation(self.act_buf)
        self.term_buf[:] = self.reset_buf

        if self.print_hash:
            print(
                "self.reset_buf:     {}".format(
                    sha256(
                        self.reset_buf.cpu().numpy().tobytes()).hexdigest()))

    def compute_observations(self, actions):

        self.gym.get_articulation_kinematic_states(self.gpu_get_kinematic_state_command_array)

        self.gym.get_sensor_forces(self.gpu_get_sensor_forces_command_array)

        # Conversion operation for compatibility with handle-based sensor API
        self.sensor_force_buf[:] = torch.cat(self.force_sensor_buffers, dim=-1)

        if self.print_hash:
            print(
                "get_dof_pos_buf:    {}".format(
                    sha256(
                        self.get_dof_pos_buf.cpu().numpy().tobytes()).hexdigest()))
            print(
                "get_dof_vel_buf:    {}".format(
                    sha256(
                        self.get_dof_vel_buf.cpu().numpy().tobytes()).hexdigest()))
            print(
                "root_pos_buf:       {}".format(
                    sha256(
                        self.root_pos_buf.cpu().numpy().tobytes()).hexdigest()))
            print(
                "root_vel_buf:       {}".format(
                    sha256(
                        self.root_vel_buf.cpu().numpy().tobytes()).hexdigest()))
            print(
                "sensor_force_buf:   {}".format(
                    sha256(
                        self.sensor_force_buf.cpu().numpy().tobytes()).hexdigest()))

        self.obs_buf[:], self.potentials[:], self.prev_potentials[:], _, _ = compute_observations_isaac_gym_helper(
            self.obs_buf, self.root_pos_buf, self.root_vel_buf, self.targets, self.potentials,
            self.inv_start_rot, self.get_dof_pos_buf, self.get_dof_vel_buf,
            self.dof_limits_lower, self.dof_limits_upper, self.dof_vel_scale,
            self.sensor_force_buf, actions, self.dt, self.contact_force_scale,
            self.basis_vec0, self.basis_vec1, self.up_axis_idx, self.gpu_zero_y)

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
            self.termination_height,
            self.max_episode_length
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

    num_envs = 2

    reset_noise_scale = 1
    spacing = 2

    assert torch.cuda.is_available()
    device = torch.device("cuda:0")

    envs = AntEnvironmentIsaacGym(num_envs, device,
                                  reset_noise_scale=reset_noise_scale, spacing=spacing,
                                  rendering=rendering, with_window=with_window)

    obs, _ = envs.reset()

    gym = v.get_gym()
    render = gym.get_render()

    # Define control
    if render is not None:
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
                envs.reset()

            forces = torch.tensor([slider.get_value() for slider in sliders],
                                  dtype=torch.float32, device=device)

            return torch.tile(forces, (num_envs, 1))

        control_fn = control_by_menu

    else:
        def control_fn(n=num_envs, m=envs.art_def.get_num_motor_defs()): return torch.zeros((n, m))

    if render is not None:
        render.capped_step = True
    finished = False
    idx = 0
    while not finished:

        # actions = torch.zeros((num_envs, 8), dtype=torch.float32, device=device)
        actions = control_fn()

        # print("action: {}".format(action))

        obs, reward, terminated, truncated, info = envs.step(actions)

        # print("obs: {}".format(obs))
        # print("reward: {}".format(reward))

        # print("{}".format(obs[:,0])) # y-coordinate
        # print("{}".format(obs[:,1:5])) # Rotation
        # print("{}".format(obs[:,5:8])) # Linear velocity
        # print("{}".format(obs[:,8:11])) # Angular velocity
        # print("{}".format(obs[:,11:19])) # DOF position
        # print("{}".format(obs[:,19:27])) # DOF velocity
        # print("{}".format(obs[:,27:51])) # Contact forces

        # print(obs, reward, terminated, truncated, info)

        finished = envs.render_finished

        idx += 1
        if idx >= num_iter:
            finished = True
