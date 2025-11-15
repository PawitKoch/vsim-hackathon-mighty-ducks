import torch
import numpy as np
from vlearn.spaces import Box

import vlearn as v

if __name__ == "__main__":
    from environment import EnvironmentGpu
    from vlearn.torch_utils.torch_jit_utils import *
    from common import create_plane, create_heightfield
else:
    from .environment import EnvironmentGpu
    from vlearn.torch_utils.torch_jit_utils import *
    from .common import create_plane, create_heightfield


@torch.jit.script
def reset_noise_helper(init_dof_val, noise_scale: float, scale1: float, scale2: float):
    return (noise_scale * (torch.rand(init_dof_val.shape, dtype=torch.float32,
            device=init_dof_val.device) * scale1 - scale2))


@torch.jit.script
def reset_potentials(targets, gpu_init_root_translation, dt: float,
                     reset_buf, prev_potentials, potentials, zero_y):
    to_target = (targets - gpu_init_root_translation) * zero_y

    new_potentials = -torch.norm(to_target, p=2, dim=-1) / dt

    prev_potentials[:] = torch.where(reset_buf, new_potentials, prev_potentials)
    potentials[:] = torch.where(reset_buf, new_potentials, potentials)


@torch.jit.script
def compute_humanoid_observations(
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
        joint_force_scale: float,
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

    # obs_buf shapes: 1, 3, 3, 1, 1, 1, 1, 1, num_dofs (21), num_dofs (21),
    # num_dofs (21), 12, num_acts (21)
    obs = torch.cat((torso_position[:, 1].view(-1, 1), vel_loc, angvel_loc * angular_velocity_scale,
                     yaw, roll, angle_to_target, up_proj.unsqueeze(-1), heading_proj.unsqueeze(-1),
                     dof_pos_scaled, dof_vel * dof_vel_scale, dof_force * joint_force_scale,
                     sensor_force_torques.view(-1, 12) * contact_force_scale, actions), dim=-1)

    return obs, potentials, prev_potentials_new, up_vec, heading_vec


@torch.jit.script
def compute_humanoid_reward(
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
    scaled_cost = joints_at_limit_cost_scale * (torch.abs(obs_buf[:, 12:33]) - 0.9) / 0.1
    dof_at_limit_cost = torch.sum(
        (torch.abs(obs_buf[:, 12:33]) > 0.9) * scaled_cost * motor_effort_ratio.unsqueeze(0), dim=-1)

    electricity_cost = torch.sum(
        torch.abs(actions * obs_buf[:, 33:54]) * motor_effort_ratio.unsqueeze(0), dim=-1)

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


class HumanoidEnvironmentIsaacGym(EnvironmentGpu):

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
                 termination_height: float = 0.8,
                 reset_noise_scale: float = 1,
                 dof_vel_scale: float = 0.1,
                 contact_force_scale: float = 0.01,
                 joint_force_scale: float = 0.01,
                 up_weight: float = 0.1,
                 heading_weight: float = 0.5,
                 actions_cost_scale: float = 0.01,
                 energy_cost_scale: float = 0.02,
                 angular_velocity_scale: float = 0.25,
                 joints_at_limit_cost_scale: float = 0.25,
                 death_cost: float = -1.0,
                 power_scale: float = 1.0,
                 initial_is_paused: bool = False,
                 send_interrupt: bool = False,
                 static_friction: float = 1.0,
                 dynamic_friction: float = 1.0,
                 report_speed: bool = False,
                 print_hash: bool = False,
                 max_contact_pairs_per_env: int = 128,
                 with_window: bool = True,
                 ):

        self.num_dofs = 21

        super().__init__(num_envs, device, rendering, enable_scene_query, max_episode_length,
                         timestep, frame_skip, spacing, gravity, self.num_dofs,
                         initial_is_paused=initial_is_paused, send_interrupt=send_interrupt,
                         print_hash=print_hash, max_contact_pairs_per_env=max_contact_pairs_per_env,
                         with_window=with_window)

        # Store environment-specific configuration
        self.reset_noise_scale = reset_noise_scale
        self.report_speed = report_speed

        # Isaac Gym extra stuff
        # Static
        # self.dt = self.timestep
        self.termination_height = termination_height
        self.dof_vel_scale = dof_vel_scale
        self.contact_force_scale = contact_force_scale
        self.joint_force_scale = joint_force_scale
        self.up_weight = up_weight
        self.heading_weight = heading_weight
        self.actions_cost_scale = actions_cost_scale
        self.energy_cost_scale = energy_cost_scale
        self.angular_velocity_scale = angular_velocity_scale
        self.joints_at_limit_cost_scale = joints_at_limit_cost_scale
        self.death_cost = death_cost
        self.power_scale = power_scale

        # Hardcode these for now
        self.num_motors = 21
        self.num_sensors = 2

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

        self.single_action_space = Box(low=low, high=high, dtype=np.float32)

        # Save motor efforts (also Isaac Gym)
        self.motor_efforts = torch.tensor(motor_efforts, dtype=torch.float32, device=self.device)
        self.max_motor_effort = self.motor_efforts.max()

        # Create plane
        create_plane(self.gym, static_friction=static_friction, dynamic_friction=dynamic_friction)
        # create_heightfield(self.gym)

        # Store DOF data
        self.store_initial_conditions()

        # Allocate buffers
        self.allocate_buffers()

        # Set gravity
        self.gym.set_gravity(self.gravity)

        # Finalize gym
        self.gym.gym_finalize()

        # Isaac Gym stuff
        self.dof_limits_lower = self.dof_pos_low
        self.dof_limits_upper = self.dof_pos_high

        self.targets = torch.tensor([1000, 0, 0], device=self.device,
                                    dtype=torch.float32).repeat((self.num_envs, 1))

        inv_start_rot = self.root_trans_init.q.get_conjugate()
        self.inv_start_rot = torch.tensor([inv_start_rot.x,
                                           inv_start_rot.y,
                                           inv_start_rot.z,
                                           inv_start_rot.w],
                                          device=self.device,
                                          dtype=torch.float32).repeat((self.num_envs,
                                                                       1))

        self.up_axis_idx = 1  # up axis is Y
        self.basis_vec0 = torch.tensor([1, 0, 0], device=self.device,
                                       dtype=torch.float32).repeat((self.num_envs, 1))
        self.basis_vec1 = torch.tensor([0, 1, 0], device=self.device,
                                       dtype=torch.float32).repeat((self.num_envs, 1))

        # Dynamic
        self.potentials = torch.tensor([-1000.0 / self.dt],
                                       device=self.device,
                                       dtype=torch.float32).repeat(self.num_envs)
        self.prev_potentials = self.potentials.clone()

        self.gym.set_num_solver_iterations(6)

    def store_initial_conditions(self):

        # Store dof limits and initial values
        dof_pos_low = []
        dof_pos_high = []
        dof_pos_init = []
        for dofdef in self.art_def.get_joint_dof_defs():
            low, high = dofdef.get_limits()
            init = np.clip(0, low, high)

            dof_pos_low.append(low)
            dof_pos_high.append(high)
            dof_pos_init.append(init)

        self.dof_pos_low = torch.tensor(dof_pos_low, dtype=torch.float32, device=self.device)
        self.dof_pos_high = torch.tensor(dof_pos_high, dtype=torch.float32, device=self.device)

        self.dof_pos_init = torch.tensor(dof_pos_init, dtype=torch.float32, device=self.device)

        # Store initial root transform
        rot = v.shortest_rotation(v.Vec3(0, 0, 1), v.Vec3(0, 1, 0))
        pos = v.Vec3(0, 1.34, 0)
        self.root_trans_init = v.Transform(rot, pos)

        # Store initial root velocity
        self.root_vel_init = torch.tensor(
            [0, 0, 0, 0, 0, 0], dtype=torch.float32, device=self.device)

    def allocate_buffers(self):

        super().allocate_buffers()

        # Root transforms buffers
        self.root_pos_buf = torch.zeros((self.num_envs, 7), device=self.device, dtype=torch.float32)
        self.old_root_pos_buf = torch.zeros(
            (self.num_envs, 7), device=self.device, dtype=torch.float32)

        # Root velocities buffer
        self.root_vel_buf = torch.zeros((self.num_envs, 6), device=self.device, dtype=torch.float32)

        # Sensor forces buffer
        self.sensor_force_buf = torch.zeros(
            (self.num_envs, self.num_sensors * 6), device=self.device, dtype=torch.float32)

        # Joint forces buffer
        self.joint_force_buf = torch.zeros(
            (self.num_envs, self.num_dofs), device=self.device, dtype=torch.float32)

        # Reset state
        self.gpu_init_dof_pos = torch.tile(self.dof_pos_init, (self.num_envs, 1))
        self.gpu_init_dof_vel = torch.zeros_like(self.gpu_init_dof_pos)

        self.gpu_init_root_velocities = torch.tile(self.root_vel_init, (self.num_envs, 1))
        self.gpu_init_root_transforms = torch.empty(7, dtype=torch.float32, device=self.device)
        self.gpu_zero_y = torch.empty(3, dtype=torch.float32, device=self.device)
        self.gpu_init_root_translation = torch.empty(3, dtype=torch.float32, device=self.device)

        self.gpu_init_root_transforms[0] = self.root_trans_init.q.x
        self.gpu_init_root_transforms[1] = self.root_trans_init.q.y
        self.gpu_init_root_transforms[2] = self.root_trans_init.q.z
        self.gpu_init_root_transforms[3] = self.root_trans_init.q.w
        self.gpu_init_root_transforms[4] = self.root_trans_init.p.x
        self.gpu_init_root_transforms[5] = self.root_trans_init.p.y
        self.gpu_init_root_transforms[6] = self.root_trans_init.p.z

        self.gpu_zero_y[0] = 1.0
        self.gpu_zero_y[1] = 0.0
        self.gpu_zero_y[2] = 1.0
        self.gpu_init_root_translation[0] = self.root_trans_init.p.x
        self.gpu_init_root_translation[1] = self.root_trans_init.p.y
        self.gpu_init_root_translation[2] = self.root_trans_init.p.z

        self.gpu_init_root_transforms = torch.tile(
            self.gpu_init_root_transforms, (self.num_envs, 1))
        self.gpu_zero_y = torch.tile(self.gpu_zero_y, (self.num_envs, 1))
        self.gpu_init_root_translation = torch.tile(
            self.gpu_init_root_translation, (self.num_envs, 1))

        # Set motor GPU command
        set_motor_data = self.env_group.create_motor_control_command(
            v.wrap_gpu_buffer(self.act_buf),
            self.arti_handle)

        self.gpu_set_motor_data_array = self.gym.create_motor_control_command_gpu_array([
                                                                                        set_motor_data])

        # Set kinematic state GPU command
        set_kinematic_state_command = self.env_group.create_articulation_kinematic_state_command(
            v.wrap_gpu_buffer(self.set_dof_pos_buf),
            v.wrap_gpu_buffer(self.set_dof_vel_buf),
            v.wrap_gpu_buffer(self.gpu_init_root_transforms),
            v.wrap_gpu_buffer(self.gpu_init_root_velocities),
            self.arti_handle,
            link_index_range=(0, 1),
            masks_buffer=v.wrap_gpu_buffer(self.reset_buf))

        self.gpu_set_kinematic_state_command_array = self.gym.create_articulation_kinematic_state_command_gpu_array(
            [set_kinematic_state_command])

        # Get kinematic state GPU command
        get_kinematic_state_command = self.env_group.create_articulation_kinematic_state_command(
            v.wrap_gpu_buffer(self.get_dof_pos_buf),
            v.wrap_gpu_buffer(self.get_dof_vel_buf),
            v.wrap_gpu_buffer(self.root_pos_buf),
            v.wrap_gpu_buffer(self.root_vel_buf),
            self.arti_handle,
            link_index_range=(0, 1))

        self.gpu_get_kinematic_state_command_array = \
            self.gym.create_articulation_kinematic_state_command_gpu_array(
                [get_kinematic_state_command])

        # Get sensor forces GPU command
        num_force_sensors = self.art_def.get_num_force_sensor_defs()
        env_def = self.gym.get_environment_def(self.env_def_handle)
        articulation = env_def.get_articulation(self.arti_handle)

        self.force_sensor_handles = []
        self.force_sensor_buffers = []
        self.force_sensor_cmds = []
        for i in range(num_force_sensors):
            self.force_sensor_handles.append(articulation.get_force_sensor_handle(i))
            self.force_sensor_buffers.append(torch.zeros((self.num_envs, 6), device=self.device,
                                                         dtype=torch.float32))
            self.force_sensor_cmds.append(self.env_group.create_force_sensor_command(
                v.wrap_gpu_buffer(self.force_sensor_buffers[-1]), self.force_sensor_handles[-1]))

        self.gpu_get_sensor_forces_command_array = self.gym.create_force_sensor_command_gpu_array(
            self.force_sensor_cmds)

        # Get joint force sensor GPU command
        get_joint_force_command = self.env_group.create_joint_state_command(
            v.wrap_gpu_buffer(self.joint_force_buf),
            self.arti_handle)

        self.gpu_get_joint_forces_command_array = self.gym.create_joint_state_command_gpu_array([
                                                                                                get_joint_force_command])

        self.one_masks_buf = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.zero_masks_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.alive_reward_buf = self.one_masks_buf * 2.0
        self.death_cost_buf = self.one_masks_buf * self.death_cost

    def reset_idx(self):

        if self.report_speed and torch.any(self.reset_buf):
            env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
            for env_id in env_ids:
                distance = torch.norm(self.root_pos_buf[env_id, 4:7])
                time_elapsed = self.progress_buf[env_id] * self.dt
                speed = distance / time_elapsed
                print(
                    "distance, time elapsed, avg speed of #{}: {}, {}, {}".format(
                        env_id, distance, time_elapsed, speed))

        # Set kinematic state
        self.set_dof_pos_buf[:] = torch.clamp(
            self.gpu_init_dof_pos +
            reset_noise_helper(
                self.gpu_init_dof_pos,
                self.reset_noise_scale,
                0.4,
                0.2),
            self.dof_limits_lower,
            self.dof_limits_upper)
        self.set_dof_vel_buf[:] = reset_noise_helper(
            self.gpu_init_dof_vel, self.reset_noise_scale, 0.2, 0.1)
        # self.set_dof_pos_buf[:] = self.gpu_init_dof_pos
        # self.set_dof_vel_buf[:] = self.gpu_init_dof_vel

        self.gym.set_articulation_kinematic_states(self.gpu_set_kinematic_state_command_array)

        reset_potentials(self.targets, self.gpu_init_root_translation, self.dt,
                         self.reset_buf, self.prev_potentials, self.potentials, self.gpu_zero_y)

        # # Reset remaining buffers
        # self.old_root_pos_buf[:] = torch.where(self.reset_buf.view(-1, 1),
        #         self.gpu_init_root_transforms, self.old_root_pos_buf)

        self.act_buf[:] = torch.where(self.reset_buf.view(-1, 1), 0, self.act_buf)
        self.progress_buf[:] = torch.where(self.reset_buf, 0, self.progress_buf)

        # This is necessary because the reset buffer is not the logical_or of
        # term_buf and trunc_buf, like in the new Gymnasium API, and it is not
        # fully reset in compute_reward
        self.reset_buf[:] = torch.zeros_like(self.reset_buf)

    def reset(self):

        super().reset()

        # Return observations
        self.compute_isaac_gym_observations()

        return self.obs_buf.clone(), {}

    def pre_physics_step(self, actions):

        assert isinstance(actions, torch.Tensor)

        self.reset_idx()

        self.act_buf[:] = self.power_scale * actions

        self.gym.set_motor_forces(self.gpu_set_motor_data_array)

    def post_physics_step(self):
        self.progress_buf += 1

        self.compute_isaac_gym_observations()
        self.compute_isaac_gym_reward(self.act_buf)
        self.term_buf[:] = self.reset_buf

    def compute_isaac_gym_observations(self):

        self.gym.get_articulation_kinematic_states(self.gpu_get_kinematic_state_command_array)
        self.gym.get_sensor_forces(self.gpu_get_sensor_forces_command_array)

        # Conversion operation for compatibility with handle-based sensor API
        self.sensor_force_buf[:] = torch.cat(self.force_sensor_buffers, dim=-1)

        self.gym.get_joint_forces(self.gpu_get_joint_forces_command_array)

        self.obs_buf[:], self.potentials[:], self.prev_potentials[:], _, _ = compute_humanoid_observations(
            self.obs_buf, self.root_pos_buf, self.root_vel_buf, self.targets,
            self.potentials, self.inv_start_rot, self.get_dof_pos_buf, self.get_dof_vel_buf,
            self.joint_force_buf, self.dof_limits_lower, self.dof_limits_upper,
            self.dof_vel_scale, self.sensor_force_buf, self.act_buf, self.dt,
            self.contact_force_scale, self.joint_force_scale, self.angular_velocity_scale,
            self.basis_vec0, self.basis_vec1, self.up_axis_idx,
            self.gpu_zero_y)

    def compute_isaac_gym_reward(self, actions):
        rew_buf, reset_buf = compute_humanoid_reward(
            self.obs_buf,
            self.reset_buf,
            self.one_masks_buf,
            self.zero_masks_buf,
            self.progress_buf,
            self.alive_reward_buf,
            self.death_cost_buf,
            self.act_buf,
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

        self.rew_buf[:] = rew_buf
        self.reset_buf[:] = reset_buf

    def create_envs(self):

        # Environment def
        self.env_def_handle = self.gym.create_environment_def("humanoid")
        self.env_def = self.gym.get_environment_def(self.env_def_handle)

        filename = "assets/vsim/humanoid.vsim"
        num_arti, num_rigid = self.env_def.import_definitions(filename, fixed=False)
        assert num_arti == 1
        assert num_rigid == 0
        def_handle = self.env_def.get_articulation_def_handle_by_name("torso")

        rot = v.shortest_rotation(v.Vec3(0, 0, 1), v.Vec3(0, 1, 0))
        pos = v.Vec3(0, 1.34, 0)
        transform = v.Transform(rot, pos)

        self.arti_handle = self.env_def.create_articulation(def_handle, transform, 'runner')

        self.art_def = self.env_def.get_articulation_def(def_handle)
        self.art_def.enable_control_type(v.ArticulationControlType.MOTOR, True)

        self.art_def.contact_offset = 0.01

        self.gym.get_environment_def(self.env_def_handle).finalize()

        super().create_envs(self.env_def_handle)


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
    enable_scene_query = True

    device = torch.device("cuda:0")

    envs = HumanoidEnvironmentIsaacGym(
        num_envs,
        device,
        rendering=rendering,
        max_episode_length=max_episode_length,
        enable_scene_query=enable_scene_query,
        with_window=with_window)

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
        # print("obs[:, 0]: {}".format(obs[:, 0])) # y-position
        # print("obs[:, 1:4]: {}".format(obs[:, 1:4])) # torso linear velocity
        # print("obs[:, 4:7]: {}".format(obs[:, 4:7])) # torso angular velocity
        # print("obs[:, 7]: {}".format(obs[:, 7])) # yaw
        # print("obs[:, 8]: {}".format(obs[:, 8])) # roll
        # print("obs[:, 9]: {}".format(obs[:, 9])) # angle_to_target
        # print("obs[:, 10]: {}".format(obs[:, 10])) # up_proj
        # print("obs[:, 11]: {}".format(obs[:, 11])) # heading_proj
        # print("obs[:, 12:33]: {}".format(obs[:, 12:33])) # dof_pos_scaled
        # print("obs[:, 33:54]: {}".format(obs[:, 33:54])) # dof_vel_scaled
        # print("obs[:, 54:75]: {}".format(obs[:, 54:75])) # dof_force
        # print("obs[:, 75:87]: {}".format(obs[:, 75:87])) # sensor forces
        # print("obs[:, 87:108]: {}".format(obs[:, 87:108])) # actions
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
