# Adapted from https://github.com/pairlab/leibnizgym/
import torch
import numpy as np
from vlearn.spaces import Box
from typing import List, Tuple
import torch.profiler as profiler

import vlearn as v

if __name__ == "__main__":
    from environment import EnvironmentGpu
    from vlearn.torch_utils.torch_jit_utils import unscale_transform, saturate, quat_from_euler_xyz, quat_diff_rad
    from common import create_plane, reset_noise_helper
else:
    from .environment import EnvironmentGpu
    from vlearn.torch_utils.torch_jit_utils import unscale_transform, saturate, quat_from_euler_xyz, quat_diff_rad
    from .common import create_plane, reset_noise_helper


@torch.jit.script
def random_xy(num: int, max_com_distance_to_center: float,
              device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns sampled uniform positions in circle (https://stackoverflow.com/a/50746409)"""
    # sample radius of circle
    radius = torch.sqrt(torch.rand(num, dtype=torch.float, device=device))
    radius *= max_com_distance_to_center
    # sample theta of point
    theta = 2 * np.pi * torch.rand(num, dtype=torch.float, device=device)
    # x,y-position of the cube
    x = radius * torch.cos(theta)
    y = radius * torch.sin(theta)

    return x, y


@torch.jit.script
def random_z(num: int, min_height: float, max_height: float, device: torch.device) -> torch.Tensor:
    """Returns sampled height of the goal object."""
    z = torch.rand(num, dtype=torch.float, device=device)
    z = (max_height - min_height) * z + min_height

    return z


@torch.jit.script
def default_orientation(num: int, device: torch.device) -> torch.Tensor:
    """Returns identity rotation transform."""
    quat = torch.zeros((num, 4,), dtype=torch.float, device=device)
    quat[..., -1] = 1.0

    return quat


@torch.jit.script
def random_orientation(num: int, device: torch.device) -> torch.Tensor:
    """Returns sampled rotation in 3D as quaternion.
    Ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.random.html
    """
    # sample random orientation from normal distribution
    quat = torch.randn((num, 4,), dtype=torch.float, device=device)
    # normalize the quaternion
    quat = torch.nn.functional.normalize(quat, p=2., dim=-1, eps=1e-12)

    return quat


@torch.jit.script
def lgsk_kernel(x: torch.Tensor, scale: float = 50.0) -> torch.Tensor:
    """Defines logistic kernel function to bound input to [-0.25, 0)

    Ref: https://arxiv.org/abs/1901.08652 (page 15)

    Args:
        x: Input tensor.
        scale: Scaling of the kernel function.

    Returns:
        Output tensor computed using kernel.
    """
    scaled = x * scale
    return 1.0 / (scaled.exp() + 2 + (-scaled).exp())


@torch.jit.script
def random_yaw_orientation(num: int, device: torch.device) -> torch.Tensor:
    """Returns sampled rotation around z-axis."""
    roll = torch.zeros(num, dtype=torch.float, device=device)
    pitch = torch.zeros(num, dtype=torch.float, device=device)
    yaw = 2 * np.pi * torch.rand(num, dtype=torch.float, device=device)

    return quat_from_euler_xyz(roll, pitch, yaw)


@torch.jit.script
def compute_reward_helper(
    goal_cube_pos_buf: torch.Tensor,
    get_cube_pos_buf: torch.Tensor,
    old_cube_pos_buf: torch.Tensor,
    get_fingertip_pos_buffers: List[torch.Tensor],
    old_fingertip_pos_buffers: List[torch.Tensor],
    dt: float,
    object_dist_weight: float,
    object_move_weight: float,
    object_rot_weight: float,
    object_rot_delta_weight: float,
    fingertip_reach_weight: float,
    fingertip_move_weight: float,
        ) -> torch.Tensor:

    # Distance reward
    dist = torch.norm(get_cube_pos_buf[:, 4:7] - goal_cube_pos_buf[:, 4:7], p=2, dim=-1)
    dist_reward = dt * lgsk_kernel(dist)

    # Object move reward
    curr_move_norms = torch.norm(get_cube_pos_buf[:, 4:7] - goal_cube_pos_buf[:, 4:7], dim=-1)
    prev_move_norms = torch.norm(old_cube_pos_buf[:, 4:7] - goal_cube_pos_buf[:, 4:7], dim=-1)
    move_reward = curr_move_norms - prev_move_norms

    # Object rotation reward
    quat_a = get_cube_pos_buf[:, 0:4]
    quat_b = goal_cube_pos_buf[:, 0:4]

    angles = torch.abs(quat_diff_rad(quat_a, quat_b))
    rotation_reward = dt / (angles + 1)

    # Object rotation delta reward
    last_quat_a = old_cube_pos_buf[:, 0:4]
    last_angles = torch.abs(quat_diff_rad(last_quat_a, quat_b))

    rotation_delta_reward = angles - last_angles

    # Finger reach object reward
    curr_reach_norms = torch.stack([
        torch.norm(get_fingertip_pos_buffers[i][:, 4:7] - get_cube_pos_buf[:, 4:7], p=2, dim=-1)
        for i in range(3)], dim=-1)
    prev_reach_norms = torch.stack([
        torch.norm(old_fingertip_pos_buffers[i][:, 4:7] - old_cube_pos_buf[:, 4:7], p=2, dim=-1)
        for i in range(3)], dim=-1)
    reach_reward = (curr_reach_norms - prev_reach_norms).sum(dim=-1)

    # Finger movement penalty
    vel = torch.concatenate([(get_fingertip_pos_buffers[i][:, 4:7] -
                              old_fingertip_pos_buffers[i][:, 4:7]) / dt for i in range(3)], dim=-1)
    fingertip_move_penalty = (vel * vel).sum(dim=-1)

    reward = object_dist_weight * dist_reward + \
        object_move_weight * move_reward + \
        object_rot_weight * rotation_reward + \
        object_rot_delta_weight * rotation_delta_reward + \
        fingertip_reach_weight * reach_reward + \
        fingertip_move_weight * fingertip_move_penalty

    return reward


@torch.jit.script
def compute_termination_helper(
        goal_cube_pos_buf: torch.Tensor,
        get_cube_pos_buf: torch.Tensor,
        rew_buf: torch.Tensor,
        frames_on_goal: torch.Tensor,
        position_tolerance: float,
        orientation_tolerance: float,
        on_target_bonus: float,
        difficulty: int,
        goal_reset_delay: int,
        success_bonus: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    object_goal_position_dist = torch.norm(goal_cube_pos_buf[:, 4:7] -
                                           get_cube_pos_buf[:, 4:7], p=2, dim=-1)

    goal_position_reset = torch.le(object_goal_position_dist, position_tolerance)

    if difficulty < 4:
        task_completion_reset = goal_position_reset
    elif difficulty == 4:
        # Compute the difference in orientation between object and goal pose
        object_goal_orientation_dist = quat_diff_rad(get_cube_pos_buf[:, 0:4],
                                                     goal_cube_pos_buf[:, 0:4])

        # Check for distance within tolerance
        goal_orientation_reset = torch.le(object_goal_orientation_dist,
                                          orientation_tolerance)

        task_completion_reset = torch.logical_and(goal_position_reset, goal_orientation_reset)
    # This was added for pytorch jit compilation
    else:
        task_completion_reset = goal_position_reset

    # Add reward bonus for being on the goal
    rew_buf[:] = torch.where(task_completion_reset, rew_buf + on_target_bonus, rew_buf)

    # Termination delay
    frames_on_goal[:] = torch.where(task_completion_reset, frames_on_goal + 1, 0)
    termination = frames_on_goal > goal_reset_delay

    # Add reward bonus for being on the goal for long enough
    rew_buf[:] = torch.where(termination, rew_buf + success_bonus, rew_buf)

    return termination, rew_buf, frames_on_goal


class TrifingerEnvironmentGpu(EnvironmentGpu):

    def __init__(self,
                 num_envs: int,
                 device: torch.device,
                 rendering: bool = False,
                 enable_scene_query: bool = False,
                 max_episode_length: int = 750,
                 gravity: v.Vec3 = v.Vec3(0, -9.81, 0),
                 timestep: float = 0.01667,
                 frame_skip: int = 1,
                 spacing: float = 1,
                 reset_noise_scale: float = 1,
                 max_torque: float = 0.36,
                 position_tolerance: float = 0.01,
                 orientation_tolerance: float = 0.2,
                 on_target_bonus: float = 100,
                 success_bonus: float = 5000,
                 difficulty: int = 1,
                 distribution: str = "random",
                 object_dist_weight: float = 2000,
                 object_move_weight: float = -750,
                 object_rot_weight: float = 300,
                 object_rot_delta_weight: float = -250,
                 fingertip_reach_weight: float = -750,
                 fingertip_move_weight: float = -0.1,
                 draw_goal: bool = False,
                 record_goal: bool = False,
                 reset_progress: bool = False,
                 initial_is_paused: bool = False,
                 send_interrupt: bool = False,
                 goal_reset_delay: int = 0,
                 max_contact_pairs_per_env: int = 256,
                 with_window: bool = True,
                 ):

        num_dofs = 9

        super().__init__(num_envs, device, rendering, enable_scene_query, max_episode_length,
                         timestep, frame_skip, spacing, gravity, num_dofs, treat_warning_as_error=True,
                         initial_is_paused=initial_is_paused, send_interrupt=send_interrupt,
                         max_contact_pairs_per_env=max_contact_pairs_per_env, with_window=with_window)

        # Store configuration
        self.reset_noise_scale = reset_noise_scale
        self.max_torque = torch.tensor(max_torque, dtype=torch.float32, device=self.device)
        self.position_tolerance = position_tolerance
        self.orientation_tolerance = orientation_tolerance
        self.on_target_bonus = on_target_bonus
        self.success_bonus = success_bonus
        self.difficulty = difficulty
        self.distribution = distribution
        self.object_dist_weight = object_dist_weight
        self.object_move_weight = object_move_weight
        self.object_rot_weight = object_rot_weight
        self.object_rot_delta_weight = object_rot_delta_weight
        self.fingertip_reach_weight = fingertip_reach_weight
        self.fingertip_move_weight = fingertip_move_weight
        self.draw_goal = draw_goal
        self.record_goal = record_goal
        self.reset_progress = reset_progress
        self.goal_reset_delay = goal_reset_delay

        # constants
        self.cube_size = 0.065
        self.radius_3d = self.cube_size * np.sqrt(3) / 2
        self.arena_radius = 0.195
        self.max_com_distance_to_center = self.arena_radius - self.radius_3d

        # Observation and action spaces
        num_obs = 0
        num_obs += self.num_dofs    # DOF positions
        num_obs += self.num_dofs    # DOF velocities
        num_obs += 7                # Object pose
        num_obs += 7                # Goal object pose
        num_obs += self.num_dofs    # Actions

        print("num_obs: {}".format(num_obs))

        self.single_observation_space = Box(
            low=np.array([np.finfo('f').min] * num_obs, dtype=np.float32),
            high=np.array([np.finfo('f').max] * num_obs, dtype=np.float32),
            dtype=np.float32)
        self.single_action_space = Box(
            low=np.array([-1] * self.num_dofs, dtype=np.float32),
            high=np.array([1] * self.num_dofs, dtype=np.float32),
            dtype=np.float32)

        # State space
        self.num_states = num_obs
        self.num_states += 6                # Object velocity
        self.num_states += 3 * 7            # Fingertip transforms
        self.num_states += 3 * 6            # Fingertip velocities
        self.num_states += self.num_dofs    # Joint sensor forces
        self.num_states += 3 * 6            # Fingertip forces

        self.single_state_space = Box(
            low=np.array([np.finfo('f').min] * self.num_states, dtype=np.float32),
            high=np.array([np.finfo('f').max] * self.num_states, dtype=np.float32),
            dtype=np.float32)

        # Create environments
        self.create_envs()

        # Store initial conditions
        self.store_initial_conditions()

        # Allocate buffers
        self.allocate_buffers()

        # Create access commands
        self.create_access_commands()

        # Create plane
        create_plane(self.gym)

        # Finalize gym
        self.gym.gym_finalize()

    def create_envs(self):
        # Environment def
        self.env_def_handle = self.gym.create_environment_def("trifinger")
        env_def = self.gym.get_environment_def(self.env_def_handle)

        # Trifinger
        filename = "assets/trifinger/robot_properties_fingers/vsim/trifingerpro.vsim"
        num_arti, num_rigid = env_def.import_definitions(
            filename, fixed=True, merge_fixed_joints=False, use_visual_mesh=False)
        assert num_arti == 1 and num_rigid == 0

        tri_def_handle = env_def.get_articulation_def_handle_by_name("trifingerpro")
        self.art_def = env_def.get_articulation_def(tri_def_handle)
        self.art_def.has_self_collisions = True

        rot = v.Quat(0, 0, 0, 1)
        pos = v.Vec3(0, 0, 0)
        transform = v.Transform(rot, pos)

        self.arti_handle = env_def.create_articulation(tri_def_handle, transform, 'trifinger')

        # Stage
        filename = "assets/trifinger/robot_properties_fingers/urdf/high_table_boundary.urdf"
        num_arti, num_rigid = env_def.import_definitions(filename, True)
        assert num_arti == 1 and num_rigid == 1

        stage_def_handle = env_def.get_rigid_body_def_handle_by_name("stage")

        rot = v.Quat(0, 0, 0, 1)
        pos = v.Vec3(0, 0, 0)
        transform = v.Transform(rot, pos)

        env_def.create_rigid_body(stage_def_handle, transform, 'stage')

        # Cube
        filename = "assets/trifinger/objects/urdf/cube_multicolor_rrc.urdf"
        name = "object"
        num_arti, num_rigid = env_def.import_definitions(filename, False)
        assert num_arti == 1 and num_rigid == 2

        cube_def_handle = env_def.get_rigid_body_def_handle_by_name(name)

        rot = v.Quat(0, 0, 0, 1)
        pos = v.Vec3(0, 0, self.cube_size / 2)
        transform = v.Transform(rot, pos)

        self.cube_handle = env_def.create_rigid_body(cube_def_handle, transform, 'cube')

        # Goal
        if self.record_goal:
            filename = "assets/trifinger/objects/urdf/cube_multicolor_rrc_no_visual_no_collision.urdf"
            name = "goal.object"
            num_arti, num_rigid = env_def.import_definitions(filename, True, alias="goal")
            assert num_arti == 1 and num_rigid == 3

            goal_def_handle = env_def.get_rigid_body_def_handle_by_name(name)

            rot = v.Quat(0, 0, 0, 1)
            pos = v.Vec3(0, 0, self.cube_size / 2)
            transform = v.Transform(rot, pos)

            self.goal_handle = env_def.create_rigid_body(goal_def_handle, transform, 'goal')

        env_rot = v.shortest_rotation(v.Vec3(0, 0, 1), v.Vec3(0, 1, 0))

        self.gym.get_environment_def(self.env_def_handle).finalize()

        super().create_envs(self.env_def_handle, env_rot=env_rot)

    def store_initial_conditions(self):

        # Trifinger
        # Initial dof positions are hardcoded
        self.dof_pos_init = torch.tensor([0.0, 0.9, -1.7] * 3, dtype=torch.float32,
                                         device=self.device)
        self.dof_vel_init = torch.zeros(9, dtype=torch.float32, device=self.device)

        # Cube
        rot = v.Quat(0, 0, 0, 1)
        pos = v.Vec3(0, 0, self.cube_size / 2)
        self.cube_pos_init = v.Transform(rot, pos)
        self.cube_vel_init = v.SpatialVector(0)

    def allocate_buffers(self):

        super().allocate_buffers()

        # State buffer
        self.state_buf = torch.zeros((self.num_envs, self.num_states), dtype=torch.float32,
                                     device=self.device)

        # Initial kinematic state buffers
        self.dof_pos_init_buf = torch.tile(self.dof_pos_init, (self.num_envs, 1))
        self.dof_vel_init_buf = torch.tile(self.dof_vel_init, (self.num_envs, 1))

        self.cube_pos_init_buf = torch.zeros(7, dtype=torch.float32, device=self.device)
        self.cube_pos_init_buf[0] = self.cube_pos_init.q.x
        self.cube_pos_init_buf[1] = self.cube_pos_init.q.y
        self.cube_pos_init_buf[2] = self.cube_pos_init.q.z
        self.cube_pos_init_buf[3] = self.cube_pos_init.q.w
        self.cube_pos_init_buf[4] = self.cube_pos_init.p.x
        self.cube_pos_init_buf[5] = self.cube_pos_init.p.y
        self.cube_pos_init_buf[6] = self.cube_pos_init.p.z
        self.cube_pos_init_buf = torch.tile(self.cube_pos_init_buf, (self.num_envs, 1))

        self.cube_vel_init_buf = torch.zeros(6, dtype=torch.float32, device=self.device)
        self.cube_vel_init_buf[0] = self.cube_vel_init.top.x
        self.cube_vel_init_buf[1] = self.cube_vel_init.top.y
        self.cube_vel_init_buf[2] = self.cube_vel_init.top.z
        self.cube_vel_init_buf[3] = self.cube_vel_init.bottom.x
        self.cube_vel_init_buf[4] = self.cube_vel_init.bottom.y
        self.cube_vel_init_buf[5] = self.cube_vel_init.bottom.z
        self.cube_vel_init_buf = torch.tile(self.cube_vel_init_buf, (self.num_envs, 1))

        # Goal position buffer
        self.goal_cube_pos_buf = torch.zeros((self.num_envs, 7), dtype=torch.float32,
                                             device=self.device)

        # Trifinger set joint forces buffer
        self.set_joint_forces_buf = torch.zeros((self.num_envs, self.num_dofs), device=self.device,
                                                dtype=torch.float32)

        # Cube kinematic state buffers
        self.set_cube_pos_buf = torch.zeros((self.num_envs, 7), device=self.device,
                                            dtype=torch.float32)
        self.set_cube_vel_buf = torch.zeros((self.num_envs, 6), device=self.device,
                                            dtype=torch.float32)

        self.get_cube_pos_buf = torch.zeros((self.num_envs, 7), device=self.device,
                                            dtype=torch.float32)
        self.get_cube_vel_buf = torch.zeros((self.num_envs, 6), device=self.device,
                                            dtype=torch.float32)

        # Fingertip transforms and velocities
        self.get_fingertip_pos_buffers = []
        self.get_fingertip_vel_buffers = []
        for i in range(3):
            self.get_fingertip_pos_buffers.append(torch.zeros(
                (self.num_envs, 7), device=self.device, dtype=torch.float32))
            self.get_fingertip_vel_buffers.append(torch.zeros(
                (self.num_envs, 6), device=self.device, dtype=torch.float32))

        # Joint sensor force buffer
        self.get_joint_force_buf = torch.zeros((self.num_envs, self.num_dofs), device=self.device,
                                               dtype=torch.float32)

        # Force sensor buffer
        # num fingers x (force + torque)
        self.get_force_sensor_buf = torch.zeros((self.num_envs, 3 * 6), device=self.device,
                                                dtype=torch.float32)

        # Line cubes
        if self.draw_goal:
            assert self.rendering, "Rendering must be enabled to draw goal"
            self.line_cubes = []
            self.line_cube_color = v.Vec3(0.63, 0.13, 0.94)
            transform = v.Transform(v.Quat(0, 0, 0, 1), v.Vec3(0, 0, 0))
            for i in range(self.num_envs):
                env_handle = self.env_set.get_environment_handle(i)

                cube = self.gym_render.create_user_line_cube(
                    self.cube_size, transform, self.line_cube_color, env_handle=env_handle)

                self.gym_render.register_line_shape(cube)
                self.line_cubes.append(cube)

        # Frames on goal buffer (used for delaying goal reset)
        self.frames_on_goal = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # Help buffer
        self.new_pos_buf = torch.zeros(self.num_envs, 7, dtype=torch.float32, device=self.device)

    def create_access_commands(self):

        # Set joint forces GPU command
        set_joint_forces_cmd = self.env_group.create_joint_state_command(
            v.wrap_gpu_buffer(self.set_joint_forces_buf), self.arti_handle)
        self.set_joint_forces_cmd_arr = self.gym.create_joint_state_command_gpu_array([
                                                                                      set_joint_forces_cmd])

        # Set joint positions GPU command
        set_joint_pos_cmd = self.env_group.create_joint_state_command(
            v.wrap_gpu_buffer(self.set_dof_pos_buf), self.arti_handle,
            masks_buffer=v.wrap_gpu_buffer(self.trunc_buf))
        self.set_joint_pos_cmd_arr = self.gym.create_joint_state_command_gpu_array([
                                                                                   set_joint_pos_cmd])

        # Set joint velocities GPU command
        set_joint_vel_cmd = self.env_group.create_joint_state_command(
            v.wrap_gpu_buffer(self.set_dof_vel_buf), self.arti_handle,
            masks_buffer=v.wrap_gpu_buffer(self.trunc_buf))
        self.set_joint_vel_cmd_arr = self.gym.create_joint_state_command_gpu_array([
                                                                                   set_joint_vel_cmd])

        # Get joint positions GPU command
        get_joint_pos_cmd = self.env_group.create_joint_state_command(
            v.wrap_gpu_buffer(self.get_dof_pos_buf), self.arti_handle)
        self.get_joint_pos_cmd_arr = self.gym.create_joint_state_command_gpu_array([
                                                                                   get_joint_pos_cmd])

        # Get joint velocities GPU command
        get_joint_vel_cmd = self.env_group.create_joint_state_command(
            v.wrap_gpu_buffer(self.get_dof_vel_buf), self.arti_handle)
        self.get_joint_vel_cmd_arr = self.gym.create_joint_state_command_gpu_array([
                                                                                   get_joint_vel_cmd])

        # Set kinematic states of cube GPU command
        set_cube_kine_cmd = self.env_group.create_rigid_body_kinematic_state_command(
            v.wrap_gpu_buffer(self.set_cube_pos_buf), v.wrap_gpu_buffer(self.set_cube_vel_buf),
            self.cube_handle, masks_buffer=v.wrap_gpu_buffer(self.trunc_buf))
        self.set_cube_kine_cmd_arr = self.gym.create_rigid_body_kinematic_state_command_gpu_array(
            [set_cube_kine_cmd])

        # Get kinematic states of cube GPU command
        get_cube_kine_cmd = self.env_group.create_rigid_body_kinematic_state_command(
            v.wrap_gpu_buffer(self.get_cube_pos_buf), v.wrap_gpu_buffer(self.get_cube_vel_buf),
            self.cube_handle)
        self.get_cube_kine_cmd_arr = self.gym.create_rigid_body_kinematic_state_command_gpu_array(
            [get_cube_kine_cmd])

        # Get fingertip transforms & velocities command
        names = ["finger_tip_link_0", "finger_tip_link_120", "finger_tip_link_240"]
        pos_cmds = []
        vel_cmds = []
        for name, pos_buffer, vel_buffer in zip(names, self.get_fingertip_pos_buffers,
                                                self.get_fingertip_vel_buffers):
            index = self.art_def.get_link_def_index_by_name(name)
            pos_cmds.append(self.env_group.create_link_transform_command(
                v.wrap_gpu_buffer(pos_buffer), self.arti_handle, (index, index + 1)))
            vel_cmds.append(self.env_group.create_link_velocity_command(
                v.wrap_gpu_buffer(vel_buffer), self.arti_handle, (index, index + 1)))
        self.get_fingertip_pos_cmd_arr = self.gym.create_link_transform_command_gpu_array(pos_cmds)
        self.get_fingertip_vel_cmd_arr = self.gym.create_link_velocity_command_gpu_array(vel_cmds)

        # Get joint forces command
        get_joint_forces_cmd = self.env_group.create_joint_force_sensor_command(
            v.wrap_gpu_buffer(self.get_joint_force_buf), self.arti_handle)
        self.get_joint_forces_cmd_arr = self.gym.create_joint_force_sensor_command_gpu_array(
            [get_joint_forces_cmd])

        # Get fingertip force sensor command
        env_def = self.gym.get_environment_def(self.env_def_handle)
        articulation = env_def.get_articulation(self.arti_handle)

        self.force_sensor_handles = []
        self.force_sensor_buffers = []
        self.force_sensor_cmds = []
        for i in range(3):
            self.force_sensor_handles.append(articulation.get_force_sensor_handle(i))
            self.force_sensor_buffers.append(torch.zeros((self.num_envs, 6), device=self.device,
                                                         dtype=torch.float32))
            self.force_sensor_cmds.append(self.env_group.create_force_sensor_command(
                v.wrap_gpu_buffer(self.force_sensor_buffers[-1]), self.force_sensor_handles[-1]))

        self.get_force_sensor_cmd_arr = self.gym.create_force_sensor_command_gpu_array(
            self.force_sensor_cmds)

        # Set position of goal cube GPU command
        if self.record_goal:
            set_goal_pos_cmd = self.env_group.create_rigid_body_transform_command(
                v.wrap_gpu_buffer(self.goal_cube_pos_buf), self.goal_handle)
            self.set_goal_pos_cmd_arr = self.gym.create_rigid_body_transform_command_gpu_array([
                                                                                               set_goal_pos_cmd])

    def reset_idx(self):
        with profiler.record_function("RESET IDX"):

            # Set kinematic states of trifinger
            self.set_dof_pos_buf[:] = self.dof_pos_init_buf + \
                reset_noise_helper(self.dof_pos_init_buf, self.reset_noise_scale, 0.4, 0.2)
            self.set_dof_vel_buf[:] = self.dof_vel_init_buf + \
                reset_noise_helper(self.dof_vel_init_buf, self.reset_noise_scale, 0.2, 0.1)

            self.gym.set_joint_positions(self.set_joint_pos_cmd_arr)
            self.gym.set_joint_velocities(self.set_joint_vel_cmd_arr)

            # Set kinematic states of cube
            self.sample_object_pos(self.distribution)

            # Reset goal position
            self.sample_object_goal_pos(self.trunc_buf, self.difficulty)

            # Auxiliary buffers
            self.act_buf[:] = torch.where(self.trunc_buf.view(-1, 1), 0, self.act_buf)
            self.progress_buf[:] = torch.where(self.trunc_buf, 0, self.progress_buf)

    def reset_goal_idx(self):
        with profiler.record_function("RESET GOAL IDX"):
            self.sample_object_goal_pos(self.term_buf, self.difficulty)

            if self.reset_progress:
                self.progress_buf[:] = torch.where(self.term_buf, 0, self.progress_buf)

    def sample_object_pos(self, distribution="random"):
        with profiler.record_function("SAMPLE OBJECT POS"):

            if distribution == "none":
                return
            elif distribution == "default":
                self.set_cube_pos_buf[:] = self.cube_pos_init_buf
                self.set_cube_vel_buf[:] = self.cube_vel_init_buf
            elif distribution == "random":
                pos_x, pos_y = random_xy(
                    self.num_envs, self.max_com_distance_to_center, self.device)
                pos_z = self.cube_size / 2
                orientation = random_yaw_orientation(self.num_envs, self.device)

                self.set_cube_pos_buf[:, 4] = pos_x
                self.set_cube_pos_buf[:, 5] = pos_y
                self.set_cube_pos_buf[:, 6] = pos_z
                self.set_cube_pos_buf[:, 0:4] = orientation
                self.set_cube_vel_buf[:] = self.cube_vel_init_buf
            else:
                msg = f"Invalid object initial state distribution. Input: {distribution} " \
                      "not in [`default`, `random`, `none`]."
                raise ValueError(msg)

            self.gym.set_rigid_body_kinematic_states(self.set_cube_kine_cmd_arr)

    def reset(self):

        # Setting trunc buf in order to trigger resets across all environments, since
        self.trunc_buf[:] = True

        super().reset()

        # Fetch joint positions, velocities, and forces
        self.gym.get_joint_positions(self.get_joint_pos_cmd_arr)
        self.gym.get_joint_velocities(self.get_joint_vel_cmd_arr)
        self.gym.get_joint_sensor_forces(self.get_joint_forces_cmd_arr)

        # Fetch rigid body kinematic states
        self.gym.get_rigid_body_kinematic_states(self.get_cube_kine_cmd_arr)

        # Fetch fingertip transforms & velocities & forces
        self.gym.get_link_transforms(self.get_fingertip_pos_cmd_arr)
        self.gym.get_link_velocities(self.get_fingertip_vel_cmd_arr)
        self.gym.get_sensor_forces(self.get_force_sensor_cmd_arr)

        # Conversion operation for compatibility with handle-based sensor API
        self.get_force_sensor_buf[:] = torch.cat(self.force_sensor_buffers, dim=-1)

        self.obs_buf[:] = self.compute_observation(self.act_buf)
        self.state_buf[:] = self.compute_state()

        return {"obs": self.obs_buf.clone(), "states": self.state_buf.clone()}, {}

    # Overwrite step() to implement asymmetric PPO
    def step(self, actions):
        with profiler.record_function("ENVIRONMENT STEP"):
            obs, rew, term, trunc, info = super().step(actions)
            return {"obs": obs, "states": self.state_buf.clone()}, rew, term, trunc, info

    def pre_physics_step(self, actions: torch.Tensor):
        with profiler.record_function("PRE PHYSICS STEP"):
            # Reset robot, object, goal
            self.reset_idx()

            # Reset goal
            self.reset_goal_idx()

            self.act_buf[:] = actions

            actions_transformed = unscale_transform(self.act_buf, lower=-self.max_torque,
                                                    upper=self.max_torque)
            actions_clamped = saturate(actions_transformed, lower=-self.max_torque,
                                       upper=self.max_torque)

            self.set_joint_forces_buf[:] = actions_clamped

            self.gym.set_joint_forces(self.set_joint_forces_cmd_arr)

            # Save fingertip and object state
            self.gym.get_rigid_body_kinematic_states(self.get_cube_kine_cmd_arr)
            self.gym.get_link_transforms(self.get_fingertip_pos_cmd_arr)
            self.old_fingertip_pos_buffers = [self.get_fingertip_pos_buffers[i].clone() for i in
                                              range(3)]
            self.old_cube_pos_buf = self.get_cube_pos_buf.clone()

    def post_physics_step(self):
        with profiler.record_function("POST PHYSICS STEP"):
            self.progress_buf += 1

            # Fetch joint positions and velocities
            self.gym.get_joint_positions(self.get_joint_pos_cmd_arr)
            self.gym.get_joint_velocities(self.get_joint_vel_cmd_arr)
            self.gym.get_joint_sensor_forces(self.get_joint_forces_cmd_arr)

            # Fetch rigid body kinematic states
            self.gym.get_rigid_body_kinematic_states(self.get_cube_kine_cmd_arr)

            # Fetch fingertip transforms & velocities & forces
            self.gym.get_link_transforms(self.get_fingertip_pos_cmd_arr)
            self.gym.get_link_velocities(self.get_fingertip_vel_cmd_arr)
            self.gym.get_sensor_forces(self.get_force_sensor_cmd_arr)

            self.obs_buf[:] = self.compute_observation(self.act_buf)
            self.state_buf[:] = self.compute_state()

            self.rew_buf[:] = self.compute_reward()
            self.term_buf[:] = self.compute_termination()
            self.trunc_buf[:] = self.compute_truncation()

            self.reset_buf[:] = torch.logical_or(self.term_buf, self.trunc_buf)

    def compute_observation(self, actions):
        return torch.cat((
            self.get_dof_pos_buf,           # DOF positions
            self.get_dof_vel_buf,           # DOF velocities
            self.get_cube_pos_buf,          # Cube pose
            self.goal_cube_pos_buf,         # Goal pose
            actions), dim=-1)

    def compute_state(self):
        """
        Compute state.

        Call this after self.gym.get_rigid_body_kinematic_states(self.get_cube_kine_cmd_arr)
        and self.obs_buf[:] = self.compute_observations(self.act_buf)
        """
        return torch.cat(
            (self.obs_buf,
             self.get_cube_vel_buf,
             *self.get_fingertip_pos_buffers,
             *self.get_fingertip_vel_buffers,
             self.get_joint_force_buf,
             self.get_force_sensor_buf),
            dim=-1)

    def compute_reward(self):
        return compute_reward_helper(
            self.goal_cube_pos_buf,
            self.get_cube_pos_buf,
            self.old_cube_pos_buf,
            self.get_fingertip_pos_buffers,
            self.old_fingertip_pos_buffers,
            self.dt,
            self.object_dist_weight,
            self.object_move_weight,
            self.object_rot_weight,
            self.object_rot_delta_weight,
            self.fingertip_reach_weight,
            self.fingertip_move_weight)

    def compute_termination(self):
        termination, self.rew_buf[:], self.frames_on_goal[:] = compute_termination_helper(
            self.goal_cube_pos_buf,
            self.get_cube_pos_buf,
            self.rew_buf,
            self.frames_on_goal,
            self.position_tolerance,
            self.orientation_tolerance,
            self.on_target_bonus,
            self.difficulty,
            self.goal_reset_delay,
            self.success_bonus)
        return termination

    def compute_truncation(self):
        return self.progress_buf >= self.max_episode_length

    def sample_object_goal_pos(self, reset_buffer, difficulty=0):
        with profiler.record_function("SAMPLE OBJECT GOAL POS"):

            # Custom: fixed goal
            if difficulty == 0:

                pos_x = 0.1
                pos_y = 0
                pos_z = self.cube_size / 2
                orientation = default_orientation(self.num_envs, self.device)

            # Random position on the ground
            elif difficulty == 1:
                pos_x, pos_y = random_xy(
                    self.num_envs, self.max_com_distance_to_center, self.device)
                pos_z = self.cube_size / 2
                orientation = default_orientation(self.num_envs, self.device)

            # Fixed position in the air
            elif difficulty == 2:
                pos_x, pos_y = 0.0, 0.0
                pos_z = self.cube_size / 2 + 0.05
                orientation = default_orientation(self.num_envs, self.device)

            # Random position in the air
            elif difficulty == 3:
                pos_x, pos_y = random_xy(
                    self.num_envs, self.max_com_distance_to_center, self.device)
                pos_z = random_z(self.num_envs, self.cube_size / 2, 0.1, self.device)
                orientation = default_orientation(self.num_envs, self.device)

            # Random position and orientation in the air
            elif difficulty == 4:
                pos_x, pos_y = random_xy(
                    self.num_envs, self.max_com_distance_to_center, self.device)
                pos_z = random_z(self.num_envs, self.radius_3d, 0.1, self.device)
                orientation = random_orientation(self.num_envs, self.device)

            else:
                raise NotImplementedError

            self.new_pos_buf[:, 0:4] = orientation
            self.new_pos_buf[:, 4] = pos_x
            self.new_pos_buf[:, 5] = pos_y
            self.new_pos_buf[:, 6] = pos_z

            self.goal_cube_pos_buf[:] = torch.where(reset_buffer.view(-1, 1), self.new_pos_buf,
                                                    self.goal_cube_pos_buf)

            if self.record_goal:
                self.gym.set_rigid_body_transforms(self.set_goal_pos_cmd_arr)

            if self.draw_goal:
                for i in torch.nonzero(reset_buffer).squeeze(-1):

                    transform = v.Transform(v.Quat(
                        self.goal_cube_pos_buf[i][0],
                        self.goal_cube_pos_buf[i][1],
                        self.goal_cube_pos_buf[i][2],
                        self.goal_cube_pos_buf[i][3]),
                        v.Vec3(
                        self.goal_cube_pos_buf[i][4],
                        self.goal_cube_pos_buf[i][5],
                        self.goal_cube_pos_buf[i][6]))

                    self.line_cubes[i].set_transform(transform)


if __name__ == "__main__":
    from time import sleep
    from sys import argv
    from vlearn.utils import get_VL_VISUAL_TESTS

    rendering = True
    with_window = get_VL_VISUAL_TESTS()

    num_iter = np.inf
    if len(argv) >= 2:
        num_iter = int(argv[1])

    difficulty = 1
    num_envs = 2
    max_episode_length = 750
    # max_episode_length = 10
    enable_scene_query = True
    draw_goal = rendering
    goal_reset_delay = 0
    # goal_reset_delay = 60

    assert torch.cuda.is_available()
    device = torch.device("cuda:0")

    envs = TrifingerEnvironmentGpu(
        num_envs,
        device,
        rendering=rendering,
        max_episode_length=max_episode_length,
        enable_scene_query=enable_scene_query,
        draw_goal=draw_goal,
        difficulty=difficulty,
        goal_reset_delay=goal_reset_delay,
        with_window=with_window)

    obs, _ = envs.reset()

    gym = v.get_gym()
    render = gym.get_render()

    # Define controls
    reset_box = v.UserCheckbox("Reset", False)
    if render is not None:
        render.register_menu_item(reset_box)

    sliders = []

    for name in envs.art_def.get_joint_dof_def_names():

        sliders.append(v.UserSlider(name, -1, 1, 0))

        if render is not None:
            render.register_menu_item(sliders[-1])

    def control_by_menu():

        if reset_box.get_value():
            envs.reset()

        action = torch.tensor([slider.get_value() for slider in sliders], dtype=torch.float32,
                              device=device)

        return torch.tile(action, (num_envs, 1))

    control_fn = control_by_menu

    # Start simulation loop
    if render is not None:
        render.capped_step = True
    finished = False
    idx = 0
    while not finished:

        actions = control_fn()

        obs, reward, terminated, truncated, info = envs.step(actions)

        # print("obs: {}".format(obs))
        # print("obs[:,0:9]: {}".format(obs[:,0:9]))         # dof pos
        # print("obs[:,9:18]: {}".format(obs[:,9:18]))       # dof vel
        # print("obs[:,18:25]: {}".format(obs[:,18:25]))     # object pose
        # print("obs[:,25:32]: {}".format(obs[:,25:32]))     # object goal pose
        # print("obs[:,32:41]: {}".format(obs[:,32:41]))     # actions

        # print("reward: {}".format(reward))

        finished = envs.render_finished

        idx += 1
        if idx >= num_iter:
            finished = True
