import torch
import numpy as np
from vlearn.spaces import Box
from typing import List, Tuple
from time import time

import vlearn as v

if __name__ == "__main__":
    from environment import EnvironmentGpu
    from common import create_plane, reset_noise_helper
    from vlearn.torch_utils.torch_jit_utils import (
        scale, torch_rand_float, quat_mul, quat_from_angle_axis, quat_conjugate)
else:
    from .environment import EnvironmentGpu
    from .common import create_plane, reset_noise_helper
    from vlearn.torch_utils.torch_jit_utils import (
        scale, torch_rand_float, quat_mul, quat_from_angle_axis, quat_conjugate)


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))


def check_for_nans(tensor, name="default"):

    if torch.isnan(tensor).any():
        print(f"NaN found in {name}: {tensor}")

        print("torch.nonzero(torch.isnan(tensor)): {}".format(torch.nonzero(torch.isnan(tensor))))

        # raise Exception


@torch.jit.script
def compute_reward_termination_truncation_helper(
        actions: torch.Tensor,
        rew_buf: torch.Tensor,
        term_buf: torch.Tensor,
        trunc_buf: torch.Tensor,
        progress_buf: torch.Tensor,
        reset_goal_buf: torch.Tensor,
        object_rot: torch.Tensor,
        object_pos: torch.Tensor,
        target_rot: torch.Tensor,
        target_pos: torch.Tensor,
        successes: torch.Tensor,
        last_actions: torch.Tensor,
        last_last_actions: torch.Tensor,
        last_dof_vel: torch.Tensor,
        success_tolerance: float,
        max_consecutive_successes: int,
        max_episode_length: int,
        fall_dist: float,
        dist_reward_scale: float,
        rot_eps: float,
        rot_reward_scale: float,
        action_penalty_scale: float,
        action_smoothness_penalty_scale: float,
        reach_goal_bonus: float,
        fall_down_penalty: float):

    # Distance from hand to object
    goal_dist = torch.norm(object_pos - target_pos, p=2, dim=-1)

    # Orientation alignment for the cube in hand and goal cube
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

    dist_rew = goal_dist * dist_reward_scale
    rot_rew = 1.0 / (torch.abs(rot_dist) + rot_eps) * rot_reward_scale

    # Action penalty
    action_penalty = action_penalty_scale * torch.sum(torch.square(actions), dim=-1)

    # Action smoothness penalty
    term1 = torch.sum(torch.square(last_actions), dim=-1)
    term2 = torch.sum(torch.square(actions - 2 * last_actions + last_last_actions), dim=-1)
    term3 = 0.05 * torch.sum(torch.abs(actions), dim=-1)
    action_smoothness_penalty = action_smoothness_penalty_scale * (term1 + term2 + term3)

    # Determine goal resets
    reset_goal_buf[:] = torch.abs(rot_dist) <= success_tolerance
    successes += reset_goal_buf

    # Success bonus: orientation is within `success_tolerance` of goal orientation
    success_bonus = reset_goal_buf * reach_goal_bonus

    # Termination condition
    term_buf[:] = goal_dist >= fall_dist

    # Fall penalty: distance to the goal is larger than a threshold
    fall_penalty = term_buf * fall_down_penalty

    # Total reward is: position distance + orientation alignment + action regularization +
    # success bonus + fall penalty
    rew_buf[:] = (dist_rew + rot_rew + action_penalty + action_smoothness_penalty + success_bonus +
                  fall_penalty)

    # Truncation condition
    trunc_buf[:] = progress_buf >= max_episode_length

    # If max_consecutive_successes > 0, i) reset progress upon goal reset, and ii) reset env
    # when max_consecutive_successes is exceeded
    # Else: terminate only when cube falls or environment times out
    if max_consecutive_successes > 0:
        progress_buf[:] = torch.where(reset_goal_buf, 0, progress_buf)
        term_buf[:] = torch.where(successes >= max_consecutive_successes, True, term_buf)

        # Apply penalty for not reaching the goal
        rew_buf += trunc_buf * 0.5 * fall_down_penalty


class AllegroEnvironment(EnvironmentGpu):

    def __init__(self,
                 num_envs: int,
                 device: torch.device,
                 control_mode: str = 'pid',
                 pid_stiffness: float = 3,
                 pid_damping: float = 0.1,
                 pid_max_force: float = 0.7,  # Taken from tech specs
                 rendering: bool = False,
                 enable_scene_query: bool = False,
                 max_episode_length: int = 480,
                 gravity: v.Vec3 = v.Vec3(0, 0, -9.81),
                 timestep: float = 0.01667,
                 frame_skip: int = 1,
                 spacing: float = 0.5,
                 initial_is_paused: bool = False,
                 send_interrupt: bool = False,
                 reset_noise_scale: float = 1,
                 fall_dist: float = 0.24,
                 max_consecutive_successes: int = 50,
                 success_tolerance: float = 0.1,
                 draw_goal: bool = False,
                 dist_reward_scale: float = -10.0,
                 rot_eps: float = 0.1,
                 rot_reward_scale: float = 1.0,
                 action_penalty_scale: float = -0.001,
                 action_smoothness_penalty_scale: float = -0.001,
                 reach_goal_bonus: float = 250.0,
                 fall_down_penalty: float = 0.0,
                 max_contact_pairs_per_env: int = 128,
                 has_self_collisions: bool = True,
                 force_mass_inertia_computation: bool = True,
                 allegro_density: float = 1377.9,  # only has effect if mass and inertia is recomputed
                 # This value was computed by comparing link masses to
                 # tech specs and calculating the density such that the
                 # combined mass of fingers and thumb correspond to the
                 # combined mass as predicted by the technical
                 # specifications
                 joint_friction: float = 0.2,  # Ignored if equal to -1.0
                 max_joint_velocity: float = 9.52,  # Ignored if equal to -1.0
                 # This value was taken from the tech specs
                 armature: float = 0.0,
                 checking_for_nans: bool = False,
                 print_memory_frequency: int = 0,  # do not print if equal to 0
                 print_minutes_passed: bool = False,
                 interactive_armature: bool = False,
                 armature_range: tuple[float, float] = (0, 0.05),
                 with_window: bool = True
                 ):

        num_dofs = 16

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
            num_dofs,
            treat_warning_as_error=True,
            initial_is_paused=initial_is_paused,
            send_interrupt=send_interrupt,
            up_axis=v.Vec3(
                0,
                0,
                1),
            max_contact_pairs_per_env=max_contact_pairs_per_env,
            with_window=with_window)

        self.device_str = f'{self.device.type}:{self.device.index}'

        # Only PID control for now
        assert control_mode in ['pid']
        self.control_mode = control_mode
        self.pid_stiffness = pid_stiffness
        self.pid_damping = pid_damping
        self.pid_max_force = pid_max_force
        self.reset_noise_scale = reset_noise_scale
        self.fall_dist = fall_dist
        self.max_consecutive_successes = max_consecutive_successes
        self.success_tolerance = success_tolerance
        self.draw_goal = draw_goal
        self.dist_reward_scale = dist_reward_scale
        self.rot_eps = rot_eps
        self.rot_reward_scale = rot_reward_scale
        self.action_penalty_scale = action_penalty_scale
        self.action_smoothness_penalty_scale = action_smoothness_penalty_scale
        self.reach_goal_bonus = reach_goal_bonus
        self.fall_down_penalty = fall_down_penalty
        self.has_self_collisions = has_self_collisions
        self.force_mass_inertia_computation = force_mass_inertia_computation
        self.allegro_density = allegro_density
        self.joint_friction = joint_friction
        self.max_joint_velocity = max_joint_velocity
        self.armature = armature
        self.checking_for_nans = checking_for_nans
        self.print_memory_frequency = print_memory_frequency
        self.print_minutes_passed = print_minutes_passed
        self.interactive_armature = interactive_armature
        self.armature_range = armature_range

        self.counter = 0

        if self.print_minutes_passed:
            self.minutes_passed = 0
            self.init_time = time()

        # constants
        self.cube_size = 0.065

        self.num_obs = 0
        self.num_obs += 7  # block pos
        self.num_obs += 6  # block vel
        self.num_obs += self.num_dofs  # dof pos
        self.num_obs += self.num_dofs  # dof vel

        self.num_obs += 4  # goal quat

        self.num_obs += self.num_dofs  # last actions

        self.single_observation_space = Box(
            low=np.array([np.finfo('f').min] * self.num_obs, dtype=np.float32),
            high=np.array([np.finfo('f').max] * self.num_obs, dtype=np.float32),
            dtype=np.float32)
        self.single_action_space = Box(
            low=np.array([-1] * self.num_dofs, dtype=np.float32),
            high=np.array([1] * self.num_dofs, dtype=np.float32),
            dtype=np.float32)

        # Create environments
        self.create_envs()

        # Store initial conditions
        self.store_initial_conditions()

        # Allocate buffers
        self.allocate_buffers()

        # Create GPU commands
        self.create_gpu_commands()

        # Initialize goal drawing
        if self.draw_goal:
            self.initialize_goal_drawing()

        # Initialize interactive armature
        if self.interactive_armature:
            self.initialize_interactive_armature()

        # Create plane
        # create_plane(self.gym)

        # Finalize gym
        self.gym.gym_finalize()

        # Set camera
        if self.rendering:
            self.gym_render.reset_camera(v.Vec3(-1, -1, 2), v.Vec3(1, 1, -1))

    def create_envs(self):

        # Environment def
        self.env_def_handle = self.gym.create_environment_def("allegro")
        env_def = self.gym.get_environment_def(self.env_def_handle)

        # Allegro hand
        filename = "assets/kuka_allegro_description/allegro_touch_sensor.vsim"
        env_def.import_definitions(filename, fixed=True, use_visual_mesh=True,
                                   force_mass_computation=self.force_mass_inertia_computation,
                                   force_inertia_computation=self.force_mass_inertia_computation,
                                   density=self.allegro_density
                                   )
        self.def_handle = env_def.get_articulation_def_handle_by_name("kuka_allegro")

        self.art_def = env_def.get_articulation_def(self.def_handle)
        self.art_def.has_self_collisions = self.has_self_collisions
        self.art_def.contact_offset = 0.005

        rot = v.Quat(0.283045, 0.683330, -0.621782, 0.257551)
        pos = v.Vec3(0, 0, 0.5)
        self.allegro_root_trans_init = v.Transform(rot, pos)

        self.arti_handle = env_def.create_articulation(self.def_handle,
                                                       self.allegro_root_trans_init, 'allegro')

        # Configure PID and joints
        self.configure_pid_joint()

        # Block
        block_filename = "assets/kuka_allegro_description/cube_multicolor_dextreme.urdf"
        env_def.import_definitions(block_filename, fixed=False)
        block_def_handle = env_def.get_rigid_body_def_handle_by_name("object")

        self.block_root_trans_init = v.Transform(v.Quat(0, 0, 0, 1), v.Vec3(0.0, -0.15, 0.06 + 0.5))
        self.block_root_vel_init = v.SpatialVector(0)

        self.block_handle = env_def.create_rigid_body(block_def_handle,
                                                      self.block_root_trans_init, 'block')

        env_def.finalize()

        super().create_envs(self.env_def_handle)

    def store_initial_conditions(self):

        # Allegro hand
        self.allegro_root_vel_init = v.SpatialVector(0)

        dof_pos_low = []
        dof_pos_high = []
        dof_pos_init = []

        for dofdef in self.art_def.get_joint_dof_defs():
            dof_pos_low.append(dofdef.low_limit)
            dof_pos_high.append(dofdef.high_limit)
            dof_pos_init.append(np.clip(0, dofdef.low_limit, dofdef.high_limit))

        self.dof_pos_low = torch.tensor(dof_pos_low, dtype=torch.float32, device=self.device)
        self.dof_pos_high = torch.tensor(dof_pos_high, dtype=torch.float32, device=self.device)

        self.dof_pos_init = torch.tensor(dof_pos_init, dtype=torch.float32, device=self.device)
        self.dof_vel_init = torch.zeros_like(self.dof_pos_init, dtype=torch.float32,
                                             device=self.device)

    def configure_pid_joint(self):

        if self.control_mode == 'pid':

            for pid_def in self.art_def.get_pid_defs():
                pid_def.stiffness = self.pid_stiffness
                pid_def.damping = self.pid_damping
                pid_def.max_force = self.pid_max_force

        else:
            raise Exception("This should not be reached")

        for joint_def in self.art_def.get_joint_defs():
            if not self.joint_friction == -1.0:
                joint_def.joint_friction = self.joint_friction
            if not self.max_joint_velocity == -1.0:
                joint_def.max_joint_velocity = self.max_joint_velocity

        for dof_def in self.art_def.get_joint_dof_defs():
            dof_def.armature = self.armature

    def allocate_buffers(self):

        super().allocate_buffers()

        # Set allegro kinematic state
        self.gpu_init_dof_pos = torch.tile(self.dof_pos_init, (self.num_envs, 1))
        self.gpu_init_dof_vel = torch.zeros_like(self.gpu_init_dof_pos)

        self.gpu_init_root_velocities = torch.empty(6, dtype=torch.float32, device=self.device)
        self.gpu_init_root_velocities[0] = self.allegro_root_vel_init.top.x
        self.gpu_init_root_velocities[1] = self.allegro_root_vel_init.top.y
        self.gpu_init_root_velocities[2] = self.allegro_root_vel_init.top.z
        self.gpu_init_root_velocities[3] = self.allegro_root_vel_init.bottom.x
        self.gpu_init_root_velocities[4] = self.allegro_root_vel_init.bottom.y
        self.gpu_init_root_velocities[5] = self.allegro_root_vel_init.bottom.z
        self.gpu_init_root_velocities = torch.tile(
            self.gpu_init_root_velocities, (self.num_envs, 1))

        self.gpu_init_root_transforms = torch.empty(7, dtype=torch.float32, device=self.device)
        self.gpu_init_root_transforms[0] = self.allegro_root_trans_init.q.x
        self.gpu_init_root_transforms[1] = self.allegro_root_trans_init.q.y
        self.gpu_init_root_transforms[2] = self.allegro_root_trans_init.q.z
        self.gpu_init_root_transforms[3] = self.allegro_root_trans_init.q.w
        self.gpu_init_root_transforms[4] = self.allegro_root_trans_init.p.x
        self.gpu_init_root_transforms[5] = self.allegro_root_trans_init.p.y
        self.gpu_init_root_transforms[6] = self.allegro_root_trans_init.p.z
        self.gpu_init_root_transforms = torch.tile(
            self.gpu_init_root_transforms, (self.num_envs, 1))

        # Get block kinematic state
        self.get_block_pos_buf = torch.zeros((self.num_envs, 7), device=self.device,
                                             dtype=torch.float32)

        self.get_block_vel_buf = torch.zeros((self.num_envs, 6), device=self.device,
                                             dtype=torch.float32)

        # Set block kinematic state
        self.set_block_pos_buf = torch.zeros((self.num_envs, 7), device=self.device,
                                             dtype=torch.float32)

        self.gpu_init_block_velocities = torch.empty(6, dtype=torch.float32, device=self.device)
        self.gpu_init_block_velocities[0] = self.block_root_vel_init.top.x
        self.gpu_init_block_velocities[1] = self.block_root_vel_init.top.y
        self.gpu_init_block_velocities[2] = self.block_root_vel_init.top.z
        self.gpu_init_block_velocities[3] = self.block_root_vel_init.bottom.x
        self.gpu_init_block_velocities[4] = self.block_root_vel_init.bottom.y
        self.gpu_init_block_velocities[5] = self.block_root_vel_init.bottom.z
        self.gpu_init_block_velocities = torch.tile(self.gpu_init_block_velocities, (self.num_envs,
                                                                                     1))

        self.gpu_init_block_transforms = torch.empty(7, dtype=torch.float32, device=self.device)
        self.gpu_init_block_transforms[0] = self.block_root_trans_init.q.x
        self.gpu_init_block_transforms[1] = self.block_root_trans_init.q.y
        self.gpu_init_block_transforms[2] = self.block_root_trans_init.q.z
        self.gpu_init_block_transforms[3] = self.block_root_trans_init.q.w
        self.gpu_init_block_transforms[4] = self.block_root_trans_init.p.x
        self.gpu_init_block_transforms[5] = self.block_root_trans_init.p.y
        self.gpu_init_block_transforms[6] = self.block_root_trans_init.p.z
        self.gpu_init_block_transforms = torch.tile(self.gpu_init_block_transforms, (self.num_envs,
                                                                                     1))

        self.set_block_pos_buf[:] = self.gpu_init_block_transforms

        # Set armature
        self.set_armature_buf = torch.zeros(self.num_dofs, dtype=torch.float32, device=self.device)

        # Block rotation randomization
        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float,
                                          device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float,
                                          device=self.device).repeat((self.num_envs, 1))

        # Goal
        self.goal_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float32)
        self.goal_quat[:, 3] = 1.0

        self.reset_goal_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.successes = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)

        # Last buffers
        self.last_actions = torch.zeros_like(self.act_buf)
        self.last_last_actions = torch.zeros_like(self.act_buf)
        self.last_dof_vel = torch.zeros_like(self.get_dof_vel_buf)

    def create_gpu_commands(self):

        # Actions
        if self.control_mode == 'pid':
            set_pid_cmd = self.env_group.create_pid_control_command(
                v.wrap_gpu_buffer(self.act_buf), self.arti_handle, (0, self.num_dofs))
            self.set_pid_cmd_arr = self.gym.create_pid_control_command_gpu_array([set_pid_cmd])
        else:
            raise Exception("This should not be reached")

        # Get allegro joint positions
        get_joint_positions_command = self.env_group.create_joint_state_command(
            v.wrap_gpu_buffer(self.get_dof_pos_buf), self.arti_handle)

        self.gpu_get_joint_positions_command_array = self.gym.create_joint_state_command_gpu_array(
            [get_joint_positions_command])

        # Get allegro joint velocities
        get_joint_velocities_command = self.env_group.create_joint_state_command(
            v.wrap_gpu_buffer(self.get_dof_vel_buf), self.arti_handle)

        self.gpu_get_joint_velocities_command_array = self.gym.create_joint_state_command_gpu_array(
            [get_joint_velocities_command])

        # Set allegro kinematic state
        set_kinematic_state_command = self.env_group.create_articulation_kinematic_state_command(
            v.wrap_gpu_buffer(self.set_dof_pos_buf), v.wrap_gpu_buffer(self.set_dof_vel_buf),
            v.wrap_gpu_buffer(self.gpu_init_root_transforms),
            v.wrap_gpu_buffer(self.gpu_init_root_velocities),
            self.arti_handle, (0, self.num_dofs), (0, 1),
            masks_buffer=v.wrap_gpu_buffer(self.reset_buf))

        self.gpu_set_kinematic_state_command_array = \
            self.gym.create_articulation_kinematic_state_command_gpu_array(
                [set_kinematic_state_command])

        # Get block kinematic state
        get_block_rigid_body_kinematic_state_command = \
            self.env_group.create_rigid_body_kinematic_state_command(
                v.wrap_gpu_buffer(self.get_block_pos_buf),
                v.wrap_gpu_buffer(self.get_block_vel_buf), self.block_handle)

        self.gpu_get_block_kinematic_state_command_array = \
            self.gym.create_rigid_body_kinematic_state_command_gpu_array(
                [get_block_rigid_body_kinematic_state_command])

        # Set block kinematic state
        set_block_rigid_body_kinematic_state_command = \
            self.env_group.create_rigid_body_kinematic_state_command(
                v.wrap_gpu_buffer(self.set_block_pos_buf),
                v.wrap_gpu_buffer(self.gpu_init_block_velocities),
                self.block_handle, masks_buffer=v.wrap_gpu_buffer(self.reset_buf))

        self.gpu_set_block_kinematic_state_command_array = \
            self.gym.create_rigid_body_kinematic_state_command_gpu_array(
                [set_block_rigid_body_kinematic_state_command])

        # Set armature
        set_armature_command = self.env_group.create_joint_dof_property_command(
            v.JointDofProperty.ARMATURE, v.wrap_gpu_buffer(self.set_armature_buf),
            self.def_handle)
        self.gpu_set_armature_command_array = self.gym.create_joint_dof_property_command_gpu_array(
            [set_armature_command])

    def initialize_goal_drawing(self):
        assert self.rendering

        self.goal_cubes = []
        self.success_cubes = []
        line_cube_color = v.Vec3(0.63, 0.13, 0.94)

        for i in range(self.num_envs):

            env_handle = self.env_set.get_environment_handle(i)

            goal_cube = self.gym_render.create_user_line_cube(
                self.cube_size, self.block_root_trans_init, line_cube_color, env_handle=env_handle)

            self.gym_render.register_line_shape(goal_cube)
            self.goal_cubes.append(goal_cube)

            success_cube = self.gym_render.create_user_line_cube(
                self.cube_size, self.block_root_trans_init, line_cube_color, env_handle=env_handle)

            self.gym_render.register_line_shape(success_cube)
            self.success_cubes.append(success_cube)

    def draw_goals(self):

        for i in torch.nonzero(self.reset_goal_buf).squeeze(-1):

            success_transform = v.Transform(
                self.goal_cubes[i].get_transform().q,
                self.goal_cubes[i].get_transform().p + v.Vec3(0, 0, 0.2))

            self.success_cubes[i].set_transform(success_transform)

            transform = v.Transform(v.Quat(
                self.goal_quat[i][0],
                self.goal_quat[i][1],
                self.goal_quat[i][2],
                self.goal_quat[i][3]),
                self.block_root_trans_init.p)

            self.goal_cubes[i].set_transform(transform)

        for i in torch.nonzero(self.reset_buf).squeeze(-1):

            transform = v.Transform(v.Quat(
                self.goal_quat[i][0],
                self.goal_quat[i][1],
                self.goal_quat[i][2],
                self.goal_quat[i][3]),
                self.block_root_trans_init.p)

            self.goal_cubes[i].set_transform(transform)
            self.success_cubes[i].set_transform(transform)

    def set_armature(self, new_armature):
        self.set_armature_buf[:] = new_armature
        self.gym.set_joint_dof_properties(self.gpu_set_armature_command_array)

    def initialize_interactive_armature(self):
        assert self.rendering

        self.armature_value = self.armature
        self.armature_slider = v.UserSlider("armature", self.armature_range[0],
                                            self.armature_range[1], self.armature_value)
        self.gym_render.register_menu_item(self.armature_slider)

    def set_interactive_armature(self):

        new_armature = self.armature_slider.get_value()

        if self.armature_value != new_armature:
            self.armature_value = new_armature
            print("Setting armature to {}".format(self.armature_value))
            self.set_armature(self.armature_value)

    def reset_idx(self):

        # Sample articulation kinematic state
        self.set_dof_pos_buf[:] = self.gpu_init_dof_pos + \
            reset_noise_helper(self.gpu_init_dof_pos, self.reset_noise_scale, 0.4, 0.2)
        self.set_dof_vel_buf[:] = reset_noise_helper(self.gpu_init_dof_vel, self.reset_noise_scale,
                                                     0.2, 0.1)

        # Set articulation kinematic state
        self.gym.set_articulation_kinematic_states(self.gpu_set_kinematic_state_command_array)

        # Reset actions and progress
        if self.control_mode == 'pid':
            actions = self.set_dof_pos_buf
        else:
            raise Exception("This should not be reached")

        # This only has effect on the initial observations and the last buffers,
        # since these actions are overwritten by the policy in the first environment step
        self.act_buf[:] = torch.where(self.reset_buf.view(-1, 1), actions, self.act_buf)
        self.progress_buf[:] = torch.where(self.reset_buf, 0, self.progress_buf)

        # Sample block kinematic state
        rand_floats = torch_rand_float(-1.0, 1.0, (4, self.num_envs), self.device_str)
        init_quat_floats = randomize_rotation(rand_floats[0, :], rand_floats[1, :],
                                              self.x_unit_tensor, self.y_unit_tensor)

        self.set_block_pos_buf[:, 0:4] = init_quat_floats

        # Set block kinematic state
        self.gym.set_rigid_body_kinematic_states(self.gpu_set_block_kinematic_state_command_array)

        # Set goal and successes
        new_rand_quats = randomize_rotation(rand_floats[2, :], rand_floats[3, :],
                                            self.x_unit_tensor, self.y_unit_tensor)
        self.goal_quat[:] = torch.where((self.reset_buf | self.reset_goal_buf).view(-1, 1),
                                        new_rand_quats, self.goal_quat)

        self.successes[:] = torch.where(self.reset_buf, 0, self.successes)

        # Set last buffers
        self.last_last_actions[:] = torch.where(self.reset_buf.view(-1, 1), actions,
                                                self.last_last_actions)
        self.last_actions[:] = torch.where(self.reset_buf.view(-1, 1), actions, self.last_actions)
        self.last_dof_vel[:] = torch.where(self.reset_buf.view(-1, 1), self.set_dof_vel_buf,
                                           self.last_dof_vel)

        if self.draw_goal:
            self.draw_goals()

        if self.interactive_armature:
            self.set_interactive_armature()

    def reset(self):

        self.reset_buf[:] = True
        super().reset()

        return self.obs_buf, {}

    def pre_physics_step(self, actions: torch.Tensor):

        if self.control_mode == 'pid':
            self.act_buf[:] = scale(actions, self.dof_pos_low, self.dof_pos_high)
            self.gym.set_joint_target_positions(self.set_pid_cmd_arr)
        else:
            raise Exception("This should not be reached")

    def refresh_buffers(self):

        self.gym.get_joint_positions(self.gpu_get_joint_positions_command_array)

        self.gym.get_joint_velocities(self.gpu_get_joint_velocities_command_array)

        self.gym.get_rigid_body_kinematic_states(self.gpu_get_block_kinematic_state_command_array)

    def post_physics_step(self):

        self.progress_buf += 1

        # Reset
        self.reset_buf[:] = torch.logical_or(self.term_buf, self.trunc_buf)
        self.reset_idx()

        # Observations and reward
        self.refresh_buffers()

        self.compute_observation(self.act_buf)
        self.compute_reward_termination_truncation(self.act_buf)

        # Update last buffers
        self.last_last_actions[:] = self.last_actions
        self.last_actions[:] = self.act_buf
        self.last_dof_vel[:] = self.get_dof_vel_buf

        # Print memory stats
        if self.print_memory_frequency > 0 and self.counter % self.print_memory_frequency == 0:
            print(torch.cuda.memory_summary(device=self.device, abbreviated=False))

        self.counter += 1

        if self.print_minutes_passed:
            minutes_passed = int((time() - self.init_time) // 60)
            if minutes_passed > self.minutes_passed:
                self.minutes_passed = minutes_passed
                print("AllegroEnvironment: minutes passed = {}".format(self.minutes_passed))

    def compute_observation(self, actions):

        if self.checking_for_nans:
            check_for_nans(self.get_block_pos_buf, "self.get_block_pos_buf")
            check_for_nans(self.get_block_vel_buf, "self.get_block_vel_buf")
            check_for_nans(self.get_dof_pos_buf, "self.get_dof_pos_buf")
            check_for_nans(self.get_dof_vel_buf, "self.get_dof_vel_buf")
            check_for_nans(self.goal_quat, "self.goal_quat")
            check_for_nans(actions, "actions")

        self.obs_buf[:] = torch.cat(
            (self.get_block_pos_buf,
             self.get_block_vel_buf,
             self.get_dof_pos_buf,
             self.get_dof_vel_buf,
             self.goal_quat,
             actions),
            dim=-1)

    def compute_reward_termination_truncation(self, actions):

        compute_reward_termination_truncation_helper(
            actions,
            self.rew_buf,
            self.term_buf,
            self.trunc_buf,
            self.progress_buf,
            self.reset_goal_buf,
            self.get_block_pos_buf[:, 0:4],
            self.get_block_pos_buf[:, 4:7],
            self.goal_quat,
            self.gpu_init_block_transforms[:, 4:7],
            self.successes,
            self.last_actions,
            self.last_last_actions,
            self.last_dof_vel,
            self.success_tolerance,
            self.max_consecutive_successes,
            self.max_episode_length,
            self.fall_dist,
            self.dist_reward_scale,
            self.rot_eps,
            self.rot_reward_scale,
            self.action_penalty_scale,
            self.action_smoothness_penalty_scale,
            self.reach_goal_bonus,
            self.fall_down_penalty)


if __name__ == "__main__":
    from time import sleep
    from sys import argv
    from vlearn.utils import get_VL_VISUAL_TESTS

    rendering = True
    enable_scene_query = True
    with_window = get_VL_VISUAL_TESTS()

    num_iter = np.inf
    if len(argv) >= 2:
        num_iter = int(argv[1])

    assert torch.cuda.is_available()
    device = torch.device("cuda:0")

    # Set to true for testing
    test_goal = False
    initial_is_paused = False

    num_envs = 2
    draw_goal = True

    envs = AllegroEnvironment(num_envs, device, rendering=rendering,
                              enable_scene_query=enable_scene_query, draw_goal=draw_goal,
                              initial_is_paused=initial_is_paused, with_window=with_window)

    obs, _ = envs.reset()

    gym = v.get_gym()
    render = gym.get_render()

    # Start simulation loop
    if render is not None:
        render.capped_step = True
        render.reset_camera(v.Vec3(0, 0, 2), v.Vec3(0, 0, -1))

        reset_box = v.UserCheckbox("Reset", False)
        render.register_menu_item(reset_box)

        sliders = []

        if test_goal:

            block_pos = torch.zeros((envs.num_envs, 7), dtype=torch.float32, device=device)
            block_pos[:] = envs.gpu_init_block_transforms
            block_vel = torch.zeros((envs.num_envs, 6), dtype=torch.float32, device=device)

            set_block_kine_cmd = envs.env_group.create_rigid_body_kinematic_state_command(
                v.wrap_gpu_buffer(block_pos), v.wrap_gpu_buffer(block_vel),
                envs.block_handle)

            set_block_kine_cmd_arr = envs.gym.create_rigid_body_kinematic_state_command_gpu_array(
                [set_block_kine_cmd])

            sliders = []

            sliders.append(v.UserSlider("block x", -2, 2, envs.block_root_trans_init.p.x))
            sliders.append(v.UserSlider("block y", -2, 2, envs.block_root_trans_init.p.y))
            sliders.append(v.UserSlider("block z", -2, 2, envs.block_root_trans_init.p.z))

            sliders.append(v.UserSlider("block roll", -torch.pi, torch.pi, 0))
            sliders.append(v.UserSlider("block pitch", -torch.pi, torch.pi, 0))
            sliders.append(v.UserSlider("block yaw", -torch.pi, torch.pi, 0))

            for slider in sliders:
                render.register_menu_item(slider)

            def test_goal_pre_physics_step(envs, actions):

                quat = v.quat_from_rpy(v.Vec3(
                    sliders[3].get_value(),
                    sliders[4].get_value(),
                    sliders[5].get_value()))

                block_pos[:, 0] = quat.x
                block_pos[:, 1] = quat.y
                block_pos[:, 2] = quat.z
                block_pos[:, 3] = quat.w
                block_pos[:, 4] = sliders[0].get_value()
                block_pos[:, 5] = sliders[1].get_value()
                block_pos[:, 6] = sliders[2].get_value()

                envs.gym.set_rigid_body_kinematic_states(set_block_kine_cmd_arr)

                # Print goal RPY
                rpy = v.rpy_from_quat(v.Quat(*envs.goal_quat[0, :]))
                if rpy != test_goal_pre_physics_step.old_rpy:
                    print("Goal roll pitch yaw: {}".format(rpy))
                    test_goal_pre_physics_step.old_rpy = rpy

            test_goal_pre_physics_step.old_rpy = v.rpy_from_quat(v.Quat(0, 0, 0, 1))

            AllegroEnvironment.pre_physics_step = test_goal_pre_physics_step

            def control_fn(): return torch.zeros(envs.num_dofs, dtype=torch.float32, device=device)

        elif envs.control_mode == 'pid':

            for pid_def in envs.art_def.get_pid_defs():
                joint_index = envs.art_def.get_joint_dof_index(pid_def.link_index,
                                                               pid_def.local_dof_index)
                joint_name = envs.art_def.get_joint_dof_def_name(joint_index)

                sliders.append(v.UserSlider(joint_name, -1, 1, 0))

                render.register_menu_item(sliders[-1])

            def control_by_menu():

                if reset_box.get_value():
                    envs.reset()

                return torch.tensor([slider.get_value() for slider in sliders], dtype=torch.float32,
                                    device=device)

            control_fn = control_by_menu

        else:
            raise Exception("This should not be reached")

    else:
        if envs.control_mode == 'pid':
            def control_fn(
                n=num_envs, m=envs.art_def.get_num_pid_defs()): return torch.zeros(
                (n, m), dtype=torch.float32, device=device)
        else:
            raise Exception("This should not be reached")

    def print_obs(envs):
        obs = envs.obs_buf
        # print("obs[0, 0:7]: {}".format(obs[0, 0:7])) # block pos
        # print("obs[0, 7:13]: {}".format(obs[0, 7:13])) # block vel
        # print("obs[0, 13:29]: {}".format(obs[0, 13:29])) # dof pos
        # print("obs[0, 29:45]: {}".format(obs[0, 29:45])) # dof vel
        # print("obs[0, 45:49]: {}".format(obs[0, 45:49])) # goal quat
        # print("obs[0, 49:65]: {}".format(obs[0, 49:65])) # actions

    finished = False
    idx = 0
    while not finished:

        actions = control_fn()

        obs, reward, terminated, truncated, info = envs.step(actions)

        if idx % 60 == 0:
            print_obs(envs)

        # print("reward[0]: {}".format(reward[0]))

        finished = envs.render_finished

        idx += 1
        if idx >= num_iter:
            finished = True
