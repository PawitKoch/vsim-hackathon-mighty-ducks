from typing import Deque
import torch
import numpy as np
import statistics
import typing

import torch.profiler as profiler

from vlearn.spaces import Box

from collections import deque

import vlearn as v

import sys
import os

if __name__ == "__main__":
    from environment import EnvironmentGpu
    from common import create_plane, reset_noise_helper, quat_tensor
    from h1_environment_common import (create_envs_helper, store_initial_conditions_helper,
                                       allocate_gpu_buffers)
    from vlearn.torch_utils.torch_jit_utils import *
else:
    from .environment import EnvironmentGpu
    from .common import create_plane, reset_noise_helper, quat_tensor
    from .h1_environment_common import (create_envs_helper, store_initial_conditions_helper,
                                        allocate_gpu_buffers)
    from vlearn.torch_utils.torch_jit_utils import *


@torch.jit.script
def get_euler_xyz_tensor(quat):
    r, p, w = get_euler_xyz(quat)
    # stack r, p, w in dim1
    euler_xyz = torch.stack((r, p, w), dim=1)
    euler_xyz[euler_xyz > np.pi] -= 2 * np.pi
    return euler_xyz


@torch.jit.script
def reset_idx_helper(
        gpu_init_dof_pos,
        set_dof_pos_buf,
        set_dof_vel_buf,
        reset_noise_scale: float,
        reset_buf,
        feet_air_time,
        act_buf,
        progress_buf,
        last_last_actions,
        last_actions,
        last_dof_vel,
        last_root_vel,
        last_contacts,
        last_feet_z,
        init_last_feet_z: float,
        base_quat,
        base_pos,
        gpu_init_root_transforms,
        base_lin_vel,
        base_ang_vel,
        gpu_init_root_velocities,
        projected_gravity,
        gravity_vec):

    reset_buf_view = reset_buf.view(-1, 1)
    set_dof_pos_buf[:] = gpu_init_dof_pos + \
        reset_noise_helper(gpu_init_dof_pos, reset_noise_scale, 0.2, 0.1)
    set_dof_vel_buf[:] = torch.zeros_like(gpu_init_dof_pos)
    feet_air_time[:] = torch.where(reset_buf_view, 0, feet_air_time)
    act_buf[:] = torch.where(reset_buf_view, 0, act_buf)
    progress_buf[:] = torch.where(reset_buf, 0, progress_buf)
    last_last_actions[:] = torch.where(reset_buf_view, 0, last_last_actions)
    last_actions[:] = torch.where(reset_buf_view, 0, last_actions)
    last_dof_vel[:] = torch.where(reset_buf_view, 0, last_dof_vel)

    # The following two buffers are not reset in Roboterax but I don't why they shouldn't
    last_root_vel[:] = torch.where(reset_buf_view, 0, last_root_vel)
    last_contacts[:] = torch.where(reset_buf_view, 0, last_contacts)
    last_feet_z[:] = torch.where(reset_buf_view, init_last_feet_z, last_feet_z)

    base_quat[:] = torch.where(reset_buf_view, gpu_init_root_transforms[:, 0:4], base_quat)
    base_pos[:] = torch.where(reset_buf_view, gpu_init_root_transforms[:, 4:7], base_pos)
    base_lin_vel[:] = torch.where(reset_buf_view,
                                  quat_rotate_inverse(base_quat, gpu_init_root_velocities[:, 3:6]),
                                  base_lin_vel)
    base_ang_vel[:] = torch.where(reset_buf_view,
                                  quat_rotate_inverse(base_quat, gpu_init_root_velocities[:, 0:3]),
                                  base_ang_vel)

    projected_gravity[:] = torch.where(reset_buf_view,
                                       quat_rotate_inverse(base_quat, gravity_vec),
                                       projected_gravity)

    base_euler_xyz = get_euler_xyz_tensor(base_quat)
    return base_euler_xyz

    # return set_dof_pos_buf, set_dof_vel_buf, feet_air_time, act_buf, progress_buf, last_last_actions, \
    #     last_actions, last_dof_vel, last_root_vel, last_contacts, last_feet_z, base_quat, base_pos, base_lin_vel, \
    #     base_ang_vel, projected_gravity, base_euler_xyz


@torch.jit.script
def _resample_commands_helper(reset_mask,
                              commands,
                              command_ranges: dict[str,
                                                   tuple[float,
                                                         float]],
                              num_envs: int,
                              device: torch.device):

    commands[:, 0] = torch.where(reset_mask,
                                 torch_rand_float(
                                     command_ranges["lin_vel_x"][0],
                                     command_ranges["lin_vel_x"][1],
                                     (num_envs, 1),
                                     device=str(device)).squeeze(1),
                                 commands[:, 0])

    commands[:, 1] = torch.where(reset_mask,
                                 torch_rand_float(
                                     command_ranges["lin_vel_y"][0],
                                     command_ranges["lin_vel_y"][1],
                                     (num_envs, 1),
                                     device=str(device)).squeeze(1),
                                 commands[:, 1])

    commands[:, 3] = torch.where(reset_mask,
                                 torch_rand_float(
                                     command_ranges["heading"][0],
                                     command_ranges["heading"][1],
                                     (num_envs, 1),
                                     device=str(device)).squeeze(1),
                                 commands[:, 3])

    commands[:, :2] *= (torch.norm(commands[:, :2], dim=1) > 0.2).unsqueeze(1)


def reset_history_helper(obs_history, critic_history, reset_buf):
    for i in range(obs_history.maxlen):
        obs_history[i][:] = torch.where(reset_buf.view(-1, 1), 0, obs_history[i])
    for i in range(critic_history.maxlen):
        critic_history[i][:] = torch.where(reset_buf.view(-1, 1), 0,
                                           critic_history[i])


@torch.jit.script
def compute_ref_state_helper(phase, get_dof_pos_buf, use_xbot: bool, target_joint_pos_scale: float,
                             fix_upper_body: bool):

    sin_pos = torch.sin(2 * torch.pi * phase)
    sin_pos_l = sin_pos.clone()
    sin_pos_r = sin_pos.clone()

    ref_dof_pos = torch.zeros_like(get_dof_pos_buf)

    if use_xbot:
        scale_1 = target_joint_pos_scale
        scale_2 = 2 * scale_1
        # left foot stance phase set to default joint pos
        sin_pos_l[sin_pos_l > 0] = 0
        ref_dof_pos[:, 2] = sin_pos_l * scale_1
        ref_dof_pos[:, 3] = sin_pos_l * scale_2
        ref_dof_pos[:, 4] = sin_pos_l * scale_1
        # right foot stance phase set to default joint pos
        sin_pos_r[sin_pos_r < 0] = 0
        ref_dof_pos[:, 8] = sin_pos_r * scale_1
        ref_dof_pos[:, 9] = sin_pos_r * scale_2
        ref_dof_pos[:, 10] = sin_pos_r * scale_1

    elif fix_upper_body:

        # I had to change the signs here so that the dof values would be consistent
        scale_l1 = 2 * target_joint_pos_scale
        scale_l2 = - 2 * scale_l1
        scale_l3 = scale_l1

        scale_r1 = - scale_l1
        scale_r2 = - 2 * scale_r1
        scale_r3 = scale_r1

        # left foot stance phase set to default joint pos
        sin_pos_l[sin_pos_l > 0] = 0
        ref_dof_pos[:, 7] = sin_pos_l * scale_l1  # left_hip_pitch
        ref_dof_pos[:, 8] = sin_pos_l * scale_l2  # left_knee
        ref_dof_pos[:, 9] = sin_pos_l * scale_l3  # left_ankle

        # right foot stance phase set to default joint pos
        sin_pos_r[sin_pos_r < 0] = 0
        ref_dof_pos[:, 2] = sin_pos_r * scale_r1  # right_hip_pitch
        ref_dof_pos[:, 3] = sin_pos_r * scale_r2  # right_knee
        ref_dof_pos[:, 4] = sin_pos_r * scale_r3  # right_ankle

    else:
        # I had to change the signs here so that the dof values would be consistent
        scale_l1 = 2 * target_joint_pos_scale
        scale_l2 = - 2 * scale_l1
        scale_l3 = scale_l1

        scale_r1 = - scale_l1
        scale_r2 = - 2 * scale_r1
        scale_r3 = scale_r1

        # left foot stance phase set to default joint pos
        sin_pos_l[sin_pos_l > 0] = 0
        ref_dof_pos[:, 16] = sin_pos_l * scale_l1  # left_hip_pitch
        ref_dof_pos[:, 17] = sin_pos_l * scale_l2  # left_knee
        ref_dof_pos[:, 18] = sin_pos_l * scale_l3  # left_ankle

        # right foot stance phase set to default joint pos
        sin_pos_r[sin_pos_r < 0] = 0
        ref_dof_pos[:, 11] = sin_pos_r * scale_r1  # right_hip_pitch
        ref_dof_pos[:, 12] = sin_pos_r * scale_r2  # right_knee
        ref_dof_pos[:, 13] = sin_pos_r * scale_r3  # right_ankle

    # Double support phase
    ref_dof_pos[torch.abs(sin_pos) < 0.1] = 0

    ref_action = 2 * ref_dof_pos
    return ref_dof_pos, ref_action


@torch.jit.script
def compute_obs_helper(
        phase,
        stance_mask,
        sensor_force_buf,
        feet_force_indices: list[int],
        commands,
        commands_scale,
        get_dof_pos_buf,
        default_dof_pos,
        dof_pos_obs_scale: float,
        get_dof_vel_buf,
        dof_vel_obs_scale: float,
        ref_dof_pos,
        base_ang_vel,
        ang_vel_obs_scale: float,
        base_euler_xyz,
        quat_obs_scale: float,
        base_lin_vel,
        lin_vel_obs_scale: float,
        actions):
    sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
    cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)

    # stance_mask = self._get_gait_phase()
    contact_mask = sensor_force_buf[:, feet_force_indices, 1] < -5.

    command_input = torch.cat(
        (sin_pos, cos_pos, commands[:, :3] * commands_scale), dim=1)

    q = (get_dof_pos_buf - default_dof_pos) * dof_pos_obs_scale
    dq = get_dof_vel_buf * dof_vel_obs_scale

    diff = get_dof_pos_buf - ref_dof_pos

    obs_now = torch.cat((
        command_input,
        q,
        dq,
        actions,
        base_ang_vel * ang_vel_obs_scale,
        base_euler_xyz * quat_obs_scale,
        ), dim=-1)

    privileged_obs_now = torch.cat((
        obs_now,                                    # 68
        diff,                                       # 19
        base_lin_vel * lin_vel_obs_scale,  # 3
        # self.rand_push_force[:, :2],  # Leave pushing out for now
        # self.rand_push_torque,        # Leave pushing out for now
        # self.env_frictions,           # Leave domain randomisation out for now
        # self.body_mass / 30,          # Leave domain randomisation out for now
        stance_mask,                                # 2
        contact_mask,                               # 2
        ), dim=-1)

    return command_input, obs_now, privileged_obs_now


@torch.jit.script
def _get_phase_helper(cycle_time: float, progress_buf, dt: float):
    phase = progress_buf * dt / cycle_time
    return phase


@torch.jit.script
def _get_gait_phase_helper(
        cycle_time: float,
        progress_buf,
        dt: float,
        num_envs: int,
        device: torch.device):

    # return float mask 1 is stance, 0 is swing
    phase = _get_phase_helper(cycle_time, progress_buf, dt)
    sin_pos = torch.sin(2 * torch.pi * phase)

    # Add double support phase
    stance_mask = torch.zeros((num_envs, 2), device=device)

    # left foot stance
    stance_mask[:, 0] = sin_pos >= 0

    # right foot stance
    stance_mask[:, 1] = sin_pos < 0

    # Double support phase
    stance_mask[torch.abs(sin_pos) < 0.1] = 1

    return stance_mask


@torch.jit.script
def _reward_joint_pos_helper(get_dof_pos_buf, ref_dof_pos):
    """
    Calculates the reward based on the difference between the current joint positions and the
    target joint positions.
    """
    diff = get_dof_pos_buf - ref_dof_pos
    r = torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)
    return r


@torch.jit.script
def _reward_feet_distance_helper(
        get_left_foot_pos_buf,
        get_right_foot_pos_buf,
        min_dist: float,
        max_dist: float):
    """
    Calculates the reward based on the distance between the feet. Penalize feet get close to
    each other or too far away.
    """
    foot_pos = torch.stack([get_left_foot_pos_buf, get_right_foot_pos_buf],
                           dim=1)[:, :, 4:6]
    foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
    fd = min_dist
    max_df = max_dist
    d_min = torch.clamp(foot_dist - fd, -0.5, 0.)
    d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
    return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2


@torch.jit.script
def _reward_knee_distance_helper(
        get_left_knee_pos_buf,
        get_right_knee_pos_buf,
        min_dist: float,
        max_dist: float):
    """
    Calculates the reward based on the distance between the knee of the humanoid.
    """
    knee_pos = torch.stack([get_left_knee_pos_buf, get_right_knee_pos_buf],
                           dim=1)[:, :, 4:6]
    knee_dist = torch.norm(knee_pos[:, 0, :] - knee_pos[:, 1, :], dim=1)
    fd = min_dist
    max_df = max_dist
    d_min = torch.clamp(knee_dist - fd, -0.5, 0.)
    d_max = torch.clamp(knee_dist - max_df, 0, 0.5)
    return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2


@torch.jit.script
def _reward_foot_slip_helper(sensor_force_buf, feet_force_indices: list[int],
                             get_left_foot_vel_buf, get_right_foot_vel_buf):
    """
    Calculates the reward for minimizing foot slip. The reward is based on the contact forces
    and the speed of the feet. A contact threshold is used to determine if the foot is in
    contact with the ground. The speed of the foot is calculated and scaled by the contact
    condition.
    """
    contact = sensor_force_buf[:, feet_force_indices, 1] < -5.
    foot_vel = torch.stack([get_left_foot_vel_buf, get_right_foot_vel_buf],
                           dim=1)[:, :, 3:5]
    foot_speed_norm = torch.norm(foot_vel, dim=2)
    rew = torch.sqrt(foot_speed_norm)
    rew *= contact
    return torch.sum(rew, dim=1)


@torch.jit.script
def _reward_feet_air_time_helper(
        sensor_force_buf,
        feet_force_indices: list[int],
        cycle_time: float,
        progress_buf,
        dt: float,
        num_envs: int,
        device: torch.device,
        last_contacts,
        feet_air_time):
    """
    Calculates the reward for feet air time, promoting longer steps. This is achieved by
    checking the first contact with the ground after being in the air. The air time is
    limited to a maximum value for reward calculation.
    """
    contact = sensor_force_buf[:, feet_force_indices, 1] < -5.
    stance_mask = _get_gait_phase_helper(cycle_time, progress_buf, dt, num_envs, device)
    contact_filt = torch.logical_or(torch.logical_or(contact, stance_mask),
                                    last_contacts)
    first_contact = (feet_air_time > 0.) * contact_filt
    feet_air_time += dt
    air_time = feet_air_time.clamp(0, 0.5) * first_contact
    feet_air_time *= ~contact_filt
    return air_time.sum(dim=1), contact_filt, contact, feet_air_time


@torch.jit.script
def _reward_feet_contact_number_helper(
        sensor_force_buf,
        feet_force_indices: list[int],
        cycle_time: float,
        progress_buf,
        dt: float,
        num_envs: int,
        device: torch.device):
    """
    Calculates a reward based on the number of feet contacts aligning with the gait phase.
    Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
    """
    contact = sensor_force_buf[:, feet_force_indices, 1] < -5.
    stance_mask = _get_gait_phase_helper(cycle_time, progress_buf, dt, num_envs, device)
    reward = torch.where(contact == stance_mask, 1, -0.3)
    return torch.mean(reward, dim=1)


@torch.jit.script
def _reward_orientation_helper(base_euler_xyz, projected_gravity):
    """
    Calculates the reward for maintaining a flat base orientation. It penalizes deviation from
    the desired base orientation using the base euler angles and the projected gravity vector.
    """
    quat_mismatch = torch.exp(-torch.sum(torch.abs(base_euler_xyz[:, :2]), dim=1) * 10)
    orientation = torch.exp(-torch.norm(projected_gravity[:, :2], dim=1) * 20)
    return (quat_mismatch + orientation) / 2.


@torch.jit.script
def _reward_feet_contact_forces_helper(
        sensor_force_buf,
        feet_force_indices: list[int],
        max_contact_force: float):
    """
    Calculates the reward for keeping contact forces within a specified range. Penalizes
    high contact forces on the feet.
    """
    return torch.sum((torch.norm(sensor_force_buf[:, feet_force_indices, :], dim=-1) -
                      max_contact_force).clip(0, 400), dim=1)


@torch.jit.script
def _reward_default_joint_pos_helper(
        get_dof_pos_buf,
        default_dof_pos,
        use_xbot: bool,
        fix_upper_body: bool):
    """
    Calculates the reward for keeping joint positions close to default positions, with a focus
    on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main
    penalty.
    """
    joint_diff = get_dof_pos_buf - default_dof_pos
    if use_xbot:
        left_yaw_roll = joint_diff[:, 0:2]
        right_yaw_roll = joint_diff[:, 6:8]
    elif fix_upper_body:
        left_yaw_roll = joint_diff[:, 5:7]
        right_yaw_roll = joint_diff[:, 0:2]
    else:
        left_yaw_roll = joint_diff[:, 14:16]
        right_yaw_roll = joint_diff[:, 9:11]
    yaw_roll = torch.norm(left_yaw_roll, dim=1) + torch.norm(right_yaw_roll, dim=1)
    yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)
    if use_xbot or fix_upper_body:
        return torch.exp(-yaw_roll * 100) - 0.01 * torch.norm(joint_diff, dim=1)
    else:
        # use joint_diff from 9th dof onwards to include only locomotion related joints
        return torch.exp(-yaw_roll * 100) - 0.01 * torch.norm(joint_diff[:, 9:], dim=1)


@torch.jit.script
def _reward_base_height_helper(
        cycle_time: float,
        progress_buf,
        dt: float,
        num_envs: int,
        device: torch.device,
        get_left_foot_pos_buf,
        get_right_foot_pos_buf,
        root_pos_buf,
        base_height_target: float):
    """
    Calculates the reward based on the robot's base height. Penalizes deviation from a target
    base height. The reward is computed based on the height difference between the robot's base
    and the average height of its feet when they are in contact with the ground.
    """
    stance_mask = _get_gait_phase_helper(cycle_time, progress_buf, dt, num_envs, device)
    foot_pos = torch.stack([get_left_foot_pos_buf, get_right_foot_pos_buf],
                           dim=1)[:, :, 6]
    measured_heights = torch.sum(foot_pos * stance_mask, dim=1) / torch.sum(stance_mask, dim=1)
    base_height = root_pos_buf[:, 6] - (measured_heights - 0.05)
    return torch.exp(-torch.abs(base_height - base_height_target) * 100)


@torch.jit.script
def _reward_base_acc_helper(last_root_vel, root_vel_buf):
    """
    Computes the reward based on the base's acceleration. Penalizes high accelerations of the
    robot's base, encouraging smoother motion.
    """
    root_acc = last_root_vel - root_vel_buf
    rew = torch.exp(-torch.norm(root_acc, dim=1) * 3)
    return rew


@torch.jit.script
def _reward_vel_mismatch_exp_helper(base_lin_vel, base_ang_vel):
    """
    Computes a reward based on the mismatch in the robot's linear and angular velocities.
    Encourages the robot to maintain a stable velocity by penalizing large deviations.
    """
    lin_mismatch = torch.exp(-torch.square(base_lin_vel[:, 2]) * 10)
    ang_mismatch = torch.exp(-torch.norm(base_ang_vel[:, :2], dim=1) * 5.)

    c_update = (lin_mismatch + ang_mismatch) / 2.

    return c_update


@torch.jit.script
def _reward_track_vel_hard_helper(commands, base_lin_vel, base_ang_vel):
    """
    Calculates a reward for accurately tracking both linear and angular velocity commands.
    Penalizes deviations from specified linear and angular velocity targets.
    """
    # Tracking of linear velocity commands (xy axes)
    lin_vel_error = torch.norm(commands[:, :2] - base_lin_vel[:, :2], dim=1)
    lin_vel_error_exp = torch.exp(-lin_vel_error * 10)

    # Tracking of angular velocity commands (yaw)
    ang_vel_error = torch.abs(commands[:, 2] - base_ang_vel[:, 2])
    ang_vel_error_exp = torch.exp(-ang_vel_error * 10)

    linear_error = 0.2 * (lin_vel_error + ang_vel_error)

    return (lin_vel_error_exp + ang_vel_error_exp) / 2. - linear_error


@torch.jit.script
def _reward_tracking_lin_vel_helper(commands, base_lin_vel, tracking_sigma: float):
    """
    Tracks linear velocity commands along the xy axes.
    Calculates a reward based on how closely the robot's linear velocity matches the commanded
    values.
    """
    lin_vel_error = torch.sum(torch.square(
        commands[:, :2] - base_lin_vel[:, :2]), dim=1)
    return torch.exp(-lin_vel_error * tracking_sigma)


@torch.jit.script
def _reward_tracking_ang_vel_helper(commands, base_ang_vel, tracking_sigma: float):
    """
    Tracks angular velocity commands for yaw rotation.
    Computes a reward based on how closely the robot's angular velocity matches the commanded
    yaw values.
    """
    ang_vel_error = torch.square(
        commands[:, 2] - base_ang_vel[:, 2])
    return torch.exp(-ang_vel_error * tracking_sigma)


@torch.jit.script
def _reward_feet_clearance_helper(
        sensor_force_buf,
        feet_force_indices: list[int],
        get_left_foot_pos_buf,
        get_right_foot_pos_buf,
        last_feet_z,
        feet_height,
        cycle_time: float,
        progress_buf,
        dt: float,
        num_envs: int,
        device: torch.device,
        target_feet_height: float):
    """
    Calculates reward based on the clearance of the swing leg from the ground during movement.
    Encourages appropriate lift of the feet during the swing phase of the gait.
    """
    # Compute feet contact mask
    contact = sensor_force_buf[:, feet_force_indices, 1] < -5.

    # Get the z-position of the feet and compute the change in z-position
    feet_z = torch.stack([get_left_foot_pos_buf, get_right_foot_pos_buf],
                         dim=1)[:, :, 6] - 0.05
    delta_z = feet_z - last_feet_z
    feet_height += delta_z
    # self.last_feet_z = feet_z

    # Compute swing mask
    swing_mask = 1 - _get_gait_phase_helper(cycle_time, progress_buf, dt, num_envs, device)

    # feet height should be closed to target feet height at the peak
    rew_pos = torch.abs(feet_height - target_feet_height) < 0.01
    rew_pos = torch.sum(rew_pos * swing_mask, dim=1)
    feet_height *= ~contact
    return rew_pos, feet_height, feet_z


@torch.jit.script
def _reward_low_speed_helper(base_lin_vel, commands):
    """
    Rewards or penalizes the robot based on its speed relative to the commanded speed.
    This function checks if the robot is moving too slow, too fast, or at the desired speed,
    and if the movement direction matches the command.
    """
    # Calculate the absolute value of speed and command for comparison
    absolute_speed = torch.abs(base_lin_vel[:, 0])
    absolute_command = torch.abs(commands[:, 0])

    # Define speed criteria for desired range
    speed_too_low = absolute_speed < 0.5 * absolute_command
    speed_too_high = absolute_speed > 1.2 * absolute_command
    speed_desired = ~(speed_too_low | speed_too_high)

    # Check if the speed and command directions are mismatched
    sign_mismatch = torch.sign(
        base_lin_vel[:, 0]) != torch.sign(commands[:, 0])

    # Initialize reward tensor
    reward = torch.zeros_like(base_lin_vel[:, 0])

    reward = torch.where(speed_too_low, -1.0, reward)
    reward = torch.where(speed_too_high, .0, reward)
    reward = torch.where(speed_desired, 1.2, reward)
    reward = torch.where(sign_mismatch, -2.0, reward)

    # Assign rewards based on conditions
    # Speed too low
    # reward[speed_too_low] = -1.0
    # # Speed too high
    # reward[speed_too_high] = 0.
    # # Speed within desired range
    # reward[speed_desired] = 1.2
    # # Sign mismatch has the highest priority
    # reward[sign_mismatch] = -2.0
    return reward * (commands[:, 0].abs() > 0.1)


@torch.jit.script
def _reward_torques_helper(torques):
    """
    Penalizes the use of high torques in the robot's joints. Encourages efficient movement by
    minimizing the necessary force exerted by the motors.
    """
    return torch.sum(torch.square(torques), dim=1)


@torch.jit.script
def _reward_dof_vel_helper(get_dof_vel_buf):
    """
    Penalizes high velocities at the degrees of freedom (DOF) of the robot. This encourages
    smoother and more controlled movements.
    """
    return torch.sum(torch.square(get_dof_vel_buf), dim=1)


@torch.jit.script
def _reward_dof_acc_helper(last_dof_vel, get_dof_vel_buf, dt: float):
    """
    Penalizes high accelerations at the robot's degrees of freedom (DOF). This is important for
    ensuring smooth and stable motion, reducing wear on the robot's mechanical parts.
    """
    return torch.sum(torch.square((last_dof_vel - get_dof_vel_buf) / dt), dim=1)


@torch.jit.script
def _reward_collision_helper(sensor_force_buf, penalised_contact_indices: list[int]):
    """
    Penalizes collisions of the robot with the environment, specifically focusing on selected
    body parts. This encourages the robot to avoid undesired contact with objects or surfaces.
    """
    return torch.sum(1. * (torch.norm(sensor_force_buf[:, penalised_contact_indices, :],
                                      dim=-1) > 0.1), dim=1)


@torch.jit.script
def _reward_action_smoothness_helper(last_actions, act_buf, last_last_actions):
    """
    Encourages smoothness in the robot's actions by penalizing large differences between
    consecutive actions. This is important for achieving fluid motion and reducing mechanical
    stress.
    """
    term_1 = torch.sum(torch.square(last_actions - act_buf), dim=1)
    term_2 = torch.sum(torch.square(act_buf + last_last_actions - 2 *
                                    last_actions), dim=1)
    term_3 = 0.05 * torch.sum(torch.abs(act_buf), dim=1)
    return term_1 + term_2 + term_3


class H1EnvironmentHumanoidGym(EnvironmentGpu):

    def __init__(self,
                 num_envs: int,
                 device: torch.device,
                 rendering: bool = False,
                 enable_scene_query: bool = False,
                 max_episode_length_s: float = 24.,
                 gravity: v.Vec3 = v.Vec3(0, -9.81, 0),
                 timestep: float = 0.01,
                 frame_skip: int = 1,  # called cfg.control.decimation in humanoid-gym
                 spacing: float = 2.,
                 reset_noise_scale: float = 1.,
                 cycle_time: float = 0.64,
                 target_joint_pos_scale: float = 0.17,
                 command_ranges: dict = {"lin_vel_x": [-0.3, 0.6],
                                         "lin_vel_y": [-0.3, 0.3],
                                         "ang_vel_yaw": [-0.3, 0.3],
                                         "heading": [-3.14, 3.14], },
                 manual_command: bool = False,
                 manual_mode: str = 'points',
                 manual_targets: list[tuple[float, float]] = [(1., 0.), (2., 0.), (3., 0.)],
                 spline_file: str = 'spline.txt',
                 spline_cutoff: float = 2.0,
                 draw_target: bool = False,
                 target_tolerance: float = 0.5,
                 terminate_at_last_target: bool = True,
                 initial_mode: str = "grid",
                 lin_vel_obs_scale: float = 2.,
                 ang_vel_obs_scale: float = 1.,
                 dof_pos_obs_scale: float = 1.,
                 dof_vel_obs_scale: float = 0.05,
                 quat_obs_scale: float = 1.,
                 resampling_time: float = 8.,
                 frame_stack: int = 15,
                 c_frame_stack: int = 3,
                 initial_is_paused: bool = False,
                 send_interrupt: bool = False,
                 only_positive_rewards: bool = True,
                 min_height: float = 0.7,
                 fix_upper_body: bool = True,
                 # == Reward weights == #
                 # Reference motion tracking
                 joint_pos_weight: float = 1.6,
                 feet_clearance_weight: float = 1.0,
                 feet_contact_number_weight: float = 1.2,
                 # Gait
                 feet_air_time_weight: float = 1.0,
                 foot_slip_weight: float = -0.05,
                 feet_distance_weight: float = 0.2,
                 knee_distance_weight: float = 0.2,
                 # Contact
                 feet_contact_forces_weight: float = -0.01,
                 # Velocity tracking
                 tracking_lin_vel_weight: float = 1.2,
                 tracking_ang_vel_weight: float = 1.1,
                 vel_mismatch_exp_weight: float = 0.5,
                 low_speed_weight: float = 0.2,
                 track_vel_hard_weight: float = 0.5,
                 # Base position
                 default_joint_pos_weight: float = 0.5,
                 orientation_weight: float = 1.0,
                 base_height_weight: float = 0.2,
                 base_acc_weight: float = 0.2,
                 # Energy
                 action_smoothness_weight: float = -0.002,
                 torques_weight: float = -1e-5,
                 dof_vel_weight: float = -5e-4,
                 dof_acc_weight: float = -1e-7,
                 collision_weight: float = -1.0,
                 # === Auxiliary reward parameters
                 min_dist: float = 0.2,
                 max_dist: float = 0.5,
                 max_contact_force: float = 700.,
                 base_height_target: float = 1.0241,
                 tracking_sigma: float = 5,
                 target_feet_height: float = 0.04,
                 init_last_feet_z: float = 0.01,
                 # === PD controller parameters === #
                 stiffness: dict = {
                     'hip_roll': 200.0,
                     'hip_pitch': 350.0,
                     'hip_yaw': 200.0,
                     'knee': 350.0,
                     'ankle': 15,
                     'torso': 200,
                     'shoulder_pitch': 100,
                     'shoulder_roll': 100,
                     'shoulder_yaw': 100,
                     'elbow': 50,
                     },
                 damping: dict = {
                     'hip_roll': 10,
                     'hip_pitch': 10,
                     'hip_yaw': 10,
                     'knee': 10,
                     'ankle': 10,
                     'torso': 10,
                     'shoulder_pitch': 10,
                     'shoulder_roll': 10,
                     'shoulder_yaw': 10,
                     'elbow': 10,
                     },
                 action_scale: float = 1.,
                 torque_limit: float = 0.85,
                 clip_actions=18.0,
                 use_xbot: bool = False,
                 print_info: bool = False,
                 log_period: int = 60,
                 friction_scale: float = 1.,
                 max_contact_pairs_per_env: int = 128,
                 with_window: bool = True,
                 ):

        self.num_dofs = 19
        self.max_episode_length_s = max_episode_length_s
        max_episode_length = np.ceil(self.max_episode_length_s / (frame_skip * timestep))

        self.fix_upper_body = fix_upper_body
        if self.fix_upper_body:
            self.num_dofs = 10

        self.use_xbot = use_xbot
        if self.use_xbot:
            self.num_dofs = 12
            base_height_target = 0.89
            target_feet_height = 0.06
            init_last_feet_z = 0.05
            stiffness = {'leg_roll': 200.0, 'leg_pitch': 350.0, 'leg_yaw': 200.0,
                         'knee': 350.0, 'ankle': 15}
            damping = {'leg_roll': 10, 'leg_pitch': 10, 'leg_yaw':
                       10, 'knee': 10, 'ankle': 10}
            min_height = 0.6

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
            treat_warning_as_error=True,
            max_contact_pairs_per_env=max_contact_pairs_per_env,
            with_window=with_window)

        # Store configuration
        self.reset_noise_scale = reset_noise_scale
        self.cycle_time = cycle_time
        self.target_joint_pos_scale = target_joint_pos_scale
        self.command_ranges = command_ranges
        self.lin_vel_obs_scale = lin_vel_obs_scale
        self.ang_vel_obs_scale = ang_vel_obs_scale
        self.dof_pos_obs_scale = dof_pos_obs_scale
        self.dof_vel_obs_scale = dof_vel_obs_scale
        self.quat_obs_scale = quat_obs_scale
        self.resampling_time = resampling_time
        self.commands_scale = torch.tensor(
            [self.lin_vel_obs_scale, self.lin_vel_obs_scale, self.ang_vel_obs_scale], device=self.device)
        self.frame_stack = frame_stack
        self.c_frame_stack = c_frame_stack
        self.only_positive_rewards = only_positive_rewards
        self.min_height = min_height
        self.manual_command = manual_command
        self.manual_targets = torch.tensor(manual_targets, dtype=torch.float32, device=self.device)
        self.num_targets = len(self.manual_targets)
        self.manual_targets = torch.concatenate((self.manual_targets, self.manual_targets[-1:]))
        self.draw_target = draw_target
        self.target_tolerance = target_tolerance
        self.terminate_at_last_target = terminate_at_last_target
        self.initial_mode = initial_mode

        assert manual_mode in ["points", "spline"]
        self.manual_mode = manual_mode

        if self.manual_mode == "spline":
            self.spline_cutoff = spline_cutoff
            self.spline_points = []
            with open(spline_file, 'r') as f:
                for line in f:
                    x, y, z = tuple(float(num) for num in line.split())
                    self.spline_points.append(v.Vec3(x, y, z))

            print("self.spline_points: {}".format(self.spline_points))
            self.gym.create_spline(self.spline_points)

        # Reward dict
        self.d = {}

        self.d["joint_pos"] = (joint_pos_weight, self._reward_joint_pos)
        self.d["feet_clearance"] = (feet_clearance_weight, self._reward_feet_clearance)
        self.d["feet_contact_number"] = (feet_contact_number_weight,
                                         self._reward_feet_contact_number)
        self.d["feet_air_time"] = (feet_air_time_weight, self._reward_feet_air_time)
        self.d["foot_slip"] = (foot_slip_weight, self._reward_foot_slip)
        self.d["feet_distance"] = (feet_distance_weight, self._reward_feet_distance)
        self.d["knee_distance"] = (knee_distance_weight, self._reward_knee_distance)
        self.d["feet_contact_forces"] = (feet_contact_forces_weight,
                                         self._reward_feet_contact_forces)
        self.d["tracking_lin_vel"] = (tracking_lin_vel_weight, self._reward_tracking_lin_vel)
        self.d["tracking_ang_vel"] = (tracking_ang_vel_weight, self._reward_tracking_ang_vel)
        self.d["vel_mismatch_exp"] = (vel_mismatch_exp_weight, self._reward_vel_mismatch_exp)
        self.d["low_speed"] = (low_speed_weight, self._reward_low_speed)
        self.d["track_vel_hard"] = (track_vel_hard_weight, self._reward_track_vel_hard)
        self.d["default_joint_pos"] = (default_joint_pos_weight, self._reward_default_joint_pos)
        self.d["orientation"] = (orientation_weight, self._reward_orientation)
        self.d["base_height"] = (base_height_weight, self._reward_base_height)
        self.d["base_acc"] = (base_acc_weight, self._reward_base_acc)
        self.d["action_smoothness"] = (action_smoothness_weight, self._reward_action_smoothness)
        self.d["torques"] = (torques_weight, self._reward_torques)
        self.d["dof_vel"] = (dof_vel_weight, self._reward_dof_vel)
        self.d["dof_acc"] = (dof_acc_weight, self._reward_dof_acc)
        self.d["collision"] = (collision_weight, self._reward_collision)

        # Auxiliary reward parameters
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.max_contact_force = max_contact_force
        self.base_height_target = base_height_target
        self.tracking_sigma = tracking_sigma
        self.target_feet_height = target_feet_height
        self.init_last_feet_z = init_last_feet_z

        # PD controller parameters
        self.stiffness = stiffness
        self.damping = damping
        self.action_scale = action_scale
        self.torque_limit = torque_limit

        self.clip_actions = clip_actions

        # Hardcode these for now
        self.num_actions = self.num_dofs
        self.num_sensors = 12
        self.num_links = 20

        if self.use_xbot:
            self.num_sensors = 3
            self.num_links = 13

        # Number of observations
        self.num_single_obs = 0
        self.num_single_obs += 5                # command input
        self.num_single_obs += self.num_dofs    # joint positions
        self.num_single_obs += self.num_dofs    # joint velocities
        self.num_single_obs += self.num_actions  # PD controllers
        self.num_single_obs += 3                # Root angular velocity
        self.num_single_obs += 3                # Root euler angles

        # history of single observations composes full observation tensor
        self.num_obs = self.num_single_obs * self.frame_stack

        # Define observation space
        self.single_observation_space = Box(
            low=np.array([np.finfo('f').min] * self.num_obs, dtype=np.float32),
            high=np.array([np.finfo('f').max] * self.num_obs, dtype=np.float32),
            dtype=np.float32)

        # State space
        self.num_single_state = self.num_single_obs
        self.num_single_state += self.num_dofs    # Diff between DOF and ref DOF
        self.num_single_state += 3                # Root linear velocity
        self.num_single_state += 2                # Stance mask
        self.num_single_state += 2                # Contact mask

        # history of single states composes full state tensor
        self.num_states = self.num_single_state * self.c_frame_stack

        # Define state space
        self.single_state_space = Box(
            low=np.array([np.finfo('f').min] * self.num_states, dtype=np.float32),
            high=np.array([np.finfo('f').max] * self.num_states, dtype=np.float32),
            dtype=np.float32)

        # Create environment
        self.create_envs()

        # First two sensors are feet
        self.feet_force_indices = [0, 1]
        # Third sensor is torso
        self.penalised_contact_indices = range(2, 12)
        if self.use_xbot:
            self.penalised_contact_indices = [2]

        # Link indices
        if self.use_xbot:
            self.feet_link_indices = [
                self.art_def.get_link_def_index_by_name('left_ankle_roll_link'),
                self.art_def.get_link_def_index_by_name('right_ankle_roll_link')]
            self.knee_link_indices = [self.art_def.get_link_def_index_by_name('left_knee_link'),
                                      self.art_def.get_link_def_index_by_name('right_knee_link')]
        else:
            self.feet_link_indices = [self.art_def.get_link_def_index_by_name('left_ankle_link'),
                                      self.art_def.get_link_def_index_by_name('right_ankle_link')]
            self.knee_link_indices = [self.art_def.get_link_def_index_by_name('left_knee_link'),
                                      self.art_def.get_link_def_index_by_name('right_knee_link')]

        # Define action space
        low_array = []
        high_array = []
        for i in range(self.art_def.get_num_joint_dof_defs()):
            dofdef = self.art_def.get_joint_dof_def(i)
            low, high = dofdef.get_limits()
            low_array.append(low)
            high_array.append(high)
            # name = self.art_def.get_joint_dof_def_name(i)
            # print("joint {}: {}".format(i, name))
        low_array = np.array(low_array, dtype=np.float32)
        high_array = np.array(high_array, dtype=np.float32)

        self.single_action_space = Box(low=low_array, high=high_array, dtype=np.float32)

        # Store DOF data
        self.store_initial_conditions()

        # Allocate buffers
        self.allocate_buffers()

        # Initialize PD controller
        self.initialize_pd_controller()

        # Create plane
        create_plane(self.gym,
                     dynamic_friction=0.75 * friction_scale,
                     static_friction=0.75 * friction_scale)

        # Finalize gym
        self.gym.gym_finalize()

        # Print info
        self.print_info = print_info
        if self.print_info:
            self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float32,
                                                   device=self.device) for name in self.d.keys()}

            self.extras = {}

            self.log_period = log_period
            self.ep_infos = []
            self.rewbuffer = deque(maxlen=100)
            self.lenbuffer = deque(maxlen=100)
            self.cur_reward_sum = torch.zeros(
                self.num_envs, dtype=torch.float32, device=self.device
                )
            self.cur_episode_length = torch.zeros(
                self.num_envs, dtype=torch.float32, device=self.device
                )
            self.counter = 0
            self.width = 80
            self.pad = 35

    def create_envs(self):

        enable_self_collisions = True

        # fixed = True
        fixed = False

        self.env_def_handle, self.art_def, self.arti_handle = create_envs_helper(self.gym,
                                                                                 enable_motor_control=False, enable_self_collisions=enable_self_collisions,
                                                                                 fixed=fixed, use_xbot=self.use_xbot, fix_upper_body=self.fix_upper_body)

        self.gym.get_environment_def(self.env_def_handle).finalize()

        # Rotate environment space so that z-axis is up
        super().create_envs(
            self.env_def_handle, v.shortest_rotation(
                v.Vec3(
                    0, 0, 1), v.Vec3(
                    0, 1, 0)), mode=self.initial_mode)

    def store_initial_conditions(self):

        self.dof_pos_init, self.root_trans_init, self.root_vel_init = \
            store_initial_conditions_helper(self.art_def, self.device)

        # Initial root transform is defined in z-axis up
        rot = v.Quat(0, 0, 0, 1)
        pos = v.Vec3(0, 0, 1.08)
        if self.use_xbot:
            pos = v.Vec3(0, 0, 0.95)
        self.root_trans_init = v.Transform(rot, pos)

        # Default dof positions are the same as initial dof positions
        self.default_dof_pos = self.dof_pos_init

    def allocate_buffers(self):

        super().allocate_buffers()

        allocate_gpu_buffers(self)

        # State buffer
        self.state_buf = torch.zeros((self.num_envs, self.num_states), dtype=torch.float32,
                                     device=self.device)

        # Set joint forces buffer
        self.torques = torch.zeros((self.num_envs, self.num_actions), dtype=torch.float32,
                                   device=self.device)

        # Set joint forces GPU command
        set_joint_forces_command = self.env_group.create_joint_state_command(
            v.wrap_gpu_buffer(self.torques), self.arti_handle, (0, self.num_actions))

        self.gpu_set_joint_forces_command_array = self.gym.create_joint_state_command_gpu_array(
            [set_joint_forces_command])

        # Auxiliary buffers
        self.commands = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self.forward_vec = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32,
                                        device=self.device).repeat((self.num_envs, 1))
        # z-axis is up in environment frame
        self.up_vec = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32,
                                   device=self.device).repeat((self.num_envs, 1))

        self.gravity_vec = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32,
                                        device=self.device).repeat((self.num_envs, 1))
        self.p_gains = torch.zeros((self.num_envs, self.num_actions), dtype=torch.float32,
                                   device=self.device)
        self.d_gains = torch.zeros((self.num_envs, self.num_actions), dtype=torch.float32,
                                   device=self.device)
        self.feet_air_time = torch.zeros((self.num_envs, len(self.feet_force_indices)),
                                         dtype=torch.float32, device=self.device)
        self.feet_height = torch.zeros((self.num_envs, len(self.feet_link_indices)),
                                       dtype=torch.float32, device=self.device)
        self.direction = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self.forward = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)

        self.torque_limits = torch.zeros((self.num_dofs,), dtype=torch.float32, device=self.device)

        self.last_actions = torch.zeros_like(self.act_buf, dtype=torch.float32, device=self.device)
        self.last_last_actions = torch.zeros_like(self.act_buf, dtype=torch.float32,
                                                  device=self.device)
        self.last_dof_vel = torch.zeros_like(self.get_dof_vel_buf, dtype=torch.float32,
                                             device=self.device)
        self.last_root_vel = torch.zeros((self.num_envs, 6), dtype=torch.float32,
                                         device=self.device)
        self.last_contacts = torch.zeros((self.num_envs, len(self.feet_force_indices)),
                                         dtype=torch.bool, device=self.device)
        self.last_feet_z = torch.zeros_like(self.feet_height, dtype=torch.float,
                                            device=self.device)
        self.last_feet_z[:] = self.init_last_feet_z

        # Prepare quantities related to the root (called base in humanoid-gym)
        self.base_quat = self.gpu_init_root_transforms[:, 0:4].clone()
        self.base_pos = self.gpu_init_root_transforms[:, 4:7].clone()

        offset_range = [-1.0, 0.0, -1.0]
        offsets = []

        for i in range(self.num_envs):
            offsets.append([offset_range[i % len(offset_range)]])

        self.spline_offset = torch.tensor(offsets, dtype=torch.float32,
                                          device=self.device)

        self.spline_up_vec = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32,
                                          device=self.device).repeat((self.num_envs, 1))

        self.base_lin_vel = quat_rotate_inverse(self.base_quat,
                                                self.gpu_init_root_velocities[:, 3:6])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat,
                                                self.gpu_init_root_velocities[:, 0:3])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)

        # Observation and state history
        self.obs_history = deque(maxlen=self.frame_stack)
        self.critic_history = deque(maxlen=self.c_frame_stack)
        for _ in range(self.frame_stack):
            self.obs_history.append(
                torch.zeros(
                    self.num_envs,
                    self.num_single_obs,
                    dtype=torch.float32,
                    device=self.device))
        for _ in range(self.c_frame_stack):
            self.critic_history.append(
                torch.zeros(
                    self.num_envs,
                    self.num_single_state,
                    dtype=torch.float32,
                    device=self.device))

        # == Get link velocities commands == #
        link_vel_cmds = []

        # Get root local velocity
        self.root_local_vel_buf = torch.zeros((self.num_envs, 6), device=self.device,
                                              dtype=torch.float32)

        link_vel_cmds.append(self.env_group.create_link_velocity_command(v.wrap_gpu_buffer(
            self.root_local_vel_buf), self.arti_handle, (0, 1), v.FrameType.LOCAL))

        # Get feet link velocity
        self.get_left_foot_vel_buf = torch.zeros((self.num_envs, 6), dtype=torch.float32,
                                                 device=self.device)
        self.get_right_foot_vel_buf = torch.zeros((self.num_envs, 6), dtype=torch.float32,
                                                  device=self.device)

        link_vel_cmds.append(
            self.env_group.create_link_velocity_command(
                v.wrap_gpu_buffer(
                    self.get_left_foot_vel_buf),
                self.arti_handle,
                (self.feet_link_indices[0],
                 self.feet_link_indices[0] + 1)))
        link_vel_cmds.append(
            self.env_group.create_link_velocity_command(
                v.wrap_gpu_buffer(
                    self.get_right_foot_vel_buf),
                self.arti_handle,
                (self.feet_link_indices[1],
                 self.feet_link_indices[1] + 1)))

        self.get_link_vel_cmd_arr = self.gym.create_link_velocity_command_gpu_array(link_vel_cmds)

        # == Get link transform commands == #
        link_pos_cmds = []

        # Get feet link transform
        self.get_left_foot_pos_buf = torch.zeros((self.num_envs, 7), dtype=torch.float32,
                                                 device=self.device)
        self.get_right_foot_pos_buf = torch.zeros((self.num_envs, 7), dtype=torch.float32,
                                                  device=self.device)

        link_pos_cmds.append(
            self.env_group.create_link_transform_command(
                v.wrap_gpu_buffer(
                    self.get_left_foot_pos_buf),
                self.arti_handle,
                (self.feet_link_indices[0],
                 self.feet_link_indices[0] + 1)))
        link_pos_cmds.append(
            self.env_group.create_link_transform_command(
                v.wrap_gpu_buffer(
                    self.get_right_foot_pos_buf),
                self.arti_handle,
                (self.feet_link_indices[1],
                 self.feet_link_indices[1] + 1)))

        # Get knee link transform
        self.get_left_knee_pos_buf = torch.zeros((self.num_envs, 7), dtype=torch.float32,
                                                 device=self.device)
        self.get_right_knee_pos_buf = torch.zeros((self.num_envs, 7), dtype=torch.float32,
                                                  device=self.device)

        link_pos_cmds.append(
            self.env_group.create_link_transform_command(
                v.wrap_gpu_buffer(
                    self.get_left_knee_pos_buf),
                self.arti_handle,
                (self.knee_link_indices[0],
                 self.knee_link_indices[0] + 1)))
        link_pos_cmds.append(
            self.env_group.create_link_transform_command(
                v.wrap_gpu_buffer(
                    self.get_right_knee_pos_buf),
                self.arti_handle,
                (self.knee_link_indices[1],
                 self.knee_link_indices[1] + 1)))

        self.get_link_pos_cmd_arr = self.gym.create_link_transform_command_gpu_array(link_pos_cmds)

        # Save environment positions on the grid
        self.env_pos = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        for i in range(self.num_envs):
            self.env_pos[i, 0] = self.env_transforms[i].p.x
            self.env_pos[i, 1] = self.env_transforms[i].p.z

        # Store target and target_index
        self.target_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.target = self.manual_targets[self.target_index]
        self.terminal = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Spline: store closest points and gradients
        if self.manual_mode == "spline":
            self.closest_points = torch.zeros((self.num_envs, 3), dtype=torch.float32,
                                              device=self.device)
            self.gradients = torch.zeros(
                (self.num_envs, 3), dtype=torch.float32, device=self.device)
            self.last_control_point = torch.zeros((self.num_envs, 2), dtype=torch.float32,
                                                  device=self.device)
            self.last_control_point[:, 0] = self.spline_points[-1].x
            self.last_control_point[:, 1] = self.spline_points[-1].z

        if self.draw_target:
            size = 0.5
            if self.manual_mode == "points":
                color_vals = torch.linspace(0, 1, self.num_targets)
                for i in range(self.num_targets):
                    c = color_vals[i]
                    color = v.Vec3(c, 0, 1.0 - c)
                    manual_target = self.manual_targets[i]
                    transform = v.Transform(v.Quat(0, 0, 0, 1), v.Vec3(manual_target[0], 1.25,
                                                                       manual_target[1]))
                    self.cube = self.gym_render.create_user_line_cube(size, transform, color)
                    self.gym_render.register_line_shape(self.cube)
            elif self.manual_mode == "spline":
                color_vals = torch.linspace(0, 1, len(self.spline_points))
                for i in range(len(self.spline_points)):
                    c = color_vals[i]
                    color = v.Vec3(c, 0, 1.0 - c)
                    transform = v.Transform(v.Quat(0, 0, 0, 1), self.spline_points[i])
                    self.cube = self.gym_render.create_user_line_cube(size, transform, color)
                    self.gym_render.register_line_shape(self.cube)
            else:
                raise Exception("This should not be reached")

    def _fetch_data(self):
        self.gym.get_articulation_kinematic_states(self.gpu_get_kinematic_state_command_array)
        self.gym.get_link_transforms(self.get_link_pos_cmd_arr)
        self.gym.get_link_velocities(self.get_link_vel_cmd_arr)
        self.gym.get_sensor_forces(self.gpu_get_sensor_forces_command_array)

        # Conversion operation for compatibility with handle-based sensor API
        self.sensor_force_buf = torch.stack(self.force_sensor_buffers, dim=1)

    def initialize_pd_controller(self):
        with profiler.record_function("initialize_pd_controller"):
            for i in range(self.art_def.get_num_joint_dof_defs()):
                dofdef = self.art_def.get_joint_dof_def(i)
                name = self.art_def.get_joint_dof_def_name(i)

                found = False

                for dof_name in self.stiffness.keys():

                    if dof_name in name:
                        self.p_gains[:, i] = self.stiffness[dof_name]
                        self.d_gains[:, i] = self.damping[dof_name]
                        found = True

                if not found:
                    self.p_gains[:, i] = 0
                    self.d_gains[:, i] = 0
                    print(f"PD gain of joint {name} were not defined, setting them to zero")

            for i in range(self.art_def.get_num_motor_defs()):
                motor_def = self.art_def.get_motor_def(i)
                self.torque_limits[motor_def.dof_index] = motor_def.gear_ratio * self.torque_limit

            # Vlearn doesn't import effort so I am specifying this manually
            if self.use_xbot:
                efforts = {
                    "left_ankle_pitch_link_to_left_ankle_roll_link": 100,
                    "left_knee_link_to_left_ankle_pitch_link": 100,
                    "left_leg_pitch_link_to_left_knee_link": 250,
                    "left_leg_yaw_link_to_left_leg_pitch_link": 250,
                    "left_leg_roll_link_to_left_leg_yaw_link": 100,
                    "waist_roll_link_to_left_leg_roll_link": 100,
                    "right_ankle_pitch_link_to_right_ankle_roll_link": 100,
                    "right_knee_link_to_right_ankle_pitch_link": 100,
                    "right_leg_pitch_link_to_right_knee_link": 250,
                    "right_leg_yaw_link_to_right_leg_pitch_link": 250,
                    "right_leg_roll_link_to_right_leg_yaw_link": 100,
                    "waist_roll_link_to_right_leg_roll_link": 100,
                    }
                for key, val in efforts.items():
                    index = self.art_def.get_joint_dof_def_index_by_name(key)
                    self.torque_limits[index] = val * self.torque_limit

    def reset_idx(self):
        with profiler.record_function("reset_idx"):
            if torch.sum(self.reset_buf) == 0:
                return

            # self.set_dof_pos_buf[:], self.set_dof_vel_buf[:], self.feet_air_time[:], self.act_buf[:], self.progress_buf[:], \
            # self.last_last_actions[:], self.last_actions[:], self.last_dof_vel[:], self.last_root_vel[:], self.last_contacts[:], self.last_feet_z[:], \
            # self.base_quat[:], self.base_pos[:], self.base_lin_vel[:], self.base_ang_vel[:], self.projected_gravity[:], self.base_euler_xyz[:] \
            # =
            self.base_euler_xyz[:] = reset_idx_helper(
                self.gpu_init_dof_pos,
                self.set_dof_pos_buf,
                self.set_dof_vel_buf,
                self.reset_noise_scale,
                self.reset_buf,
                self.feet_air_time,
                self.act_buf,
                self.progress_buf,
                self.last_last_actions,
                self.last_actions,
                self.last_dof_vel,
                self.last_root_vel,
                self.last_contacts,
                self.last_feet_z,
                self.init_last_feet_z,
                self.base_quat,
                self.base_pos,
                self.gpu_init_root_transforms,
                self.base_lin_vel,
                self.base_ang_vel,
                self.gpu_init_root_velocities,
                self.projected_gravity,
                self.gravity_vec)

            self.gym.set_articulation_kinematic_states(self.gpu_set_kinematic_state_command_array)

            with profiler.record_function("reset_commands"):
                # Reset commands
                self._resample_commands(self.reset_buf)

                forward = quat_rotate(self.base_quat, self.forward_vec)
                if self.manual_command:
                    self.target_index[:] = torch.where(self.reset_buf, 0, self.target_index)
                    self.target[:] = self.manual_targets[self.target_index]
                    self.compute_manual_commands(forward)
                else:
                    heading = torch.atan2(forward[:, 1], forward[:, 0])
                    self.commands[:, 2] = torch.clip(
                        0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1, 1)

            reset_history_helper(self.obs_history, self.critic_history, self.reset_buf)

 # Record episode reward sums
            if self.print_info:
                self.extras["episode"] = {}
                for key in self.episode_sums.keys():
                    self.extras["episode"]["rew_" + key] = torch.sum(
                        torch.where(self.reset_buf, self.episode_sums[key], 0)) / \
                        torch.sum(self.reset_buf) / self.max_episode_length_s
                    self.episode_sums[key] = torch.where(self.reset_buf, 0, self.episode_sums[key])

    def reset(self):

        super().reset()

        self._fetch_data()

        self.obs_buf[:], self.state_buf[:] = self.compute_observation_and_state(self.act_buf)

        return {"obs": self.obs_buf.clone(), "states": self.state_buf.clone()}, {}

    # Overwrite step() for custom PD control
    def step(self, actions):
        # Apply actions
        self.pre_physics_step(actions)

        # Step simulator
        for i in range(self.frame_skip):
            self.torques[:] = self._compute_torques(self.act_buf)
            self.gym.set_joint_forces(self.gpu_set_joint_forces_command_array)

            self.gym.step()

            self.render()

            # dof pos and dof vel are used to compute torques, so I need to refresh
            self.gym.get_articulation_kinematic_states(self.gpu_get_kinematic_state_command_array)

            # print("self.get_dof_pos_buf: {}".format(self.get_dof_pos_buf))

        # Compute observations, rewards, termination and truncation status
        self.post_physics_step()

        if self.print_info:
            self.print_info_fn()

        return ({"obs": self.obs_buf.clone(), "states": self.state_buf.clone()},
                self.rew_buf.clone(), self.term_buf.clone(), self.trunc_buf.clone(), {})

    def print_info_fn(self):

        self.ep_infos.append(self.extras["episode"])
        self.cur_reward_sum += self.rew_buf
        self.cur_episode_length += 1
        new_ids = (self.term_buf > 0).nonzero(as_tuple=False)
        self.rewbuffer.extend(
            self.cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
            )
        self.lenbuffer.extend(
            self.cur_episode_length[new_ids][:, 0].cpu().numpy().tolist()
            )
        self.cur_reward_sum[new_ids] = 0
        self.cur_episode_length[new_ids] = 0

        self.counter += 1

        if self.counter % self.log_period == 0:
            ep_string = f""
            for key in self.ep_infos[0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in self.ep_infos:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                ep_string += f"""{f'Mean episode {key}:':>{self.pad}} {value:.4f}\n"""

            if len(self.rewbuffer) > 0:
                log_string = (
                    f"""{'#' * self.width}\n"""
                    f"""{''.center(self.width, ' ')}\n\n"""
                    f"""{'Mean reward:':>{self.pad}} {statistics.mean(self.rewbuffer):.2f}\n"""
                    f"""{'Mean episode length:':>{self.pad}} {statistics.mean(self.lenbuffer):.2f}\n""")
            else:
                log_string = (
                    f"""{'#' * self.width}\n"""
                    f"""{''.center(self.width, ' ')}\n\n"""
                    )

            log_string += ep_string
            print(log_string)

            self.ep_infos.clear()

    def pre_physics_step(self, actions):

        assert isinstance(actions, torch.Tensor)

        self.act_buf[:] = torch.clip(actions, - self.clip_actions, self.clip_actions)

    def post_physics_step(self):
        with profiler.record_function("post_physics_step"):
            self.progress_buf += 1

            self._fetch_data()

            # Update auxiliary data
            self.base_quat[:] = self.root_pos_buf[:, 0:4]
            self.base_pos[:] = self.root_pos_buf[:, 4:7]
            self.base_lin_vel[:] = self.root_local_vel_buf[:, 3:6]
            self.base_ang_vel[:] = self.root_local_vel_buf[:, 0:3]
            self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
            self.base_euler_xyz[:] = get_euler_xyz_tensor(self.base_quat)

            self._post_physics_step_callback()

            # Compute termination and truncation
            self.term_buf[:] = self.compute_termination()
            self.trunc_buf[:] = self.compute_truncation()

            # Reset
            self.reset_buf[:] = torch.logical_or(self.term_buf, self.trunc_buf)
            self.reset_idx()

            # Compute rewards
            self.rew_buf[:] = self.compute_reward(self.act_buf)

            # Compute observation and privileged observations
            self.obs_buf[:], self.state_buf[:] = self.compute_observation_and_state(self.act_buf)

            # Update last buffers
            self.last_last_actions[:] = self.last_actions
            self.last_actions[:] = self.act_buf
            self.last_dof_vel[:] = self.get_dof_vel_buf
            self.last_root_vel[:] = self.root_vel_buf
            # self.last_contacts is updated in self._reward_feet_air_time()
            # self.last_feet_z is updated in self._reward_feet_clearance()

    def compute_observation_and_state(self, actions):
        with profiler.record_function("compute_observation_and_state"):
            phase = self._get_phase()
            # self.compute_ref_state()
            self.ref_dof_pos, self.ref_action = compute_ref_state_helper(self._get_phase(
                ), self.get_dof_pos_buf, self.use_xbot, self.target_joint_pos_scale, self.fix_upper_body)

            # sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
            # cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)

            # stance_mask = self._get_gait_phase()
            # contact_mask = self.sensor_force_buf[:, self.feet_force_indices, 1] < -5.

            # self.command_input = torch.cat(
            #     (sin_pos, cos_pos, self.commands[:, :3] * self.commands_scale), dim=1)

            # q = (self.get_dof_pos_buf - self.default_dof_pos) * self.dof_pos_obs_scale
            # dq = self.get_dof_vel_buf * self.dof_vel_obs_scale

            # diff = self.get_dof_pos_buf - self.ref_dof_pos

            # obs_now = torch.cat((
            #     self.command_input,
            #     q,
            #     dq,
            #     actions,
            #     self.base_ang_vel * self.ang_vel_obs_scale,
            #     self.base_euler_xyz * self.quat_obs_scale,
            #     ), dim=-1)

            # privileged_obs_now = torch.cat((
            #     obs_now,                                    # 68
            #     diff,                                       # 19
            #     self.base_lin_vel * self.lin_vel_obs_scale, # 3
            #     # self.rand_push_force[:, :2],  # Leave pushing out for now
            #     # self.rand_push_torque,        # Leave pushing out for now
            #     # self.env_frictions,           # Leave domain randomisation out for now
            #     # self.body_mass / 30,          # Leave domain randomisation out for now
            #     stance_mask,                                # 2
            #     contact_mask,                               # 2
            #     ), dim=-1)

            self.command_input, obs_now, privileged_obs_now = compute_obs_helper(phase, self._get_gait_phase(), self.sensor_force_buf, self.feet_force_indices, self.commands, self.commands_scale,
                                                                                 self.get_dof_pos_buf, self.default_dof_pos, self.dof_pos_obs_scale, self.get_dof_vel_buf, self.dof_vel_obs_scale,
                                                                                 self.ref_dof_pos, self.base_ang_vel, self.ang_vel_obs_scale, self.base_euler_xyz, self.quat_obs_scale,
                                                                                 self.base_lin_vel, self.lin_vel_obs_scale, actions)

            self.obs_history.append(obs_now)
            self.critic_history.append(privileged_obs_now)

            obs_buf_all = torch.cat([self.obs_history[i] for i in range(self.obs_history.maxlen)],
                                    dim=1)
            privileged_obs_buf_all = torch.cat([self.critic_history[i] for i in
                                                range(self.critic_history.maxlen)], dim=1)

            return obs_buf_all, privileged_obs_buf_all

    def compute_termination(self):
        with profiler.record_function("compute_termination"):
            term = self.root_pos_buf[:, 6] < self.min_height

            if self.manual_command and self.terminate_at_last_target:
                term = torch.logical_or(self.terminal, term)

            return term

    def compute_truncation(self):
        return self.progress_buf >= self.max_episode_length

    def compute_reward(self, actions):
        with profiler.record_function("rewards"):
            reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

            for name, (weight, func) in self.d.items():
                rew = weight * func()
                reward += rew
                if self.print_info:
                    self.episode_sums[name] += rew

            if self.only_positive_rewards:
                reward = torch.clip(reward, min=0)

            return reward

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured
            terrain heights and randomly push robots
        """

        with profiler.record_function("_post_physics_step_callback"):

            if np.isfinite(self.resampling_time):
                reset_mask = self.progress_buf % int(self.resampling_time / self.dt) == 0
                self._resample_commands(reset_mask)

            forward = quat_rotate(self.base_quat, self.forward_vec)
            if self.manual_command:
                self.compute_manual_commands(forward)
            else:
                heading = torch.atan2(forward[:, 1], forward[:, 0])
                self.commands[:, 2] = torch.clip(
                    0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1, 1)

    def _get_phase(self):
        cycle_time = self.cycle_time
        phase = self.progress_buf * self.dt / cycle_time
        return phase

    def _get_gait_phase(self):
        return _get_gait_phase_helper(
            self.cycle_time,
            self.progress_buf,
            self.dt,
            self.num_envs,
            self.device)

    def compute_manual_commands(self, forward):
        with profiler.record_function("compute_manual_commands"):
            if self.manual_mode == "points":
                robot_pos = torch.concatenate(
                    (self.base_pos[:, 0:1], -self.base_pos[:, 1:2]), dim=1) + self.env_pos
                direction = self.target - robot_pos

                hit_target = torch.logical_and(torch.norm(direction, dim=1) < self.target_tolerance,
                                               self.target_index < self.num_targets)

                while torch.any(hit_target):

                    self.target_index[:] = torch.where(
                        hit_target, self.target_index + 1, self.target_index)

                    self.target[:] = self.manual_targets[self.target_index]

                    direction = self.target - robot_pos

                    hit_target = torch.logical_and(
                        torch.norm(
                            direction,
                            dim=1) < self.target_tolerance,
                        self.target_index < self.num_targets)

                # Sanity check
                assert torch.all(self.target_index <= self.num_targets)
            elif self.manual_mode == "spline":
                # Convert base_pos to world space (self.base_pos is in environment space)
                robot_pos = torch.concatenate((self.base_pos[:, 0:1] +
                                               self.env_pos[:, 0:1], self.base_pos[:, 2:3], -
                                               self.base_pos[:, 1:2] +
                                               self.env_pos[:, 1:2]), dim=1)

                self.gym.find_closest_points(v.wrap_gpu_buffer(robot_pos),
                                             v.wrap_gpu_buffer(self.closest_points),
                                             v.wrap_gpu_buffer(self.gradients), self.num_envs)

                # Project to x-z space
                robot_pos = robot_pos[:, [0, 2]]
                closest_points = self.closest_points
                closest_points = self.closest_points[:, [0, 2]]
                gradients = self.gradients[:, [0, 2]]
                gradients3 = self.gradients[:, [0, 1, 2]]
                gradients_norm = torch.norm(gradients, dim=1).view(-1, 1)

                tangents = gradients3.cross(self.spline_up_vec)
                tangents = tangents[:, [0, 2]]
                closest_points += tangents * self.spline_offset

                # Delta points towards closest point on the spline
                delta = closest_points - robot_pos
                delta_norm = torch.norm(delta, dim=1).view(-1, 1)

                # Direction is linear interpolation of gradient and delta when robot is within cutoff
                # distance from spline
                # Direction is delta when robot it outside cutoff distance from spline
                direction = torch.where(delta_norm > self.spline_cutoff,
                                        delta / delta_norm,
                                        gradients / gradients_norm * (self.spline_cutoff - delta_norm) /
                                        self.spline_cutoff + delta / self.spline_cutoff)

                # print("delta_norm > self.spline_cutoff,: {}".format(delta_norm > self.spline_cutoff,))
                # print("robot_pos: {}".format(robot_pos))
                # print("closest_points: {}".format(closest_points))
                # print("gradients: {}".format(gradients))
                # print("delta: {}".format(delta))
                # print("direction: {}".format(direction))
                # print("======")

            # Use dot and cross products to determine angular velocity command
            f, d = self.forward, self.direction
            d[:, 0:2] = direction
            f[:, 0] = forward[:, 0]
            f[:, 1] = - forward[:, 1]

            cos_heading = torch.sum(f * d, dim=1) / (torch.norm(f, dim=1) * torch.norm(d, dim=1))
            cos_heading = torch.clamp(cos_heading, -1, 1)
            heading = torch.acos(cos_heading)

            sign = - torch.sign((torch.cross(f, d, dim=1) * self.up_vec)[:, 2])

            ang_vel_cmd = 2 * heading * sign

            self.commands[:, 2] = torch.clip(0.5 * ang_vel_cmd, -1, 1)
            # self.commands[:, 0] = torch.where(torch.abs(self.commands[:, 2]) < 1, 1.0, 0)
            self.commands[:, 0] = torch.where(torch.abs(self.commands[:, 2]) < 1, 0.6, 0)
            self.commands[:, 1] = 0.0

            if self.manual_mode == "points":
                self.terminal[:] = self.target_index == self.num_targets
            elif self.manual_mode == "spline":
                dist_to_last_control_point = torch.norm(self.last_control_point - robot_pos, dim=1)
                self.terminal[:] = dist_to_last_control_point < self.target_tolerance
            self.commands[:, 0:3] = torch.where(self.terminal.view(-1, 1), 0, self.commands[:, 0:3])

    def _resample_commands(self, reset_mask):
        with profiler.record_function("_resample_commands"):
            if self.manual_command:
                return

            _resample_commands_helper(
                reset_mask,
                self.commands,
                self.command_ranges,
                self.num_envs,
                self.device)

            # self.commands[:, 0] = torch.where(reset_mask,
            #         torch_rand_float(
            #             self.command_ranges["lin_vel_x"][0],
            #             self.command_ranges["lin_vel_x"][1],
            #             (self.num_envs, 1),
            #             device=str(self.device)).squeeze(1),
            #         self.commands[:, 0])

            # self.commands[:, 1] = torch.where(reset_mask,
            #         torch_rand_float(
            #             self.command_ranges["lin_vel_y"][0],
            #             self.command_ranges["lin_vel_y"][1],
            #             (self.num_envs, 1),
            #             device=str(self.device)).squeeze(1),
            #         self.commands[:, 1])

            # self.commands[:, 3] = torch.where(reset_mask,
            #         torch_rand_float(
            #             self.command_ranges["heading"][0],
            #             self.command_ranges["heading"][1],
            #             (self.num_envs, 1),
            #             device=str(self.device)).squeeze(1),
            #         self.commands[:, 3])

            # self.commands[:, :2] *= (torch.norm(self.commands[:, :2], dim=1) > 0.2).unsqueeze(1)

    def _compute_torques(self, actions):
        actions_scaled = actions * self.action_scale
        torques = self.p_gains * (actions_scaled + self.default_dof_pos - self.get_dof_pos_buf) - \
            self.d_gains * self.get_dof_vel_buf
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

## ======================== REWARDS ======================== ##

    def _reward_joint_pos(self):
        """
        Calculates the reward based on the difference between the current joint positions and the
        target joint positions.
        """
        self.dof_pos = self.get_dof_pos_buf
        return _reward_joint_pos_helper(self.get_dof_pos_buf, self.ref_dof_pos)

    def _reward_feet_distance(self):
        """
        Calculates the reward based on the distance between the feet. Penalize feet get close to
        each other or too far away.
        """

        return _reward_feet_distance_helper(
            self.get_left_foot_pos_buf,
            self.get_right_foot_pos_buf,
            self.min_dist,
            self.max_dist)

    def _reward_knee_distance(self):
        """
        Calculates the reward based on the distance between the knee of the humanoid.
        """

        return _reward_knee_distance_helper(
            self.get_left_foot_pos_buf,
            self.get_right_foot_pos_buf,
            self.min_dist,
            self.max_dist)

    def _reward_foot_slip(self):
        """
        Calculates the reward for minimizing foot slip. The reward is based on the contact forces
        and the speed of the feet. A contact threshold is used to determine if the foot is in
        contact with the ground. The speed of the foot is calculated and scaled by the contact
        condition.
        """
        return _reward_foot_slip_helper(
            self.sensor_force_buf,
            self.feet_force_indices,
            self.get_left_foot_pos_buf,
            self.get_right_foot_pos_buf)

    def _reward_feet_air_time(self):
        """
        Calculates the reward for feet air time, promoting longer steps. This is achieved by
        checking the first contact with the ground after being in the air. The air time is
        limited to a maximum value for reward calculation.
        """
        result, self.contact_filt, self.last_contacts, self.feet_air_time = _reward_feet_air_time_helper(
            self.sensor_force_buf, self.feet_force_indices, self.cycle_time, self.progress_buf, self.dt, self.num_envs, self.device, self.last_contacts, self.feet_air_time)
        return result

    def _reward_feet_contact_number(self):
        """
        Calculates a reward based on the number of feet contacts aligning with the gait phase.
        Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
        """
        return _reward_feet_contact_number_helper(
            self.sensor_force_buf,
            self.feet_force_indices,
            self.cycle_time,
            self.progress_buf,
            self.dt,
            self.num_envs,
            self.device)

    def _reward_orientation(self):
        """
        Calculates the reward for maintaining a flat base orientation. It penalizes deviation from
        the desired base orientation using the base euler angles and the projected gravity vector.
        """
        return _reward_orientation_helper(self.base_euler_xyz, self.projected_gravity)

    def _reward_feet_contact_forces(self):
        """
        Calculates the reward for keeping contact forces within a specified range. Penalizes
        high contact forces on the feet.
        """
        return _reward_feet_contact_forces_helper(
            self.sensor_force_buf,
            self.feet_force_indices,
            self.max_contact_force)

    def _reward_default_joint_pos(self):
        """
        Calculates the reward for keeping joint positions close to default positions, with a focus
        on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main
        penalty.
        """
        return _reward_default_joint_pos_helper(
            self.get_dof_pos_buf,
            self.default_dof_pos,
            self.use_xbot,
            self.fix_upper_body)

    def _reward_base_height(self):
        """
        Calculates the reward based on the robot's base height. Penalizes deviation from a target
        base height. The reward is computed based on the height difference between the robot's base
        and the average height of its feet when they are in contact with the ground.
        """

        return _reward_base_height_helper(
            self.cycle_time,
            self.progress_buf,
            self.dt,
            self.num_envs,
            self.device,
            self.get_left_foot_pos_buf,
            self.get_right_foot_pos_buf,
            self.root_pos_buf,
            self.base_height_target)

    def _reward_base_acc(self):
        """
        Computes the reward based on the base's acceleration. Penalizes high accelerations of the
        robot's base, encouraging smoother motion.
        """
        return _reward_base_acc_helper(self.last_root_vel, self.root_vel_buf)

    def _reward_vel_mismatch_exp(self):
        """
        Computes a reward based on the mismatch in the robot's linear and angular velocities.
        Encourages the robot to maintain a stable velocity by penalizing large deviations.
        """
        return _reward_vel_mismatch_exp_helper(self.base_lin_vel, self.base_ang_vel)

    def _reward_track_vel_hard(self):
        """
        Calculates a reward for accurately tracking both linear and angular velocity commands.
        Penalizes deviations from specified linear and angular velocity targets.
        """
        return _reward_track_vel_hard_helper(self.commands, self.base_lin_vel, self.base_ang_vel)

    def _reward_tracking_lin_vel(self):
        """
        Tracks linear velocity commands along the xy axes.
        Calculates a reward based on how closely the robot's linear velocity matches the commanded
        values.
        """
        return _reward_tracking_lin_vel_helper(
            self.commands, self.base_lin_vel, self.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        """
        Tracks angular velocity commands for yaw rotation.
        Computes a reward based on how closely the robot's angular velocity matches the commanded
        yaw values.
        """
        return _reward_tracking_ang_vel_helper(
            self.commands, self.base_ang_vel, self.tracking_sigma)

    def _reward_feet_clearance(self):
        """
        Calculates reward based on the clearance of the swing leg from the ground during movement.
        Encourages appropriate lift of the feet during the swing phase of the gait.
        """

        rew, self.feet_height, self.last_feet_z = _reward_feet_clearance_helper(self.sensor_force_buf, self.feet_force_indices, self.get_left_foot_pos_buf, self.get_right_foot_pos_buf, self.last_feet_z,
                                                                                self.feet_height, self.cycle_time, self.progress_buf, self.dt, self.num_envs, self.device, self.target_feet_height)
        return rew

    def _reward_low_speed(self):
        """
        Rewards or penalizes the robot based on its speed relative to the commanded speed.
        This function checks if the robot is moving too slow, too fast, or at the desired speed,
        and if the movement direction matches the command.
        """

        return _reward_low_speed_helper(self.base_lin_vel, self.commands)

    def _reward_torques(self):
        """
        Penalizes the use of high torques in the robot's joints. Encourages efficient movement by
        minimizing the necessary force exerted by the motors.
        """
        return _reward_torques_helper(self.torques)

    def _reward_dof_vel(self):
        """
        Penalizes high velocities at the degrees of freedom (DOF) of the robot. This encourages
        smoother and more controlled movements.
        """
        return _reward_dof_vel_helper(self.get_dof_vel_buf)

    def _reward_dof_acc(self):
        """
        Penalizes high accelerations at the robot's degrees of freedom (DOF). This is important for
        ensuring smooth and stable motion, reducing wear on the robot's mechanical parts.
        """
        return _reward_dof_acc_helper(self.last_dof_vel, self.get_dof_vel_buf, self.dt)

    def _reward_collision(self):
        """
        Penalizes collisions of the robot with the environment, specifically focusing on selected
        body parts. This encourages the robot to avoid undesired contact with objects or surfaces.
        """
        return _reward_collision_helper(self.sensor_force_buf, self.penalised_contact_indices)

    def _reward_action_smoothness(self):
        """
        Encourages smoothness in the robot's actions by penalizing large differences between
        consecutive actions. This is important for achieving fluid motion and reducing mechanical
        stress.
        """
        return _reward_action_smoothness_helper(
            self.last_actions, self.act_buf, self.last_last_actions)


if __name__ == "__main__":
    from time import sleep
    from sys import argv
    from vlearn.utils import get_VL_VISUAL_TESTS

    rendering = True
    with_window = get_VL_VISUAL_TESTS()

    num_iter = np.inf
    if len(argv) >= 2:
        num_iter = int(argv[1])

    mode = "none"
    # mode = "joint_monkey"
    # mode = "pd"

    if len(argv) >= 3:
        mode = argv[2]

    assert mode in ["pd", "joint_monkey", "none"]

    num_envs = 1
    max_episode_length_s = 24
    initial_is_paused = False
    timestep = 0.01667
    frame_skip = 1
    # timestep = 0.001
    # frame_skip = 10
    use_xbot = True
    use_xbot = False
    fix_upper_body = True
    # fix_upper_body = False

    device = torch.device("cuda:0")

    envs = H1EnvironmentHumanoidGym(
        num_envs,
        device,
        rendering=rendering,
        max_episode_length_s=max_episode_length_s,
        enable_scene_query=True,
        initial_is_paused=initial_is_paused,
        timestep=timestep,
        frame_skip=frame_skip,
        use_xbot=use_xbot,
        fix_upper_body=fix_upper_body,
        with_window=with_window)

    obs, _ = envs.reset()

    gym = v.get_gym()
    render = gym.get_render()

    # Reset interface
    reset_box = v.UserCheckbox("Reset", False)
    if render is not None:
        render.register_menu_item(reset_box)

    # Set PD targets
    if mode == "pd":
        sliders = []

        for i in range(envs.art_def.get_num_joint_dof_defs()):
            name = envs.art_def.get_joint_dof_def_name(i)

            low = -envs.clip_actions
            high = envs.clip_actions

            sliders.append(v.UserSlider(name, low, high, 0))

            if render is not None:
                render.register_menu_item(sliders[-1])

        def control_by_menu():

            if reset_box.get_value():
                obs, _ = envs.reset()
                print_obs(envs)

            return torch.tensor([slider.get_value() for slider in sliders])

        control_fn = control_by_menu

    # Joint monkey
    elif mode == "joint_monkey":
        joint_pos = torch.zeros((envs.num_envs, envs.num_dofs), dtype=torch.float32, device=device)
        joint_vel = torch.zeros((envs.num_envs, envs.num_dofs), dtype=torch.float32, device=device)
        root_pos = torch.zeros((envs.num_envs, 7), dtype=torch.float32, device=device)
        root_vel = torch.zeros((envs.num_envs, 6), dtype=torch.float32, device=device)

        rot = v.Quat(0, 0, 0, 1)
        pos = v.Vec3(0, 0, 1.4)
        root_pos[:, 0] = rot.x
        root_pos[:, 1] = rot.y
        root_pos[:, 2] = rot.z
        root_pos[:, 3] = rot.w
        root_pos[:, 4] = pos.x
        root_pos[:, 5] = pos.y
        root_pos[:, 6] = pos.z

        set_kine_cmd = envs.env_group.create_articulation_kinematic_state_command(
            v.wrap_gpu_buffer(joint_pos), v.wrap_gpu_buffer(joint_vel),
            v.wrap_gpu_buffer(root_pos), v.wrap_gpu_buffer(root_vel), envs.arti_handle,
            link_index_range=(0, 1))

        cmd_array = gym.create_articulation_kinematic_state_command_gpu_array([set_kine_cmd])

        def joint_monkey_pre_physics_step(envs, actions):

            assert isinstance(actions, torch.Tensor)

            offset = 0

            root_pos[:, 4:7] = actions[:, offset:offset + 3]
            offset += 3

            for i in range(envs.num_envs):
                rpy = actions[i, offset:offset + 3]
                quat = v.quat_from_rpy(v.Vec3(rpy[0], rpy[1], rpy[2]))
                quat = rot * quat
                root_pos[i, 0] = quat.x
                root_pos[i, 1] = quat.y
                root_pos[i, 2] = quat.z
                root_pos[i, 3] = quat.w
            offset += 3

            root_vel[:, :] = actions[:, offset:offset + 6]
            offset += 6

            joint_pos[:, :] = actions[:, offset:offset + envs.num_dofs]
            joint_pos[:, :] = envs.ref_dof_pos  # uncomment to impose reference gait
            # print("joint_pos: {}".format(joint_pos))
            offset += envs.num_dofs

            joint_vel[:, :] = actions[:, offset:offset + envs.num_dofs]
            offset += envs.num_dofs

            envs.act_buf[:] = actions[:, offset:]

            envs.gym.set_articulation_kinematic_states(cmd_array)

            envs.gym.get_articulation_kinematic_states(envs.gpu_get_kinematic_state_command_array)
            envs.gym.get_link_velocities(envs.get_link_vel_cmd_arr)
            envs.gym.get_sensor_forces(envs.gpu_get_sensor_forces_command_array)

            # Update base quaternion
            envs.base_quat[:] = envs.root_pos_buf[:, 0:4]
            envs.base_pos[:] = envs.root_pos_buf[:, 4:7]
            envs.base_lin_vel[:] = envs.root_local_vel_buf[:, 3:6]
            envs.base_ang_vel[:] = envs.root_local_vel_buf[:, 0:3]
            envs.base_euler_xyz[:] = get_euler_xyz_tensor(envs.base_quat)

            envs.pre_step_obs, envs.pre_step_state = envs.compute_observation_and_state(
                envs.act_buf)

            # pre step reward
            # print("envs._reward_joint_pos(): {}".format(envs._reward_joint_pos()))

        H1EnvironmentHumanoidGym.pre_physics_step = joint_monkey_pre_physics_step
        # H1EnvironmentHumanoidGym.reset_idx = lambda self: None

        sliders = []

        # Set root position
        sliders.append(v.UserSlider("root x", -5, 5, 0))
        sliders.append(v.UserSlider("root y", -5, 5, 0))
        sliders.append(v.UserSlider("root z", -5, 5, 1.046))

        # Set root orientation
        sliders.append(v.UserSlider("root roll", -torch.pi, torch.pi, 0))
        sliders.append(v.UserSlider("root pitch", -torch.pi, torch.pi, 0))
        sliders.append(v.UserSlider("root yaw", -torch.pi, torch.pi, 0))

        # Set root angular velocity
        sliders.append(v.UserSlider("root angular vel x", -5, 5, 0))
        sliders.append(v.UserSlider("root angular vel y", -5, 5, 0))
        sliders.append(v.UserSlider("root angular vel z", -5, 5, 0))

        # Set root linear velocity
        sliders.append(v.UserSlider("root linear vel x", -5, 5, 0))
        sliders.append(v.UserSlider("root linear vel y", -5, 5, 0))
        sliders.append(v.UserSlider("root linear vel z", -5, 5, 0))

        # Set joint position
        for i in range(envs.art_def.get_num_joint_dof_defs()):

            dofdef = envs.art_def.get_joint_dof_def(i)
            name = envs.art_def.get_joint_dof_def_name(i)

            low, high = dofdef.get_limits()
            init = np.clip(0, low, high)

            sliders.append(v.UserSlider("dof_pos." + name, low, high, init))

        # Set joint velocity
        for i in range(envs.art_def.get_num_joint_dof_defs()):

            dofdef = envs.art_def.get_joint_dof_def(i)
            name = envs.art_def.get_joint_dof_def_name(i)

            sliders.append(v.UserSlider("dof_vel." + name, -5, 5, 0))

        # Set joint target position
        for i in range(envs.art_def.get_num_joint_dof_defs()):

            dofdef = envs.art_def.get_joint_dof_def(i)
            name = envs.art_def.get_joint_dof_def_name(i)

            low, high = -18, 18
            init = 0

            sliders.append(v.UserSlider("dof_target." + name, low, high, init))

        for slider in sliders:
            if render is not None:
                render.register_menu_item(slider)

        def control_by_menu():

            if reset_box.get_value():
                obs, _ = envs.reset()
                print_obs(envs)

            return torch.tile(torch.tensor([slider.get_value()
                              for slider in sliders]), (envs.num_envs, 1))

        control_fn = control_by_menu

    elif mode == "none":

        def control_by_menu():

            if reset_box.get_value():
                obs, _ = envs.reset()
                print_obs(envs)

            return torch.zeros((envs.num_envs, envs.num_actions))

        control_fn = control_by_menu

    # Print humanoid observations
    def print_obs(envs):
        obs = envs.obs_buf.view((envs.num_envs, envs.frame_stack, -1))
        # print("obs[0,-1,0:5]: {}".format(obs[0,-1,0:5]))     # command input
        # print("obs[0,-1,5:24]: {}".format(obs[0,-1,5:24]))   # joint positions
        # print("obs[0,-1,24:43]: {}".format(obs[0,-1,24:43])) # joint velocities
        # print("obs[0,-1,43:62]: {}".format(obs[0,-1,43:62])) # motor forces
        # print("obs[0,-1,62:65]: {}".format(obs[0,-1,62:65]))  # root ang vel
        # print("obs[0,-1,65:68]: {}".format(obs[0,-1,65:68]))  # root euler angles

        try:
            pre_step_obs = envs.pre_step_obs.view(
                (envs.num_envs, envs.frame_stack, -1))
            # print("pre_step_obs[0, -1, 24:43]: {}".format(
            #    pre_step_obs[0, -1, 24:43])) # joint velocities
        except BaseException:
            pass

        state = envs.state_buf.view((envs.num_envs, envs.c_frame_stack, -1))
        # print("state[0,-1,0:5]: {}".format(state[0,-1,0:5]))     # command input
        # print("state[0,-1,5:24]: {}".format(state[0,-1,5:24]))   # joint positions
        # print("state[0,-1,24:43]: {}".format(state[0,-1,24:43])) # joint velocities
        # print("state[0,-1,43:62]: {}".format(state[0,-1,43:62])) # motor forces
        # print("state[0,-1,62:65]: {}".format(state[0,-1,62:65]))  # root ang vel
        # print("state[0,-1,65:68]: {}".format(state[0,-1,65:68]))  # root euler angles
        # print("state[0,-1,68:87]: {}".format(state[0,-1,68:87]))  # diff
        # print("state[0,-1,87:90]: {}".format(state[0,-1,87:90]))    # root lin vel
        # print("state[0,-1,90:92]: {}".format(state[0,-1,90:92]))    # stance mask
        # print("state[0,-1,92:94]: {}".format(state[0,-1,92:94]))    # contact mask

        # envs.gym.get_link_transforms(envs.get_foot_pos_cmd_arr)
        # left_foot_z = envs.get_left_foot_pos_buf[0, 6]
        # right_foot_z = envs.get_right_foot_pos_buf[0, 6]
        # phase = envs._get_phase()
        # print("left_foot_z, right_foot_z, phase: {:5.4f}, {:5.4f}, {:5.4f}".format(left_foot_z,
        #     right_foot_z, phase.item()))

    print_obs(envs)

    # Environment loop
    if render is not None:
        render.capped_step = True
    finished = False
    idx = 0
    while not finished:

        actions = control_fn()

        # print("action: {}".format(action))

        obs, reward, terminated, truncated, info = envs.step(actions)

        print_obs(envs)

        # print("reward: {}".format(reward))
        # print("envs.episode_sums: {}".format(envs.episode_sums))
        # print("envs.extras: {}".format(envs.extras))

        # print(obs, reward, terminated, truncated, info)

        finished = envs.render_finished

        # print("===================== STEP DONE =====================")

        idx += 1
        if idx >= num_iter:
            finished = True
