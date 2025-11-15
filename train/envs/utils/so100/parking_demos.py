import os
import sys
import math

import torch
import vlearn as v
import numpy as np
import h5py
from math import ceil
import matplotlib.pyplot as plt

import torchvision.transforms.v2 as v2
from typing import Tuple, Dict, List
from vlearn.spaces import Box

from vlearn.torch_utils.torch_jit_utils import v_quat_from_rpy, v_rpy_from_quat

# from lerobot.find_cameras import save_image
from torchvision.utils import save_image

current_dir = os.path.dirname(os.path.abspath(__file__))
train_root = os.path.abspath(os.path.join(current_dir, "../../../"))
if train_root not in sys.path:
    sys.path.insert(0, train_root)

from envs.common import create_plane
from envs.so100_parking import SO100Parking
from vlearn.rl_algos.sac.vsim_training_data import DatasetWriterHDF5

assert torch.cuda.is_available()
device = torch.device("cuda:0")


def normalize(vec):
    norm = torch.linalg.vector_norm(vec, dim=-1)
    return vec / norm.unsqueeze(-1)

def demo_reach(
        state,
        ik_pose,
        curr_ee_pose,
        car_start_pos,):

    curr_ee_pos = curr_ee_pose[:, 4:7]
    curr_ee_quat = curr_ee_pose[:, 0:4]
    target_pos = ik_pose[:, 4:7]
    target_quat = ik_pose[:, 0:4]
    # Reach offset moves just behind the box before pushing
    approach_offset = torch.tensor([0.0, 0.02, 0.01], dtype=torch.float32, device=device)
    car_start_pos_offset = car_start_pos + approach_offset

    ###################
    # State 0: Reaching
    ###################

    mask_reach = (state == 0)

    # Position error
    pos_err = torch.linalg.norm(curr_ee_pos - target_pos, dim=-1)
    pos_tol = 0.005  # 5mm tolerance
    mask_pos = pos_err < pos_tol

    # Orientation error
    # Compute angular difference between quaternions
    dot = torch.abs(torch.sum(curr_ee_quat * target_quat, dim=-1)) # Get quat diff
    dot = torch.clamp(dot, -1.0, 1.0)
    ang_err = 2 * torch.acos(dot)
    ang_tol = 0.1  # about 6 degrees tolerance
    mask_rot = ang_err < ang_tol

    # If we've reached the current target, update the target
    mask_update_target = mask_pos & mask_rot

    # Direction of travel
    direction = car_start_pos_offset - curr_ee_pos
    direction_norm = torch.linalg.norm(direction, dim=-1, keepdim=True) + 1e-8
    direction_unit = direction / direction_norm

    # How fast we want to move along the direction of travel
    step_size = 0.004
    updated_target = curr_ee_pos + direction_unit * step_size

    # New target with updated target
    target_pos = torch.where(mask_update_target.unsqueeze(-1) & mask_reach.unsqueeze(-1), updated_target, target_pos)

    # Move to the object
    ik_pose[:, 4:7] = torch.where(mask_reach.unsqueeze(-1), target_pos, ik_pose[:, 4:7])

    # check for overall reach
    err = car_start_pos_offset - curr_ee_pos
    pos_err = torch.linalg.vector_norm(err, dim=-1)

    # Check for specifically 
    z = torch.zeros_like(car_start_pos_offset)
    z = z + torch.tensor([0.0, 0.0, 1.0], device=ik_pose.device, dtype=ik_pose.dtype)
    z_diff = err * z
    reached_height = torch.linalg.vector_norm(z_diff, dim=-1) < 0.005 # error on height keeps us within box bounds
    reached = pos_err < 0.02 # Box origin is 1.5cm embedded into object, so 2cm is arriving at object
    state[:] = torch.where(mask_reach & reached & reached_height, 1, state)


def demo_push(state, 
              done_pose,
              ik_pose, 
              curr_ee_pose, 
              car_start_pos, 
              goal_pos,
              car_pos,):

    demo_reach(
        state,
        ik_pose,
        curr_ee_pose,
        car_start_pos,)
    
    # print("[State]:", torch.sum(state)/len(state))
    # print("[State]:", int(state[0]))

    curr_ee_pos = curr_ee_pose[:, 4:7]
    curr_ee_quat = curr_ee_pose[:, 0:4]
    target_pos = ik_pose[:, 4:7]
    target_quat = ik_pose[:, 0:4]
    # Gripper often hits the table when pushing, need to offset the height
    pushing_offset = torch.tensor([0.0, 0.0, -0.03], dtype=torch.float32, device=device)
    goal_pos_offset = goal_pos + pushing_offset
    car_pos_offset = car_pos + pushing_offset
    
    ##################
    # State 1: Parking
    ##################

    # establish state for pushing
    mask_push = (state == 1)

    # Position error
    pos_err = torch.linalg.norm(curr_ee_pos - target_pos, dim=-1)
    pos_tol = 0.004  # 5mm tolerance
    mask_pos = pos_err < pos_tol

    # Orientation error
    # Compute angular difference between quaternions
    dot = torch.abs(torch.sum(curr_ee_quat * target_quat, dim=-1)) # Get quat diff
    dot = torch.clamp(dot, -1.0, 1.0)
    ang_err = 2 * torch.acos(dot)
    ang_tol = 0.05  # about 3 degrees tolerance
    mask_rot = ang_err < ang_tol
    # print("ang_err", ang_err)

    # --- If we've reached the current target, update the target ---
    mask_update_target = mask_pos & mask_rot

    # Establish our direction of push
    push_vec = goal_pos - curr_ee_pos

    # Direction of travel
    direction = goal_pos_offset - curr_ee_pos
    direction_norm = torch.linalg.norm(direction, dim=-1, keepdim=True) + 1e-8
    direction_unit = direction / direction_norm

    # How fast we want to move along the direction of travel
    step_size = 0.01
    updated_target = curr_ee_pos + direction_unit * step_size

    # New target with updated target
    target_pos = torch.where(mask_update_target.unsqueeze(-1) & mask_push.unsqueeze(-1), updated_target, target_pos)

    ik_pose[:, 4:7] = torch.where(mask_push.unsqueeze(-1), target_pos, ik_pose[:, 4:7])

    # print("step", step)
    # print("state", state)

    ####################
    # State 2: Finishing
    ####################

    # --- Move back to neutral pose ---
    goal_err = torch.linalg.norm(car_pos - goal_pos, dim=-1)
    print("goal_err", float(goal_err[0]))
    promote_mask = (state == 1) & (goal_err < 0.005)
    state[:] = torch.where(promote_mask, 2, state)

    mask_finish = (state == 2)
    print("Finishing:", sum(mask_finish) / len(mask_finish))

    ik_pose[:] = torch.where(mask_finish.unsqueeze(-1), done_pose, ik_pose)

def init_dataset_writer(writer, envs):
    max_size = envs.max_episode_length - 1
    print('DatasetWriterHDF5 max_size: ', max_size)

    if envs.num_obs >= 1:
        writer.create_dataset('obses', (envs.total_num_envs, envs.num_obs), max_size, np.float32)
    writer.create_dataset('actions', (envs.total_num_envs, envs.num_dofs), max_size, np.float32)
    writer.create_dataset('rewards', (envs.total_num_envs, ), max_size, np.float32)
    writer.create_dataset('terminals', (envs.total_num_envs, ), max_size, bool)
    writer.create_dataset('timeouts', (envs.total_num_envs, ), max_size, bool)

    i = 0
    for cam in envs.rgb_cameras:
        image_buf = envs.info['image'][i]
        writer.create_dataset(cam, image_buf.shape, max_size, np.uint8)


def write_observations(writer, envs):
    # WARN: Omit timeouts for now
    if envs.trunc_buf.any():
        print('write_data_post: truncation, no writing')
        return

    if envs.num_obs >= 1:
        writer.append('obses', envs.obs_buf.cpu().unsqueeze(0))

    i = 0
    for cam in envs.rgb_cameras:
        writer.append(cam, envs.info['image'][i].cpu().unsqueeze(0))
        i += 1


def write_action_reward_term(writer, envs):
    # WARN: Omit timeouts for now
    if envs.trunc_buf.any():
        print('write_data_post: truncation, no writing')
        return

    writer.append('actions', envs.act_buf.cpu().unsqueeze(0))
    writer.append('rewards', envs.rew_buf.cpu().unsqueeze(0))
    writer.append('terminals', envs.term_buf.cpu().unsqueeze(0))
    writer.append('timeouts', envs.trunc_buf.cpu().unsqueeze(0))


if __name__ == "__main__":
    from argparse import ArgumentParser, BooleanOptionalAction
    import yaml

    parser = ArgumentParser()
    parser.add_argument("--yml", default="so100_parking.yaml", help="env config")
    parser.add_argument("--replay_buf", default=None, help="env config")
    parser.add_argument("--replay_env_idx", default=None, help="env config")
    parser.add_argument('--no_write', action='store_true')
    args = parser.parse_args()

    write_demos = not args.no_write
    print("write_demos ", write_demos)

    ## Read config file ##
    class_kwargs = {}
    yml_file = args.yml
    if yml_file is not None:
        if os.path.exists(yml_file):
            yml_path = yml_file
        else:
            yml_path = "../../../train_config/" + yml_file
        with open(yml_path, "r") as stream:
            config = yaml.safe_load(stream)

        env_params = config.get("envs", {})
        for key, value in env_params.items():
            if key == "num_envs" or key == "_target_":
                continue
            class_kwargs[key] = value

    # Environment
    num_envs = [4, 4, 4, 4]

    if yml_file is not None:
        num_envs = env_params["num_envs"]

    # Replay buffer path
    replay_buf_path = args.replay_buf
    replay_actions = True if replay_buf_path is not None else False
    replay_env_idx = int(args.replay_env_idx) if args.replay_env_idx is not None else None

    if replay_actions:
        with h5py.File(replay_buf_path, 'r') as hdf5_file:
            actions = torch.tensor(hdf5_file['action'][:]).to(device=device)

        if replay_env_idx is not None:
            actions = actions[:, replay_env_idx, ...]

        replay_length = len(actions)

    envs = SO100Parking(num_envs, **class_kwargs)

    obs, info = envs.reset()

    # record rewards for stats
    demo_rewards = envs.rew_buf.clone()

    # Dataset writer
    writer = DatasetWriterHDF5(envs.__class__.__name__, "demos.h5")
    if write_demos:
        init_dataset_writer(writer, envs)
        write_observations(writer, envs)

    # Rendering
    render = envs.gym.get_render()

    if render is not None:
        render.capped_step = True

    # Solver loop
    finished = False
    idx = 0
    push_step = torch.zeros(num_envs, dtype=torch.float, device=device)

    im_dir = './runs/images'
    os.makedirs(im_dir, exist_ok=True)

    state = torch.zeros((envs.total_num_envs, ), dtype=torch.uint8, device=device)
    ik_pose = torch.zeros((envs.total_num_envs, 7), dtype=torch.float32, device=device)
    gripper_pos = torch.zeros((envs.total_num_envs, ), dtype=torch.float32, device=device)
    envs.kinematic_sensor_handler.get_state()
    envs.ik_handler.ik_target_pose[:] = envs.kinematic_sensor_handler.ee_pose_value
    envs.approach_quat = envs.approach_quat.repeat(envs.num_envs_int, 1)

    while not finished:
        
        if replay_actions:
            action = actions[idx].unsqueeze(0)

        else:
            demo_push(state,
                        envs.done_pose,
                        envs.ik_handler.ik_target_pose,
                        envs.kinematic_sensor_handler.ee_pose_value,
                        envs.car_handler.pose_reset[:, 4:],
                        envs.goal_pose[:, 4:],
                        envs.car_handler.pose_value[:, 4:],
                        )
            
            # push_step[:] = torch.where(state == 1, push_step + 1, push_step)
            push_step += 1

            # Assign the actions
            # envs.ik_handler.ik_target_pose[:, :4] = torch.where((state >= 1).unsqueeze(-1), envs.approach_quat, curr_ee_pose[:, :4])
            rpy = v_rpy_from_quat(envs.kinematic_sensor_handler.ee_pose_value[:, :4])
            yaw = rpy[:, 2]
            approach_rpy = v_rpy_from_quat(envs.approach_quat)
            approach_rpy[:, 2] = rpy[:, 2] # allow swinging yaw
            envs.approach_quat = v_quat_from_rpy(approach_rpy) 
            envs.ik_handler.ik_target_pose[:, :4] = envs.approach_quat

            # compute target joint pos with IK
            envs.ik_handler.compute_inverse_kinematics()

            # assign the computed pid targets
            action = torch.zeros((envs.total_num_envs, envs.num_dofs),
                                 dtype=torch.float32, device=device)

            action[:] = envs.ik_handler.ik_dof_pos_target - envs.joint_handler.dof_pos_value

            # # take the highest ratio between the delta joint pos and the allowed delta step
            # ratios = torch.abs(action/ envs.pid_delta_pos)
            # max_ratio = torch.max(ratios)

            action[:] = torch.clamp(
                action,
                -envs.pid_delta_pos,
                envs.pid_delta_pos
            )

        # Step
        obs, rew, reset, timeout, info = envs.step(action)
        state[:] = torch.where(envs.reset_buf, 0, state)
        push_step[:] = torch.where(envs.reset_buf, 0, push_step)
        envs.ik_handler.ik_target_pose[:] = torch.where(envs.reset_buf.unsqueeze(-1), envs.kinematic_sensor_handler.ee_pose_value, envs.ik_handler.ik_target_pose)

        demo_rewards += envs.rew_buf

        if replay_actions:
            has_crash = envs.crash_callback()
            if has_crash:
                print("FOUND A CRASH")
                render.set_paused(True)

        if write_demos and not replay_actions:
            write_observations(writer, envs)
            write_action_reward_term(writer, envs)

        idx += 1

        if replay_actions and replay_length <= idx:
            print("WARN: the episode will not reset, please rerun the script")
            render.set_paused(True)
            idx = 0

        if idx >= envs.max_episode_length:
            print("mean demo_rewards: ", torch.mean(demo_rewards).item() / idx)
            render.set_paused(True)
            idx = 0
            state[:] = 0
            demo_rewards[:] = 0

        finished = envs.render_finished