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
from envs.so100_tower import SO100Tower
from vlearn.rl_algos.sac.vsim_training_data import DatasetWriterHDF5

assert torch.cuda.is_available()
device = torch.device("cuda:0")


def normalize(vec):
    norm = torch.linalg.vector_norm(vec, dim=-1)
    return vec / norm.unsqueeze(-1)

def move_towards_target(
    ik_pose: torch.Tensor,
    curr_ee_pos: torch.Tensor,
    target_pos: torch.Tensor,
    active_mask: torch.Tensor,
    update_mask: torch.Tensor,
    step_size: float = 0.004,
) -> torch.Tensor:
    direction = target_pos - curr_ee_pos
    direction_norm = torch.linalg.norm(direction, dim=-1, keepdim=True) + 1e-8
    direction_unit = direction / direction_norm

    updated_target = curr_ee_pos + direction_unit * step_size

    # Update only for active and update_masked envs
    combined_mask = (active_mask & update_mask).unsqueeze(-1)
    ik_pose[:, 4:] = torch.where(combined_mask, updated_target, ik_pose[:, 4:])
    return ik_pose


def reach(
        state,
        ik_pose,
        curr_ee_pose,
        box_pos,
        gripper_dof_target,
        locked_box_pos):

    curr_ee_pos = curr_ee_pose[:, 4:]
    
    # Reach offset moves just above the box before descending and grabbing
    approach_offset = torch.tensor([0.0, 0.0, 0.06], dtype=torch.float32, device=device)
    box_pos_offset = box_pos + approach_offset

    mask_reach = (state == REACH)

    ik_pose[:, 4:] = torch.where(mask_reach.unsqueeze(-1), box_pos_offset, ik_pose[:, 4:])
    gripper_dof_target[:] = torch.where(mask_reach, 0.5, gripper_dof_target)

    # check for overall reach
    err = box_pos_offset - curr_ee_pos
    pos_err = torch.linalg.vector_norm(err, dim=-1)

    # Check for specifically 
    reached = pos_err < 0.01 # Box origin is 1.5cm embedded into object, so 2cm is arriving at object
    state[:] = torch.where(mask_reach & reached, PICKUP, state)

    # When state moves to pickup, we lock in the box position
    locked_box_pos[:, 4:] = torch.where((mask_reach & reached).unsqueeze(-1), box_pos, locked_box_pos[:, 4:])

def pickup(state,
           ik_pose,
           curr_ee_pose,
           box_pos,
           gripper_dof_target,
           jaw_cube_force_val,
           delay_counter,
           grabbed_flag):

    # State 1: Picking up the cube
    curr_ee_pos = curr_ee_pose[:, 4:]
    curr_ee_quat = curr_ee_pose[:, 0:4]
    target_pos = ik_pose[:, 4:]
    target_quat = ik_pose[:, 0:4]

    mask_pickup = (state == PICKUP)

    # Position error
    pos_err = torch.linalg.norm(curr_ee_pos - target_pos, dim=-1)
    pos_tol = 0.002  # 5mm tolerance
    mask_pos = pos_err < pos_tol

    # Orientation error
    # Compute angular difference between quaternions
    dot = torch.abs(torch.sum(curr_ee_quat * target_quat, dim=-1)) # Get quat diff
    dot = torch.clamp(dot, -1.0, 1.0)
    ang_err = 2 * torch.acos(dot)
    ang_tol = 0.1  # about 6 degrees tolerance
    mask_rot = ang_err < ang_tol

    # If we've reached the current target, update the target
    mask_update_target = mask_pos # & mask_rot

    ik_pose = move_towards_target(
        ik_pose,
        curr_ee_pos,
        box_pos,
        mask_pickup,
        mask_update_target,
        step_size=0.006
    )

    # ik_pose[:, 4:] = torch.where(mask_pickup.unsqueeze(-1), box_pos, ik_pose[:, 4:])

    # check for reaching target
    err = box_pos - curr_ee_pose[:, 4:]
    pos_err = torch.linalg.vector_norm(err, dim=-1)

    # Check for specifically 
    grab = pos_err < 0.003 # 3mm error should be accuracte enough to pickup the cube

    gripper_dof_target[:] = torch.where(grab & mask_pickup, torch.zeros_like(gripper_dof_target), gripper_dof_target)

    relax_grab = pos_err < 0.005

    new_grab = (jaw_cube_force_val > 0).any(dim=1) & relax_grab
    grabbed_flag[:] = torch.where(mask_pickup & (grabbed_flag | new_grab), True, grabbed_flag)

    delay_counter[:] = torch.where((mask_pickup & grabbed_flag) | delay_counter > 0,
                                      delay_counter + 1,
                                      torch.zeros_like(delay_counter))
    
    stabilized = delay_counter > 20

    delay_counter[:] = torch.where(mask_pickup & grabbed_flag & stabilized, 0, delay_counter)
    state[:] = torch.where(mask_pickup & grabbed_flag & stabilized, CENTER, state)
    grabbed_flag[:] = torch.where(mask_pickup & grabbed_flag & stabilized, False, delay_counter)

def align_with_stack(
        state,
        ik_pose,
        curr_ee_pose,
        top_of_stack_pose,
        target_box_pose,
        aligned_flag,
    ):
    curr_ee_pos = curr_ee_pose[:, 4:]
    beside_stack = top_of_stack_pose[:, 4:] + torch.tensor([0.0, 0.0, 0.04], dtype=torch.float32, device=state.device)
    beside_stack[:, 2] = 0.303 # just above the table
    beside_stack[:, 0] -= 0.07 # To the side of the stack
    beside_stack[:, 1] += 0.03 # get a better view

    # Check for reaching target pose
    err = beside_stack - curr_ee_pos
    pos_err = torch.linalg.vector_norm(err, dim=-1)
    aligned = pos_err < 0.01
    mask_align = (state == CENTER)

    # If we have aligned, we want this state locked in so that we can proceed to move up the side of the stack
    new_align = aligned
    aligned_flag[:] = torch.where(mask_align & (aligned_flag | new_align), True, aligned_flag)

    # --- Set target pose to be next to bottom of stack if we have not aligned already ---
    ik_pose[:, 4:] = torch.where(mask_align.unsqueeze(-1) & ~aligned_flag.unsqueeze(-1), beside_stack, ik_pose[:, 4:])

    # If we've aligned, move to the top of the stack with the offset
    stack_offset = torch.tensor([0.0, 0.0, 0.04], dtype=torch.float32, device=state.device)
    target_pos = top_of_stack_pose[:, 4:] + stack_offset
    target_pos[:, 0] = beside_stack[:, 0]
    target_pos[:, 1] = beside_stack[:, 1]

    err = target_pos - curr_ee_pos
    pos_err = torch.linalg.vector_norm(err, dim=-1)
    top_of_stack = pos_err < 0.005

    ik_pose[:, 4:] = torch.where(mask_align.unsqueeze(-1) & aligned_flag.unsqueeze(-1), target_pos, ik_pose[:, 4:])

    # If we drop the cube, move back to reach state
    cube_ee_dist = curr_ee_pose[:, 4:] - target_box_pose[:, 4:]
    cube_ee_err = torch.linalg.vector_norm(cube_ee_dist, dim=-1)
    fumble = cube_ee_err > 0.05
    state[:] = torch.where(mask_align & fumble, REACH, state)
    
    state[:] = torch.where(mask_align & top_of_stack, STACK, state)


def stack(
        state,
        ik_pose,
        curr_ee_pose,
        gripper_dof_target,
        top_of_stack_pose,
        delay_counter,
        let_go_counter,
    ):

    curr_ee_pos = curr_ee_pose[:, 4:]

    mask_stack = state == STACK

    # Choose between base and extra offset
    stack_offset = torch.tensor([0.0, 0.0, 0.04], dtype=torch.float32, device=state.device)

    top_of_stack = top_of_stack_pose[:, 4:]
    target_pos = top_of_stack + stack_offset

    ik_pose[:, 4:] = torch.where(mask_stack.unsqueeze(-1), target_pos, ik_pose[:, 4:])

    err = target_pos - curr_ee_pos
    pos_err = torch.linalg.vector_norm(err, dim=-1)
    stacked = pos_err < 0.001

    delay_counter[:] = torch.where((mask_stack & stacked) | delay_counter > 0,
                                      delay_counter + 1,
                                      torch.zeros_like(delay_counter))
    
    stabilized = delay_counter > 5
    
    gripper_dof_target[:] = torch.where(mask_stack & stabilized, torch.ones_like(gripper_dof_target), gripper_dof_target)

    let_go_counter[:] = torch.where((mask_stack & stabilized) | let_go_counter > 0,
                                      let_go_counter + 1,
                                      torch.zeros_like(let_go_counter))
    
    released = let_go_counter > 5

    # Mark that we have stacked this box
    state[:] = torch.where(mask_stack & stacked & released, RESET, state)
    delay_counter[:] = torch.where(mask_stack & stacked & stabilized, 0, delay_counter)
    let_go_counter[:] = torch.where(mask_stack & stacked & released, 0, let_go_counter)

def reset_position(
        state,
        ik_pose,
        curr_ee_pose,
        rest_pose,
        target_box_pose,
        stacked_flag,
    ):
    ik_pos = ik_pose[:, 4:]
    rest_pos = rest_pose[:, 4:]
    curr_ee_pos = curr_ee_pose[:, 4:]

    mask_reset = state == RESET

    stack_offset = torch.tensor([0.0, 0.0, 0.07], dtype=torch.float32, device=state.device)
    top_of_stack = target_box_pose[:, 4:]
    target_pos = top_of_stack + stack_offset

    err = target_pos - curr_ee_pos
    pos_err = torch.linalg.vector_norm(err, dim=-1)
    moved = pos_err < 0.01

    stacked_flag[:] = torch.where(mask_reset & (stacked_flag | moved), True, stacked_flag)

    ik_pos[:] = torch.where(mask_reset.unsqueeze(-1) & ~stacked_flag.unsqueeze(-1), target_pos, ik_pos)
    ik_pose[:, :4] = torch.where(mask_reset.unsqueeze(-1) & ~stacked_flag.unsqueeze(-1), curr_ee_pose[:, :4], ik_pose[:, :4])

    err = rest_pos - curr_ee_pos
    pos_err = torch.linalg.vector_norm(err, dim=-1)
    rested = pos_err < 0.005

    ik_pose = move_towards_target(
        ik_pose,
        curr_ee_pos,
        rest_pos,
        stacked_flag,
        ~rested,
        step_size=0.004
    )

    # ik_pos[:] = torch.where(mask_reset.unsqueeze(-1) & stacked_flag.unsqueeze(-1), rest_pos, ik_pos)
    state[:] = torch.where(mask_reset & rested, FINISH, state)


def build_tower(state,
        boxes_stacked, 
        top_of_stack_pose,
        rest_pose,
        ik_pose, 
        curr_ee_pose, 
        target_box_pose,
        gripper_dof_target,
        delay_counter,
        let_go_counter,
        grabbed_flag,
        jaw_cube_force_val,
        stacked_flag,
        locked_box_pos,
        aligned_flag,
        ):
    
    # --- Debugging ---
    print("state mean", torch.mean(state.float()).item())
    
    # --- Move to target box ---
    reach(
        state,
        ik_pose,
        curr_ee_pose,
        target_box_pose[:, 4:],
        gripper_dof_target,
        locked_box_pos)
    
    # --- Pickup target box ---
    pickup(
        state,
        ik_pose,
        curr_ee_pose,
        locked_box_pos[:, 4:],
        gripper_dof_target,
        jaw_cube_force_val,
        delay_counter,
        grabbed_flag
    )

    # --- Move next to stack ---
    align_with_stack(
        state,
        ik_pose,
        curr_ee_pose,
        top_of_stack_pose,
        target_box_pose,
        aligned_flag,
    )

    # --- Stack target box ---
    stack(
        state,
        ik_pose,
        curr_ee_pose,
        gripper_dof_target,
        top_of_stack_pose,
        delay_counter,
        let_go_counter,
    )

    # --- Move to rest pose ---
    reset_position(
        state,
        ik_pose,
        curr_ee_pose,
        rest_pose,
        target_box_pose,
        stacked_flag,
    )

    finished = state == FINISH
    print("Finished:", torch.mean(finished.float()).item())
    ik_pose[:] = torch.where(finished.unsqueeze(-1), rest_pose, ik_pose)
    
    
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

def make_stack_order(num_envs: int, num_boxes: int, base_idx: int, device) -> torch.Tensor:
    others = torch.tensor([i for i in range(num_boxes) if i != base_idx], dtype=torch.long, device=device)
    # Shuffle 'others' independently per env
    rand = torch.rand(num_envs, others.numel(), device=device)
    perm = torch.argsort(rand, dim=1)  # different order per env
    shuffled_others = others.unsqueeze(0).expand(num_envs, -1).gather(1, perm)
    order = torch.cat(
        [
            torch.full((num_envs, 1), base_idx, dtype=torch.long, device=device), 
            shuffled_others
        ],
        dim=1
    )
    return order


if __name__ == "__main__":
    from argparse import ArgumentParser, BooleanOptionalAction
    import yaml

    parser = ArgumentParser()
    parser.add_argument("--yml", default="so100_tower.yaml", help="env config")
    parser.add_argument("--replay_buf", default=None, help="env config")
    parser.add_argument("--replay_env_idx", default=None, help="env config")
    parser.add_argument('--no_write', action='store_true')
    parser.add_argument('--folder', default='SO100Tower')
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
            actions = torch.tensor(hdf5_file['actions'][:]).to(device=device)

        if replay_env_idx is not None:
            actions = actions[:, replay_env_idx, ...]

        replay_length = len(actions)

    envs = SO100Tower(num_envs, **class_kwargs)
    envs.gym.step()

    obs, info = envs.reset()

    # record rewards for stats
    demo_rewards = envs.rew_buf.clone()

    # Dataset writer
    writer = DatasetWriterHDF5(args.folder, "demos.h5")
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
    delay_counter = torch.zeros((envs.total_num_envs,), dtype=torch.int32, device=device)
    let_go_counter = torch.zeros_like(delay_counter)
    grabbed_flag = torch.zeros((envs.total_num_envs,), dtype=torch.bool, device=device)
    stacked_flag = torch.zeros_like(grabbed_flag)
    aligned_flag = torch.zeros_like(stacked_flag)
    ik_pose = torch.zeros((envs.total_num_envs, 7), dtype=torch.float32, device=device)
    locked_box_pos = torch.zeros_like(ik_pose)
    locked_box_pos[:] = envs.boxes_handler.pose_values[0]
    gripper_pos = torch.zeros((envs.total_num_envs, ), dtype=torch.float32, device=device)
    envs.kinematic_sensor_handler.get_state()
    envs.ik_handler.ik_target_pose[:] = envs.kinematic_sensor_handler.ee_pose_value
    envs.approach_quat = envs.approach_quat.repeat(envs.num_envs_int, 1)
    gripper_dof_target = torch.zeros_like(envs.rew_buf)

    # The middle box is our starting point
    boxes_stacked = torch.tensor([[False, False, True, False, False]] * envs.num_envs_int, dtype=torch.bool, device=envs.device)

    # Tensor of (5, num_envs, 7)
    box_poses = envs.boxes_handler.pose_values 
    
    top_of_stack_box_idx = torch.tensor([2] *envs.num_envs_int, dtype=torch.int, device=envs.device)

    num_boxes = box_poses.shape[0]
    BASE_IDX = 2
    stack_order = make_stack_order(envs.num_envs_int, num_boxes, BASE_IDX, envs.device)
    
    # --- Define States --- 
    REACH = 0
    PICKUP = 1
    CENTER = 2
    STACK = 3
    RESET = 4
    FINISH = 5
    

    while not finished:
        
        if replay_actions:
            action = actions[idx].unsqueeze(0)

        else:
            build_tower(state,
                boxes_stacked,
                envs.top_of_stack_pose,
                envs.rest_pose,
                envs.ik_handler.ik_target_pose,
                envs.kinematic_sensor_handler.ee_pose_value,
                envs.target_box_pose,
                gripper_dof_target,
                delay_counter,
                let_go_counter,
                grabbed_flag,
                envs.jaw_cube_force_sensor_handler.force_sensor_val,
                stacked_flag,
                locked_box_pos,
                aligned_flag,
                )           
    
            # push_step[:] = torch.where(state == 1, push_step + 1, push_step)
            push_step += 1

            # Assign the actions
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

            envs.ik_handler.ik_dof_pos_target[:, envs.gripper_idx] = gripper_dof_target
            action[:] = envs.ik_handler.ik_dof_pos_target - envs.joint_handler.dof_pos_value

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

        print("Reward:", rew[0])

        demo_rewards += rew

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
            boxes_stacked = torch.tensor([[False, False, True, False, False]] * envs.num_envs_int, dtype=torch.bool, device=envs.device)
            top_of_stack_box_idx[:] = 2
            delay_counter.zero_()
            grabbed_flag.zero_()
            stacked_flag.zero_()
            locked_box_pos[:] = envs.boxes_handler.pose_values[0]

            render.set_paused(True)
            idx = 0
            state[:] = 0
            demo_rewards[:] = 0

        finished = envs.render_finished