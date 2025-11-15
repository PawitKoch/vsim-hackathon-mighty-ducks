from dataclasses import dataclass
from typing import Optional, Sequence
import torch
import vlearn as v
import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
train_root = os.path.abspath(os.path.join(current_dir, "..", "..", "train"))
if train_root not in sys.path:
    sys.path.insert(0, train_root)

from envs.environment import EnvironmentGpu

class InverseKinematicsHandler:
    def __init__(self,
                 envs: EnvironmentGpu,
                 robot_handle: int,
                 end_effector_index: int,
                 ee_sensor_local_offset: any,
                 ik_orn_dofs
                 ):
        self.envs = envs
        self.robot_handle = robot_handle
        self.end_effector_index = end_effector_index
        self.ee_sensor_local_offset = ee_sensor_local_offset
        self.ik_target_pose = torch.zeros((envs.num_envs_int, 7), dtype=torch.float32, device=envs.device)
        self.ik_dof_pos_target = torch.zeros((envs.num_envs_int, envs.num_dofs), dtype=torch.float32, device=envs.device)

        max_num_iter = 50
        ik_cmd = envs.env_group.create_inverse_kinematics_command(
            v.wrap_gpu_buffer(self.ik_target_pose),
            v.wrap_gpu_buffer(self.ik_dof_pos_target),
            self.robot_handle,
            [self.end_effector_index], # Using the desired link
            [self.ee_sensor_local_offset], # Using kinematic sensor offset calibrated in Vlab
            max_num_iter,
            ik_orn_dofs)

        self.ik_cmd_arr = envs.gym.create_inverse_kinematics_command_gpu_array([ik_cmd])

    def compute_inverse_kinematics(self):
        self.envs.gym.compute_inverse_kinematics(self.ik_cmd_arr)