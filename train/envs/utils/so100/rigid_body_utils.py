from dataclasses import dataclass
from typing import Optional, Sequence
import torch
import vlearn as v
import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
train_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
if train_root not in sys.path:
    sys.path.insert(0, train_root)

from envs.environment import EnvironmentGpu

class RigidBodiesHandler:
    def __init__(self,
                 envs: EnvironmentGpu,
                 rigid_body_handle_objects: list,
                 ):
        self.envs = envs
        self.rigid_body_handle_objects = rigid_body_handle_objects
        
        self.pose_values = torch.zeros(
            (len(rigid_body_handle_objects), envs.num_envs_int, 7), dtype=torch.float32, device=envs.device)  # quat + pos
        self.velocity_values = torch.zeros(
            (len(rigid_body_handle_objects), envs.num_envs_int, 6), dtype=torch.float32, device=envs.device)  # linear + angular

        get_object_cmds = []
        for i, handle in enumerate(self.rigid_body_handle_objects):
            get_object_cmds.append(
                envs.env_group.create_rigid_body_kinematic_state_command(
                    v.wrap_gpu_buffer(self.pose_values[i]),
                    v.wrap_gpu_buffer(self.velocity_values[i]),
                    handle,
                )
            )

        self.get_object_kine_state_cmd_array = envs.gym.create_rigid_body_kinematic_state_command_gpu_array(get_object_cmds)

        # Call get object pose
        envs.gym.get_rigid_body_kinematic_states(self.get_object_kine_state_cmd_array)

        # Set object kinematic state
        self.pose_resets = torch.zeros(
            (len(rigid_body_handle_objects), envs.num_envs_int, 7), dtype=torch.float32, device=envs.device)
        self.velocity_targets = torch.zeros(
            (len(rigid_body_handle_objects), envs.num_envs_int, 6), dtype=torch.float32, device=envs.device)

        set_object_cmds = []
        for i, handle in enumerate(self.rigid_body_handle_objects):
            set_object_cmds.append(
                envs.env_group.create_rigid_body_kinematic_state_command(
                    v.wrap_gpu_buffer(self.pose_resets[i]),
                    v.wrap_gpu_buffer(self.velocity_targets[i]),
                    handle,
                    masks_buffer=v.wrap_gpu_buffer(envs.reset_buf)
                )
            )

        self.set_object_kine_state_cmd_array = envs.gym.create_rigid_body_kinematic_state_command_gpu_array(set_object_cmds)
        
        # print("set_object_cmd", set_object_cmds)
        
    def get_states(self):
        self.envs.gym.get_rigid_body_kinematic_states(self.get_object_kine_state_cmd_array)
    
    def set_states(self, poses, velocities):
        self.pose_resets[:] = poses
        self.velocity_targets[:] = velocities
        self.envs.gym.set_rigid_body_kinematic_states(self.set_object_kine_state_cmd_array)