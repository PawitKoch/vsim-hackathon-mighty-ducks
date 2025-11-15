import os
import sys
import math
import random
import torchvision.transforms.v2 as v2

import torch
import vlearn as v
import numpy as np
from vlearn.spaces import Box
from vlearn.torch_utils.torch_jit_utils import v_quat_from_rpy, v_rpy_from_quat, quat_mul

# This should be the convention here on out
current_dir = os.path.dirname(os.path.abspath(__file__))
train_root = os.path.abspath(os.path.join(current_dir, "../"))
if train_root not in sys.path:
    sys.path.insert(0, train_root)

# Convention enables standardization of import structure, no more conditional switching
from envs.common import create_plane
from envs.environment import EnvironmentGpu
from envs.utils.sam2_utils import process_sam2
from envs.utils.so100.debug_tower import print_step_debug, init_debug_draw, update_draw_targets, debug_save_images
from envs.utils.so100.rigid_body_utils import RigidBodiesHandler
from vlearn.rl_algos.sac.vsim_training_data import *
from sam2.sam2wrapper import Sam2Wrapper
from vlearn.components.gym_command_handlers import (JointStateHandler,
                                                    KinematicSensorHandler,
                                                    RGBCameraHandler,
                                                    ForceSensorHandler,)
from envs.utils.so100.ik_handler import InverseKinematicsHandler

assert torch.cuda.is_available()
device = torch.device("cuda:0")


class SO100Tower(EnvironmentGpu):

    def __init__(self,
                 num_envs: int,
                 device: torch.device,
                 num_solver_iterations: int = 64,
                 venv_file: str = 'assets/donotship/SO100/tower/tower.venv',
                 dof_pos_home: list = [],
                 box1_pose_initial: list = [],
                 box2_pose_initial: list = [],
                 box3_pose_initial: list = [],
                 box4_pose_initial: list = [],
                 box5_pose_initial: list = [],
                 approach_quat: list = [],
                 robot_root_pos: list = [],
                 rest_pose: list = [],
                 debug: bool = False,
                 max_episode_length: int = 180,
                 timestep: float = 0.01667,
                 frame_skip: int = 4,
                 rgb_cameras: list[str] = [],
                 use_segmentation_for_inference: bool = False,
                 spacing: float = 1.5,
                 gravity: v.Vec3 = v.Vec3(0, 0, -9.81),
                 enable_scene_query: bool = True,
                 initial_is_paused: bool = True,
                 send_interrupt: bool = False,
                 up_axis: v.Vec3 = v.Vec3(0, 0, 1),
                 rendering: bool = True,
                 max_contact_pairs_per_env: int = 256,
                 with_window: bool = True,
                 num_segmentation_objs: int = 3,
                 max_delta_deg: list = [],
                 ):

        num_dofs = 6
        self.end_effector_index = 5  # Link index of gripper
        self.gripper_idx = 5

        assert (len(dof_pos_home) == num_dofs)
        assert (len(box1_pose_initial) == 7)

        device = torch.device(device)

        self.num_env_sets = 1
        self.state = torch.zeros((num_envs), device=device, dtype=torch.int32)  # models stages of stacking cubes
        self.venv_file = venv_file
        self.dof_pos_home = torch.tensor([dof_pos_home], dtype=torch.float32, device=device)
        self.box1_pose_initial = torch.tensor([box1_pose_initial], dtype=torch.float32, device=device)
        self.box2_pose_initial = torch.tensor([box2_pose_initial], dtype=torch.float32, device=device)
        self.box3_pose_initial = torch.tensor([box3_pose_initial], dtype=torch.float32, device=device)
        self.box4_pose_initial = torch.tensor([box4_pose_initial], dtype=torch.float32, device=device)
        self.box5_pose_initial = torch.tensor([box5_pose_initial], dtype=torch.float32, device=device)
        self.box_poses_initial = torch.zeros((5, num_envs, 7), dtype=torch.float32, device=device)
        self.box_poses_initial[0, :] = self.box1_pose_initial.expand(num_envs, 7)
        self.box_poses_initial[1, :] = self.box2_pose_initial.expand(num_envs, 7)
        self.box_poses_initial[2, :] = self.box3_pose_initial.expand(num_envs, 7)
        self.box_poses_initial[3, :] = self.box4_pose_initial.expand(num_envs, 7)
        self.box_poses_initial[4, :] = self.box5_pose_initial.expand(num_envs, 7)
        self.env_idx = torch.arange(num_envs, device=device)

        self.approach_quat = torch.tensor([approach_quat], dtype=torch.float32, device=device)
        self.rest_pose = torch.tensor([rest_pose] * num_envs, dtype=torch.float32, device=device)
        self.robot_root_pos = robot_root_pos
        self.gripper_open = 2.0  # Used in demos
        self.debug = debug
        assert (max_episode_length % frame_skip == 0)
        num_training_steps = max_episode_length // frame_skip
        self.first_step = True
        self.rgb_cameras = rgb_cameras
        self.use_segmentation_for_inference = use_segmentation_for_inference
        self.num_segmentation_objs = num_segmentation_objs
        self.max_delta_deg = max_delta_deg
        self.num_rgb_cameras = len(rgb_cameras)
        self.idx = 0

        super().__init__(
            [num_envs],
            device,
            rendering,
            enable_scene_query,
            num_training_steps,
            timestep,
            frame_skip,
            spacing,
            gravity,
            num_dofs,
            initial_is_paused=initial_is_paused,
            send_interrupt=send_interrupt,
            up_axis=up_axis,
            max_contact_pairs_per_env=max_contact_pairs_per_env,
            with_window=with_window,
            update_scene_dependent_components_in_step=False,
            return_clones=False)

        self.num_envs_int = num_envs  # super() saves self.num_envs as a list for cases with multiple env sets

        # Create environments
        self.create_envs()

        # Ground and sky
        create_plane(self.gym)

        # Finalize
        self.gym.gym_finalize()

        # Initialise observation and action space
        self.init_obs_and_act_spaces()

        # Allocate buffers
        self.allocate_buffers()

        # Store initial conditions
        self.store_initial_conditions()

        # Debug draw
        if self.debug:
            init_debug_draw(self)
            debug_save_images(self)

        # Finalize gym
        self.gym.set_num_solver_iterations(num_solver_iterations)
        self.gym.gym_finalize()

        # Our POV when rendering the simulation
        if self.rendering:
            self.gym_render.reset_camera(v.Vec3(0.804267, -0.946838, 0.819094), v.Vec3(-0.671851, 0.584034, -0.455545))

        self.info['cameras'] = rgb_cameras

        # Segment anything wrapper
        if self.num_envs_int == 1 and self.use_segmentation_for_inference:
            self.sam2 = Sam2Wrapper(num_objs=num_segmentation_objs,
                                    model_size='medium',
                                    device=device,
                                    compile=False)

    def create_envs(self):
        # Environment def
        self.env_def_handle = self.gym.create_environment_def("so100")
        env_def = self.gym.get_environment_def(self.env_def_handle)
        self.env_def = env_def

        # Import env assets from file
        merge_fixed_joints = False
        merge_meshes_inside_files = True
        use_visual_meshes = False
        create_env_instances = False
        force_inertia_computation = True
        env_def.import_environment(self.venv_file,
                                   merge_fixed_joints,
                                   merge_meshes_inside_files,
                                   use_visual_meshes,
                                   create_env_instances,
                                   force_inertia_computation)

        # Articulation
        self.robot_def_handle = env_def.get_articulation_def_handle_by_name("so_arm100")  # From .vsim
        self.robot_def = env_def.get_articulation_def(self.robot_def_handle)
        self.robot_handle = env_def.get_articulation_handle_by_name("so100")  # From .venv
        self.robot_def.has_self_collisions = False
        self.robot = env_def.get_articulation(self.robot_handle)

        # Table
        self.table_handle = env_def.get_rigid_body_handle_by_name("Platform")
        self.table = env_def.get_rigid_body(self.table_handle)
        table_transform_handle = self.table.get_transform_handle()

        # Boxes
        self.box1_handle = env_def.get_rigid_body_handle_by_name("box1")
        box1 = self.env_def.get_rigid_body(self.box1_handle)
        box1_transform_handle = box1.get_transform_handle()
        self.box2_handle = env_def.get_rigid_body_handle_by_name("box2")
        box2 = self.env_def.get_rigid_body(self.box2_handle)
        box2_transform_handle = box2.get_transform_handle()
        self.box3_handle = env_def.get_rigid_body_handle_by_name("box3")
        box3 = self.env_def.get_rigid_body(self.box3_handle)
        box3_transform_handle = box3.get_transform_handle()
        self.box4_handle = env_def.get_rigid_body_handle_by_name("box4")
        box4 = self.env_def.get_rigid_body(self.box4_handle)
        box4_transform_handle = box4.get_transform_handle()
        self.box5_handle = env_def.get_rigid_body_handle_by_name("box5")
        box5 = self.env_def.get_rigid_body(self.box5_handle)
        box5_transform_handle = box5.get_transform_handle()

        # RGB camera
        camera_def = self.robot_def.get_rgb_camera_def(0)
        self.res_x = camera_def.resolution_x
        self.res_y = camera_def.resolution_y

        # Rendering of camera feed in sim
        camera = self.robot_def.get_rgb_camera(0)
        camera.render_relative_transform = v.Transform(
            v.Quat(v.Vec3(1, 0, 0), np.pi / 2) * v.Quat(v.Vec3(0, 1, 0), np.pi / 2),  # rotation
            v.Vec3(-0.4, 0, 1)  # translation
            )
        ratio = self.res_y / self.res_x
        camera.render_width = 1
        camera.render_height = 1 * ratio
        self.res_y_policy = self.res_y
        self.res_x_policy = self.res_x

        # Segmentation array later used to filter observation image to mimic SAM2 inference
        self.robot.set_segmentation(index=5, segmentation=1)
        self.robot.set_segmentation(index=6, segmentation=2)
        box1.segmentation = 3  # Target box
        box2.segmentation = 4  # All other boxes for the tower
        box3.segmentation = 4
        box4.segmentation = 4
        box5.segmentation = 4
        self.segmentation_arr = [
            self.robot.get_segmentation(5),
            self.robot.get_segmentation(6),
            box1.segmentation,
            box2.segmentation,
            box3.segmentation,
            box4.segmentation,
            box5.segmentation,
            ]

        # Force sensing
        self.num_sensors = self.robot_def.get_num_force_sensor_defs()
        for force_sensor in self.robot_def.get_force_sensor_defs():
            force_sensor.max_num_transform_handles = 5

        # Contact filter used later to facilitate gripper feeling the table
        self.contact_filter_handles = [env_def.create_contact_filter([table_transform_handle])]
        self.cube_contact_filter_handles = [
            env_def.create_contact_filter([
                box1_transform_handle,
                box2_transform_handle,
                box3_transform_handle,
                box4_transform_handle,
                box5_transform_handle
                ])
            ]

        # Finalize environment def
        env_def.finalize()

        super().create_envs(self.env_def_handle)

    def init_obs_and_act_spaces(self):
        self.num_obs = self.num_dofs * 2  # current and previous dof pos

        self.single_observation_space = Box(
            low=np.array([np.finfo('f').min] * self.num_obs, dtype=np.float32),
            high=np.array([np.finfo('f').max] * self.num_obs, dtype=np.float32),
            dtype=np.float32)

        # get dof low and high limits for action space
        dof_pos_low = []
        dof_pos_high = []
        for dofdef in self.robot_def.get_joint_dof_defs():
            dof_pos_low.append(dofdef.low_limit)
            dof_pos_high.append(dofdef.high_limit)

        # take the 98% of the ranges
        delta = [(h - l) for h, l in zip(dof_pos_high, dof_pos_low)]

        dof_pos_low = [l + d * 0.05 for l, d in zip(dof_pos_low, delta)]
        dof_pos_high = [h - d * 0.05 for h, d in zip(dof_pos_high, delta)]

        # torch tensors for ease of use
        self.angle_limits_low = torch.tensor([dof_pos_low], dtype=torch.float32, device=self.device)
        self.angle_limits_high = torch.tensor([dof_pos_high], dtype=torch.float32, device=self.device)

        assert len(self.angle_limits_low[0]) == self.num_dofs
        assert len(self.angle_limits_high[0]) == self.num_dofs

        # actions spaces
        per_joint_max_deltas = [joint_max_delta / 180.0 * torch.pi for joint_max_delta in self.max_delta_deg]
        delta = per_joint_max_deltas

        self.pid_delta_pos = torch.tensor([per_joint_max_deltas], dtype=torch.float32, device=self.device)

        assert len(self.pid_delta_pos[0]) == self.num_dofs

        # define action space
        self.single_action_space = Box(
            low=np.array([-d for d in delta], dtype=np.float32),
            high=np.array(delta, dtype=np.float32),
            dtype=np.float32)

    def allocate_buffers(self):

        super().allocate_buffers()

        # Rew guarantee allows us to sequentially move through picking up boxes without losing reward of progress so far
        self.rew_guarantee = torch.zeros_like(self.rew_buf)
        self.has_fallen = torch.zeros_like(self.rew_buf)
        self.placed_flag = torch.zeros(self.num_envs_int, dtype=torch.bool, device=self.device)

        # Robot Joints
        self.joint_handler = JointStateHandler(self, self.robot_handle)
        self.last_dof_pos_value = torch.zeros_like(self.joint_handler.dof_pos_value)

        # Inverse Kinematics used in demonstrations
        kine_sensor_def = self.robot_def.get_kinematic_sensor_def(0)
        self.ee_sensor_local_offset = kine_sensor_def.offset
        ik_orn_dofs = [False, False, False, False, True, False]
        self.ik_handler = InverseKinematicsHandler(self,
                                                   self.robot_handle,
                                                   self.end_effector_index,
                                                   self.ee_sensor_local_offset,
                                                   ik_orn_dofs)

        # Kinematic sensor = end effector pose
        kine_sensor_handle = self.robot.get_kinematic_sensor_handle(0)
        self.kinematic_sensor_handler = KinematicSensorHandler(self, kine_sensor_handle)

        # Objects
        box_handles = [
            self.box1_handle,
            self.box2_handle,
            self.box3_handle,
            self.box4_handle,
            self.box5_handle
            ]
        self.boxes_handler = RigidBodiesHandler(self, box_handles)
        self.target_box_pose = torch.zeros((self.num_envs_int, 7), dtype=torch.float32, device=self.device)
        self.top_of_stack_pose = torch.zeros_like(self.target_box_pose)
        self.iniitial_top_of_stack_pose = torch.zeros_like(self.top_of_stack_pose)
        self.target_box_idx = torch.zeros_like(self.rew_buf, dtype=torch.long)
        self.top_of_stack_box_idx = torch.zeros_like(self.target_box_idx)

        # Image buffers for SAC
        self.info['image'] = [torch.zeros((self.num_envs_int, 1, self.res_y, self.res_x), dtype=torch.uint8, device=self.device)]
        self.info['image_scalars'] = [1.0 / len(self.segmentation_arr)]

        # Camera(s)
        self.rgb_camera_handle = self.robot.get_rgb_camera_handle(0)
        self.rgb_camera_handler = RGBCameraHandler(self,
                                                   self.rgb_camera_handle,
                                                   self.res_y,
                                                   self.res_x,
                                                   )
        self.camera_transform_initial = torch.zeros_like(self.rgb_camera_handler.transform_val)
        self.rgb_camera_handler.init_segmented_rgb_camera(self.segmentation_arr)

        # Get raw rather than segmented rgb to pass to SAM2 segmentor during inference
        if self.use_segmentation_for_inference and self.num_envs_int == 1:
            self.rgb_camera_handler.init_rgb_camera()

        # Table force sensors
        jaw_force_sensor_handle = self.robot.get_force_sensor_handle_by_name("jaw_sensor")
        self.jaw_force_sensor_handler = ForceSensorHandler(self,
                                                           jaw_force_sensor_handle,
                                                           self.contact_filter_handles,)
        gripper_force_sensor_handle = self.robot.get_force_sensor_handle_by_name("gripper_sensor")
        self.gripper_force_sensor_handler = ForceSensorHandler(self,
                                                               gripper_force_sensor_handle,
                                                               self.contact_filter_handles,)
        wrist_force_sensor_handle = self.robot.get_force_sensor_handle_by_name("wrist_sensor")
        self.wrist_force_sensor_handler = ForceSensorHandler(self,
                                                             wrist_force_sensor_handle,
                                                             self.contact_filter_handles,)
        self.jaw_force_sensor_handler.set_contact_filters()
        self.gripper_force_sensor_handler.set_contact_filters()
        self.wrist_force_sensor_handler.set_contact_filters()

        jaw_cube_force_sensor_handle = self.robot.get_force_sensor_handle_by_name("jaw_cube_sensor")
        self.jaw_cube_force_sensor_handler = ForceSensorHandler(self,
                                                                jaw_cube_force_sensor_handle,
                                                                self.cube_contact_filter_handles,)
        self.jaw_cube_force_sensor_handler.set_contact_filters()

        # --- pre-allocations for reset to save kernels ---
        self.all_stack_boxes = torch.arange(1, 5, device=device).unsqueeze(0).expand(self.num_envs_int, 4)
        self.stack_indexes = torch.arange(4, device=device).unsqueeze(0)
        self.target_pose_idx = torch.full((self.num_envs_int,), 1, device=device, dtype=torch.long)
        self.stack_pose_idx = torch.full((self.num_envs_int,), 3, device=device, dtype=torch.long)
        self.target_xy_noise = torch.normal(0.0, 0.02, (self.num_envs_int, 2), device=device)
        self.target_yaw_noise = torch.normal(0.0, 10.0 / 180.0 * math.pi, (self.num_envs_int, 1), device=device)
        self.target_euler = v_rpy_from_quat(self.box_poses_initial[self.target_pose_idx, self.env_idx, 0:4])


    def store_initial_conditions(self):

        self.dof_pos_home[:] = torch.clamp(
            self.dof_pos_home,
            self.angle_limits_low,
            self.angle_limits_high)

        # Root position
        self.root_position = torch.tensor([self.robot_root_pos], dtype=torch.float32, device=self.device)

        # Save home pos
        self.last_dof_pos_value[:] = self.dof_pos_home
        self.joint_handler.dof_pos_value[:] = self.dof_pos_home
        self.joint_handler.dof_pos_reset[:] = self.dof_pos_home
        self.ik_handler.ik_dof_pos_target[:] = self.dof_pos_home

        # Store initial target positions
        self.boxes_handler.pose_resets[:] = self.box_poses_initial

        # Init target transforms
        self.ik_handler.ik_target_pose[:] = torch.concat((self.approach_quat, self.root_position), dim=1)

        # Initial camera transforms
        self.rgb_camera_handler.get_transform()
        self.camera_transform_initial[:] = self.rgb_camera_handler.transform_val

        # Stacked box heights
        ground_level = 0.265
        box_height = 0.03
        err_margin = 0.005
        self.stack_heights = torch.tensor(
            [
                ground_level,  # bottom box
                ground_level + box_height,
                ground_level + (box_height * 2),
                ground_level + (box_height * 3),
                ground_level + (box_height * 4)
                ],
            dtype=torch.float32, device=self.device
            )
        # Thresholds for tower collapse per-box
        self.min_heights = self.stack_heights - err_margin

    @torch.compile
    def pre_physics_step(self, actions: torch.Tensor):
        self.act_buf[:] = actions

        # apply delta joint pos
        self.joint_handler.dof_pos_target[:] = self.joint_handler.dof_pos_value
        self.joint_handler.dof_pos_target[:] += self.act_buf

        # clamp the sim at the limits
        self.joint_handler.dof_pos_target[:] = torch.clamp(
            self.joint_handler.dof_pos_target[:],
            self.angle_limits_low,
            self.angle_limits_high)

        if self.debug and self.num_envs_int == 1:
            print_step_debug(self)
            debug_save_images(self)

        # update pid targets
        self.joint_handler.set_target_positions()

    @torch.compile
    def post_physics_step(self):
        # Eager reset
        self.progress_buf[:] += 1
        self.trunc_buf[:] = self.progress_buf >= self.max_episode_length
        self.reset_buf[:] = torch.logical_or(self.term_buf, self.trunc_buf)

        # Reset
        # if self.reset_buf.any():
        self.reset_idx()

        # Kinematics and queries
        self.gym.compute_kinematics()
        self.gym.update_scene_dependent_components()

        # Observations
        self.compute_observations()
        self.compute_rewards()

        if self.debug:
            update_draw_targets(self)

    def compute_observations(self):

        self.last_dof_pos_value[:] = self.joint_handler.dof_pos_value

        # Update buffer states
        self.joint_handler.get_positions()
        self.kinematic_sensor_handler.get_state()
        self.boxes_handler.get_states()
        self.gym.finalize_filtered_force_sensors()
        self.jaw_force_sensor_handler.get_sensor_forces()
        self.gripper_force_sensor_handler.get_sensor_forces()
        self.wrist_force_sensor_handler.get_sensor_forces()
        self.jaw_cube_force_sensor_handler.get_sensor_forces()

        self.target_box_pose[:] = self.boxes_handler.pose_values[self.target_box_idx, self.env_idx]
        self.top_of_stack_pose[:] = self.boxes_handler.pose_values[self.top_of_stack_box_idx, self.env_idx]

        # Observations (num_envs, 12)
        self.obs_buf[:] = torch.cat((
            self.joint_handler.dof_pos_value,
            self.last_dof_pos_value,
            ), dim=-1)

        # SAM2 is only available for a single instance
        if self.use_segmentation_for_inference and self.num_envs_int == 1:
            self.rgb_camera_handler.get_rgb_camera_images()
            img = self.rgb_camera_handler.rgb_images[:, :, :3]
            self.info["image"][0][:] = process_sam2(self.sam2, img)
        # For training we use automatically segmented images from sim
        else:
            self.rgb_camera_handler.get_segmented_rgb_camera_images()
            # Add random noise for sim2real
            # prepare_images_jit(self.info['image'], self.rgb_camera_handler.segmented_images,)
            self.info['image'][0][:] = self.rgb_camera_handler.segmented_images[0].permute(0, 3, 1, 2)


    def compute_rewards(self):
        extras = compute_rewards_jit(
            self.rew_buf
            )
        self.info["extras"] = extras

        # for k, v in extras.items():
        #     print(f"{k}: {v[0]}")

    def reset(self):
        self.reset_buf[:] = True

        # Reset all envs
        self.reset_idx()

        # Compute kinematics
        self.gym.compute_kinematics()
        self.gym.update_scene_dependent_components()

        # Return observations
        self.compute_observations()

        if self.debug:
            update_draw_targets(self)
            debug_save_images(self)

        return self.obs_buf, self.info

    def reset_idx(self):
        # Reset joint state
        self.joint_handler.set_positions(self.joint_handler.dof_pos_reset)
        self.joint_handler.set_velocities(self.joint_handler.dof_vel_reset)

        reset_idx_jit(self.state,
                    self.last_dof_pos_value,
                    self.dof_pos_home,
                    self.joint_handler.dof_pos_value,
                    self.reset_buf,
                    self.boxes_handler.pose_resets,
                    self.num_envs_int,
                    self.device,
                    self.box_poses_initial,
                    self.progress_buf,
                    self.camera_transform_initial,
                    self.rgb_camera_handler.transform_reset,
                    self.rew_guarantee,
                    self.stack_heights,
                    self.target_box_idx,
                    self.top_of_stack_box_idx,
                    self.iniitial_top_of_stack_pose,
                    self.has_fallen,
                    self.placed_flag,
                    # --- new ---
                    self.all_stack_boxes,
                    self.stack_indexes,
                    self.target_pose_idx,
                    self.stack_pose_idx,
                    self.target_xy_noise,
                    self.target_yaw_noise,
                    self.target_euler,
                    self.env_idx
                    )

        # Reset car, obstacles & camera with domain randomisation
        self.boxes_handler.set_states(self.boxes_handler.pose_resets, self.boxes_handler.velocity_targets)
        self.rgb_camera_handler.set_transform(self.rgb_camera_handler.transform_reset)

@torch.jit.script
def reset_idx_jit(
        state,
        last_dof_pos_value,
        dof_pos_home,
        dof_pos_value,
        reset_buf,
        box_pose_resets,
        num_envs_int: int,
        device: torch.device,
        box_poses_initial,
        progress_buf,
        camera_transform_initial,
        camera_transform_reset,
        rew_guarantee,
        stack_heights,
        target_box_idx,
        top_of_stack_box_idx,
        iniitial_top_of_stack_pose,
        has_fallen,
        placed_flag,
        # -- new --
        all_stack_boxes,
        stack_indexes,
        target_pose_idx,
        stack_pose_idx,
        target_xy_noise,
        target_yaw_noise,
        target_euler,
        env_idx,
        ):

    device = last_dof_pos_value.device

    resets = torch.nonzero(reset_buf)
    if resets.numel() == 0:
        return
    
    env_ids = env_idx[resets] 

    # --- Reset vartiables to default ---
    state[resets] = 0.0
    has_fallen[resets] = 0.0
    placed_flag[resets] = 0.0
    progress_buf[resets] = 0.0
    target_box_idx[resets] = 0
    last_dof_pos_value[resets] = dof_pos_home
    dof_pos_value[resets] = dof_pos_home
    rew_guarantee[resets] = 0.0

    # --- Random number (1–4) of stacked boxes ---
    num_stacked_boxes = torch.randint(1, 5, (num_envs_int,), device=device)
    # num_stacked_boxes = torch.tensor([4] * num_envs_int, device=device, dtype=torch.long)

    # Use boxes [1, 2, 3, 4] and mask out extras if not needed
    stack_mask = stack_indexes < num_stacked_boxes.unsqueeze(1)
    stacked_boxes = torch.where(stack_mask, all_stack_boxes, torch.full_like(all_stack_boxes, -1))

    # --- Randomize start positions for target and stack ---
    target_euler[:, 2:3] += target_yaw_noise
    box_pose_resets[target_box_idx, env_idx, 0:4] = v_quat_from_rpy(target_euler)
    box_pose_resets[target_box_idx, env_idx, 4:] = box_poses_initial[target_pose_idx, env_idx, 4:]
    box_pose_resets[target_box_idx, env_idx, 4:6] += target_xy_noise

    # --- Place stack boxes ---
    # Use base xy from stack start pose with small noise
    base_xy = box_poses_initial[stack_pose_idx, env_idx, 4:6]
    base_xy_noise = torch.normal(0.0, 0.02, base_xy.shape, device=device)
    base_xy = base_xy + base_xy_noise

    # Get flat tensors for pairwise indexing later
    valid = stacked_boxes != -1
    env_flat = env_idx.unsqueeze(1).expand_as(stacked_boxes)[valid]
    box_flat = stacked_boxes[valid]
    level_flat = torch.arange(4, device=device).unsqueeze(0).expand_as(stacked_boxes)[valid]

    box_pose_resets[box_flat, env_flat, 4:6] = base_xy[env_flat]
    box_pose_resets[box_flat, env_flat, 6] = stack_heights[level_flat]

    yaw_noise = torch.normal(0.0, 5.0 / 180.0 * math.pi, (box_flat.shape[0], 1), device=device)
    euler_stacked = v_rpy_from_quat(box_pose_resets[box_flat, env_flat, 0:4])
    euler_stacked[:, 2:3] += yaw_noise
    box_pose_resets[box_flat, env_flat, 0:4] = v_quat_from_rpy(euler_stacked)

    # --- Record top of stack ---
    top_of_stack_box_idx[:] = stacked_boxes[env_idx, num_stacked_boxes - 1]
    
    iniitial_top_of_stack_pose[:] = box_pose_resets[top_of_stack_box_idx, env_idx]

    # --- Move unused boxes out of the way ---
    safe_pose = torch.zeros_like(box_pose_resets)
    safe_pose[:, :, 3] = 1.0  # Valid quat
    safe_pose[:, :, 4] = 2.0  # Away
    safe_pose[:, :, 6] = 0.2  # Just above ground
    box_ids = torch.arange(5, device=device).unsqueeze(0).expand(num_envs_int, -1)
    combined = torch.cat((stacked_boxes, target_box_idx.unsqueeze(1)), dim=1)
    keep_mask = (box_ids.unsqueeze(1) == combined.unsqueeze(2)).any(dim=1)
    move_mask = ~keep_mask.T.unsqueeze(-1)  # (5, num_envs, 1)
    box_pose_resets[:] = torch.where(move_mask, safe_pose, box_pose_resets)

    # --- Camera randomization ---
    camera_transform_var = 0.002
    camera_angle_var = torch.pi * 1.0 / 180
    if num_envs_int == 1:
        camera_transform_var = 0.0
        camera_angle_var = 0.0

    camera_transform_noise = torch.normal(0.0, camera_transform_var, (num_envs_int, 3), device=device)
    camera_angle_noise = torch.normal(0.0, camera_angle_var, (num_envs_int, 3), device=device)
    camera_transform_reset[:, 4:] = camera_transform_initial[:, 4:] + camera_transform_noise
    camera_euler_val = v_rpy_from_quat(camera_transform_initial[:, :4]).clone()
    camera_euler_target = camera_euler_val + camera_angle_noise
    camera_transform_reset[:, :4] = v_quat_from_rpy(camera_euler_target)

@torch.jit.script
def prepare_images_jit(info_image: list[torch.Tensor],
                    image_buffers: list[torch.Tensor],
                    ):

    img = image_buffers[0].permute(0, 3, 1, 2)
    h = img.shape[-2]
    w = img.shape[-1]
    num_envs = img.shape[0]

    # One mask per segmentation mask in the image
    mask_1 = (img == 1)
    mask_2 = (img == 2)
    mask_3 = (img == 3)
    mask_4 = (img == 4)
    masks = torch.stack([mask_1, mask_2, mask_3, mask_4], dim=1)

    # Creating random numbers between -2 and 2
    # dx = torch.randint(-2, 3, (num_envs, 4), device=img.device) # high value is exclusive on upper bound
    # dy = torch.randint(-2, 3, (num_envs, 4), device=img.device)
    dx = torch.randint(-1, 2, (num_envs, 4), device=img.device) # smaller jitter
    dy = torch.randint(-1, 2, (num_envs, 4), device=img.device)

    # change the regular dx dy coordinates to the -1, 1 range
    shift_x_norm = dx * 2 / (w - 1)
    shift_y_norm = dy * 2 / (h - 1)

    # Need to change our actual pixel coordinates out for those -1, 1 values
    ys, xs = torch.meshgrid(
        torch.linspace(-1, 1, h, device=img.device),
        torch.linspace(-1, 1, w, device=img.device),
        indexing="ij", # makes the coordinate system row, column like array indexing
    )
    base_grid = torch.stack((xs, ys), dim=-1) # grid to map actual coordinates to -1, 1 values 
    grid = base_grid.unsqueeze(0).unsqueeze(1) # (1, 1, h, w, 2)
    grid = grid.repeat(num_envs, 4, 1, 1, 1) # (num_envs, 4, h, w, 2)

    # Specify new coordinates to move each mask
    grid[..., 0] = grid[..., 0] - shift_x_norm.unsqueeze(-1).unsqueeze(-1)
    grid[..., 1] = grid[..., 1] - shift_y_norm.unsqueeze(-1).unsqueeze(-1)

    # Change shapes for grid sample
    masks = masks.float().contiguous().view(num_envs * 4, 1, h, w)
    grid = grid.view(num_envs * 4, h, w, 2)

    # Perform shift
    masks_shifted = torch.nn.functional.grid_sample(
        masks,
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=True,
    )
    masks_shifted = masks_shifted.view(num_envs, 4, h, w)
    # masks_shifted has values 0 or 1, we get the index of the 1 for each mask and add 1 to get values 1,2,3,4
    presence = (masks_shifted > 0.5).any(dim=1) 
    labels_idx = torch.argmax(masks_shifted, dim=1) + 1
    segmentation = labels_idx * presence.to(labels_idx.dtype)

    # img[:] = segmentation
    img[:, 0, :, :] = segmentation.to(img.dtype)

    # # --- Add pixel mask static --- 

    # # Mask pixels become background
    # noise = torch.rand(img.shape, device=img.device)
    # mask_hit = ((noise > 0.95) & (img != 0))
    # img[:] = torch.where(mask_hit, 0, img)

    # # Background pixels become mask
    # mask_nohit = ((noise < 0.01) & (img == 0))
    # noise = torch.ceil(noise * 100 * 4)
    # img[:] = torch.where(mask_nohit, noise, img)

    # Copy to buffer
    info_image[0][:] = img


@torch.jit.script
def compute_rewards_jit(
        # Find vaiables from the environment
        rew_buf
        ):
    rew_buf.zero_()

    # Construct reward function components to teach the robot
    
    dummy_component = torch.ones_like(rew_buf)

    rew_buf[:] = rew_buf + dummy_component

    return {
        "dummy component": dummy_component,
        }
