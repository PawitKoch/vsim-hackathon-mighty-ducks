import os
import sys
import math

import torch
import vlearn as v
import numpy as np
from vlearn.spaces import Box
from vlearn.torch_utils.torch_jit_utils import v_quat_from_rpy, v_rpy_from_quat

# This should be the convention here on out
current_dir = os.path.dirname(os.path.abspath(__file__))
train_root = os.path.abspath(os.path.join(current_dir, "../"))
if train_root not in sys.path:
    sys.path.insert(0, train_root)

# Convention enables standardization of import structure, no more conditional switching
from envs.common import create_plane
from envs.environment import EnvironmentGpu
from envs.utils.sam2_utils import process_sam2
from envs.utils.so100.debug_parking import print_step_debug, init_debug_draw, update_draw_targets, debug_save_images
from vlearn.rl_algos.sac.vsim_training_data import *
from sam2.sam2wrapper import Sam2Wrapper
from vlearn.components.gym_command_handlers import (JointStateHandler,
                                                    KinematicSensorHandler,
                                                    RigidBodyHandler,
                                                    RGBCameraHandler,
                                                    ForceSensorHandler,
                                                    Context)
from envs.utils.so100.ik_handler import InverseKinematicsHandler

assert torch.cuda.is_available()
device = torch.device("cuda:0")


class SO100Parking(EnvironmentGpu):

    def __init__(self,
                 num_envs: int,
                 device: torch.device,
                 num_solver_iterations: int = 64,
                 venv_file: str = 'assets/SO100/parking.venv',
                 dof_pos_home: list = [],
                 car_pose_initial: list = [],
                 obstacle1_pose_initial: list = [],
                 obstacle2_pose_initial: list = [],
                 approach_quat: list = [],
                 robot_root_pos: list = [],
                 done_pose: list = [],
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

        assert (len(dof_pos_home) == 5)
        assert (len(car_pose_initial) == 7)

        num_dofs = 5
        self.end_effector_index = 5  # Link index of gripper

        device = torch.device(device)

        self.num_env_sets = 1
        self.state = torch.zeros((num_envs), device=device, dtype=torch.int32)  # Approaching car = 0, Picking up car = 1
        self.venv_file = venv_file
        self.dof_pos_home = torch.tensor([dof_pos_home], dtype=torch.float32, device=device)
        self.car_pose_initial = torch.tensor([car_pose_initial], dtype=torch.float32, device=device)
        self.obstacle1_pose_initial = torch.tensor([obstacle1_pose_initial], dtype=torch.float32, device=device)
        self.obstacle2_pose_initial = torch.tensor([obstacle2_pose_initial], dtype=torch.float32, device=device)
        self.approach_quat = torch.tensor([approach_quat], dtype=torch.float32, device=device)
        self.done_pose = torch.tensor([done_pose], dtype=torch.float32, device=device)
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

        self.num_envs_int = num_envs

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

        # Box "car"
        self.car_handle = env_def.get_rigid_body_handle_by_name("car")
        car = self.env_def.get_rigid_body(self.car_handle)
        car_transform_handle = car.get_transform_handle()

        # Obstacle boxes
        self.obstacle1_handle = env_def.get_rigid_body_handle_by_name("obstacle1")
        obstacle1 = self.env_def.get_rigid_body(self.obstacle1_handle)
        self.obstacle2_handle = env_def.get_rigid_body_handle_by_name("obstacle2")
        obstacle2 = self.env_def.get_rigid_body(self.obstacle2_handle)

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
        car.segmentation = 1  # leave zero as the background
        obstacle1.segmentation = 2
        obstacle2.segmentation = 3
        self.robot.set_segmentation(index=5, segmentation=4)
        self.robot.set_segmentation(index=6, segmentation=5)
        self.segmentation_arr = [
            car.segmentation,
            obstacle1.segmentation,
            obstacle2.segmentation,
            self.robot.get_segmentation(5),
            self.robot.get_segmentation(6)
            ]

        # Force sensing
        self.num_sensors = self.robot_def.get_num_force_sensor_defs()
        for force_sensor in self.robot_def.get_force_sensor_defs():
            force_sensor.max_num_transform_handles = 2

        # Contact filter used later to facilitate gripper feeling the table
        self.contact_filter_handles = [env_def.create_contact_filter([table_transform_handle])]
        self.car_contact_filter_handles = [env_def.create_contact_filter([car_transform_handle])]

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

        self.context = Context(self.gym, self.env_group, self.device, self.num_envs_int,)

        # Robot Joints
        self.joint_handler = JointStateHandler(self, self.robot_handle)
        self.last_dof_pos_value = torch.zeros_like(self.joint_handler.dof_pos_value)

        # Inverse Kinematics used in demonstrations
        kine_sensor_def = self.robot_def.get_kinematic_sensor_def(0)
        self.ee_sensor_local_offset = kine_sensor_def.offset
        ik_orn_dofs = [False, False, False, True, True]
        self.ik_handler = InverseKinematicsHandler(self,
                                                   self.robot_handle,
                                                   self.end_effector_index,
                                                   self.ee_sensor_local_offset,
                                                   ik_orn_dofs)

        # Kinematic sensor = end effector pose
        kine_sensor_handle = self.robot.get_kinematic_sensor_handle(0)
        self.kinematic_sensor_handler = KinematicSensorHandler(self, kine_sensor_handle)

        # Objects
        self.car_handler = RigidBodyHandler(self, self.car_handle)
        self.obstacle1_handler = RigidBodyHandler(self, self.obstacle1_handle)
        self.obstacle2_handler = RigidBodyHandler(self, self.obstacle2_handle)

        # Goal state
        self.goal_pose = torch.zeros((self.num_envs_int, 7), dtype=torch.float32, device=self.device)

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

        # Car force sensors
        self.jaw_car_sensor_handler = ForceSensorHandler(self,
                                                         jaw_force_sensor_handle,
                                                         self.car_contact_filter_handles)
        self.gripper_car_sensor_handler = ForceSensorHandler(self,
                                                             gripper_force_sensor_handle,
                                                             self.car_contact_filter_handles)
        self.jaw_car_sensor_handler.set_contact_filters()
        self.gripper_car_sensor_handler.set_contact_filters()

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
        self.car_handler.pose_reset[:] = self.car_pose_initial
        self.obstacle1_handler.pose_reset[:] = self.obstacle1_pose_initial
        self.obstacle2_handler.pose_reset[:] = self.obstacle2_pose_initial

        # Init target transforms
        self.ik_handler.ik_target_pose[:] = self.goal_pose

        # Initial camera transforms
        self.rgb_camera_handler.get_transform()
        self.camera_transform_initial[:] = self.rgb_camera_handler.transform_val

    def pre_physics_step(self, actions: torch.Tensor):
        self.act_buf[:] = actions
        # Don't move the wrist roll
        self.act_buf[:, -1] = 0.0

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

    def post_physics_step(self):
        # Eager reset
        self.progress_buf[:] += 1
        self.trunc_buf[:] = self.progress_buf >= self.max_episode_length
        self.reset_buf[:] = torch.logical_or(self.term_buf, self.trunc_buf)

        # Reset
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
        self.car_handler.get_state()
        self.obstacle1_handler.get_state()
        self.obstacle2_handler.get_state()
        self.gym.finalize_filtered_force_sensors()
        self.jaw_force_sensor_handler.get_sensor_forces()
        self.gripper_force_sensor_handler.get_sensor_forces()
        self.wrist_force_sensor_handler.get_sensor_forces()
        self.jaw_car_sensor_handler.get_sensor_forces()
        self.gripper_car_sensor_handler.get_sensor_forces()

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
            self.info['image'][0][:] = self.rgb_camera_handler.segmented_images[0].permute(0, 3, 1, 2)

    def compute_rewards(self):

        extras = compute_rewards_jit(self.rew_buf,
                                     )

        self.info["extras"] = extras

        # # Debugging
        # for k, v in extras.items():
        #     print(f"{k}:", v.mean())

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

        reset_idx_jit(self.last_dof_pos_value,
                      self.dof_pos_home,
                      self.joint_handler.dof_pos_value,
                      self.reset_buf,
                      self.car_handler.pose_reset,
                      self.obstacle1_handler.pose_reset,
                      self.obstacle2_handler.pose_reset,
                      self.num_envs_int,
                      self.device,
                      self.car_pose_initial,
                      self.obstacle1_pose_initial,
                      self.obstacle2_pose_initial,
                      self.progress_buf,
                      self.camera_transform_initial,
                      self.rgb_camera_handler.transform_reset,
                      self.goal_pose,
                      self.state
                      )

        # Reset car, obstacles & camera with domain randomisation
        self.car_handler.set_state(self.car_handler.pose_reset, self.car_handler.velocity_target)
        self.obstacle1_handler.set_state(self.obstacle1_handler.pose_reset, self.obstacle1_handler.velocity_target)
        self.obstacle2_handler.set_state(self.obstacle2_handler.pose_reset, self.obstacle2_handler.velocity_target)
        self.rgb_camera_handler.set_transform(self.rgb_camera_handler.transform_reset)


@torch.jit.script
def reset_idx_jit(last_dof_pos_value,
                  dof_pos_home,
                  dof_pos_value,
                  reset_buf,
                  car_pose_reset,
                  obstacle1_pose_reset,
                  obstacle2_pose_reset,
                  num_envs_int: int,
                  device: torch.device,
                  car_pose_initial,
                  obstacle1_pose_initial,
                  obstacle2_pose_initial,
                  progress_buf,
                  camera_transform_initial,
                  camera_transform_reset,
                  goal_pose,
                  state,
                  ):
    
    state[:] = torch.where(reset_buf, 0, state)

    # Reset last and get dof pos
    last_dof_pos_value[:] = torch.where(reset_buf.view(-1, 1),
                                        dof_pos_home,
                                        last_dof_pos_value)

    dof_pos_value[:] = torch.where(reset_buf.view(-1, 1),
                                   dof_pos_home,
                                   dof_pos_value)

    # Set initial target pose with noise and zero out vel
    car_pose_reset[:] = torch.where(reset_buf.view(-1, 1),
                                    car_pose_initial,
                                    car_pose_reset)

    obstacle1_pose_reset[:] = torch.where(reset_buf.view(-1, 1),
                                          obstacle1_pose_initial,
                                          obstacle1_pose_reset)
    obstacle2_pose_reset[:] = torch.where(reset_buf.view(-1, 1),
                                          obstacle2_pose_initial,
                                          obstacle2_pose_reset)

    # Randomize target position
    noise = torch.normal(0.0, 0.015, (num_envs_int, 2), device=device)
    car_pose_reset[:, 4:6] = torch.where(reset_buf.view(-1, 1),
                                         car_pose_reset[:, 4:6],
                                         car_pose_reset[:, 4:6])

    # Give obstacles positional noise but not angular
    obstacle_noise = torch.normal(0.0, 0.015, (num_envs_int, 2), device=device)
    # obstacle_noise = obstacle_noise + noise
    obstacle1_pose_reset[:, 4:6] = torch.where(reset_buf.view(-1, 1),
                                               obstacle1_pose_reset[:, 4:6] + obstacle_noise,
                                               obstacle1_pose_reset[:, 4:6])
    obstacle2_pose_reset[:, 4:6] = torch.where(reset_buf.view(-1, 1),
                                               obstacle2_pose_reset[:, 4:6] + obstacle_noise,
                                               obstacle2_pose_reset[:, 4:6])

    # --- If car placed behind space, move it in front of space ---
    behind_space_mask = reset_buf & (car_pose_reset[:, 5] < obstacle1_pose_reset[:, 5])
    behind_space_dist = obstacle1_pose_reset[:, 5] - car_pose_reset[:, 5]
    correction = behind_space_dist * 2
    new_y = car_pose_reset[:, 5] + correction
    car_pose_reset[:, 5] = torch.where(behind_space_mask, new_y, car_pose_reset[:, 5])

    euler = v_rpy_from_quat(car_pose_reset[:, 0:4])
    noise = torch.normal(0.0, 10.0 / 180.0 * torch.pi, (num_envs_int, 1), device=device)
    # euler[:, 2:3] += noise
    car_pose_reset[:, 0:4] = torch.where(
        reset_buf.view(-1, 1), v_quat_from_rpy(euler), car_pose_reset[:, 0:4])

    euler = v_rpy_from_quat(obstacle1_pose_reset[:, 0:4])
    noise = torch.normal(0.0, 3.0 / 180.0 * torch.pi, (num_envs_int, 1), device=device)
    euler[:, 2:3] += noise
    obstacle1_pose_reset[:, 0:4] = torch.where(
        reset_buf.view(-1, 1), v_quat_from_rpy(euler), obstacle1_pose_reset[:, 0:4])

    euler = v_rpy_from_quat(obstacle1_pose_reset[:, 0:4])
    euler[:, 2:3] += noise
    obstacle2_pose_reset[:, 0:4] = torch.where(
        reset_buf.view(-1, 1), v_quat_from_rpy(euler), obstacle2_pose_reset[:, 0:4])

    # --- Goal state update ---

    # Goal position
    goal_pose[:, 4:] = torch.where(
        reset_buf.view(-1, 1),
        0.5 * (obstacle1_pose_reset[:, 4:] + obstacle2_pose_reset[:, 4:]),
        goal_pose[:, 4:]
        )

    # Goal orientation
    goal_pose[:, :4] = torch.where(
        reset_buf.view(-1, 1),
        (obstacle1_pose_reset[:, :4]),
        goal_pose[:, :4]
        )

    # Randomise camera position
    camera_transform_var = 0.002  # 2mm
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

    # Reset progress
    progress_buf[:] = torch.where(reset_buf, 0, progress_buf)


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