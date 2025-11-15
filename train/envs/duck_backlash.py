import torch
import vlearn as v
import numpy as np
from math import ceil, cos, sin, floor
import json
import platform
import os
import matplotlib.pyplot as plt
import numpy as np


from vlearn.torch_utils.torch_jit_utils import v_quat_from_rpy, v_rpy_from_quat, quaternion_to_matrix
from vlearn.spaces import Box


if __name__ == "__main__":
    from common import create_plane
    from environment import EnvironmentGpu
else:
    from .common import create_plane
    from .environment import EnvironmentGpu

assert torch.cuda.is_available()


def linux_torch_compile(fn):
    if platform.system() == "Linux" and hasattr(torch, "compile"):
        return torch.compile(fn)
    return fn


HOME_POS_ARR = [
    # 0,
    # 0,
    # 0,
    # 0,
    0.002,
    0.053,
    -0.63,
    1.368,
    -0.784,
    -0.003,
    -0.065,
    0.635,
    1.379,
    -0.796,
    ]

ROOT_HOME_ARR = [0,
                 0,
                 0,
                 1,
                 0,
                 0,
                 0.1675]


class DuckEnv(EnvironmentGpu):
    def __init__(
            self,
            num_envs=1,
            device=torch.device("cuda:0"),
            env_name="assets/duck_mini_playground/duck_backlash.venv",
            rendering=True,
            enable_scene_query=True,
            max_episode_length=800,  # 8s at 100Hz sim
            timestep=1.0 / 100.0,
            frame_skip=2,
            apply_external_force: bool = True,
            spacing=0.5,
            gravity=v.Vec3(0, 0, -9.81),
            num_dofs=24,
            treat_warning_as_error=False,
            initial_is_paused=True,
            send_interrupt=False,
            print_hash=False,
            up_axis: v.Vec3 = v.Vec3(0, 0, 1),
            with_window=True,
            max_contact_pairs_per_env=256,
            max_patches_per_env=-1,
            max_contacts_per_patch=4,
            update_scene_dependent_components_in_step=False,
            return_clones=True,
            env_offset_mode="grid",
            sac_not_ppo=False,  # WARN: configure in your yaml!!!
            use_control_policy=False,
            ):
        assert max_episode_length % frame_skip == 0
        super().__init__(
            num_envs,
            device,
            rendering,
            enable_scene_query,
            max_episode_length // frame_skip,
            timestep,
            frame_skip,
            spacing,
            gravity,
            num_dofs,
            treat_warning_as_error,
            initial_is_paused,
            send_interrupt,
            print_hash,
            up_axis,
            with_window,
            max_contact_pairs_per_env=max_contact_pairs_per_env,
            max_patches_per_env=max_patches_per_env,
            max_contacts_per_patch=max_contacts_per_patch,
            update_scene_dependent_components_in_step=update_scene_dependent_components_in_step,
            return_clones=return_clones)

        self.use_control_policy = use_control_policy
        self.env_offset_mode = env_offset_mode
        self.sac_not_ppo = sac_not_ppo
        self.env_name = env_name
        self.num_links = 15
        self.num_active_dofs = 10
        self.apply_external_force = apply_external_force
        self.action_min_delay = 0  # env steps
        self.action_max_delay = 4  # env steps
        self.imu_min_delay = 0  # env steps
        self.imu_max_delay = 4  # env steps

        # Create environments
        self.create_envs()

        create_plane(self.gym, dynamic_friction=0.7, static_friction=0.9)

        # Finalize
        self.gym.gym_finalize()

        # Initialise observation and action space
        self.init_obs_and_act_spaces()

        # Allocate buffers
        self.allocate_buffers()

        # Store initial conditions
        self.store_initial_conditions()

        # Finalize gym
        self.gym.set_num_solver_iterations(64)
        self.gym.gym_finalize()

        # Camera
        if self.rendering:
            x = 0.5 * self.total_num_envs**0.5
            y = -2
            z = 2
            self.gym_render.reset_camera(v.Vec3(x, y, z), v.Vec3(0, 1, 0))

        if self.use_control_policy:
            assert not self.sac_not_ppo  # only for ppo atm
            self.init_ort_session()

    def init_ort_session(self):
        import onnxruntime as ort
        self.ort_session = ort.InferenceSession("policies/duck_v10/duck_gait.onnx",
                                                providers=["CUDAExecutionProvider"])

        onnx_in = self.ort_session.get_inputs()[0]
        onnx_out = self.ort_session.get_outputs()[0]
        print("tensor in: {}, onnx_in: {}".format(self.obs_gait.shape, onnx_in.shape))
        print("tensor in: {}, onnx_out: {}".format(self.actions_gait.shape, onnx_out.shape))

        # Create IO binding
        self.io_binding = self.ort_session.io_binding()

        # Bind input
        self.io_binding.bind_input(
            name=onnx_in.name,
            device_type='cuda',
            device_id=0,
            element_type=np.float32,
            shape=tuple(self.obs_gait.shape),
            buffer_ptr=self.obs_gait.data_ptr()
            )

        # Bind output
        self.io_binding.bind_output(
            name=onnx_out.name,
            device_type='cuda',
            device_id=0,
            element_type=np.float32,
            shape=tuple(self.actions_gait.shape),
            buffer_ptr=self.actions_gait.data_ptr()
            )

    def create_envs(self):
        # Environment def
        self.env_def_handle = self.gym.create_environment_def("duck_env")
        env_def = self.gym.get_environment_def(self.env_def_handle)
        self.env_def = env_def

        merge_fixed_joints = True
        merge_meshes_inside_files = True
        use_visual_meshes = False
        create_env_instances = False
        force_inertia_computation = False
        force_mass_computation = False
        env_def.import_environment(self.env_name,
                                   merge_fixed_joints,
                                   merge_meshes_inside_files,
                                   use_visual_meshes,
                                   create_env_instances,
                                   force_inertia_computation,
                                   force_mass_computation)

        # Articulation
        self.arti_def_handle = env_def.get_articulation_def_handle_by_name("duck")
        self.arti_def = env_def.get_articulation_def(self.arti_def_handle)
        self.arti_handle = env_def.get_articulation_handle_by_name("duck")

        self.arti_def.has_self_collisions = True

        # set target mass
        def get_sim_mass():
            sim_mass = 0.0
            for link_index in range(0, self.arti_def.get_num_link_defs()):
                link_def = self.arti_def.get_link_def(link_index)
                if "Dummy" in link_def.name:
                    continue
                sim_mass += link_def.mass
            return sim_mass

        real_mass = 2.3
        mass_ratio = real_mass / get_sim_mass()
        print("real to sim total mass ratio: ", mass_ratio)

        for link_index in range(0, self.arti_def.get_num_link_defs()):
            link_def = self.arti_def.get_link_def(link_index)
            if "Dummy" in link_def.name:
                continue
            link_def.mass *= mass_ratio
        print("scaling to real mass done, new sim mass: ", get_sim_mass())

        # get dof low and high limits for action space
        self.dof_pos_low_sim = []
        self.dof_pos_high_sim = []
        self.dof_dict = {}
        self.active_dof_dict = {}

        i = 0
        for dofdef in self.arti_def.get_joint_dof_defs():
            self.dof_pos_low_sim.append(dofdef.low_limit)
            self.dof_pos_high_sim.append(dofdef.high_limit)
            self.dof_dict[dofdef.name] = i

            if "backlash" not in dofdef.name and "head" not in dofdef.name and "neck" not in dofdef.name:
                self.active_dof_dict[dofdef.name] = i

            i += 1

        # active dof ids for index select
        self.active_dof_ids = torch.tensor(list(self.active_dof_dict.values()), dtype=torch.int, device=self.device)
        print("dof_dict: ", self.dof_dict)
        print("active_dof_dict: ", self.active_dof_dict)
        print("active_dof_ids: ", self.active_dof_ids)
        assert len(self.active_dof_ids) == self.num_active_dofs

        hip_ids = []
        knee_ids = []
        ankle_ids = []
        for i, dof in enumerate(self.active_dof_dict.keys()):
            if "hip" in dof:
                hip_ids.append(i)
            if "knee" in dof:
                knee_ids.append(i)
            if "ankle" in dof:
                ankle_ids.append(i)
        assert len(hip_ids) == 6
        assert len(knee_ids) == 2
        assert len(ankle_ids) == 2

        self.hip_ids = torch.tensor(hip_ids, device=self.device)
        self.knee_ids = torch.tensor(knee_ids, device=self.device)
        self.ankle_ids = torch.tensor(ankle_ids, device=self.device)

        # Load reference gait
        current_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(current_dir, "data/all_episodes_pawit.json"), 'r') as file:
            gait_data = json.load(file)

        id_map = build_id_map(gait_data, self.active_dof_dict)
        self.imitation_dt = gait_data["dt"]
        self.period = gait_data["period"]
        self.imitation_steps_per_period = gait_data["steps_per_period"]
        self.n_samples = gait_data["n_samples"]
        self.dx_dtheta = torch.tensor(gait_data["dx_dtheta"], dtype=torch.float32, device=self.device)
        self.imitation_frame_skip = 1
        self.dof_targets = torch.tensor(gait_data["joint_pos"], dtype=torch.float32, device=self.device)[:, :, id_map]
        self.foot_contacts_ref = torch.tensor(gait_data["foot_contacts"], dtype=torch.float32, device=self.device)
        assert self.dof_targets.shape[0] == self.n_samples
        assert self.foot_contacts_ref.shape[0] == self.n_samples
        assert self.dof_targets.shape[1] == self.imitation_steps_per_period
        assert self.foot_contacts_ref.shape[1] == self.imitation_steps_per_period

        # visualize a sample
        plot_polynomial(self.dof_targets[0], dt=self.imitation_dt)

        # self.dof_targets, self.dof_vel_targets, self.foot_contacts_ref = extract_gait_data(gait_data, self.active_dof_dict)
        # self.period = gait_data["gait_params"]["period"]
        # self.imitation_dt = gait_data["FrameDuration"]
        # self.imitation_frame_skip = round(self.dt / self.imitation_dt)
        # self.imitation_steps_per_period = floor(self.period / self.imitation_dt)
        # print("self.period: ", self.period)
        # print("self.imitation_steps_per_period: ", self.imitation_steps_per_period)
        # print("self.imitation_frame_skip: ", self.imitation_frame_skip)
        # print("len self.dof_targets: ", len(self.dof_targets))

        # data_dir = "runs/duck_reference_data"
        # if os.path.exists(data_dir):
        #     import shutil
        #     shutil.rmtree(data_dir)
        # os.makedirs(data_dir)
        # do_plot = False
        # self.dof_targets = self.extract_plot_reference_slice(self.dof_targets, data_dir, "dof_pos", do_plot)
        # self.dof_vel_targets = self.extract_plot_reference_slice(self.dof_vel_targets, data_dir, "dof_vel", do_plot)
        # self.foot_contacts_ref = self.extract_plot_reference_slice(self.foot_contacts_ref.float(), data_dir, "foot_contacts", do_plot)
        # self.foot_contacts_ref = (self.foot_contacts_ref > 0.5).float()
        # assert len(self.dof_targets) == self.imitation_steps_per_period
        # assert len(self.dof_vel_targets) == self.imitation_steps_per_period
        # assert len(self.foot_contacts_ref) == self.imitation_steps_per_period

        # # Helps to smooth the signal a bit
        # polys = fit_polynomial(self.dof_targets, self.imitation_dt)
        # self.dof_targets = eval_plot_polynomial(polys, self.dof_targets, dt=self.imitation_dt).to(self.device)
        # assert len(self.dof_targets) == self.imitation_steps_per_period

        self.foot_ids = {}
        for id, link_def in enumerate(self.arti_def.get_link_defs()):
            if "foot" in link_def.name:
                self.foot_ids[link_def.name] = id
        print("foot_ids: ", self.foot_ids)
        assert len(self.foot_ids) == 2

        # Offsets
        env_set_offsets = []
        if isinstance(self.num_envs, list):
            self.num_env_sets = len(self.num_envs)
            envs_per_set = self.num_envs[0]
        elif isinstance(self.num_envs, int):
            self.num_env_sets = 1
            envs_per_set = self.num_envs
        else:
            raise NotImplementedError

        for i in range(self.num_env_sets):
            dim_set = ceil(envs_per_set**0.5)
            N = ceil(self.num_env_sets**0.5)
            x = (i % N) * dim_set
            y = (i // N) * dim_set
            offset = self.spacing * v.Vec3(x, y, 0)
            env_set_offsets.append(offset)

        self.articulation = env_def.get_articulation(self.arti_handle)

        # Finalize environment def
        env_def.finalize()

        super().create_envs(self.env_def_handle, env_set_offsets=env_set_offsets, mode=self.env_offset_mode)

    def init_obs_and_act_spaces(self):
        self.num_obs = 0
        self.num_obs += 3  # gyroscope
        self.num_obs += 3  # accelerometer
        self.num_obs += 3  # lin + ang vel cmd
        self.num_obs += self.num_active_dofs  # dof pos
        self.num_obs += self.num_active_dofs  # dof vel
        self.num_obs += self.num_active_dofs * (self.action_max_delay - self.action_min_delay)  # action history
        self.num_obs += self.num_active_dofs  # pid targets
        self.num_obs += 2  # feet contacts
        self.num_obs += 2  # phase cos, sin

        self.num_obs_gait = self.num_obs
        if self.use_control_policy:
            self.num_obs = 9

        # Privileged
        self.priv_obs_handle = self.num_obs
        # self.num_obs += 3  # gyroscope
        # self.num_obs += 3  # accelerometer
        # self.num_obs += self.num_active_dofs  # dof pos
        # self.num_obs += self.num_active_dofs  # dof vel
        self.num_obs += 3  # root lin vel
        self.num_obs += 1  # root z
        self.num_obs += self.num_active_dofs  # joint forces
        self.num_obs += 4 * 3  # feet velocities
        self.num_obs += self.num_active_dofs  # imitation joint targets

        if not self.sac_not_ppo:  # ppo
            self.num_states = self.num_obs - self.priv_obs_handle
            self.num_obs = self.priv_obs_handle

            self.single_state_space = Box(
                low=np.array([np.finfo('f').min] * self.num_states, dtype=np.float32),
                high=np.array([np.finfo('f').max] * self.num_states, dtype=np.float32),
                dtype=np.float32)

        print('num_obs: ', self.num_obs)
        print('priv_obs_handle: ', self.priv_obs_handle)

        self.single_observation_space = Box(
            low=np.array([np.finfo('f').min] * self.num_obs, dtype=np.float32),
            high=np.array([np.finfo('f').max] * self.num_obs, dtype=np.float32),
            dtype=np.float32)

        # import hardware dof ranges
        current_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(current_dir, "data/duck_pid_limits.json"), 'r') as file:
            data_dict = json.load(file)

        names = list(data_dict.keys())
        assert names == list(self.active_dof_dict.keys())
        self.dof_pos_high = []
        self.dof_pos_low = []
        for _, v in data_dict.items():
            self.dof_pos_high.append(v["high"])
            self.dof_pos_low.append(v["low"])
        print("data_dict joint names ", names)
        print("dof_pos_high    ", self.dof_pos_high)
        print("dof_pos_low     ", self.dof_pos_low)
        print("dof_pos_low_sim ", torch.as_tensor(self.dof_pos_low_sim)[self.active_dof_ids.cpu()])
        print("dof_pos_low_sim ", torch.as_tensor(self.dof_pos_low_sim)[self.active_dof_ids.cpu()])

        # take the 80% of the ranges - this avoids crash in real
        delta = [(h - l) for h, l in zip(self.dof_pos_high, self.dof_pos_low)]

        self.dof_pos_low = [l + d * 0.1 for l, d in zip(self.dof_pos_low, delta)]
        self.dof_pos_high = [h - d * 0.1 for h, d in zip(self.dof_pos_high, delta)]

        assert len(self.dof_pos_low) == self.num_active_dofs
        assert len(self.dof_pos_high) == self.num_active_dofs

        print("self.dof_pos_low: ", self.dof_pos_low)
        print("self.dof_pos_high: ", self.dof_pos_high)

        # torch tensors for ease of use
        self.pid_low = torch.tensor(self.dof_pos_low, dtype=torch.float32, device=self.device)
        self.pid_high = torch.tensor(
            self.dof_pos_high, dtype=torch.float32, device=self.device)

        print("pid_low ", self.pid_low)
        print("pid_high ", self.pid_high)

        assert len(self.pid_low) == self.num_active_dofs
        assert len(self.pid_high) == self.num_active_dofs

        # # actions spaces
        # delta_rad_max = 20.0 / 180.0 * torch.pi
        # delta = [delta_rad_max] * self.num_dofs

        # self.pid_delta_pos = torch.tensor([delta], dtype=torch.float32, device=self.device)
        # print("pid_delta_pos rad ", self.pid_delta_pos)
        # print("pid_delta_pos deg ", self.pid_delta_pos * 180.0 / torch.pi)

        # assert len(self.pid_delta_pos[0]) == self.num_dofs

        # # define action space
        # self.single_action_space = Box(
        #     low=np.array([-d for d in delta], dtype=np.float32),
        #     high=np.array(delta, dtype=np.float32),
        #     dtype=np.float32)

        # define action space
        # delta = 30.0 / 180.0 * torch.pi
        delta = 0.25  # almost 15 degreees
        print("action range min max: ", delta)
        self.single_action_space = Box(
            low=np.array([-delta] * self.num_active_dofs, dtype=np.float32),
            high=np.array([delta] * self.num_active_dofs, dtype=np.float32),
            dtype=np.float32)

        ####################################################################################################
        # HACKATHON TODO
        ####################################################################################################
        if self.use_control_policy:
            # x, theta cmd, delta x pos
            # WARN: be careful if changing action 1 or 2, which are the control commands
            low = [0.0, -0.5, -1.0]
            high = [0.15, 0.5, 1.0]
            self.single_action_space = Box(
                low=np.array(low, dtype=np.float32),
                high=np.array(high, dtype=np.float32),
                dtype=np.float32)

        print('single_observation_space: ', self.single_observation_space.shape)
        print('single_action_space: ', self.single_action_space.shape)

    def allocate_buffers(self):

        super().allocate_buffers()

        # root home
        self.root_home = torch.as_tensor(ROOT_HOME_ARR, device=self.device)
        print("root_home ", self.root_home)

        # clamp the home_pos at the limits
        self.home_pos_active = torch.as_tensor(HOME_POS_ARR, device=self.device)
        print("home_pos_active before clamp ", self.home_pos_active)
        self.home_pos_active[:] = torch.clamp(
            self.home_pos_active,
            self.pid_low,
            self.pid_high)
        print("home_pos_active after clamp ", self.home_pos_active)

        # to avoid slicing everything
        self.home_pos = torch.tensor([0.0] * self.num_dofs, dtype=torch.float32, device=self.device)
        self.home_pos[self.active_dof_ids] = self.home_pos_active

        # Extra buffers
        self.obs_gait = torch.zeros((self.total_num_envs, self.num_obs_gait), dtype=torch.float32, device=self.device)

        self.actions_gait = torch.zeros((self.total_num_envs, self.num_active_dofs), dtype=torch.float32, device=self.device)
        self.action_hist = self.actions_gait.unsqueeze(1).repeat(1, self.action_max_delay - self.action_min_delay, 1)

        self.pos_actions = torch.zeros((self.total_num_envs, 3), dtype=torch.float32, device=self.device)

        self.imu_gyro_hist = torch.zeros(
            (self.total_num_envs, self.imu_max_delay - self.imu_min_delay, 3), dtype=torch.float32, device=self.device)

        self.imu_accelero_hist = torch.zeros(
            (self.total_num_envs, self.imu_max_delay - self.imu_min_delay, 3), dtype=torch.float32, device=self.device)

        # assume standing at start
        self.left_foot_contact = torch.ones((self.total_num_envs, ), dtype=torch.bool, device=self.device)
        self.right_foot_contact = torch.ones((self.total_num_envs, ), dtype=torch.bool, device=self.device)

        # Velocity commands
        self.sample_ids = torch.randint(low=0, high=self.n_samples, size=(self.total_num_envs, ), device=self.device)

        self.vel_xy_cmd = torch.zeros(
            (self.total_num_envs, 2), dtype=torch.float32, device=self.device)

        self.yaw_rate_cmd = torch.zeros(
            (self.total_num_envs, 1), dtype=torch.float32, device=self.device)

        self.yaw_cmd = torch.zeros(
            (self.total_num_envs, 1), dtype=torch.float32, device=self.device)

        # PID targets
        self.set_pid_buf = torch.zeros(
            (self.total_num_envs,
             self.num_dofs),
            dtype=torch.float32,
            device=self.device)

        set_pid_cmd = self.env_group.create_pid_control_command(
            v.wrap_gpu_buffer(self.set_pid_buf), self.arti_handle)

        self.set_pid_cmd_arr = self.gym.create_pid_control_command_gpu_array([set_pid_cmd])

        # Get feet transforms
        self.get_foot_1_transform_buf = torch.zeros(
            (self.total_num_envs, 7), dtype=torch.float32, device=self.device)

        self.get_foot_2_transform_buf = torch.zeros(
            (self.total_num_envs, 7), dtype=torch.float32, device=self.device)

        foot_id = self.foot_ids["foot_assembly"]
        get_foot_transform_cmd_1 = self.env_group.create_link_transform_command(
            v.wrap_gpu_buffer(self.get_foot_1_transform_buf),
            self.arti_handle,
            (foot_id, foot_id + 1),
            transform_type=v.TransformType.MODEL,
            frame_type=v.FrameType.ENVIRONMENT)

        foot_id = self.foot_ids["foot_assembly_2"]
        get_foot_transform_cmd_2 = self.env_group.create_link_transform_command(
            v.wrap_gpu_buffer(self.get_foot_2_transform_buf),
            self.arti_handle,
            (foot_id, foot_id + 1),
            transform_type=v.TransformType.MODEL,
            frame_type=v.FrameType.ENVIRONMENT)

        self.get_foot_transform_cmd_arr = self.gym.create_link_transform_command_gpu_array([
            get_foot_transform_cmd_1, get_foot_transform_cmd_2])

        # Last foot position
        self.foot_1_xy = self.get_foot_1_transform_buf[:, 4:6].clone()
        self.foot_2_xy = self.get_foot_2_transform_buf[:, 4:6].clone()

        # Get articulation states
        self.get_root_vel_buf = torch.zeros(
            (self.total_num_envs, 6), dtype=torch.float32, device=self.device)
        self.get_dof_pos_buf[:] = self.home_pos.view(1, -1)

        self.last_dof_pos = self.get_dof_pos_buf.clone()
        self.last_last_dof_pos = self.get_dof_pos_buf.clone()

        self.get_root_transform_buf = torch.zeros(
            (self.total_num_envs, 7), dtype=torch.float32, device=self.device)
        self.get_root_transform_buf[:] = self.root_home.view(1, -1)

        self.last_root_transform = self.get_root_transform_buf.clone()
        self.last_last_root_transform = self.get_root_transform_buf.clone()

        get_kinematic_state_command = self.env_group.create_articulation_kinematic_state_command(
            v.wrap_gpu_buffer(self.get_dof_pos_buf),
            v.wrap_gpu_buffer(self.get_dof_vel_buf),
            v.wrap_gpu_buffer(self.get_root_transform_buf),
            v.wrap_gpu_buffer(self.get_root_vel_buf),
            self.arti_handle,
            link_index_range=(0, 1))

        self.get_kinematic_state_command_array = self.gym.create_articulation_kinematic_state_command_gpu_array(
            [get_kinematic_state_command])

        get_link_vel_cmd = self.env_group.create_link_velocity_command(
            v.wrap_gpu_buffer(self.get_root_vel_buf),
            self.arti_handle,
            index_range=(0, 1),
            frame_type=v.FrameType.LOCAL)

        self.get_link_vel_cmd_arr = self.gym.create_link_velocity_command_gpu_array([get_link_vel_cmd])

        # Reset articulation
        self.set_dof_pos_buf[:] = self.home_pos.view(1, -1)
        self.set_dof_vel_buf[:] = 0.0

        self.set_root_transform_buf = torch.zeros(
            (self.total_num_envs, 7), dtype=torch.float32, device=self.device)
        self.set_root_transform_buf[:] = self.root_home.view(1, -1)

        self.set_root_vel_buf = torch.zeros(
            (self.total_num_envs, 6), dtype=torch.float32, device=self.device)

        set_kinematic_state_command = self.env_group.create_articulation_kinematic_state_command(
            v.wrap_gpu_buffer(self.set_dof_pos_buf),
            v.wrap_gpu_buffer(self.set_dof_vel_buf),
            v.wrap_gpu_buffer(self.set_root_transform_buf),
            v.wrap_gpu_buffer(self.set_root_vel_buf),
            self.arti_handle,
            link_index_range=(0, 1),
            masks_buffer=v.wrap_gpu_buffer(self.reset_buf))

        self.set_kinematic_state_command_array = self.gym.create_articulation_kinematic_state_command_gpu_array(
            [set_kinematic_state_command])

        # Force sensors
        num_fs = 15

        # make sure the order is right
        self.force_sensor_dict = {}
        for id, fs in enumerate(self.arti_def.get_force_sensor_defs()):
            self.force_sensor_dict[id] = fs.name

        env_def = self.gym.get_environment_def(self.env_def_handle)
        articulation = env_def.get_articulation(self.arti_handle)

        force_sensor_feet_names = []

        self.force_sensor_feet_buf = [None, None]
        self.force_sensor_other_buf = []
        self.force_sensor_cmds = []
        for i in range(num_fs):
            name = self.force_sensor_dict[i]
            fs_handle = articulation.get_force_sensor_handle(i)

            if name in ["fs_foot_left", "fs_foot_right"]:
                id = -1
                if name == "fs_foot_left":
                    id = 0
                if name == "fs_foot_right":
                    id = 1

                self.force_sensor_feet_buf[id] = torch.zeros(
                    (self.total_num_envs, 6), device=self.device, dtype=torch.float32)

                self.force_sensor_cmds.append(self.env_group.create_force_sensor_command(
                    v.wrap_gpu_buffer(self.force_sensor_feet_buf[id]),
                    fs_handle))

                force_sensor_feet_names.append(name)

            else:
                self.force_sensor_other_buf.append(torch.zeros(
                    (self.total_num_envs, 6), device=self.device, dtype=torch.float32))

                self.force_sensor_cmds.append(self.env_group.create_force_sensor_command(
                    v.wrap_gpu_buffer(self.force_sensor_other_buf[-1]),
                    fs_handle))

        print("force_sensor_feet_names: ", force_sensor_feet_names)
        assert force_sensor_feet_names == [
            "fs_foot_left", "fs_foot_right"] or force_sensor_feet_names == [
            "fs_foot_right", "fs_foot_left"]

        self.get_force_sensors_command_array = self.gym.create_force_sensor_command_gpu_array(
            self.force_sensor_cmds)

        self.get_joint_sensor_buf = torch.zeros(
            (self.total_num_envs, self.num_dofs), dtype=torch.float32, device=self.device)

        get_joint_sensor_cmd = self.env_group.create_joint_force_sensor_command(
            v.wrap_gpu_buffer(self.get_joint_sensor_buf),
            self.arti_handle)

        self.get_joint_sensor_cmd_arr = self.gym.create_joint_force_sensor_command_gpu_array([
            get_joint_sensor_cmd])

        # Force & torque buffer: shape is (num_envs, num_links=1, 6)
        self.set_force_torque_buf = torch.zeros(
            (self.total_num_envs, 6), dtype=torch.float32, device=self.device)

        # Create command
        set_force_torque_cmd = self.env_group.create_link_external_force_command(
            v.wrap_gpu_buffer(self.set_force_torque_buf),
            self.arti_handle,
            [0, 1],
            force_type=v.ForceType.FORCE_TORQUE)

        self.set_force_torque_cmd_arr = self.gym.create_link_external_force_command_gpu_array([
            set_force_torque_cmd])

        self.obs_recording = torch.torch.zeros(
            (self.max_episode_length,
             self.num_obs),
            dtype=torch.float32,
            device=self.device)

    def store_initial_conditions(self):
        self.gravity_vec = torch.tensor([0.0, 0.0, -9.81], dtype=torch.float32, device=self.device).view(1, -1, 1)
        self.idx = 0

    def reset(self):
        self.reset_buf[:] = True

        # Reset all envs
        self.reset_idx()

        # Compute kinematics
        self.gym.compute_kinematics()
        self.gym.update_scene_dependent_components()

        # Return observations
        self.compute_observation_reward_termination()

        if self.sac_not_ppo:
            return self.obs_buf, self.info

        else:  # ppo
            return {"obs": self.obs_buf.clone(), "states": self.state_buf.clone()}, {}

    def reset_idx(self):
        # randomize dof pos
        noise = torch.randn_like(self.set_dof_pos_buf[:, self.active_dof_ids]) * 1.0 / 180.0 * torch.pi
        self.set_dof_pos_buf[:, self.active_dof_ids] = torch.where(self.reset_buf.view(-1, 1),
                                                                   noise + self.home_pos_active.view(1, -1),
                                                                   self.set_dof_pos_buf[:, self.active_dof_ids])

        # # randomize dof vel
        # noise = torch.randn_like(self.set_dof_vel_buf[:, self.active_dof_ids]) * 0.5  # max vel = 5.24
        # self.set_dof_vel_buf[:, self.active_dof_ids] = torch.where(self.reset_buf.view(-1, 1),
        #                                                            noise,
        #                                                            self.set_dof_vel_buf[:, self.active_dof_ids])

        self.gym.set_articulation_kinematic_states(self.set_kinematic_state_command_array)

        self.get_dof_pos_buf[:] = torch.where(self.reset_buf.view(-1, 1),
                                              self.home_pos,
                                              self.get_dof_pos_buf)

        self.last_dof_pos[:] = torch.where(self.reset_buf.view(-1, 1),
                                           self.home_pos,
                                           self.last_dof_pos)

        self.last_last_dof_pos[:] = torch.where(self.reset_buf.view(-1, 1),
                                                self.home_pos,
                                                self.last_last_dof_pos)

        self.get_root_transform_buf[:] = torch.where(self.reset_buf.view(-1, 1),
                                                     self.root_home,
                                                     self.get_root_transform_buf)

        self.last_root_transform = torch.where(self.reset_buf.view(-1, 1),
                                               self.root_home,
                                               self.last_root_transform)

        self.last_last_root_transform = torch.where(self.reset_buf.view(-1, 1),
                                                    self.root_home,
                                                    self.last_last_root_transform)

        self.action_hist[:] = torch.where(self.reset_buf.view(-1, 1, 1),
                                          0.0,
                                          self.action_hist)

        self.imu_gyro_hist[:] = torch.where(self.reset_buf.view(-1, 1, 1),
                                            0.0,
                                            self.imu_gyro_hist)

        self.imu_accelero_hist[:] = torch.where(self.reset_buf.view(-1, 1, 1),
                                                0.0,
                                                self.imu_accelero_hist)

        # random starting phase
        self.progress_buf[:] = torch.where(self.reset_buf,
                                           torch.randint_like(self.progress_buf, low=0, high=self.imitation_steps_per_period),
                                           self.progress_buf)

        # Velocity command
        if not self.use_control_policy:
            # get samples from the trajectories
            self.sample_ids[:] = torch.where(self.reset_buf,
                                             torch.randint_like(self.sample_ids, low=0, high=self.n_samples),
                                             self.sample_ids)

            noise = torch.randn_like(self.vel_xy_cmd[:, 0]) * 0.15 / 100.0  # 1% of range
            self.vel_xy_cmd[:, 0] = torch.where(self.reset_buf,
                                                self.dx_dtheta[self.sample_ids, 0] + noise,
                                                self.vel_xy_cmd[:, 0])
            # print("self.vel_xy_cmd ", self.vel_xy_cmd)

            noise = torch.randn_like(self.yaw_rate_cmd[:, 0]) * 1.0 / 100.0  # 1% of range
            self.yaw_rate_cmd[:, 0] = torch.where(self.reset_buf,
                                                  self.dx_dtheta[self.sample_ids, 1] + noise,
                                                  self.yaw_rate_cmd[:, 0])

        self.imitation_steps = (self.progress_buf * self.imitation_frame_skip) % self.imitation_steps_per_period
        self.q_ref = self.dof_targets[self.sample_ids, self.imitation_steps, :]
        # self.dq_ref = self.dof_vel_targets[self.imitation_steps, :]
        self.feet_ref = self.foot_contacts_ref[self.sample_ids, self.imitation_steps, :]

    def step(self, actions: torch.Tensor):
        if self.sac_not_ppo or self.use_control_policy:
            return super().step(actions)

        else:  # ppo
            super().step(actions)
            return ({"obs": self.obs_buf.clone(), "states": self.state_buf.clone()},
                    self.rew_buf.clone(), self.term_buf.clone(), self.trunc_buf.clone(), {})

    @linux_torch_compile
    def pre_physics_step(self, actions: torch.Tensor):
        if self.use_control_policy:
            self.act_buf[:] = actions

            self.vel_xy_cmd[:, 0] = actions[:, 0]
            self.yaw_rate_cmd[:, 0] = actions[:, 1]
            # self.pos_actions = actions[:, 2:5]

            # Run inference (everything stays on GPU)
            # input: self.obs_gait, output: self.actions_gait
            self.ort_session.run_with_iobinding(self.io_binding)

        else:
            self.actions_gait[:] = actions

        self.action_hist[:] = self.action_hist.roll(-1, 1)  # roll left along dim 1
        self.action_hist[:, -1, :] = self.actions_gait  # assign last

        # random delay
        rand_idx = torch.randint(low=self.action_min_delay, high=self.action_max_delay, size=(self.total_num_envs, ), device=self.device)
        pid_deltas = self.action_hist[torch.arange(self.total_num_envs), rand_idx, :]

        # apply delta joint pos
        self.set_pid_buf[:, self.active_dof_ids] = self.home_pos_active + pid_deltas

        # clamp the sim at the limits
        self.set_pid_buf[:, self.active_dof_ids] = torch.clamp(
            self.set_pid_buf[:, self.active_dof_ids],
            self.pid_low.view(1, -1),
            self.pid_high.view(1, -1))

        # update pid targets
        self.gym.set_joint_target_positions(self.set_pid_cmd_arr)

        # push that duck
        if self.apply_external_force:
            force = torch.normal(
                mean=0.0, std=0.5, size=(self.total_num_envs, 3), device=self.device)
            should_push = ((self.progress_buf + 0) % 50) == 0
            self.set_force_torque_buf[:, 3:6] = (should_push).float().view(-1, 1) * force
            self.gym.set_link_external_forces(self.set_force_torque_cmd_arr)

    @linux_torch_compile
    def post_physics_step(self):
        # Eager reset
        self.progress_buf[:] += 1
        self.reset_buf[:] = torch.logical_or(self.term_buf, self.trunc_buf)

        # Reset
        self.reset_idx()

        # Kinematics and queries
        self.gym.compute_kinematics()
        self.gym.update_scene_dependent_components()

        # Observations
        self.idx += 1
        self.compute_observation_reward_termination()

        self.trunc_buf[:] = self.progress_buf >= self.max_episode_length

    def compute_observation_reward_termination(self):
        ####################################################################################################
        ## START OF KINEMATICS UPDATE ##
        # HACKATHON TODO
        ####################################################################################################
        self.last_last_dof_pos = self.last_dof_pos.clone()
        self.last_dof_pos = self.get_dof_pos_buf.clone()

        self.last_last_root_transform = self.last_root_transform.clone()
        self.last_root_transform = self.get_root_transform_buf.clone()
        self.last_root_rpy = v_rpy_from_quat(self.get_root_transform_buf[:, 0:4])

        last_foot_1_tr = self.get_foot_1_transform_buf.clone()
        last_foot_2_tr = self.get_foot_2_transform_buf.clone()
        last_foot_1_rpy = v_rpy_from_quat(self.get_foot_1_transform_buf[:, 0:4])
        last_foot_2_rpy = v_rpy_from_quat(self.get_foot_2_transform_buf[:, 0:4])

        last_root_lin_vel = self.get_root_vel_buf[:, 3:6].clone()

        # update
        self.gym.get_link_transforms(self.get_foot_transform_cmd_arr)
        self.gym.get_articulation_kinematic_states(self.get_kinematic_state_command_array)
        if self.total_num_envs == 1:
            print("root_lin_vel global", self.get_root_vel_buf[0, 3:6])
            print("root_ang_vel global", self.get_root_vel_buf[0, 0:3])
        self.gym.get_link_velocities(self.get_link_vel_cmd_arr)  # WARN: overwrite root velocity buffer
        if self.total_num_envs == 1:
            print("root_lin_vel local", self.get_root_vel_buf[0, 3:6])
            print("root_ang_vel local", self.get_root_vel_buf[0, 0:3])
        self.gym.get_sensor_forces(self.get_force_sensors_command_array)
        self.gym.get_joint_sensor_forces(self.get_joint_sensor_cmd_arr)

        # post update
        dof_vel = (self.get_dof_pos_buf - self.last_dof_pos) / self.dt
        last_dof_vel = (self.last_dof_pos - self.last_last_dof_pos) / self.dt
        dof_acc = (dof_vel - last_dof_vel) / self.dt

        # WARN: using local linear velocity BUT global angular velocity
        # Linear
        # root_lin_vel = (self.get_root_transform_buf[:, 4:7] - self.last_root_transform[:, 4:7]) / self.dt
        # last_root_lin_vel = (self.last_root_transform[:, 4:7] - self.last_last_root_transform[:, 4:7]) / self.dt
        root_lin_vel = self.get_root_vel_buf[:, 3:6].clone()
        root_lin_acc = (root_lin_vel - last_root_lin_vel) / self.dt
        # Angular
        self.root_rpy = v_rpy_from_quat(self.get_root_transform_buf[:, 0:4])
        root_ang_vel = (self.root_rpy - self.last_root_rpy) / self.dt
        # root_ang_vel = self.get_root_vel_buf[:, 0:3].clone()

        # Feet
        foot_1_vel = (self.get_foot_1_transform_buf[:, 4:7] - last_foot_1_tr[:, 4:7]) / self.dt
        foot_2_vel = (self.get_foot_2_transform_buf[:, 4:7] - last_foot_2_tr[:, 4:7]) / self.dt
        foot_1_rpy = v_rpy_from_quat(self.get_foot_1_transform_buf[:, 0:4])
        foot_2_rpy = v_rpy_from_quat(self.get_foot_2_transform_buf[:, 0:4])
        foot_1_ang_vel = (foot_1_rpy - last_foot_1_rpy) / self.dt
        foot_2_ang_vel = (foot_2_rpy - last_foot_2_rpy) / self.dt

        last_left_foot_contact = self.left_foot_contact.clone()
        last_right_foot_contact = self.right_foot_contact.clone()
        self.left_foot_contact = torch.sum(self.force_sensor_feet_buf[0], dim=1) > 1e-3
        self.right_foot_contact = torch.sum(self.force_sensor_feet_buf[1], dim=1) > 1e-3
        ####################################################################################################
        ## END OF KINEMATICS UPDATE ##
        ####################################################################################################

        ####################################################################################################
        ## START OF REWARDS ##
        # HACKATHON TODO
        ####################################################################################################
        
        ########################################
        # SIMPLE L1 LOSS APPROACH (FIRST PRINCIPLES)
        ########################################
        # Simple L1 loss between commanded and actual velocities
        w_lin_vel_tracking = 10.0  # Strong weight for linear velocity tracking
        w_ang_vel_tracking = 5.0   # Weight for angular velocity tracking
        w_alive = 20.0             # Stay upright bonus
        w_joint_pos = -10.0        # Imitation learning component
        
        # L1 loss for linear velocity (x-axis, backwards motion)
        lin_vel_error = torch.abs(root_lin_vel[:, 0] - self.vel_xy_cmd[:, 0])
        lin_vel_tracking = -w_lin_vel_tracking * lin_vel_error
        
        # L1 loss for angular velocity (yaw rate)
        ang_vel_error = torch.abs(root_ang_vel[:, 2] - self.yaw_rate_cmd[:, 0])
        ang_vel_tracking = -w_ang_vel_tracking * ang_vel_error
        
        # Joint position imitation (helps with learning proper gait)
        joint_pos_rew = w_joint_pos * torch.linalg.vector_norm(self.q_ref - self.get_dof_pos_buf[:, self.active_dof_ids], dim=1)
        
        ## TERMINATION ##
        lim = 40.0
        roll_error_deg = torch.abs(self.root_rpy[:, 0]) * 180.0 / torch.pi
        pitch_error_deg = torch.abs(self.root_rpy[:, 1]) * 180.0 / torch.pi
        self.term_buf[:] = (roll_error_deg > lim) | (pitch_error_deg > lim) | (self.get_root_transform_buf[:, 6] < 0.08)
        
        alive = w_alive * (~self.term_buf).float()
        
        ## ASSEMBLE REWARDS ##
        self.rew_buf[:] = alive + lin_vel_tracking + ang_vel_tracking + joint_pos_rew
        
        ########################################
        # COMMENTED OUT: PREVIOUS COMPLEX REWARD
        ########################################
        # w_lin_vel_rew = 3.0  # Increased - backwards walking is the main goal
        # w_lin_vel_pen = -8.0  # Increased - stronger penalty for sideways drift
        # w_ang_vel_rew = 4.0  # Slightly reduced since turning less critical for backwards
        # w_ang_vel_pen = -0.001  # Increased - backwards walking requires more stability
        # w_acc_pen = -0.003  # Increased - smoother motion for backwards stability
        # w_vel_pen = -0.05  # Increased - discourage jerky movements
        # w_action_pen = -1.5  # Increased - smoother actions are critical for backwards walking
        # w_alive = 20.0  # Increased - staying upright is harder when going backwards
        # w_joint_pos = -20.0  # Increased - closer tracking needed for backwards gait
        # w_joint_vel = -1.0e-3
        # w_contact = 3.0  # Increased - proper foot contact timing is crucial for backwards walking
        # w_foot_height = 1.0
        # w_backward_bonus = 2.0  # NEW - extra reward for actually moving backwards

        # # LINEAR VELOCITY TRACKING
        # # Note: for backwards walking, velocities will be negative
        # lin_vel_tol_m = 0.03  # Tighter tolerance for backwards - it's harder to control
        # lin_vel_tol_p = 0.4  # Adjusted for backwards motion
        # lin_vel_target = self.vel_xy_cmd[:, 0]
        # error = torch.abs(root_lin_vel[:, 0] - lin_vel_target)
        # lin_vel_rew = w_lin_vel_rew - w_lin_vel_rew / lin_vel_tol_m * error
        # # For backwards walking, we want to penalize going too fast (less stable)
        # # and reward staying close to target
        # lin_vel_rew = torch.where(root_lin_vel[:, 0] < lin_vel_target,  # going more backwards than target
        #                           w_lin_vel_rew - w_lin_vel_rew / lin_vel_tol_p * error,
        #                           lin_vel_rew)
        
        # # BACKWARD MOTION BONUS - reward for actually moving backwards
        # backward_bonus = w_backward_bonus * torch.clamp(-root_lin_vel[:, 0] / 0.15, 0.0, 1.0)

        # # LINEAR VELOCITY PENALTY
        # lin_vel_pen = w_lin_vel_pen * torch.abs(root_lin_vel[:, 1])

        # # ANGULAR VELOCITY TRACKING
        # ang_vel_tol = 0.12  # Slightly looser for backwards walking
        # error = torch.abs(root_ang_vel[:, 2] - self.yaw_rate_cmd[:, 0])
        # ang_vel_rew = w_ang_vel_rew - w_ang_vel_rew / ang_vel_tol * error

        # # MOTION PENALTIES
        # # Penalize roll/pitch more heavily for backwards walking (stability critical)
        # ang_vel_pen = w_ang_vel_pen * (torch.linalg.vector_norm(root_ang_vel[:, 0:2], dim=1) ** 2)
        # acc_pen = w_acc_pen * torch.linalg.vector_norm(dof_acc, dim=1)
        # vel_pen = w_vel_pen * torch.linalg.vector_norm(dof_vel, dim=1)
        # action_pen = w_action_pen * torch.linalg.vector_norm(self.action_hist[:, -1, :] - self.action_hist[:, -2, :], dim=1)
        
        # # Additional penalty for large action changes (squared for stronger effect)
        # action_smoothness_pen = -0.5 * (torch.linalg.vector_norm(self.action_hist[:, -1, :] - self.action_hist[:, -2, :], dim=1) ** 2)

        # # FOOT HEIGHT
        # # foot_1_height = torch.where(left_foot_contact, 0.0, self.get_foot_1_transform_buf[:, 6])
        # # foot_2_height = torch.where(right_foot_contact, 0.0, self.get_foot_2_transform_buf[:, 6])
        # # foot_height_rew = w_foot_height * (foot_1_height + foot_2_height)

        # ## IMITTATION REWARDS ##
        # # JOINT POSITION
        # joint_pos_rew = w_joint_pos * (torch.linalg.vector_norm(self.q_ref - self.get_dof_pos_buf[:, self.active_dof_ids], dim=1))

        # # JOINT VELOCITY
        # # joint_vel_rew = w_joint_vel * torch.linalg.vector_norm(self.dq_ref - dof_vel[:, self.active_dof_ids], dim=1)

        # # FOOT CONTACT
        # contact_rew = w_contact * ((self.feet_ref[:, 0] == self.left_foot_contact).float() + (self.feet_ref[:, 1] == self.right_foot_contact).float())

        # # penalize for contact change
        # contact_pen = -w_contact * ((last_left_foot_contact != self.left_foot_contact).float() + (last_right_foot_contact != self.right_foot_contact).float())

        # ## TERMINATION ##
        # lim = 40.0  # Slightly more lenient - backwards walking is inherently less stable
        # roll_error_deg = torch.abs(self.root_rpy[:, 0]) * 180.0 / torch.pi
        # pitch_error_deg = torch.abs(self.root_rpy[:, 1]) * 180.0 / torch.pi

        # # More lenient height threshold for backwards walking
        # self.term_buf[:] = (roll_error_deg > lim) | (
        #     pitch_error_deg > lim) | (self.get_root_transform_buf[:, 6] < 0.08)

        # alive = w_alive * (~self.term_buf).float()

        # ## ASSEMBLE REWARDS (ADDED OFFSET SO REWARDS LEAVE NEGATIVE DOMAIN)##
        # self.rew_buf[:] = (alive + lin_vel_rew + backward_bonus + lin_vel_pen + 
        #                   joint_pos_rew + contact_rew + contact_pen + 
        #                   acc_pen + vel_pen + action_pen + action_smoothness_pen + 
        #                   ang_vel_pen + ang_vel_rew + 50)
        ####################################################################################################
        ## END OF REWARDS ##
        ####################################################################################################

        ####################################################################################################
        ## ASSEMBLE OBSERVATIONS ##
        ## DO NOT CHANGE THIS ##
        ####################################################################################################
        # Rotate the local frame of the root to compute the IMU
        # rotate vectors by the transpose of the local frame rotation
        rot = quaternion_to_matrix(self.get_root_transform_buf[:, 0:4])
        rot = torch.transpose(rot, dim0=1, dim1=2)

        # IMU random delay
        rand_idx = torch.randint(low=self.imu_min_delay, high=self.imu_max_delay, size=(self.total_num_envs, ), device=self.device)

        self.imu_gyro_hist[:] = self.imu_gyro_hist.roll(-1, 1)  # left along dim 1
        # rotated gyro vec
        gyro = torch.matmul(rot, root_ang_vel.unsqueeze(-1)).squeeze(-1)
        self.imu_gyro_hist[:, -1, :] = gyro
        gyro_delayed = self.imu_gyro_hist[torch.arange(self.total_num_envs), rand_idx, :]

        rand_idx = torch.randint(low=self.imu_min_delay, high=self.imu_max_delay, size=(self.total_num_envs, ), device=self.device)

        self.imu_accelero_hist[:] = self.imu_accelero_hist.roll(-1, 1)  # left along dim 1
        # rotated acc vec
        accelero = torch.matmul(rot, root_lin_acc.unsqueeze(-1)).squeeze(-1)
        # subtract rotated gravity vector
        accelero -= torch.matmul(rot, self.gravity_vec).squeeze(-1)
        # print("accelero ", accelero[0])
        self.imu_accelero_hist[:, -1, :] = accelero
        accelero_delayed = self.imu_accelero_hist[torch.arange(self.total_num_envs), rand_idx, :]

        # Phase + noise
        phase = (self.imitation_steps).float() / float(self.imitation_steps_per_period)  # [0 - 1) range
        phase += torch.randn_like(phase) * self.period / 50.0  # -+ 2% std

        phase_cos = torch.cos(phase * 2.0 * torch.pi)
        phase_sin = torch.sin(phase * 2.0 * torch.pi)

        # noise
        n_hip_pos = 0.03  # rad, for each hip joint
        n_knee_pos = 0.05  # rad, for each knee joint
        n_ankle_pos = 0.08  # rad, for each ankle joint
        n_joint_vel = 2.5  # rad/s # Was 1.5
        n_gyro = 0.1
        n_accelerometer = 0.05

        dof_pos = self.get_dof_pos_buf[:, self.active_dof_ids]
        dof_pos_n = dof_pos.clone()
        dof_pos_n[:, self.hip_ids] += 2.0 * (torch.rand_like(dof_pos_n[:, self.hip_ids]) - 0.5) * n_hip_pos
        dof_pos_n[:, self.knee_ids] += 2.0 * (torch.rand_like(dof_pos_n[:, self.knee_ids]) - 0.5) * n_knee_pos
        dof_pos_n[:, self.ankle_ids] += 2.0 * (torch.rand_like(dof_pos_n[:, self.ankle_ids]) - 0.5) * n_ankle_pos

        dof_vel = dof_vel[:, self.active_dof_ids]
        dof_vel_n = dof_vel + 2.0 * (torch.rand_like(dof_vel) - 0.5) * n_joint_vel

        gyro_delayed_n = gyro_delayed + 2.0 * (torch.rand_like(gyro_delayed) - 0.5) * n_gyro
        accelero_delayed_n = accelero_delayed + 2.0 * (torch.rand_like(accelero_delayed) - 0.5) * n_accelerometer

        # Scale linear velocity command to a reasonable range close to (-1, 1)
        lin_vel_x_range = [0.0, 0.15]
        scaled_vel_cmd = self.vel_xy_cmd.clone()
        mean = 0.5 * (lin_vel_x_range[0] + lin_vel_x_range[1])
        scaled_vel_cmd[:, 0] = (scaled_vel_cmd[:, 0] - mean) * 10.0

        dof_vel_scale = 0.05
        if self.sac_not_ppo:
            self.obs_buf[:] = torch.cat((
                gyro_delayed_n,  # gyro
                accelero_delayed_n,  # accelero
                scaled_vel_cmd,
                self.yaw_rate_cmd,
                dof_pos_n - self.home_pos_active.view(1, -1),
                dof_vel_scale * dof_vel_n,
                self.action_hist.reshape(self.action_hist.shape[0], -1),
                self.set_pid_buf[:, self.active_dof_ids],
                self.left_foot_contact.view(-1, 1),
                self.right_foot_contact.view(-1, 1),
                phase_cos.view(-1, 1),
                phase_sin.view(-1, 1),
                # Privileged
                root_lin_vel,  # 3
                self.get_root_transform_buf[:, 6:7],  # 1
                self.get_joint_sensor_buf[:, self.active_dof_ids],  # 14
                foot_1_vel,  # 4 * 3
                foot_2_vel,
                foot_1_ang_vel,
                foot_2_ang_vel,
                self.q_ref
                ), dim=-1)

        else:  # ppo
            self.obs_buf = torch.cat((
                gyro_delayed_n,  # gyro
                accelero_delayed_n,  # accelero
                scaled_vel_cmd,
                self.yaw_rate_cmd,
                dof_pos_n - self.home_pos_active.view(1, -1),
                dof_vel_scale * dof_vel_n,
                self.action_hist.reshape(self.action_hist.shape[0], -1),
                self.set_pid_buf[:, self.active_dof_ids],
                self.left_foot_contact.view(-1, 1),
                self.right_foot_contact.view(-1, 1),
                phase_cos.view(-1, 1),
                phase_sin.view(-1, 1),
                ), dim=-1)

            self.state_buf = torch.cat((
                root_lin_vel,  # 3
                self.get_root_transform_buf[:, 6:7],  # 1
                self.get_joint_sensor_buf[:, self.active_dof_ids],  # 14
                foot_1_vel,  # 4 * 3
                foot_2_vel,
                foot_1_ang_vel,
                foot_2_ang_vel,
                self.q_ref
                ), dim=-1)

        if self.use_control_policy:
            self.control_policy_override()

    def control_policy_override(self):
        # save the default gait observations that will be passed to the gait policy
        # NOTE that we overwrite the buffer as inference requires a static memory buffer
        self.obs_gait[:] = self.obs_buf.clone()  # DO NOT CHANGE THIS

        ####################################################################################################
        ## CONTROL POLICY ##
        # HACKATHON TODO

        # TODO: define observations that are either available in real, or you can estimate
        # NOTE: you will need a way to estimate the global position/progress of the duck
        # NOTE that you can reuse any of the gait policy observations too (may or may not be useful):
        # self.obs_buf = torch.cat((
        #     gyro_delayed_n,  # gyro
        #     accelero_delayed_n,  # accelero
        #     scaled_vel_cmd,
        #     self.yaw_rate_cmd,
        #     dof_pos_n - self.home_pos_active.view(1, -1),
        #     dof_vel_scale * dof_vel_n,
        #     self.action_hist.reshape(self.action_hist.shape[0], -1),
        #     self.set_pid_buf[:, self.active_dof_ids],
        #     self.left_foot_contact.view(-1, 1),
        #     self.right_foot_contact.view(-1, 1),
        #     phase_cos.view(-1, 1),
        #     phase_sin.view(-1, 1),
        #     ), dim=-1)

        ####################################################################################################

        ## THIS IS A VERY BASIC EXAMPLE TO GET STARTED ##

        # override the observation buffer that will be returned for training the control policy
        # WARN: these observations are only for testing and are not available in real
        global_xy = self.get_root_transform_buf[:, 4:6]
        self.obs_buf = torch.cat((self.get_root_vel_buf,
                                  global_xy,  # global x y root position
                                  self.root_rpy[:, 2:3]  # global yaw
                                  ), dim=1)

        ## REWARD BUFFER OVERWRITE ##
        # Go towards a goal position - the ducks should simply move forward and not steer sideways
        goal_pos = torch.tensor([1.0, 0.0], dtype=torch.float32, device=self.device).view(1, -1)
        pos_error = torch.linalg.vector_norm(goal_pos - global_xy, dim=1)

        # have the agent estimate the x position
        # NOTE: you need to adjsut `self.single_action_space` to define different actions
        delta_x = self.get_root_transform_buf[:, 4] - self.last_root_transform[:, 4]
        delta_x_action_scale = 0.01  # rough estimate of the max delta pos per step
        delta_x_error = torch.abs(delta_x / delta_x_action_scale - self.act_buf[:, 2])

        self.rew_buf[:] = 2.0 - pos_error - delta_x_error

    def extract_plot_reference_slice(self, data, data_dir: str, name: str, do_plot: bool):
        num_pairs = data.shape[1] // 2

        if do_plot:
            data_start = 0
            data_end = len(data)
            time = torch.linspace(0, 10, data_end - data_start)
            time_np = time.numpy()
            data_np = data.cpu().numpy()
            # Generate 5 plots: (0,5), (1,6), ..., (4,9)
            for i in range(num_pairs):
                plt.figure(figsize=(8, 5))

                plt.plot(time_np, data_np[data_start:data_end, i], label=f'Var {i}')
                plt.plot(time_np, data_np[data_start:data_end, i + num_pairs], label=f'Var {i + num_pairs}')

                plt.title(f'Columns {i} and {i + num_pairs}')
                plt.xlabel('Time')
                plt.ylabel('Value')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()

                # Save each figure
                plt.savefig(os.path.join(data_dir, name + f'_all_{i}_{i+num_pairs}.png'), dpi=300)
                plt.close()

        time = torch.linspace(0, self.period, self.imitation_steps_per_period)
        time_np = time.numpy()

        start_slice = 3
        num_slices = 10
        avg_data = data[start_slice * self.imitation_steps_per_period: (start_slice + 1) * self.imitation_steps_per_period, :]
        for i in range(start_slice + 1, start_slice + num_slices):
            avg_data += data[i * self.imitation_steps_per_period: (i + 1) * self.imitation_steps_per_period, :]
        avg_data /= float(num_slices)

        if do_plot:
            data_np = avg_data.cpu().numpy()
            # Generate 5 plots: (0,5), (1,6), ..., (4,9)
            for i in range(num_pairs):
                plt.figure(figsize=(8, 5))

                plt.plot(time_np, data_np[data_start:data_end, i], label=f'Var {i}')
                plt.plot(time_np, data_np[data_start:data_end, i + num_pairs], label=f'Var {i + num_pairs}')

                plt.title(f'Columns {i} and {i + num_pairs}')
                plt.xlabel('Time')
                plt.ylabel('Value')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()

                # Save each figure
                plt.savefig(os.path.join(data_dir, name + f'_avg_slice_{i}_{i+num_pairs}.png'), dpi=300)
                plt.close()

        return avg_data


def plot_joints(obs_all):
    N, M = obs_all.shape

    plt.figure(figsize=(10, 6))
    x = np.arange(N)
    for j in range(14):
        plt.scatter(x, obs_all[:, j], label=f"{j+1}")
    plt.title(f"dof pos")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    x = np.arange(N)
    for j in range(4 * 14, 4 * 14 + 3):
        plt.scatter(x, obs_all[:, j], label=f"{j+1}")
    plt.title(f"gyro")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    x = np.arange(N)
    for j in range(4 * 14 + 3, 4 * 14 + 6):
        plt.scatter(x, obs_all[:, j], label=f"{j+1}")
    plt.title(f"accelero")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    x = np.arange(N)
    for j in range(4 * 14 + 6, 4 * 14 + 6 + 2):
        plt.scatter(x, obs_all[:, j], label=f"{j+1}")
    plt.title(f"feet")
    plt.legend()
    plt.grid(True)
    plt.show()

# @linux_torch_compile


def evaluate_polynomials_horner_polyfit_order(coeffs, t):
    """
    Evaluate N polynomials using Horner's method.
    Assumes coeffs from np.polyfit format: [highest_degree, ..., constant_term]

    Args:
        coeffs: Tensor of shape (N, D) where coeffs[i] = [a_n, a_{n-1}, ..., a_1, a_0]
    """
    # Start with the highest degree coefficient (first element)
    result = coeffs[:, 0].view(1, -1)

    # Apply Horner's method: work forwards through coefficients
    for i in range(1, coeffs.shape[1]):
        result = result * t.view(-1, 1) + coeffs[:, i].view(1, -1)

    return result


def build_id_map(gait_data, dof_dict):
    print("creating id map for active dofs: ", dof_dict)
    id_map = []
    keys = gait_data['joint_names']
    for k in dof_dict.keys():
        if k in keys:
            id_map.append(keys.index(k))
    print("id mapping of loaded joints: ", id_map)
    return id_map


def extract_gait_data(gait_data, dof_dict, device=torch.device("cuda:0")):
    """
    Extract frames as PyTorch tensor and joint names as list from gait JSON file.
    """
    id_map = build_id_map(gait_data, dof_dict)
    id_map = torch.as_tensor(id_map)

    # Extract frames and convert to PyTorch tensor
    dof_targets = torch.tensor(gait_data["joint_pos"], dtype=torch.float32, device=device)
    dof_vel_targets = torch.tensor(gait_data["joint_vel"], dtype=torch.float32, device=device)

    dof_targets = dof_targets[:, id_map]
    dof_vel_targets = dof_vel_targets[:, id_map]

    print("new dof_targets shape: ", dof_targets.shape)
    print("dof_targets home: ", dof_targets[0])
    assert dof_targets.shape[1] == len(dof_dict)
    assert dof_vel_targets.shape[1] == len(dof_dict)

    foot_contacts_ref = torch.tensor(
        gait_data["foot_contacts"],
        dtype=torch.float32,
        device=device) > 0.5
    # left_toe_pos = torch.tensor(gait_data["left_toe_pos"], dtype=torch.float32,
    #                             device=self.device)
    # right_toe_pos = torch.tensor(gait_data["right_toe_pos"], dtype=torch.float32,
    #                              device=self.device)

    return dof_targets, dof_vel_targets, foot_contacts_ref  # , left_toe_pos, right_toe_pos


def fit_polynomial(joint_pos, dt):
    joint_pos = joint_pos.cpu().numpy()  # shape (T, J)
    T, J = joint_pos.shape
    time = np.arange(T) * dt

    degree = 15  # polynomial degree
    polys = []  # will hold coefficients for each joint
    for j in range(J):
        y = joint_pos[:, j]
        coeffs = np.polyfit(time, y, degree)
        polys.append(coeffs)

    return polys


def plot_polynomial(joint_pos, dt):
    # Ensure we have 10 joints
    joint_pos = joint_pos.cpu().numpy()  # shape (T, J)
    T, J = joint_pos.shape
    assert J == 10, f"Expected 10 joints, but got {J}"

    time = np.arange(T) * dt

    # ---- Helper function for plotting subsets ----
    def plot_subset(joint_indices, filename):
        plt.figure(figsize=(10, 6))
        for j in joint_indices:
            plt.scatter(time, joint_pos[:, j], s=10, label=f"Joint {j} data")

        plt.xlabel("Time (s)")
        plt.ylabel("Position")
        plt.title(f"Polynomial Fit for Joints {joint_indices[0]}-{joint_indices[-1]}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()

    data_dir = "runs/duck_polys"
    if os.path.exists(data_dir):
        import shutil
        shutil.rmtree(data_dir)
    os.makedirs(data_dir)

    # ---- Plot first 5 joints (1–5) ----
    plot_subset(range(0, 5), os.path.join(data_dir, "polynomial_fit_0_4.png"))

    # ---- Plot second 5 joints (6–10) ----
    plot_subset(range(5, 10), os.path.join(data_dir, "polynomial_fit_5_9.png"))

    print("Saved plots as 'polynomial_fit_0_4.png' and 'polynomial_fit_5_9.png'")


def eval_plot_polynomial(polys, joint_pos, dt):
    # Ensure we have 10 joints
    joint_pos = joint_pos.cpu().numpy()  # shape (T, J)
    T, J = joint_pos.shape
    assert J == 10, f"Expected 10 joints, but got {J}"

    time = np.arange(T) * dt

    # Evaluate fitted curves
    fitted = []
    for j, coeffs in enumerate(polys):
        p = np.poly1d(coeffs)
        fitted.append(p(time))

    fitted = np.array(fitted).T  # shape (T, J)

    # ---- Helper function for plotting subsets ----
    def plot_subset(joint_indices, filename):
        plt.figure(figsize=(10, 6))
        for j in joint_indices:
            plt.scatter(time, joint_pos[:, j], s=10, label=f"Joint {j} data")
            plt.plot(time, fitted[:, j], label=f"Joint {j} fit", linewidth=1.5)

        plt.xlabel("Time (s)")
        plt.ylabel("Position")
        plt.title(f"Polynomial Fit for Joints {joint_indices[0]}-{joint_indices[-1]}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()

    data_dir = "runs/duck_polys"
    if os.path.exists(data_dir):
        import shutil
        shutil.rmtree(data_dir)
    os.makedirs(data_dir)

    # ---- Plot first 5 joints (1–5) ----
    plot_subset(range(0, 5), os.path.join(data_dir, "polynomial_fit_0_4.png"))

    # ---- Plot second 5 joints (6–10) ----
    plot_subset(range(5, 10), os.path.join(data_dir, "polynomial_fit_5_9.png"))

    print("Saved plots as 'polynomial_fit_0_4.png' and 'polynomial_fit_5_9.png'")

    return torch.as_tensor(fitted).float()


if __name__ == "__main__":

    max_episode_length = 800
    use_control_policy = True
    if use_control_policy:
        max_episode_length *= 10

    envs = DuckEnv(num_envs=2, use_control_policy=use_control_policy, max_episode_length=max_episode_length, env_offset_mode="line_z")
    obs, info = envs.reset()

    # Rendering
    render = envs.gym.get_render()
    if render is not None:
        render.capped_step = True

    # Solver loop
    finished = False
    idx = 0

    def get_dof_pos_env_0():
        dof_pos = envs.get_dof_pos_buf[:, envs.active_dof_ids] - envs.home_pos_active.view(1, -1)
        dof_pos = dof_pos[0].view(1, -1)
        return dof_pos

    t_rec = [0.0]
    dof_pos = get_dof_pos_env_0()

    t_sim = 0.0

    while not finished:
        # same for all envs
        action = envs.q_ref - envs.home_pos_active

        if use_control_policy:
            # vel x, yaw rate + global delta x, y, yaw
            action = torch.tensor([0.1, 0.0] + [0.0] * 3, device=envs.device).view(1, -1)

        # Step
        obs, rew, reset, timeout, info = envs.step(action)
        idx += 1
        t_sim += envs.dt

        if False:
            t_rec.append(t_sim)
            dof_pos = torch.cat((dof_pos, get_dof_pos_env_0()), dim=0)
            if t_sim > 2.0:
                # --- First 5 columns ---
                plt.figure(figsize=(8, 5))
                for i in range(5):
                    plt.plot(t_rec, dof_pos[:, i].cpu().numpy(), label=f'Var {i+1}')
                plt.title("First 5 Columns")
                plt.xlabel("Time")
                plt.ylabel("Value")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig("dof_left.png", dpi=300)
                plt.close()

                # --- Last 5 columns ---
                plt.figure(figsize=(8, 5))
                for i in range(5, 10):
                    plt.plot(t_rec, dof_pos[:, i].cpu().numpy(), label=f'Var {i+1}')
                plt.title("Last 5 Columns")
                plt.xlabel("Time")
                plt.ylabel("Value")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig("dof_right.png", dpi=300)
                plt.close()

                break

        if idx >= envs.max_episode_length:
            render.set_paused(True)
            idx = 0

        finished = envs.render_finished
