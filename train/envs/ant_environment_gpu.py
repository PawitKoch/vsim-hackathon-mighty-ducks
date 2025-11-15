import torch
import numpy as np
from vlearn.spaces import Box

import vlearn as v

if __name__ == "__main__":
    from environment import EnvironmentGpu
    from common import create_plane, reset_noise_helper
    from ant_environment_common import (create_envs_helper,
                                        allocate_gpu_buffers, store_initial_conditions_helper,
                                        compute_observations_helper,
                                        compute_reward_termination_truncation_helper)
else:
    from .environment import EnvironmentGpu
    from .common import create_plane, reset_noise_helper
    from .ant_environment_common import (create_envs_helper,
                                         allocate_gpu_buffers, store_initial_conditions_helper,
                                         compute_observations_helper,
                                         compute_reward_termination_truncation_helper)


class AntEnvironmentGpu(EnvironmentGpu):

    def __init__(self,
                 num_envs: int,
                 device: torch.device,
                 rendering: bool = False,
                 enable_scene_query: bool = False,
                 max_episode_length: int = 1000,
                 ctrl_cost_weight: float = 0.5,
                 healthy_reward: float = 1,
                 healthy_y_range: tuple[float, float] = (0.3, 1.1),
                 reset_noise_scale: float = 1,
                 gravity: v.Vec3 = v.Vec3(0, -9.81, 0),
                 timestep: float = 0.01667,
                 frame_skip: int = 1,
                 spacing: float = 2,
                 control_mode: str = 'motor',  # motor or pid
                 initial_is_paused: bool = False,
                 send_interrupt: bool = False,
                 print_hash: bool = False,
                 max_contact_pairs_per_env: int = 64,
                 with_window: bool = True,
                 ):

        assert control_mode in ['motor', 'pid']
        num_dofs = 8

        super().__init__(num_envs, device, rendering, enable_scene_query, max_episode_length,
                         timestep, frame_skip, spacing, gravity, num_dofs,
                         initial_is_paused=initial_is_paused, send_interrupt=send_interrupt,
                         print_hash=print_hash, max_contact_pairs_per_env=max_contact_pairs_per_env,
                         with_window=with_window)

        # Store configuration
        self.ctrl_cost_weight = ctrl_cost_weight
        self.healthy_reward = healthy_reward
        self.healthy_y_range = healthy_y_range
        self.reset_noise_scale = reset_noise_scale
        self.control_mode = control_mode

        # Hardcode these for now
        self.num_motors = 8
        self.num_sensors = 4

        # Initialise observation and action space
        num_obs = 0
        num_obs += 1                        # y position
        num_obs += 4                        # torso rotation
        num_obs += 3                        # torso linear velocity
        num_obs += 3                        # torso angular velocity
        num_obs += self.num_dofs            # dof positions
        num_obs += self.num_dofs            # dof velocities
        if control_mode == 'motor':
            num_obs += self.num_motors      # motor forces
        elif control_mode == 'pid':
            num_obs += self.num_dofs        # pid control
        num_obs += 6 * self.num_sensors     # joint force sensors

        print("num_obs: {}".format(num_obs))

        self.single_observation_space = Box(
            low=np.array([np.finfo('f').min] * num_obs, dtype=np.float32),
            high=np.array([np.finfo('f').max] * num_obs, dtype=np.float32),
            dtype=np.float32)

        # Create environments
        self.create_envs()

        # Define action space
        if control_mode == 'motor':
            action_low = [-1] * self.num_motors
            action_high = [1] * self.num_motors
        elif control_mode == 'pid':
            action_low = []
            action_high = []

            for i in range(self.art_def.get_num_pid_defs()):

                pid_def = self.art_def.get_pid_def(i)
                joint_index = self.art_def.get_joint_dof_index(
                    pid_def.link_index, pid_def.local_dof_index)

                dof_def = self.art_def.get_joint_dof_def(joint_index)
                low, high = dof_def.get_limits()

                action_low.append(low)
                action_high.append(high)

        self.single_action_space = Box(
            low=np.array(action_low, dtype=np.float32),
            high=np.array(action_high, dtype=np.float32),
            dtype=np.float32)

        # Store initial conditions
        self.store_initial_conditions()

        # Allocate buffers
        self.allocate_buffers()

        # Create plane
        create_plane(self.gym)

        self.gym.set_num_solver_iterations(8)

        # Finalize gym
        self.gym.gym_finalize()

        # Camera
        if self.rendering:
            x = self.num_envs**0.5
            y = 2 * self.num_envs**0.5
            z = self.num_envs**0.5
            self.gym_render.reset_camera(v.Vec3(x, y, z), v.Vec3(1e-5, -1, 0))

    def create_envs(self):

        self.env_def_handle, self.art_def_handle, self.arti_handle = create_envs_helper(
            self.gym, self.control_mode)

        env_def = self.gym.get_environment_def(self.env_def_handle)
        self.art_def = env_def.get_articulation_def(self.art_def_handle)

        # Overwriting PID values
        for i in range(self.art_def.get_num_pid_defs()):

            pid_def = self.art_def.get_pid_def(i)
            pid_def.damping = 1
            pid_def.stiffness = 100

        env_def.finalize()

        super().create_envs(self.env_def_handle)

    def store_initial_conditions(self):

        self.dof_pos_init, self.root_trans_init, self.root_vel_init = \
            store_initial_conditions_helper(self.art_def, self.device)

    def allocate_buffers(self):

        super().allocate_buffers()

        allocate_gpu_buffers(self)

        self.old_root_pos_buf = torch.zeros((self.num_envs, 7), device=self.device,
                                            dtype=torch.float32)

    def reset_idx(self):

        # Set kinematic state
        self.set_dof_pos_buf[:] = self.gpu_init_dof_pos + \
            reset_noise_helper(self.gpu_init_dof_pos, self.reset_noise_scale, 0.4, 0.2)
        self.set_dof_vel_buf[:] = reset_noise_helper(self.gpu_init_dof_vel, self.reset_noise_scale,
                                                     0.2, 0.1)

        self.gym.set_articulation_kinematic_states(self.gpu_set_kinematic_state_command_array)

        self.old_root_pos_buf[:] = torch.where(self.reset_buf.view(-1, 1),
                                               self.gpu_init_root_transforms, self.old_root_pos_buf)

        self.act_buf[:] = torch.where(self.reset_buf.view(-1, 1), 0, self.act_buf)

        self.progress_buf[:] = torch.where(self.reset_buf, 0, self.progress_buf)

    def reset(self):

        super().reset()

        # Return observations
        self.compute_observations(self.act_buf)

        return self.obs_buf.clone(), {}

    def pre_physics_step(self, actions):

        assert isinstance(actions, torch.Tensor)

        # Save old root transforms for reward calculation
        self.old_root_pos_buf[:] = self.root_pos_buf

        self.act_buf[:] = actions

        if self.control_mode == 'motor':
            self.gym.set_motor_forces(self.gpu_set_motor_data_array)
        elif self.control_mode == 'pid':
            self.gym.set_joint_target_positions(self.gpu_set_pid_cmd_arr)

    def post_physics_step(self):
        self.progress_buf += 1

        self.reset_buf[:] = torch.logical_or(self.term_buf, self.trunc_buf)
        self.reset_idx()

        # Use self.act_buf instead of actions because it reflects resetted
        # environments
        self.compute_observations(self.act_buf)
        self.compute_reward_termination_truncation(self.act_buf)

    def compute_observations(self, actions):

        self.gym.get_articulation_kinematic_states(self.gpu_get_kinematic_state_command_array)

        self.gym.get_sensor_forces(self.gpu_get_sensor_forces_command_array)

        # Conversion operation for compatibility with handle-based sensor API
        self.sensor_force_buf[:] = torch.cat(self.force_sensor_buffers, dim=-1)

        self.obs_buf[:] = compute_observations_helper(
            self.root_pos_buf,
            self.root_vel_buf,
            self.get_dof_pos_buf,
            self.get_dof_vel_buf,
            actions,
            self.sensor_force_buf)

    def compute_reward_termination_truncation(self, actions):

        self.rew_buf[:], self.term_buf[:], self.trunc_buf[:] = \
            compute_reward_termination_truncation_helper(actions, self.obs_buf,
                                                         self.root_pos_buf, self.old_root_pos_buf, self.progress_buf,
                                                         self.healthy_y_range, self.healthy_reward, self.dt, self.ctrl_cost_weight,
                                                         self.max_episode_length)


if __name__ == "__main__":
    from time import sleep
    from sys import argv
    from vlearn.utils import get_VL_VISUAL_TESTS

    rendering = True
    with_window = get_VL_VISUAL_TESTS()

    num_iter = np.inf
    if len(argv) >= 2:
        num_iter = int(argv[1])

    control_mode = 'motor'
    if len(argv) >= 3:
        control_mode = argv[2]

    num_envs = 64

    reset_noise_scale = 1
    spacing = 2

    max_episode_length = 1000

    assert torch.cuda.is_available()
    device = torch.device("cuda:0")

    envs = AntEnvironmentGpu(
        num_envs,
        device,
        reset_noise_scale=reset_noise_scale,
        spacing=spacing,
        rendering=rendering,
        max_episode_length=max_episode_length,
        control_mode=control_mode,
        with_window=with_window)

    obs, _ = envs.reset()

    gym = v.get_gym()
    render = gym.get_render()

    # Define controls
    if render is not None:
        reset_box = v.UserCheckbox("Reset", False)
        render.register_menu_item(reset_box)

        sliders = []

        if envs.control_mode == 'motor':
            for i in range(envs.art_def.get_num_motor_defs()):
                name = envs.art_def.get_motor_def_name(i)
                motor_def = envs.art_def.get_motor_def(i)
                low = motor_def.low_limit
                high = motor_def.high_limit

                sliders.append(v.UserSlider(name, low, high, 0))

                render.register_menu_item(sliders[-1])
        elif envs.control_mode == 'pid':
            for i in range(envs.art_def.get_num_pid_defs()):

                pid_def = envs.art_def.get_pid_def(i)
                joint_index = envs.art_def.get_joint_dof_index(
                    pid_def.link_index, pid_def.local_dof_index)

                dof_def = envs.art_def.get_joint_dof_def(joint_index)
                name = envs.art_def.get_joint_dof_def_name(joint_index)
                low, high = dof_def.get_limits()

                sliders.append(v.UserSlider(name, low, high, 0))

                render.register_menu_item(sliders[-1])

        def control_by_menu():

            if reset_box.get_value():
                envs.reset()

            actions = torch.tensor([slider.get_value()
                                   for slider in sliders], dtype=torch.float32, device=device)

            return torch.tile(actions, (num_envs, 1))

        control_fn = control_by_menu

    else:
        def control_fn(n=num_envs, m=envs.art_def.get_num_motor_defs()): return torch.zeros((n, m))

    # Start simulation loop
    if render is not None:
        render.capped_step = True
    finished = False
    idx = 0
    while not finished:

        # actions = torch.zeros((num_envs, 8), dtype=torch.float32, device=device)
        actions = control_fn()

        # print("action: {}".format(action))

        obs, reward, terminated, truncated, info = envs.step(actions)

        # print("obs: {}".format(obs))
        # print("reward: {}".format(reward))

        # print("{}".format(obs[:,0])) # y-coordinate
        # print("{}".format(obs[:,1:5])) # Rotation
        # print("{}".format(obs[:,5:8])) # Linear velocity
        # print("{}".format(obs[:,8:11])) # Angular velocity
        # print("{}".format(obs[:,11:19])) # DOF position
        # print("{}".format(obs[:,19:27])) # DOF velocity

        # print(obs, reward, terminated, truncated, info)

        finished = envs.render_finished

        idx += 1
        if idx >= num_iter:
            finished = True
