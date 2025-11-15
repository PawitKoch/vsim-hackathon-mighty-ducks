import torch
import numpy as np
from vlearn.spaces import Box
from math import ceil

import vlearn as v

if __name__ == "__main__":
    from environment import EnvironmentGpu
    from common import create_plane, reset_noise_helper
    from ant_environment_common import (
        create_envs_helper,
        store_initial_conditions_helper,
        compute_observations_helper,
        compute_reward_termination_truncation_helper)
else:
    from .environment import EnvironmentGpu
    from .common import create_plane, reset_noise_helper
    from .ant_environment_common import (
        create_envs_helper,
        store_initial_conditions_helper,
        compute_observations_helper,
        compute_reward_termination_truncation_helper)


def allocate_dynamic_state_gpu_buffers(envs):

    total_num_envs = envs.env_group.get_total_num_environments()

    # Root transforms buffer
    envs.root_pos_buf = torch.zeros((total_num_envs, 7), device=envs.device,
                                    dtype=torch.float32)

    # Root velocities buffer
    envs.root_vel_buf = torch.zeros((total_num_envs, 6), device=envs.device,
                                    dtype=torch.float32)

    # Sensor forces buffer
    envs.sensor_force_buf = torch.zeros((total_num_envs, envs.num_sensors * 6),
                                        device=envs.device, dtype=torch.float32)

    # Reset state
    envs.gpu_init_dof_pos = torch.tile(envs.dof_pos_init, (total_num_envs, 1))

    envs.gpu_init_dof_vel = torch.zeros_like(envs.gpu_init_dof_pos)

    envs.gpu_init_root_velocities = torch.empty(6, dtype=torch.float32, device=envs.device)
    envs.gpu_init_root_velocities[0] = envs.root_vel_init.top.x
    envs.gpu_init_root_velocities[1] = envs.root_vel_init.top.y
    envs.gpu_init_root_velocities[2] = envs.root_vel_init.top.z
    envs.gpu_init_root_velocities[3] = envs.root_vel_init.bottom.x
    envs.gpu_init_root_velocities[4] = envs.root_vel_init.bottom.y
    envs.gpu_init_root_velocities[5] = envs.root_vel_init.bottom.z
    envs.gpu_init_root_velocities = torch.tile(envs.gpu_init_root_velocities, (total_num_envs,
                                                                               1))

    envs.gpu_init_root_transforms = torch.empty(7, dtype=torch.float32, device=envs.device)
    envs.gpu_init_root_transforms[0] = envs.root_trans_init.q.x
    envs.gpu_init_root_transforms[1] = envs.root_trans_init.q.y
    envs.gpu_init_root_transforms[2] = envs.root_trans_init.q.z
    envs.gpu_init_root_transforms[3] = envs.root_trans_init.q.w
    envs.gpu_init_root_transforms[4] = envs.root_trans_init.p.x
    envs.gpu_init_root_transforms[5] = envs.root_trans_init.p.y
    envs.gpu_init_root_transforms[6] = envs.root_trans_init.p.z
    envs.gpu_init_root_transforms = torch.tile(envs.gpu_init_root_transforms, (total_num_envs,
                                                                               1))

    # Set motor GPU command
    set_motor_cmd = envs.env_group.create_motor_control_command(
        v.wrap_gpu_buffer(envs.act_buf), envs.arti_handle)
    envs.gpu_set_motor_array = envs.gym.create_motor_control_command_gpu_array([set_motor_cmd])

    # Set kinematic state GPU command
    set_kine_cmd = envs.env_group.create_articulation_kinematic_state_command(
        v.wrap_gpu_buffer(
            envs.set_dof_pos_buf), v.wrap_gpu_buffer(
            envs.set_dof_vel_buf), v.wrap_gpu_buffer(
                envs.gpu_init_root_transforms), v.wrap_gpu_buffer(
                    envs.gpu_init_root_velocities), envs.arti_handle, link_index_range=(
                        0, 1), masks_buffer=v.wrap_gpu_buffer(
                            envs.reset_buf))
    envs.gpu_set_kinematic_state_command_array = envs.gym.create_articulation_kinematic_state_command_gpu_array(
        [set_kine_cmd])

    # Get kinematic state GPU command
    get_kine_cmd = envs.env_group.create_articulation_kinematic_state_command(
        v.wrap_gpu_buffer(
            envs.get_dof_pos_buf), v.wrap_gpu_buffer(
            envs.get_dof_vel_buf), v.wrap_gpu_buffer(
                envs.root_pos_buf), v.wrap_gpu_buffer(
                    envs.root_vel_buf), envs.arti_handle, link_index_range=(
                        0, 1))
    envs.gpu_get_kinematic_state_command_array = envs.gym.create_articulation_kinematic_state_command_gpu_array(
        [get_kine_cmd])

    # Get sensor forces GPU command
    num_force_sensors = envs.art_def.get_num_force_sensor_defs()
    env_def = envs.gym.get_environment_def(envs.env_def_handle)
    articulation = env_def.get_articulation(envs.arti_handle)

    envs.force_sensor_handles = []
    envs.force_sensor_buffers = []
    envs.force_sensor_cmds = []
    for i in range(num_force_sensors):
        envs.force_sensor_handles.append(articulation.get_force_sensor_handle(i))
        envs.force_sensor_buffers.append(torch.zeros((total_num_envs, 6), device=envs.device,
                                                     dtype=torch.float32))
        envs.force_sensor_cmds.append(envs.env_group.create_force_sensor_command(
            v.wrap_gpu_buffer(envs.force_sensor_buffers[-1]), envs.force_sensor_handles[-1]))

    envs.gpu_get_sensor_forces_command_array = envs.gym.create_force_sensor_command_gpu_array(
        envs.force_sensor_cmds)


def allocate_static_parameters_gpu_buffers(envs):

    num_env_sets = envs.env_group.get_num_environment_sets()
    num_links = envs.num_links

    # Link mass buffer
    envs.link_mass_buf = torch.zeros((num_env_sets, num_links), device=envs.device,
                                     dtype=torch.float32)

    # Set link mass command
    set_link_mass_cmd = envs.env_group.create_link_property_command(
        v.LinkProperty.MASS, v.wrap_gpu_buffer(
            envs.link_mass_buf), envs.art_def_handle, masks_buffer=v.wrap_gpu_buffer(
            envs.reset_buf))
    envs.gpu_set_link_mass_command_array = envs.gym.create_link_property_command_gpu_array(
        [set_link_mass_cmd])


class AntEnvironmentDomainRandomization(EnvironmentGpu):

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
                 initial_is_paused: bool = False,
                 send_interrupt: bool = False,
                 randomize_mass_range: tuple[float, float] = (0.8, 1.2),
                 max_contact_pairs_per_env: int = 32,
                 ):

        num_dofs = 8

        # We create num_envs environment sets containing one environment each
        num_envs = [1] * num_envs

        super().__init__(num_envs, device, rendering, enable_scene_query, max_episode_length,
                         timestep, frame_skip, spacing, gravity, num_dofs,
                         initial_is_paused=initial_is_paused, send_interrupt=send_interrupt,
                         max_contact_pairs_per_env=max_contact_pairs_per_env)

        self.gym.set_num_solver_iterations(4)

        # Store configuration
        self.ctrl_cost_weight = ctrl_cost_weight
        self.healthy_reward = healthy_reward
        self.healthy_y_range = healthy_y_range
        self.reset_noise_scale = reset_noise_scale
        self.randomize_mass_range = randomize_mass_range

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
        num_obs += self.num_motors          # motor forces
        num_obs += 6 * self.num_sensors     # joint force sensors

        print("num_obs: {}".format(num_obs))

        self.single_observation_space = Box(
            low=np.array([np.finfo('f').min] * num_obs, dtype=np.float32),
            high=np.array([np.finfo('f').max] * num_obs, dtype=np.float32),
            dtype=np.float32)

        # Create environments
        self.create_envs()

        # Define action space
        action_low = [-1] * self.num_motors
        action_high = [1] * self.num_motors
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

        # Finalize gym
        self.gym.gym_finalize()

    def create_envs(self):

        self.env_def_handle, self.art_def_handle, self.arti_handle = create_envs_helper(self.gym)

        env_def = self.gym.get_environment_def(self.env_def_handle)
        self.art_def = env_def.get_articulation_def(self.art_def_handle)

        env_set_offsets = []
        for i in range(len(self.num_envs)):
            N = ceil(len(self.num_envs)**0.5)
            x = i % N
            z = i // N
            offset = self.spacing * v.Vec3(x, 0, z)
            env_set_offsets.append(offset)

        env_def.finalize()

        super().create_envs(self.env_def_handle, env_set_offsets=env_set_offsets)

    def store_initial_conditions(self):

        self.dof_pos_init, self.root_trans_init, self.root_vel_init = \
            store_initial_conditions_helper(self.art_def, self.device)

        # Store initial link masses
        self.num_links = self.art_def.get_num_link_defs()
        self.link_mass = [link_def.mass for link_def in self.art_def.get_link_defs()]
        link_mass_low = self.randomize_mass_range[0] * torch.tensor(
            self.link_mass, device=self.device, dtype=torch.float32)
        link_mass_high = self.randomize_mass_range[1] * torch.tensor(
            self.link_mass, device=self.device, dtype=torch.float32)

        self.link_mass_low = torch.tile(link_mass_low, (len(self.num_envs), 1))
        self.link_mass_high = torch.tile(link_mass_high, (len(self.num_envs), 1))

    def allocate_buffers(self):

        super().allocate_buffers()

        allocate_dynamic_state_gpu_buffers(self)
        allocate_static_parameters_gpu_buffers(self)

        self.old_root_pos_buf = torch.zeros((self.total_num_envs, 7), device=self.device,
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

        # Randomize link masses
        r = torch.rand((len(self.num_envs), self.num_links),
                       device=self.device, dtype=torch.float32)
        self.link_mass_buf[:] = (self.link_mass_high - self.link_mass_low) * r + self.link_mass_low

        self.gym.set_link_properties(self.gpu_set_link_mass_command_array)

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

        self.gym.set_motor_forces(self.gpu_set_motor_array)

    def post_physics_step(self):
        self.progress_buf += 1

        self.reset_buf[:] = torch.logical_or(self.term_buf, self.trunc_buf)
        self.reset_idx()

        # Use self.act_buf instead of actions because it reflects resetted environments
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
    from vlearn.utils import str_to_env_args, get_VL_VISUAL_TESTS

    rendering = get_VL_VISUAL_TESTS()
    enable_scene_query = True

    num_iter = np.inf
    if len(argv) >= 2:
        num_iter = int(argv[1])

    d = dict()
    if len(argv) >= 3:
        d.update(str_to_env_args(AntEnvironmentDomainRandomization, argv[2]))

    num_envs = 64

    randomize_mass_range = (0.5, 1.5)

    reset_noise_scale = 1
    spacing = 2

    max_episode_length = 1000

    assert torch.cuda.is_available()
    device = torch.device("cuda:0")

    kwargs = dict(reset_noise_scale=reset_noise_scale, spacing=spacing,
                  rendering=rendering,
                  max_episode_length=max_episode_length,
                  enable_scene_query=enable_scene_query,
                  randomize_mass_range=randomize_mass_range)
    kwargs.update(d)

    envs = AntEnvironmentDomainRandomization(num_envs, device, **kwargs)

    obs, _ = envs.reset()

    gym = v.get_gym()
    render = gym.get_render()
    if render is not None:
        render.capped_step = True

    # Define controls
    if render is not None:
        reset_box = v.UserCheckbox("Reset", False)
        render.register_menu_item(reset_box)

        sliders = []

        for i in range(envs.art_def.get_num_motor_defs()):
            name = envs.art_def.get_motor_def_name(i)
            motor_def = envs.art_def.get_motor_def(i)
            low = motor_def.low_limit
            high = motor_def.high_limit

            sliders.append(v.UserSlider(name, low, high, 0))

            render.register_menu_item(sliders[-1])

        def control_by_menu():

            if reset_box.get_value():
                obs, _ = envs.reset()
                print_obs(obs)

            actions = torch.tensor([slider.get_value()
                                   for slider in sliders], dtype=torch.float32, device=device)

            actions = torch.tile(actions, (sum(envs.num_envs), 1))
            # actions = torch.stack((actions, torch.zeros_like(actions)))

            return actions

        control_fn = control_by_menu

    else:
        def control_fn(n=num_envs, m=envs.art_def.get_num_motor_defs()): return torch.zeros((n, m))

    def print_obs(obs):
        # print("obs[:, 0:1]: {}".format(obs[:, 0:1])) # y-position
        # print("obs[:, 1:5]: {}".format(obs[:, 1:5])) # torso rotation
        # print("obs[:, 5:8]: {}".format(obs[:, 5:8])) # torso linear velocity
        # print("obs[:, 8:11]: {}".format(obs[:, 8:11])) # torso angular velocity
        # print("obs[:, 11:19]: {}".format(obs[:, 11:19])) # dof positions
        # print("obs[:, 19:27]: {}".format(obs[:, 19:27])) # dof velocities
        # print("obs[:, 27:35]: {}".format(obs[:, 27:35])) # motor forces
        # print("obs[:, 35:59]: {}".format(obs[:, 35:59])) # sensor forces
        pass

    # Start simulation loop
    finished = False
    idx = 0
    while not finished:

        # actions = torch.zeros((num_envs, 8), dtype=torch.float32, device=device)
        actions = control_fn()

        obs, reward, terminated, truncated, info = envs.step(actions)

        print_obs(obs)

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
