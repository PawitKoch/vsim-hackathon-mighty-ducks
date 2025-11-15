if __name__ == '__main__':
    print("This script should not be run on its own.")
    exit(1)

import torch
from math import ceil
from abc import ABC, abstractmethod
from hashlib import sha256
from typing import Union

import vlearn as v


class Environment(ABC):

    def __init__(self,
                 num_envs: Union[int, list[int]],
                 device: torch.device,
                 rendering: bool,
                 enable_scene_query: bool,
                 max_episode_length: int,
                 timestep: float,
                 frame_skip: int,
                 spacing: float,
                 gravity: v.Vec3,
                 treat_warning_as_error: bool = False,
                 initial_is_paused: bool = False,
                 send_interrupt: bool = False,
                 print_hash: bool = False,
                 up_axis: v.Vec3 = v.Vec3(0.0, 1.0, 0.0),
                 max_contact_pairs_per_env: int = 128,
                 with_window: bool = True,
                 max_patches_per_env: int = -1,
                 max_contacts_per_patch: int = 4,
                 update_scene_dependent_components_in_step: bool = True,
                 return_clones: bool = True
                 ):

        # Store configuration
        self.num_envs = num_envs
        self.device = device
        self.rendering = rendering
        self.enable_scene_query = enable_scene_query
        self.max_episode_length = max_episode_length
        self.frame_skip = frame_skip
        self.spacing = spacing
        self.gravity = gravity
        self.send_interrupt = send_interrupt
        self.print_hash = print_hash
        self.return_clones = return_clones

        # Create gym
        if max_patches_per_env == -1:
            max_patches_per_env = max_contact_pairs_per_env
        self.gym = v.create_gym(
            self.rendering,
            self.enable_scene_query,
            treat_warning_as_error=treat_warning_as_error,
            up_axis=up_axis,
            with_window=with_window,
            max_contact_pairs=max_contact_pairs_per_env * self.total_num_envs,
            max_patches=max_patches_per_env * self.total_num_envs,
            max_contacts=max_contacts_per_patch * max_patches_per_env * self.total_num_envs,
            update_scene_dependent_components_in_step=update_scene_dependent_components_in_step,
            cuda_device=device.index)

        self.gym.set_timestep(timestep)

        self.up_axis = up_axis
        self.up_axis_rotation = v.shortest_rotation(up_axis, v.Vec3(0, 1, 0))

        # Set up rendering
        self.gym_render = self.gym.get_render() if self.rendering else None
        self._render_finished = False
        if self.rendering:
            self.gym_render.set_paused(initial_is_paused)

        # Set gravity
        self.gym.set_gravity(self.gravity)

        # Environment group
        self.env_group_handle = None
        self.env_group = None

        # Environment sets
        self.env_set_handles = None
        self.env_sets = None

        self.env_set_handle = None
        self.env_set = None

        # Environment transforms
        self.env_transforms_list = None
        self.env_transforms = None

    def render_callback(self):
        """
        This function is called after render_function()
        """
        pass

    def render(self):
        """
        Render the environment.
        """
        if not self.rendering:
            return

        while True:
            self.render_finished = self.gym_render.render_function()
            self.render_callback()

            if self.render_finished:
                if self.send_interrupt:
                    raise KeyboardInterrupt
                return

            if not self.gym_render.is_paused() or self.gym_render.is_step_set():
                return

    def allocate_buffers(self):
        """
        Create torch buffers for observations, rewards, actions,
        termination status, truncation status, progress, and any additional
        data.

        Requires self.num_envs and self.single_observation_space to be set
        """
        self.obs_buf = torch.zeros((self.total_num_envs,) + self.single_observation_space.shape,
                                   dtype=torch.float32, device=self.device)
        self.rew_buf = torch.zeros(self.total_num_envs, dtype=torch.float32, device=self.device)
        self.term_buf = torch.zeros(self.total_num_envs, dtype=torch.bool, device=self.device)
        self.trunc_buf = torch.zeros(self.total_num_envs, dtype=torch.bool, device=self.device)
        self.progress_buf = torch.zeros(self.total_num_envs, dtype=torch.long, device=self.device)
        self.act_buf = torch.zeros((self.total_num_envs,) + self.single_action_space.shape,
                                   device=self.device, dtype=torch.float32)

        self.info = {}

    def create_envs(self, env_def_handle: v.EnvironmentDefHandle, env_rot: v.Quat = v.Quat(
            0, 0, 0, 1), mode='grid', env_set_offsets: list[v.Vec3] = [v.Vec3(0, 0, 0)]):
        """
        Instantiate environments from environment definition on a grid with the given spacing.

        Args:
            env_def_handle: environment definition handle
            env_rot: rotation to apply to environment (default none)
            env_set_offset: spatial offset for grid (default null vector)
            num_envs: number of environments
            mode: grid, line_x, line_z
        """

        assert self.gym.get_environment_def(env_def_handle).is_finalized(
            ), ("You must supply a " "finalized environment def to Environment.create_envs()")

        assert mode in ['grid', 'line_x', 'line_z'], "Mode must be 'grid', 'line_x', or line_z"

        assert self.env_group is None, (
            "EnvironmentGpu currently supports only one environment group")

        # Create environment group
        self.env_group_handle = self.gym.create_environment_group(env_def_handle)
        self.env_group = self.gym.get_environment_group(self.env_group_handle)

        if isinstance(self.num_envs, list):
            for num_envs in self.num_envs:
                self.env_group.create_environment_set(num_envs)
        elif isinstance(self.num_envs, int):
            self.env_group.create_environment_set(self.num_envs)
        else:
            raise Exception("This should not be reached")
        self.env_group.finalize()

        # Get environment sets
        self.env_set_handles = list(self.env_group.get_environment_set_handles())
        self.env_sets = list(self.env_group.get_environment_sets())

        # Arrange environments on a grid
        assert len(env_set_offsets) == self.env_group.get_num_environment_sets(), (
            "You must supply a spatial offset for each environment set")

        self.env_transforms_list = []

        for env_set, env_set_offset in zip(self.env_sets, env_set_offsets):

            env_transforms = []
            num_envs_in_env_set = env_set.get_num_environments()

            for i in range(num_envs_in_env_set):
                if mode == 'grid':
                    N = ceil(num_envs_in_env_set**0.5)
                    x = i % N
                    z = i // N
                elif mode == 'line_x':
                    x = i
                    z = 0
                elif mode == 'line_z':
                    x = 0
                    z = i
                else:
                    raise Exception("This should not be reached")

                env_pos = (self.up_axis_rotation.rotate((self.spacing * v.Vec3(x, 0, z))) +
                           env_set_offset)
                env_transform = v.Transform(env_rot, env_pos)

                env_handle = env_set.get_environment_handle(i)
                env = env_set.get_environment(env_handle)
                env.set_transform(env_transform)

                env_transforms.append(env_transform)

            self.env_transforms_list.append(env_transforms)

        # Single environment-set environments
        self.env_set_handle = self.env_set_handles[-1]
        self.env_set = self.env_sets[-1]
        self.env_transforms = self.env_transforms_list[-1]

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                                   torch.Tensor, dict]:
        """
        Step the physics of the environment.

        Args:
            actions: actions to apply

        Returns:
            Observations, rewards, termination status, truncation status, info
        """
        if self.print_hash:
            print("actions:            {}".format(
                sha256(actions.cpu().numpy().tobytes()).hexdigest()))

        self.render()

        # Apply actions
        self.pre_physics_step(actions)

        # Step simulator (motor control is persistent)
        for i in range(self.frame_skip):
            self.pre_gym_step()

            self.gym.step()

            self.post_gym_step()

        # Block stepping after the frame skip steps are ready
        if self.rendering:
            self.gym_render.set_step(False)

        # Compute observations, rewards, termination and truncation status
        self.post_physics_step()

        if self.print_hash:
            print("self.obs_buf        {}".format(sha256(
                self.obs_buf.cpu().numpy().tobytes()).hexdigest()))

        if not self.return_clones:
            return self.obs_buf, self.rew_buf, self.term_buf, self.trunc_buf, self.info

        return (self.obs_buf.clone(), self.rew_buf.clone(), self.term_buf.clone(),
                self.trunc_buf.clone(), self.info)

    @abstractmethod
    def pre_physics_step(self, actions: torch.Tensor):
        """
        Apply the actions to the environment (eg by setting torques, position targets).

        Args:
            actions: the actions to apply
        """

    def pre_gym_step(self):
        """
        Callback between pre_physics_step and gym.step for action repeat
        """
        pass

    def post_gym_step(self):
        """
        Callback between gym.step and post_physics_step for action repeat
        """
        pass

    @abstractmethod
    def post_physics_step(self):
        """
        Compute reward and observations, reset any environments that require it.
        """

    @abstractmethod
    def reset(self) -> tuple[torch.Tensor, dict]:
        """
        Reset the environment.

        Return:
            Observations
            Info dictionary
        """

    @property
    def total_num_envs(self) -> int:
        """
        Get the total number of environments.
        """
        if isinstance(self.num_envs, int):
            return self.num_envs
        elif isinstance(self.num_envs, list):
            return sum(self.num_envs)
        else:
            error_msg = "self.num_envs must be of type int or list[int]."
            error_msg += " type(self.num_envs): {}".format(type(self.num_envs))
            raise ValueError(error_msg)

    @property
    def render_finished(self) -> bool:
        """
        Get whether the renderer has finished.
        """
        return self._render_finished

    @render_finished.setter
    def render_finished(self, new_val: bool) -> bool:
        """
        Once render_finished is set to True, it is always set to True
        """
        assert isinstance(new_val, bool)
        if new_val:
            self._render_finished = True

    @property
    def timestep(self) -> float:
        return self.gym.get_timestep()

    @property
    def dt(self) -> float:
        return self.timestep * self.frame_skip


class EnvironmentGpu(Environment):

    def __init__(self,
                 num_envs: Union[int, list[int]],
                 device: torch.device,
                 rendering: bool,
                 enable_scene_query: bool,
                 max_episode_length: int,
                 timestep: float,
                 frame_skip: int,
                 spacing: float,
                 gravity: v.Vec3,
                 num_dofs: int,
                 treat_warning_as_error: bool = False,
                 initial_is_paused: bool = False,
                 send_interrupt: bool = False,
                 print_hash: bool = False,
                 up_axis: v.Vec3 = v.Vec3(0, 1, 0),
                 with_window: bool = True,
                 max_contact_pairs_per_env=128,
                 max_patches_per_env: int = -1,
                 max_contacts_per_patch: int = 4,
                 update_scene_dependent_components_in_step: bool = True,
                 return_clones: bool = True
                 ):

        assert device.type == "cuda"

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
            treat_warning_as_error,
            initial_is_paused,
            send_interrupt,
            print_hash,
            up_axis,
            max_contact_pairs_per_env,
            with_window,
            max_patches_per_env,
            max_contacts_per_patch,
            update_scene_dependent_components_in_step=update_scene_dependent_components_in_step,
            return_clones=return_clones)

        self.num_dofs = num_dofs

    def allocate_buffers(self):

        super().allocate_buffers()

        # Initialize reset buf to be true
        self.reset_buf = torch.ones(self.total_num_envs, dtype=torch.bool, device=self.device)

        # Reset positions buffers
        self.set_dof_pos_buf = torch.zeros((self.total_num_envs, self.num_dofs), device=self.device,
                                           dtype=torch.float32)

        # Reset velocities buffers
        self.set_dof_vel_buf = torch.zeros((self.total_num_envs, self.num_dofs), device=self.device,
                                           dtype=torch.float32)

        # Positions buffer
        self.get_dof_pos_buf = torch.zeros((self.total_num_envs, self.num_dofs), device=self.device,
                                           dtype=torch.float32)

        # Velocities buffer
        self.get_dof_vel_buf = torch.zeros((self.total_num_envs, self.num_dofs), device=self.device,
                                           dtype=torch.float32)

    def reset(self, compute_kinematics=True):
        self.reset_buf[:] = True
        self.reset_idx()

        if compute_kinematics:
            self.gym.compute_kinematics()

        # return self.obs_buf.clone(), {}

    @abstractmethod
    def reset_idx(self):
        """
        Reset environments based on the reset buffer
        """
