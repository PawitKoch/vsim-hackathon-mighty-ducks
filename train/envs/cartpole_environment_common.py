import torch

import vlearn as v


def create_envs_helper(gym, vision=False, cartpole_asset_vision="assets/vsim/cartpole_depth.vsim"):

    # Create environment def
    env_def_handle = gym.create_environment_def("cartpole")
    env_def = gym.get_environment_def(env_def_handle)

    if vision:
        filename = cartpole_asset_vision
    else:
        filename = "assets/urdf/cartpole/cartpole.urdf"
    env_def.import_definitions(filename, True)
    art_def_handle = env_def.get_articulation_def_handle_by_name("cartpole")
    art_def = env_def.get_articulation_def(art_def_handle)

    rot = v.shortest_rotation(v.Vec3(0, 0, 1), v.Vec3(0, 1, 0))
    pos = v.Vec3(0, 1, 0)
    transform = v.Transform(rot, pos)

    arti_handle = env_def.create_articulation(art_def_handle, transform, "cartpole")

    return env_def_handle, arti_handle


def reset_idx_helper(num_envs, num_dofs, reset_noise_scale, device):

    positions = reset_noise_scale * (-0.01 + torch.rand((num_envs, num_dofs),
                                                        device=device, dtype=torch.float32) * 0.02)
    velocities = reset_noise_scale * (-0.01 + torch.rand((num_envs, num_dofs),
                                                         device=device, dtype=torch.float32) * 0.02)

    return positions, velocities


def compute_observations_helper(dof_pos, dof_vel):
    return torch.cat((dof_pos[:, 0].unsqueeze(-1),
                      dof_pos[:, 1].unsqueeze(-1),
                      dof_vel[:, 0].unsqueeze(-1),
                      dof_vel[:, 1].unsqueeze(-1)), dim=-1)


def compute_reward_termination_truncation_helper(
        obs, progress_buf, cart_bounds, pole_bounds, max_episode_length):

    cart_pos = obs[:, 0]
    pole_angle = obs[:, 1]
    cart_vel = obs[:, 2]
    pole_vel = obs[:, 3]

    # Check if Out Of Bounds (OOB)
    cart_OOB = torch.logical_or(cart_pos < cart_bounds[0], cart_bounds[1] < cart_pos)
    pole_OOB = torch.logical_or(pole_angle < pole_bounds[0], pole_bounds[1] < pole_angle)
    term = torch.logical_or(cart_OOB, pole_OOB)

    # Reward is 1 if within bounds, else zero
    rew = torch.logical_not(term)

    # Truncate if max episode length is reached
    trunc = progress_buf >= max_episode_length

    return rew, term, trunc
