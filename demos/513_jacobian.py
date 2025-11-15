import vlearn as v

import torch

# == Initialize Gym == #
gym = v.create_gym(with_render=True,
                   enable_scene_query=True,
                   max_contact_pairs=512 * 1024,
                   max_patches=512 * 1024,
                   max_contacts=4 * 512 * 1024)

# == Environment definition == #
env_def_handle = gym.create_environment_def()
env_def = gym.get_environment_def(env_def_handle)

# Import definition
filename, name = "assets/urdf/franka_description/robots/franka_panda.vsim", "panda"
env_def.import_definitions(filename, fixed=True)

art_def_handle = env_def.get_articulation_def_handle_by_name(name)

# Instantiate franka
rot = v.shortest_rotation(v.Vec3(0, 0, 1), v.Vec3(0, 1, 0))
pos = v.Vec3(0, 0, 0)

transform = v.Transform(rot, pos)
art_handle = env_def.create_articulation(art_def_handle, transform)

# Finalize environment def
env_def.finalize()

# == Environment group == #
num_envs = 1
env_group_handle = gym.create_environment_group(env_def_handle)
env_group = gym.get_environment_group(env_group_handle)
env_group.create_environment_set(num_envs)
env_group.finalize()

env_group.tile_environments()

# == Create plane == #
gym.create_plane()

# == Finalize Gym == #
gym.gym_finalize()

# ================== #
# == GPU commands == #
# ================== #
assert torch.cuda.is_available(), "Could not find GPU"
device = torch.device("cuda:0")

art_def = env_def.get_articulation_def(art_def_handle)
num_dofs = art_def.get_num_joint_dof_defs()
num_links = art_def.get_num_link_defs()

# == Getting joint velocities == #
# Joint velocities buffer: shape is (num_envs, num_dofs)
get_joint_vel_buf = torch.zeros((num_envs, num_dofs), dtype=torch.float32, device=device)

# Create command
get_joint_vel_cmd = env_group.create_joint_state_command(
    v.wrap_gpu_buffer(get_joint_vel_buf),
    art_handle)

get_joint_vel_cmd_arr = gym.create_joint_state_command_gpu_array([get_joint_vel_cmd])

# == Getting link velocities == #
# Link velocities buffer: shape is (num_envs, num_links, 6)
get_link_vel_buf = torch.zeros((num_envs, num_links - 1, 6), dtype=torch.float32,
                               device=device)

# Create command
get_link_vel_cmd = env_group.create_link_velocity_command(
    v.wrap_gpu_buffer(get_link_vel_buf),
    art_handle,
    index_range=[1, num_links]
    )

get_link_vel_cmd_arr = gym.create_link_velocity_command_gpu_array([get_link_vel_cmd])

# == Getting Jacobian == #
num_rows = 6 * (num_links - 1)
num_columns = num_dofs
get_jacobian_buf = torch.zeros((num_envs, num_rows, num_columns), dtype=torch.float32,
                               device=device)

# Create command
get_jacobian_cmd = env_group.create_jacobian_command(
    v.wrap_gpu_buffer(get_jacobian_buf),
    art_handle,
    )

get_jacobian_cmd_arr = gym.create_jacobian_command_gpu_array([get_jacobian_cmd])

# == Set up render == #
render = gym.get_render()

render.reset_camera(v.Vec3(-0.5, 6.4, -2.9), v.Vec3(0.2, -0.9, 0.4))
render.capped_step = True
render.set_paused(False)


def compare_link_velocities():

    # Compute Jacobian
    gym.compute_jacobians(get_jacobian_cmd_arr)

    # Get joint velocities
    gym.get_joint_velocities(get_joint_vel_cmd_arr)

    # Get link velocities
    gym.get_link_velocities(get_link_vel_cmd_arr)

    # Compute link velocities using Jacobian and compare to observed link velocities
    computed_link_vel = (get_jacobian_buf @ get_joint_vel_buf.unsqueeze(-1)).squeeze()
    diffnorm = torch.norm(computed_link_vel - get_link_vel_buf.view(num_envs, -1), dim=1)

    # Print
    print("L2 difference: {}".format(diffnorm))


# == Simulation loop == #
timestep = 0.01667
gym.set_timestep(timestep)

done = False
while not done:

    # Simulation step
    gym.step()

    # Compare link velocities
    if not render.is_paused() or render.is_step_set():
        compare_link_velocities()

    # Render
    done = render.render_function()
