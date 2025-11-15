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

# Instantiate franka panda arm
rot = v.shortest_rotation(v.Vec3(0, 0, 1), v.Vec3(0, 1, 0))
pos = v.Vec3(0, 0, 0)

transform = v.Transform(rot, pos)
art_handle = env_def.create_articulation(art_def_handle, transform)

# Finalize environment def
env_def.finalize()

# == Environment group == #
num_env_sets = 2
num_envs_per_env_set = 2
num_envs = num_env_sets * num_envs_per_env_set

env_group_handle = gym.create_environment_group(env_def_handle)
env_group = gym.get_environment_group(env_group_handle)
for i in range(num_env_sets):
    env_group.create_environment_set(num_envs_per_env_set)
env_group.finalize()

# Set environment transforms
env_group.tile_environments(spacing=2)

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

# == Getting joint velocities == #
# Joint velocities buffer: shape is (num_envs, num_dofs)
get_vel_buf = torch.zeros((num_envs, num_dofs), dtype=torch.float32, device=device)

# Create command
get_vel_cmd = env_group.create_joint_state_command(
    v.wrap_gpu_buffer(get_vel_buf),
    art_handle)

get_vel_cmd_arr = gym.create_joint_state_command_gpu_array([get_vel_cmd])

# == Getting joint positions == #
# Joint positions buffer: shape is (num_envs, num_dofs)
get_pos_buf = torch.zeros((num_envs, num_dofs), dtype=torch.float32, device=device)

# Create command
get_pos_cmd = env_group.create_joint_state_command(
    v.wrap_gpu_buffer(get_pos_buf),
    art_handle)

get_pos_cmd_arr = gym.create_joint_state_command_gpu_array([get_pos_cmd])

# == Setting joint forces == #
# Joint forces buffer: shape is (num_envs, num_dofs)
set_force_buf = torch.zeros((num_envs, num_dofs), dtype=torch.float32, device=device)

# Create command
set_force_cmd = env_group.create_joint_state_command(
    v.wrap_gpu_buffer(set_force_buf),
    art_handle)

set_force_cmd_arr = gym.create_joint_state_command_gpu_array([set_force_cmd])

# == Compute inverse dynamics == #
# Target joint accelerations buffer: shape is (num_envs, num_dofs)
target_acc_buf = torch.zeros((num_envs, num_dofs), dtype=torch.float32, device=device)

# Write results into same buffer as set_force_buf
compute_inverse_dynamics_cmd = env_group.create_inverse_dynamics_command(
    v.wrap_gpu_buffer(target_acc_buf),
    v.wrap_gpu_buffer(set_force_buf),
    art_handle,
    gym.get_gravity())

compute_inverse_dynamics_cmd_arr = gym.create_inverse_dynamics_command_gpu_array([
    compute_inverse_dynamics_cmd])

# == Set up render == #
render = gym.get_render()

render.reset_camera(v.Vec3(3.8, 5.7, -5.4), v.Vec3(0.1, -0.7, 0.7))
render.capped_step = True
render.set_paused(False)

# == GUI controls == #
inverse_dynamics_combo = v.UserCombo("Target acceleration", ["None", "Fixed", "Rigid"], 2)
render.register_menu_item(inverse_dynamics_combo)


def fixed_acc(x, v): return - v / timestep
def rigid_acc(x, v): return 0.5 * ((-x / timestep) - v) / timestep


def apply_inverse_dynamics():

    if inverse_dynamics_combo.get_current_item() == "None":
        set_force_buf[:] = 0

    else:
        gym.get_joint_positions(get_pos_cmd_arr)
        gym.get_joint_velocities(get_vel_cmd_arr)

        if inverse_dynamics_combo.get_current_item() == "Fixed":
            target_acc_buf[:] = fixed_acc(get_pos_buf, get_vel_buf)
        elif inverse_dynamics_combo.get_current_item() == "Rigid":
            target_acc_buf[:] = rigid_acc(get_pos_buf, get_vel_buf)

        gym.compute_inverse_dynamics(compute_inverse_dynamics_cmd_arr)

    gym.set_joint_forces(set_force_cmd_arr)


# == Simulation loop == #
timestep = 0.01667
gym.set_timestep(timestep)

done = False
while not done:

    # Apply inverse dynamics
    apply_inverse_dynamics()

    # Simulation step
    gym.step()

    # Render
    done = render.render_function()
