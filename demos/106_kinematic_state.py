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

# Import ant definition
filename = "assets/vsim/ant.vsim"
env_def.import_definitions(filename)

ant_def_handle = env_def.get_articulation_def_handle_by_name("torso")

# Instantiate ant
ant_rot = v.shortest_rotation(v.Vec3(0, 0, 1), v.Vec3(0, 1, 0))
ant_pos = v.Vec3(0, 0.75, 0)

ant_transform = v.Transform(ant_rot, ant_pos)
ant_handle = env_def.create_articulation(ant_def_handle, ant_transform)

# Finalize environment def
env_def.finalize()

# == Environment group == #
num_envs = 2

env_group_handle = gym.create_environment_group(env_def_handle)
env_group = gym.get_environment_group(env_group_handle)
env_group.create_environment_set(num_envs)
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

art_def = env_def.get_articulation_def(ant_def_handle)
num_dofs = art_def.get_num_joint_dof_defs()

# == Setting kinematic state == #
# Compute initial joint positions: 0 clamped by joint limits
dof_pos_init = []
for dof_def in art_def.get_joint_dof_defs():
    low, high = dof_def.get_limits()
    init = max(low, min(0, high))
    dof_pos_init.append(init)

# Joint positions buffer: shape is (num_envs, num_dofs)
set_dof_pos_buf = torch.tensor([dof_pos_init] * num_envs, dtype=torch.float32, device=device)

# Joint velocities buffer: shape is (num_envs, num_dofs)
set_dof_vel_buf = torch.zeros_like(set_dof_pos_buf)

# Root positions buffer: shape is (num_envs, 7)
set_root_pos_buf = torch.tensor([[ant_rot.x, ant_rot.y, ant_rot.z, ant_rot.w, ant_pos.x, ant_pos.y,
                                  ant_pos.z]] * num_envs, dtype=torch.float32, device=device)

# Root velocities buffer: shape is (num_envs, 6)
set_root_vel_buf = torch.zeros((num_envs, 6), dtype=torch.float32, device=device)

# Reset buffer: shape is (num_envs,)
reset_buf = torch.zeros(num_envs, dtype=torch.bool, device=device)

# Create command
set_kine_cmd = env_group.create_articulation_kinematic_state_command(
    v.wrap_gpu_buffer(set_dof_pos_buf),
    v.wrap_gpu_buffer(set_dof_vel_buf),
    v.wrap_gpu_buffer(set_root_pos_buf),
    v.wrap_gpu_buffer(set_root_vel_buf),
    ant_handle,
    (0, num_dofs),
    (0, 1),
    masks_buffer=v.wrap_gpu_buffer(reset_buf))

set_kine_cmd_arr = gym.create_articulation_kinematic_state_command_gpu_array([set_kine_cmd])

# == Getting kinematic state == #
# Buffers are same size as above
get_dof_pos_buf = torch.empty_like(set_dof_pos_buf)
get_dof_vel_buf = torch.empty_like(set_dof_vel_buf)
get_root_pos_buf = torch.empty_like(set_root_pos_buf)
get_root_vel_buf = torch.empty_like(set_root_vel_buf)

# Create command
get_kine_cmd = env_group.create_articulation_kinematic_state_command(
    v.wrap_gpu_buffer(get_dof_pos_buf),
    v.wrap_gpu_buffer(get_dof_vel_buf),
    v.wrap_gpu_buffer(get_root_pos_buf),
    v.wrap_gpu_buffer(get_root_vel_buf),
    ant_handle,
    (0, num_dofs),
    (0, 1),)

get_kine_cmd_arr = gym.create_articulation_kinematic_state_command_gpu_array([get_kine_cmd])

# == Set up render == #
render = gym.get_render()

render.reset_camera(v.Vec3(-0.5, 6.4, -2.9), v.Vec3(0.2, -0.9, 0.4))
render.capped_step = True
render.set_paused(False)

# == GUI controls == #
env_combo = v.UserCombo("Environment", ["All"] + [f"Environment {i}" for i in range(num_envs)], 0)
reset_box = v.UserCheckbox("Reset", True)
print_box = v.UserCheckbox("Print kinematic states", False)

render.register_menu_item(env_combo)
render.register_menu_item(reset_box)
render.register_menu_item(print_box)


def reset_environments():

    if reset_box.get_value() == False:
        return

    reset_box.set_value(False)

    if env_combo.get_current_index() == 0:
        reset_buf[:] = True
    else:
        reset_buf[:] = False
        reset_buf[env_combo.get_current_index() - 1] = True

    gym.set_articulation_kinematic_states(set_kine_cmd_arr)


def print_kinematic_states():

    if print_box.get_value() == False:
        return

    print_box.set_value(False)

    gym.get_articulation_kinematic_states(get_kine_cmd_arr)

    get_dof_pos_buf_cpu = get_dof_pos_buf.cpu()
    get_dof_vel_buf_cpu = get_dof_vel_buf.cpu()
    get_root_pos_buf_cpu = get_root_pos_buf.cpu()
    get_root_vel_buf_cpu = get_root_vel_buf.cpu()

    if env_combo.get_current_index() == 0:
        env_indices = range(num_envs)
    else:
        env_indices = [env_combo.get_current_index() - 1]

    for i in env_indices:
        print(f">> Environment #{i}")

        dof_pos = get_dof_pos_buf_cpu[i]
        dof_vel = get_dof_vel_buf_cpu[i]
        root_pos = get_root_pos_buf_cpu[i]
        root_vel = get_root_vel_buf_cpu[i]

        print("Root transform:   q({:.4f}, {:.4f}, {:.4f}, {:.4f}), "
              "p({:.4f}, {:.4f}, {:.4f})".format(*root_pos))

        print("Root velocity:    top({:.4f}, {:.4f}, {:.4f}), "
              "bottom({:.4f}, {:.4f}, {:.4f})".format(*root_vel))

        print("Joint positions:  " + ", ".join(f"{pos:.4f}" for pos in dof_pos))

        print("Joint velocities: " + ", ".join(f"{vel:.4f}" for vel in dof_vel))


# == Simulation loop == #
timestep = 0.01667
gym.set_timestep(timestep)

done = False
while not done:

    # Reset environments
    reset_environments()

    # Print kinematic states
    print_kinematic_states()

    # Simulation step
    gym.step()

    # Render
    done = render.render_function()
