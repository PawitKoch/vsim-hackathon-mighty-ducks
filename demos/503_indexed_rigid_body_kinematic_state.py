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

# Create box definition
box_def_handle = env_def.create_box_def(v.Vec3(0.4), fixed=False)

# Instantiate box
box_rot = v.Quat(0, 0, 0, 1)
box_pos = v.Vec3(0, 1, 0)

box_transform = v.Transform(box_rot, box_pos)
box_handle = env_def.create_rigid_body(box_def_handle, box_transform)

# Import capsule definition
filename = "assets/mjcf/capsule.xml"
env_def.import_definitions(filename, fixed=False, scale=5)

capsule_def_handle = env_def.get_rigid_body_def_handle_by_name("capsule")

# Instantiate capsule
capsule_rot = v.Quat(0, 0, 0, 1)
capsule_pos = v.Vec3(0, 1, 2)

capsule_transform = v.Transform(capsule_rot, capsule_pos)
capsule_handle = env_def.create_rigid_body(capsule_def_handle, capsule_transform)

# Finalize environment def
env_def.finalize()

# == Environment group == #
num_envs = 2

env_group_handle = gym.create_environment_group(env_def_handle)
env_group = gym.get_environment_group(env_group_handle)
env_group.create_environment_set(num_envs)
env_group.finalize()

# Set environment transforms
env_group.tile_environments(spacing=3)

# == Create plane == #
gym.create_plane()

# == Finalize Gym == #
gym.gym_finalize()

# ================== #
# == GPU commands == #
# ================== #
assert torch.cuda.is_available(), "Could not find GPU"
device = torch.device("cuda:0")

# == Setting kinematic state == #
# Poses buffer: shape is (num_envs, 7)
set_pose_buf = torch.empty((num_envs, 7), dtype=torch.float32, device=device)

reset_poses = torch.tensor([[box_rot.x,
                             box_rot.y,
                             box_rot.z,
                             box_rot.w,
                             box_pos.x,
                             box_pos.y,
                             box_pos.z],
                            [capsule_rot.x,
                             capsule_rot.y,
                             capsule_rot.z,
                             capsule_rot.w,
                             capsule_pos.x,
                             capsule_pos.y,
                             capsule_pos.z]],
                           dtype=torch.float32,
                           device=device)

# Velocities buffer: shape is (num_envs, 6)
set_vel_buf = torch.zeros((num_envs, 6), dtype=torch.float32, device=device)

# Reset buffer: shape is (num_envs,)
reset_buf = torch.zeros(num_envs, dtype=torch.bool, device=device)

# Indices buffer: shape is (num_envs,)
set_indices_buf = torch.empty(num_envs, dtype=torch.uint32, device=device)

# Create command
set_kine_cmd = env_group.create_rigid_body_kinematic_state_command(
    v.wrap_gpu_buffer(set_pose_buf),
    v.wrap_gpu_buffer(set_vel_buf),
    rigid_body_handle_list=[box_handle, capsule_handle],
    indices_buffer=v.wrap_gpu_buffer(set_indices_buf),
    masks_buffer=v.wrap_gpu_buffer(reset_buf))

set_kine_cmd_arr = gym.create_rigid_body_kinematic_state_command_gpu_array([set_kine_cmd])

# == Getting kinematic state == #
# Buffers are same size as above
get_pose_buf = torch.empty_like(set_pose_buf)
get_vel_buf = torch.empty_like(set_vel_buf)
print_buf = torch.zeros_like(reset_buf)
get_indices_buf = torch.empty_like(set_indices_buf)

# Create command
get_kine_cmd = env_group.create_rigid_body_kinematic_state_command(
    v.wrap_gpu_buffer(get_pose_buf),
    v.wrap_gpu_buffer(get_vel_buf),
    rigid_body_handle_list=[box_handle, capsule_handle],
    indices_buffer=v.wrap_gpu_buffer(get_indices_buf),
    masks_buffer=v.wrap_gpu_buffer(print_buf))

get_kine_cmd_arr = gym.create_rigid_body_kinematic_state_command_gpu_array([get_kine_cmd])

# == Set up render == #
render = gym.get_render()

render.reset_camera(v.Vec3(-0.5, 6.4, -2.9), v.Vec3(0.3, -0.8, 0.4))
render.capped_step = True
render.set_paused(False)

# == GUI controls == #
reset_box = v.UserCheckbox("Reset", False)
print_box = v.UserCheckbox("Print kinematic state", False)
rigid_combo = v.UserCombo("Rigid body to reset", ["Box", "Capsule"], 0)
env_combo = v.UserCombo("Environment to reset", [f"Environment {i}" for i in range(num_envs)], 0)

render.register_menu_item(reset_box)
render.register_menu_item(print_box)
render.register_menu_item(rigid_combo)
render.register_menu_item(env_combo)


def reset_rigid_body():

    if reset_box.get_value() == False:
        return

    reset_box.set_value(False)

    # Select environment
    env_index = env_combo.get_current_index()
    reset_buf[:] = False
    reset_buf[env_index] = True

    ## Select rigid body ##
    rigid_body_index = rigid_combo.get_current_index()
    set_indices_buf[env_index] = rigid_body_index
    set_pose_buf[env_index] = reset_poses[rigid_body_index]

    # Set rigid body kinematic states
    gym.set_rigid_body_kinematic_states(set_kine_cmd_arr)


def print_kinematic_state():

    if print_box.get_value() == False:
        return

    print_box.set_value(False)

    # Select environment
    env_index = env_combo.get_current_index()
    print_buf[:] = False
    print_buf[env_index] = True

    ## Select rigid body ##
    # Box
    if rigid_combo.get_current_index() == 0:
        get_indices_buf[env_index] = box_handle.index()

    # Capsule
    elif rigid_combo.get_current_index() == 1:
        get_indices_buf[env_index] = capsule_handle.index()

    else:
        raise Exception("This should not be reached")

    # Get rigid body kinematic states
    gym.get_rigid_body_kinematic_states(get_kine_cmd_arr)

    # Print rigid body kinematic state
    pose = get_pose_buf[env_index]
    vel = get_vel_buf[env_index]

    print("Rigid body transform:   q({:.4f}, {:.4f}, {:.4f}, {:.4f}), "
          "p({:.4f}, {:.4f}, {:.4f})".format(*pose))

    print("Rigid body velocity:    top({:.4f}, {:.4f}, {:.4f}), "
          "bottom({:.4f}, {:.4f}, {:.4f})".format(*vel))


# == Simulation loop == #
timestep = 0.01667
gym.set_timestep(timestep)

done = False
while not done:

    # Reset rigid body
    reset_rigid_body()

    # Print kinematic state
    print_kinematic_state()

    # Simulation step
    gym.step()

    # Render
    done = render.render_function()
