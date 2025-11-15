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

# == Setting kinematic state of box == #
# Transforms buffer: shape is (num_envs, 7)
set_box_pose_buf = torch.empty((num_envs, 7), dtype=torch.float32, device=device)
set_box_pose_buf[:] = torch.tensor([box_rot.x, box_rot.y, box_rot.z, box_rot.w, box_pos.x,
                                    box_pos.y, box_pos.z], dtype=torch.float32, device=device)

# Velocities buffer: shape is (num_envs, 6)
set_box_vel_buf = torch.zeros((num_envs, 6), dtype=torch.float32, device=device)

# Create command
set_box_kine_cmd = env_group.create_rigid_body_kinematic_state_command(
    v.wrap_gpu_buffer(set_box_pose_buf),
    v.wrap_gpu_buffer(set_box_vel_buf),
    box_handle)

# == Setting kinematic state of capsule == #
# Transforms buffer: shape is (num_envs, 7)
set_capsule_pose_buf = torch.empty((num_envs, 7), dtype=torch.float32, device=device)
set_capsule_pose_buf[:] = torch.tensor([capsule_rot.x,
                                        capsule_rot.y,
                                        capsule_rot.z,
                                        capsule_rot.w,
                                        capsule_pos.x,
                                        capsule_pos.y,
                                        capsule_pos.z],
                                       dtype=torch.float32,
                                       device=device)

# Velocities buffer: shape is (num_envs, 6)
set_capsule_vel_buf = torch.zeros((num_envs, 6), dtype=torch.float32, device=device)

# Create command
set_capsule_kine_cmd = env_group.create_rigid_body_kinematic_state_command(
    v.wrap_gpu_buffer(set_capsule_pose_buf),
    v.wrap_gpu_buffer(set_capsule_vel_buf),
    capsule_handle)

# == Command array == #
cmd_reset_buf = torch.zeros(2, dtype=torch.bool, device=device)

set_kine_cmd_arr = gym.create_rigid_body_kinematic_state_command_gpu_array(
    [set_box_kine_cmd, set_capsule_kine_cmd], masks_buffer=v.wrap_gpu_buffer(cmd_reset_buf))

# == Getting kinematic state of box == #
# Buffers are same size as above
get_box_pose_buf = torch.empty_like(set_box_pose_buf)
get_box_vel_buf = torch.empty_like(set_box_vel_buf)

# Create command
get_box_kine_cmd = env_group.create_rigid_body_kinematic_state_command(
    v.wrap_gpu_buffer(get_box_pose_buf),
    v.wrap_gpu_buffer(get_box_vel_buf),
    box_handle)

# == Getting kinematic state of capsule == #
# Buffers are same size as above
get_capsule_pose_buf = torch.empty_like(set_capsule_pose_buf)
get_capsule_vel_buf = torch.empty_like(set_capsule_vel_buf)

# Create command
get_capsule_kine_cmd = env_group.create_rigid_body_kinematic_state_command(
    v.wrap_gpu_buffer(get_capsule_pose_buf),
    v.wrap_gpu_buffer(get_capsule_vel_buf),
    capsule_handle)

# == Command array == #
cmd_print_buf = torch.zeros(2, dtype=torch.bool, device=device)

get_kine_cmd_arr = gym.create_rigid_body_kinematic_state_command_gpu_array(
    [get_box_kine_cmd, get_capsule_kine_cmd], masks_buffer=v.wrap_gpu_buffer(cmd_print_buf))

# == Set up render == #
render = gym.get_render()

render.reset_camera(v.Vec3(-0.5, 6.4, -2.9), v.Vec3(0.3, -0.8, 0.4))
render.capped_step = True
render.set_paused(False)

# == GUI controls == #
reset_box = v.UserCheckbox("Reset", False)
print_box = v.UserCheckbox("Print kinematic state", False)
rigid_combo = v.UserCombo("Rigid body", ["Box", "Capsule"], 0)

render.register_menu_item(reset_box)
render.register_menu_item(print_box)
render.register_menu_item(rigid_combo)


def reset_rigid_body():

    if reset_box.get_value() == False:
        return

    reset_box.set_value(False)

    # Select rigid body
    rigid_index = rigid_combo.get_current_index()
    cmd_reset_buf[:] = False
    cmd_reset_buf[rigid_index] = True

    # Set rigid body kinematic states
    gym.set_rigid_body_kinematic_states(set_kine_cmd_arr)


def print_kinematic_state():

    if print_box.get_value() == False:
        return

    print_box.set_value(False)

    # Select rigid body
    rigid_index = rigid_combo.get_current_index()
    cmd_print_buf[:] = False
    cmd_print_buf[rigid_index] = True

    if rigid_index == 0:
        get_pose_buf = get_box_pose_buf
        get_vel_buf = get_box_vel_buf
    elif rigid_index == 1:
        get_pose_buf = get_capsule_pose_buf
        get_vel_buf = get_capsule_vel_buf
    else:
        raise Exception("This should not be reached")

    # Get rigid body kinematic states
    gym.get_rigid_body_kinematic_states(get_kine_cmd_arr)

    # Print rigid body kinematic states
    for i in range(num_envs):

        print("> Environment {}".format(i))

        pose = get_pose_buf[i]
        vel = get_vel_buf[i]

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
