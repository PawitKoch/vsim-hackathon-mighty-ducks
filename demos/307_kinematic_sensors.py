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

# Create large box definition
large_box_def_handle = env_def.create_box_def(v.Vec3(2), fixed=False)

# Instantiate large box
rot = v.Quat(0, 0, 0, 1)
pos = v.Vec3(0, 2, 5)

large_box_transform = v.Transform(rot, pos)
large_box_handle = env_def.create_rigid_body(large_box_def_handle, large_box_transform)

large_box = env_def.get_rigid_body(large_box_handle)
large_box_transform_handle = large_box.get_transform_handle()

# Create small box definition
filename = "assets/vsim/box_sensor.vsim"
env_def.import_definitions(filename)

small_box_def_handle = env_def.get_rigid_body_def_handle_by_name("link1_0")

# Instantiate small box
rot = v.Quat(0, 0, 0, 1)
pos = v.Vec3(0, 1, -1)

small_box_transform = v.Transform(rot, pos)
small_box_handle = env_def.create_rigid_body(small_box_def_handle, small_box_transform)

# Import ant definition
filename = "assets/vsim/ant_sensor.vsim"
env_def.import_definitions(filename)

ant_def_handle = env_def.get_articulation_def_handle_by_name("torso")

# Instantiate ant
rot = v.shortest_rotation(v.Vec3(0, 0, 1), v.Vec3(0, 1, 0))
pos = v.Vec3(0, 10, 5)

ant_transform = v.Transform(rot, pos)
ant_handle = env_def.create_articulation(ant_def_handle, ant_transform)

art_def = env_def.get_articulation_def(ant_def_handle)

# Print kinematic sensor definition
print("Kinematic sensor definitions:")
for kine_def in art_def.get_kinematic_sensor_defs():
    print(kine_def)

# Finalize environment def
env_def.finalize()

# == Environment group == #
num_envs = 1
env_group_handle = gym.create_environment_group(env_def_handle)
env_group = gym.get_environment_group(env_group_handle)
env_group.create_environment_set(num_envs)
env_group.finalize()

# == Create plane == #
gym.create_plane()

# == Finalize Gym == #
gym.gym_finalize()

# ================== #
# == GPU commands == #
# ================== #
assert torch.cuda.is_available(), "Could not find GPU"
device = torch.device("cuda:0")

# == Getting articulation kinematic sensor state == #
articulation = env_def.get_articulation(ant_handle)
arti_kine_sensor_handle = articulation.get_kinematic_sensor_handle(0)

# Kinematic sensor pose buffer: shape is (num_envs, 7)
get_arti_kine_pose_sensor_buf = torch.zeros((num_envs, 7), dtype=torch.float32, device=device)

# Kinematic sensor velocity buffer: shape is (num_envs, 6)
get_arti_kine_vel_sensor_buf = torch.zeros((num_envs, 6), dtype=torch.float32, device=device)

get_arti_kine_sensor_cmd = env_group.create_kinematic_sensor_state_command(
    v.wrap_gpu_buffer(get_arti_kine_pose_sensor_buf),
    v.wrap_gpu_buffer(get_arti_kine_vel_sensor_buf),
    arti_kine_sensor_handle)

# == Getting rigid body kinematic sensor state == #
rigid_body = env_def.get_rigid_body(small_box_handle)
rigid_kine_sensor_handle = rigid_body.get_kinematic_sensor_handle(0)

# Kinematic sensor pose buffer: shape is (num_envs, 7)
get_rigid_kine_pose_sensor_buf = torch.zeros((num_envs, 7), dtype=torch.float32, device=device)

# Kinematic sensor velocity buffer: shape is (num_envs, 6)
get_rigid_kine_vel_sensor_buf = torch.zeros((num_envs, 6), dtype=torch.float32, device=device)

get_rigid_kine_sensor_cmd = env_group.create_kinematic_sensor_state_command(
    v.wrap_gpu_buffer(get_rigid_kine_pose_sensor_buf),
    v.wrap_gpu_buffer(get_rigid_kine_vel_sensor_buf),
    rigid_kine_sensor_handle)

# == Create command GPU array == #
get_kine_sensor_cmd_arr = gym.create_kinematic_sensor_state_command_gpu_array(
    [get_arti_kine_sensor_cmd, get_rigid_kine_sensor_cmd])

# == Set up render == #
render = gym.get_render()

render.reset_camera(v.Vec3(15.1, 6.8, 4.3), v.Vec3(-0.9, -0.3, 0))
render.capped_step = True
render.set_paused(True)

# == GUI controls == #
print_arti_kine_state_box = v.UserCheckbox("Print articulation kinematic sensor", False)
print_rigid_kine_state_box = v.UserCheckbox("Print rigid body kinematic sensor", False)
render.register_menu_item(print_arti_kine_state_box)
render.register_menu_item(print_rigid_kine_state_box)

timestep = 0.01667
gym.set_timestep(timestep)


def print_sensors():

    if print_arti_kine_state_box.get_value():
        gym.get_kinematic_sensor_states(get_kine_sensor_cmd_arr)

        pose = get_arti_kine_pose_sensor_buf[0]
        velocity = get_arti_kine_vel_sensor_buf[0]

        print("Articulation Transform:   q({:.4f}, {:.4f}, {:.4f}, {:.4f}), "
              "p({:.4f}, {:.4f}, {:.4f})".format(*pose))

        print("Articulation Velocity:    Angular Velocity({:.4f}, {:.4f}, {:.4f}), "
              "Linear Velocity({:.4f}, {:.4f}, {:.4f})".format(*velocity))

    if print_rigid_kine_state_box.get_value():
        gym.get_kinematic_sensor_states(get_kine_sensor_cmd_arr)

        pose = get_rigid_kine_pose_sensor_buf[0]
        velocity = get_rigid_kine_vel_sensor_buf[0]

        print("Rigid Body Transform:   q({:.4f}, {:.4f}, {:.4f}, {:.4f}), "
              "p({:.4f}, {:.4f}, {:.4f})".format(*pose))

        print("Rigid Body Velocity:    Angular Velocity({:.4f}, {:.4f}, {:.4f}), "
              "Linear Velocity({:.4f}, {:.4f}, {:.4f})".format(*velocity))


# == Simulation loop == #
done = False
while not done:

    # Simulation step
    gym.step()

    # Print sensors
    if not render.is_paused() or render.is_step_set():
        print_sensors()

    # Render
    done = render.render_function()
