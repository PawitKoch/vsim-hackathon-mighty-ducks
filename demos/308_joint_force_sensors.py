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
small_box_def_handle = env_def.create_box_def(v.Vec3(1), fixed=False)

# Instantiate small box
rot = v.Quat(0, 0, 0, 1)
pos = v.Vec3(0, 1, -1)

small_box_transform = v.Transform(rot, pos)
small_box_handle = env_def.create_rigid_body(small_box_def_handle, small_box_transform)

small_box = env_def.get_rigid_body(small_box_handle)
small_box_transform_handle = small_box.get_transform_handle()

# Import ant definition
filename = "assets/vsim/ant_sensor.vsim"
env_def.import_definitions(filename)

ant_def_handle = env_def.get_articulation_def_handle_by_name("torso")

# Instantiate ant
rot = v.shortest_rotation(v.Vec3(0, 0, 1), v.Vec3(0, 1, 0))
pos = v.Vec3(0, 5, 5)

ant_transform = v.Transform(rot, pos)
ant_handle = env_def.create_articulation(ant_def_handle, ant_transform)

art_def = env_def.get_articulation_def(ant_def_handle)

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

art_def = env_def.get_articulation_def(ant_def_handle)
articulation = env_def.get_articulation(ant_handle)

# == Getting joint force sensor data == #
num_dofs = art_def.get_num_joint_dof_defs()

# Joint force sensor buf: shape is (num_envs, num_dofs)
get_joint_sensor_buf = torch.zeros((num_envs, num_dofs), dtype=torch.float32, device=device)

get_joint_sensor_cmd = env_group.create_joint_force_sensor_command(
    v.wrap_gpu_buffer(get_joint_sensor_buf),
    ant_handle)

get_joint_sensor_cmd_arr = gym.create_joint_force_sensor_command_gpu_array([get_joint_sensor_cmd])

# == Set up render == #
render = gym.get_render()

render.reset_camera(v.Vec3(15.1, 6.8, 4.3), v.Vec3(-0.9, -0.3, 0))
render.capped_step = True
render.set_paused(True)

# == GUI controls == #
print_joint_box = v.UserCheckbox("Print joint force sensor", False)
render.register_menu_item(print_joint_box)

timestep = 0.01667
gym.set_timestep(timestep)


def print_sensors():

    if print_joint_box.get_value():
        gym.get_joint_sensor_forces(get_joint_sensor_cmd_arr)

        joint_forces = get_joint_sensor_buf[0]

        print("Joint forces: " + ", ".join(["{:.4f}"] * num_dofs).format(*joint_forces))


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
