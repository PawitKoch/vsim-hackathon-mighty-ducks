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

# Set max num transform handles
for force_sensor in art_def.get_force_sensor_defs():
    force_sensor.max_num_transform_handles = 2
    print("> Force sensor: {}".format(force_sensor))

# Create contact filters
contact_filter_handles = []
contact_filter_handles.append(env_def.create_contact_filter([]))
contact_filter_handles.append(env_def.create_contact_filter([large_box_transform_handle]))
contact_filter_handles.append(env_def.create_contact_filter([small_box_transform_handle]))
contact_filter_handles.append(env_def.create_contact_filter([large_box_transform_handle,
                                                             small_box_transform_handle]))

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

num_sensors = art_def.get_num_force_sensor_defs()

# == Getting force sensor data == #

# Force sensor buffers: shape is (num_envs, 6)
get_force_sensor_buffers = []
get_force_sensor_cmds = []

for i in range(num_sensors):
    force_sensor_handle = articulation.get_force_sensor_handle(i)

    get_force_sensor_buffers.append(torch.zeros((num_envs, 6), dtype=torch.float32, device=device))
    get_force_sensor_cmds.append(env_group.create_force_sensor_command(
        v.wrap_gpu_buffer(get_force_sensor_buffers[-1]),
        force_sensor_handle))

get_force_sensor_cmd_arr = gym.create_force_sensor_command_gpu_array(get_force_sensor_cmds)

# == Setting contact filters == #
force_sensor_handle = articulation.get_force_sensor_handle_by_name("front_left_leg_sensor")

# Contact filter buffer: shape is (num_envs,)
set_contact_filter_buf = torch.zeros(num_envs, dtype=torch.uint32, device=device)

set_contact_filter_cmd = env_group.create_contact_filter_command(
    v.wrap_gpu_buffer(set_contact_filter_buf),
    force_sensor_handle,
    contact_filter_handles)

set_contact_filter_cmd_arr = gym.create_contact_filter_command_gpu_array([set_contact_filter_cmd])

# == Set up render == #
render = gym.get_render()

render.reset_camera(v.Vec3(15.1, 6.8, 4.3), v.Vec3(-0.9, -0.3, 0))
render.capped_step = True
render.set_paused(True)

# == GUI controls == #
print_force_box = v.UserCheckbox("Print force sensor", False)
filter_combo = v.UserCombo("Contact filter", ["None", "Large box", "Small box",
                                              "Large & small boxes"], 0)

render.register_menu_item(filter_combo)
render.register_menu_item(print_force_box)

timestep = 0.01667
gym.set_timestep(timestep)


def print_force_sensors():

    if not print_force_box.get_value():
        return

    gym.finalize_filtered_force_sensors()
    gym.get_sensor_forces(get_force_sensor_cmd_arr)

    for i, get_force_sensor_buf in enumerate(get_force_sensor_buffers):
        force = get_force_sensor_buf[0, 0:3]
        torque = get_force_sensor_buf[0, 3:6]

        print("Sensor #{} Force:  {:.4f}, {:.4f}, {:.4f}".format(i, *force))
        print("Sensor #{} Torque: {:.4f}, {:.4f}, {:.4f}".format(i, *torque))

# == Simulation loop == #


done = False
last_filter_index = filter_combo.get_current_index()
while not done:

    # Set contact filters
    if not render.is_paused() or render.is_step_set():
        current_filter_index = filter_combo.get_current_index()

        if last_filter_index != current_filter_index:
            set_contact_filter_buf[:] = filter_combo.get_current_index()
            gym.set_contact_filters(set_contact_filter_cmd_arr)

            last_filter_index = current_filter_index

    # Simulation step
    gym.step()

    # Print sensors
    if not render.is_paused() or render.is_step_set():
        print_force_sensors()

    # Render
    done = render.render_function()
