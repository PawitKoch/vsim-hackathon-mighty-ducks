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
env_def.import_definitions(filename, fixed=False)

ant_def_handle = env_def.get_articulation_def_handle_by_name("torso")

# Instantiate ant
rot = v.shortest_rotation(v.Vec3(0, 0, 1), v.Vec3(0, 1, 0))
pos = v.Vec3(0, 1, 0)

ant_transform = v.Transform(rot, pos)
ant_handle = env_def.create_articulation(ant_def_handle, ant_transform)

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

link_index = 0

# == Setting external forces through force & torque == #
# Force & torque buffer: shape is (num_envs, num_links, 6)
set_force_torque_buf = torch.zeros((num_envs, 1, 6), dtype=torch.float32, device=device)

# Create command
set_force_torque_cmd = env_group.create_link_external_force_command(
    v.wrap_gpu_buffer(set_force_torque_buf),
    ant_handle,
    [link_index, link_index + 1],
    force_type=v.ForceType.FORCE_TORQUE)

set_force_torque_cmd_arr = gym.create_link_external_force_command_gpu_array([set_force_torque_cmd])

# == Setting external forces through force & position == #
# Force & position buffer: shape is (num_envs, num_links, 6)
set_force_pos_buf = torch.zeros((num_envs, 1, 6), dtype=torch.float32, device=device)

# Create command
set_force_pos_cmd = env_group.create_link_external_force_command(
    v.wrap_gpu_buffer(set_force_pos_buf),
    ant_handle,
    [link_index, link_index + 1],
    force_type=v.ForceType.FORCE_POSITION)

set_force_pos_cmd_arr = gym.create_link_external_force_command_gpu_array([set_force_pos_cmd])

# == Set up render == #
render = gym.get_render()

render.reset_camera(v.Vec3(3.5, 4.6, 10.7), v.Vec3(-0.4, -0.4, -0.8))
render.capped_step = True
render.set_paused(False)

# == GUI controls == #
force_min, force_max = -10, 10
torque_min, torque_max = -10, 10
pos_min, pos_max = -5, 5

last_index = 0
force_type_combo = v.UserCombo("Force type", ["Force & Torque", "Force & Position"], last_index)

force_x_slider = v.UserSlider("Force x", force_min, force_max, 0)
force_y_slider = v.UserSlider("Force y", force_min, force_max, 0)
force_z_slider = v.UserSlider("Force z", force_min, force_max, 0)

torque_x_slider = v.UserSlider("Torque x", torque_min, torque_max, 0)
torque_y_slider = v.UserSlider("Torque y", torque_min, torque_max, 0)
torque_z_slider = v.UserSlider("Torque z", torque_min, torque_max, 0)

pos_x_slider = v.UserSlider("Position x", pos_min, pos_max, 0)
pos_y_slider = v.UserSlider("Position y", pos_min, pos_max, 0)
pos_z_slider = v.UserSlider("Position z", pos_min, pos_max, 0)

render.register_menu_item(force_type_combo)

render.register_menu_item(force_x_slider)
render.register_menu_item(force_y_slider)
render.register_menu_item(force_z_slider)

render.register_menu_item(torque_x_slider)
render.register_menu_item(torque_y_slider)
render.register_menu_item(torque_z_slider)


def set_external_forces():

    global last_index

    if last_index != force_type_combo.get_current_index():
        if last_index == 0:

            render.unregister_menu_item(torque_x_slider)
            render.unregister_menu_item(torque_y_slider)
            render.unregister_menu_item(torque_z_slider)

            render.register_menu_item(pos_x_slider)
            render.register_menu_item(pos_y_slider)
            render.register_menu_item(pos_z_slider)

            set_force_torque_buf[:] = 0
            gym.set_link_external_forces(set_force_torque_cmd_arr)

        elif last_index == 1:

            render.unregister_menu_item(pos_x_slider)
            render.unregister_menu_item(pos_y_slider)
            render.unregister_menu_item(pos_z_slider)

            render.register_menu_item(torque_x_slider)
            render.register_menu_item(torque_y_slider)
            render.register_menu_item(torque_z_slider)

            set_force_pos_buf[:] = 0
            gym.set_link_external_forces(set_force_pos_cmd_arr)

        last_index = force_type_combo.get_current_index()

    if force_type_combo.get_current_item() == "Force & Torque":
        set_force_torque_buf[:, 0, :] = torch.tensor([
            force_x_slider.get_value(),
            force_y_slider.get_value(),
            force_z_slider.get_value(),
            torque_x_slider.get_value(),
            torque_y_slider.get_value(),
            torque_z_slider.get_value()])

        gym.set_link_external_forces(set_force_torque_cmd_arr)

    elif force_type_combo.get_current_item() == "Force & Position":
        set_force_pos_buf[:, 0, :] = torch.tensor([
            force_x_slider.get_value(),
            force_y_slider.get_value(),
            force_z_slider.get_value(),
            pos_x_slider.get_value(),
            pos_y_slider.get_value(),
            pos_z_slider.get_value()])

        gym.set_link_external_forces(set_force_pos_cmd_arr)


# == Simulation loop == #
timestep = 0.01667
gym.set_timestep(timestep)

done = False
while not done:

    # Set external forces
    set_external_forces()

    # Simulation step
    gym.step()

    # Render
    done = render.render_function()
