import vlearn as v

import torch

# == Initialize Gym == #
gym = v.create_gym(with_render=True, enable_scene_query=True)

# == Environment definition == #
env_def_handle = gym.create_environment_def()
env_def = gym.get_environment_def(env_def_handle)

# Import cartpole definition
env_def.import_definitions("assets/vsim/cartpole_rgb/cartpole.vsim", fixed=True)

cartpole_def_handle = env_def.get_articulation_def_handle_by_name("cartpole")

# Instantiate cartpole
cartpole_rot = v.shortest_rotation(v.Vec3(0, 0, 1), v.Vec3(0, 1, 0))
cartpole_pos = v.Vec3(0, 1, 0)
cartpole_transform = v.Transform(cartpole_rot, cartpole_pos)

cartpole_handle = env_def.create_articulation(cartpole_def_handle, cartpole_transform, "cartpole")

# Import textured link
filename = "assets/vsim/Link6_Blue/Link2_Blue.vsim"
env_def.import_definitions(filename, fixed=True, query_mode=v.QueryMode.USE_VISUALS,
                           import_extra_mesh_data=True)

link_def_handle = env_def.get_rigid_body_def_handle_by_name("Link6_Blue")

# Instantiate textured link
link_rot = v.Quat(v.Vec3(0, 1, 0), 3.1415926 / 2)
link_pos = v.Vec3(1, 1.5, 1.5)
link_transform = v.Transform(link_rot, link_pos)

link_handle = env_def.create_rigid_body(link_def_handle, link_transform)

# Configure RGB camera render
art_def = env_def.get_articulation_def(cartpole_def_handle)

camera_def = art_def.get_rgb_camera_def(0)
camera = art_def.get_rgb_camera(0)

rot = v.Quat(v.Vec3(0, 1, 0), torch.pi / 2)
pos = v.Vec3(2, 1.5, 0)
camera.render_relative_transform = v.Transform(rot, pos)

camera.render_width = 2
camera.render_height = 0.5

# Create RGB material
rgb_mat = v.RGBMaterial()
rgb_mat.color = v.Vec3(1, 1, 0)
rgb_mat.specular = 40
rgb_mat.spec_intensity = 0.25

rgb_mat_handle = env_def.create_rgb_material(rgb_mat)

# Control all links with the same material
env_def.assign_rgb_material_to_articulation_link(cartpole_def_handle, rgb_mat_handle, 0)
env_def.assign_rgb_material_to_articulation_link(cartpole_def_handle, rgb_mat_handle, 1)
env_def.assign_rgb_material_to_articulation_link(cartpole_def_handle, rgb_mat_handle, 2)

# Finalize environment def
env_def.finalize()

# == Environment group == #
num_env_sets = 2
num_envs_per_env_set = 1

env_group_handle = gym.create_environment_group(env_def_handle)
env_group = gym.get_environment_group(env_group_handle)

for i in range(num_env_sets):
    env_group.create_environment_set(num_envs_per_env_set)

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

# == Setting color == #
# Color buffer: shape is (num_env_sets, 3)
set_color_buf = torch.empty((num_env_sets, 3), dtype=torch.float32, device=device)

# Create command
set_color_cmd = env_group.create_rgb_material_property_command(
    v.RGBMaterialProperty.COLOR,
    v.wrap_gpu_buffer(set_color_buf),
    rgb_mat_handle)

set_color_cmd_arr = gym.create_rgb_material_property_command_gpu_array([set_color_cmd])

# == Set up render == #
render = gym.get_render()

render.reset_camera(v.Vec3(-4.5, 4.5, 3.7), v.Vec3(0.7, -0.5, -0.5))
render.capped_step = True
render.set_paused(False)

# == GUI controls == #
low, high, init = 0, 1, 0.5
R_sliders = []
G_sliders = []
B_sliders = []
for i in range(num_env_sets):
    R_sliders.append(v.UserSlider(f"Env set {i} red channel", low, high, init))
    G_sliders.append(v.UserSlider(f"Env set {i} green channel", low, high, init))
    B_sliders.append(v.UserSlider(f"Env set {i} blue channel", low, high, init))

for R_slider, G_slider, B_slider in zip(R_sliders, G_sliders, B_sliders):
    render.register_menu_item(R_slider)
    render.register_menu_item(G_slider)
    render.register_menu_item(B_slider)


def set_color():
    # Get values from sliders
    R_vals = torch.tensor([slider.get_value() for slider in R_sliders]).unsqueeze(-1)
    G_vals = torch.tensor([slider.get_value() for slider in G_sliders]).unsqueeze(-1)
    B_vals = torch.tensor([slider.get_value() for slider in B_sliders]).unsqueeze(-1)

    # Write to buffer
    set_color_buf[:] = torch.cat((R_vals, G_vals, B_vals), dim=-1)

    # Set color
    gym.set_rgb_material_properties(set_color_cmd_arr)


# == Simulation loop == #
timestep = 0.01667
gym.set_timestep(timestep)

done = False
while not done:

    # Set color
    set_color()

    # Simulation step
    gym.step()

    # Render
    done = render.render_function()
