import vlearn as v

import torch

# == Initialize Gym == #
gym = v.create_gym(with_render=True, enable_scene_query=True)

# == Environment definition == #
env_def_handle = gym.create_environment_def()
env_def = gym.get_environment_def(env_def_handle)

# Load skyboxes
tex_handle0 = env_def.load_texture("assets/skyboxes/daylight/Daylight Box_Left.bmp")
tex_handle1 = env_def.load_texture("assets/skyboxes/daylight/Daylight Box_Right.bmp")
tex_handle2 = env_def.load_texture("assets/skyboxes/daylight/Daylight Box_Top.bmp")
tex_handle3 = env_def.load_texture("assets/skyboxes/daylight/Daylight Box_Bottom.bmp")
tex_handle4 = env_def.load_texture("assets/skyboxes/daylight/Daylight Box_Front.bmp")
tex_handle5 = env_def.load_texture("assets/skyboxes/daylight/Daylight Box_Back.bmp")

tex_handle6 = env_def.load_texture("assets/skyboxes/holodeck/west.jpg")
tex_handle7 = env_def.load_texture("assets/skyboxes/holodeck/east.jpg")
tex_handle8 = env_def.load_texture("assets/skyboxes/holodeck/top.jpg")
tex_handle9 = env_def.load_texture("assets/skyboxes/holodeck/bottom.jpg")
tex_handle10 = env_def.load_texture("assets/skyboxes/holodeck/north.jpg")
tex_handle11 = env_def.load_texture("assets/skyboxes/holodeck/south.jpg")

# Import cartpole definition
env_def.import_definitions("assets/vsim/cartpole_rgb/cartpole.vsim", fixed=True)

cartpole_def_handle = env_def.get_articulation_def_handle_by_name("cartpole")

cartpole_rot = v.shortest_rotation(v.Vec3(0, 0, 1), v.Vec3(0, 1, 0))
cartpole_pos = v.Vec3(0, 1, 0)

cartpole_transform = v.Transform(cartpole_rot, cartpole_pos)
cartpole_handle = env_def.create_articulation(cartpole_def_handle, cartpole_transform, "cartpole")

# Configure RGB camera render
art_def = env_def.get_articulation_def(cartpole_def_handle)

camera_def = art_def.get_rgb_camera_def(0)
camera = art_def.get_rgb_camera(0)

rot = v.Quat(v.Vec3(0, 1, 0), torch.pi / 2)
pos = v.Vec3(2, 1.5, 0)
camera.render_relative_transform = v.Transform(rot, pos)

camera.render_width = 2
camera.render_height = 0.5

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

daylight_textures = [tex_handle0.index(), tex_handle1.index(), tex_handle2.index(),
                     tex_handle3.index(), tex_handle4.index(), tex_handle5.index()]
holodeck_textures = [tex_handle6.index(), tex_handle7.index(), tex_handle8.index(),
                     tex_handle9.index(), tex_handle10.index(), tex_handle11.index()]

daylight_tensor = torch.tensor(daylight_textures, dtype=torch.uint32, device=device)
holodeck_tensor = torch.tensor(holodeck_textures, dtype=torch.uint32, device=device)

# == Setting skybox == #
# Skybox texture buffer: shape is (num_env_sets, 6)
set_skybox_texture_buf = torch.empty((num_env_sets, 6), dtype=torch.uint32, device=device)

# Create command
camera_def_handle = art_def.get_rgb_camera_def_handle(0)
set_skybox_cmd = env_group.create_rgb_camera_skybox_command(
    v.RGBCameraSkybox.TEXTURE,
    v.wrap_gpu_buffer(set_skybox_texture_buf),
    camera_def_handle)

set_skybox_cmd_arr = gym.create_rgb_camera_skybox_command_gpu_array([set_skybox_cmd])

# == Set up render == #
render = gym.get_render()

render.reset_camera(v.Vec3(14.9, 4.7, -2.9), v.Vec3(-0.8, -0.4, 0.5))
render.capped_step = True
render.set_paused(False)

# == GUI controls == #
skybox_combos = []
for i in range(num_env_sets):
    skybox_combos.append(v.UserCombo(f"Env set {i} skybox", ["Daylight", "Holodeck"], 0))

for combo in skybox_combos:
    render.register_menu_item(combo)


def set_skyboxes():
    # Get values from combos
    combo_vals = [c.get_current_item() for c in skybox_combos]

    # Write to skybox texture buffer
    for i in range(num_env_sets):
        if combo_vals[i] == "Daylight":
            set_skybox_texture_buf[i][:] = daylight_tensor
        elif combo_vals[i] == "Holodeck":
            set_skybox_texture_buf[i][:] = holodeck_tensor
        else:
            raise Exception("This should not be reached")

    # Set skyboxes
    gym.set_rgb_camera_skyboxes(set_skybox_cmd_arr)


# == Simulation loop == #
timestep = 0.01667
gym.set_timestep(timestep)

done = False
while not done:

    # Set skyboxes
    set_skyboxes()

    # Simulation step
    gym.step()

    # Render
    done = render.render_function()
