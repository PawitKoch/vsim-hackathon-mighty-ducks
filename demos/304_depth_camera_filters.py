import vlearn as v

import torch
import matplotlib.pyplot as plt

# == Initialize Gym == #
gym = v.create_gym(with_render=True, enable_scene_query=True)

# == Environment definition == #
env_def_handle = gym.create_environment_def()
env_def = gym.get_environment_def(env_def_handle)

# Import cartpole definition
env_def.import_definitions("assets/vsim/cartpole_depth_demo.vsim", fixed=True)

cartpole_def_handle = env_def.get_articulation_def_handle_by_name("cartpole")

cartpole_rot = v.shortest_rotation(v.Vec3(0, 0, 1), v.Vec3(0, 1, 0))
cartpole_pos = v.Vec3(0, 1, 0)

cartpole_transform = v.Transform(cartpole_rot, cartpole_pos)
cartpole_handle = env_def.create_articulation(cartpole_def_handle, cartpole_transform, "cartpole")

# Configure depth camera render
art_def = env_def.get_articulation_def(cartpole_def_handle)

camera_def = art_def.get_depth_camera_def_by_name("depth_camera")
camera = art_def.get_depth_camera_by_name("slider_camera")

rot = v.Quat(v.Vec3(0, 1, 0), torch.pi / 2)
pos = v.Vec3(2, 1.5, 0)
camera.render_relative_transform = v.Transform(rot, pos)

camera.render_width = 2
camera.render_height = 0.5

camera.render_min_depth = 0
camera.render_max_depth = camera_def.far_clip

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

# == Applying filters to raw depth image data == #
# Create command
articulation = env_def.get_articulation(cartpole_handle)
camera_handle = articulation.get_depth_camera_handle_by_name("slider_camera")

# Gaussian noise
mean = 0
std = 0.1
gaussian_noise_cmd = env_group.create_gaussian_noise_depth_command(camera_handle, mean, std)
gaussian_noise_cmd_arr = gym.create_gaussian_noise_depth_command_gpu_array([gaussian_noise_cmd])

# Erase pixels
percent = 0.05
erase_pixels_cmd = env_group.create_erase_pixels_depth_command(camera_handle, percent)
erase_pixels_cmd_arr = gym.create_erase_pixels_depth_command_gpu_array([erase_pixels_cmd])

# Erase patch
size_x = 10
size_y = 5
erase_patch_cmd = env_group.create_erase_patch_depth_command(camera_handle, size_x, size_y)
erase_patch_cmd_arr = gym.create_erase_patch_depth_command_gpu_array([erase_patch_cmd])

# == Getting depth camera image == #
assert torch.cuda.is_available(), "Could not find GPU"
device = torch.device("cuda:0")

art_def = env_def.get_articulation_def(cartpole_def_handle)
camera_def = art_def.get_depth_camera_def_by_name("depth_camera")
res_x, res_y = camera_def.resolution_x, camera_def.resolution_y

# Depth image buffer: shape is (num_envs, resolution_y, resolution_x)
get_image_buf = torch.empty((num_envs, res_y, res_x), dtype=torch.float32, device=device)

# Create command
articulation = env_def.get_articulation(cartpole_handle)
camera_handle = articulation.get_depth_camera_handle_by_name("slider_camera")
get_image_cmd = env_group.create_depth_camera_command(
    v.wrap_gpu_buffer(get_image_buf),
    camera_handle)

get_image_cmd_arr = gym.create_depth_camera_command_gpu_array([get_image_cmd])

# == Set up render == #
render = gym.get_render()

render.reset_camera(v.Vec3(5.0, 4.3, 0), v.Vec3(-0.8, -0.5, 0))
render.capped_step = True
render.set_paused(False)

# == GUI controls == #
grab_box = v.UserCheckbox("Grab image", False)
filter_box_0 = v.UserCheckbox("Apply Gaussian noise", False)
filter_box_1 = v.UserCheckbox("Erase pixels", False)
filter_box_2 = v.UserCheckbox("Erase patch", False)

render.register_menu_item(grab_box)
render.register_menu_item(filter_box_0)
render.register_menu_item(filter_box_1)
render.register_menu_item(filter_box_2)


def grab_image():

    if grab_box.get_value() == False:
        return

    grab_box.set_value(False)

    gym.get_depth_camera_images(get_image_cmd_arr)

    plt.imshow(get_image_buf[0].cpu(), vmin=0, vmax=camera_def.far_clip)
    plt.gca().invert_yaxis()

    plt.savefig("304_depth_camera_filters.png")
    print("Image saved in 304_depth_camera_filters.png")

    plt.close()


def apply_filter():

    if filter_box_0.get_value():
        gym.apply_depth_camera_gaussian_noise(gaussian_noise_cmd_arr)

    if filter_box_1.get_value():
        gym.apply_depth_camera_erase_pixels(erase_pixels_cmd_arr)

    if filter_box_2.get_value():
        gym.apply_depth_camera_erase_patch(erase_patch_cmd_arr)


# == Simulation loop == #
timestep = 0.01667
gym.set_timestep(timestep)

done = False
while not done:

    # Grab image
    grab_image()

    # Simulation step
    gym.step()

    # Filter
    apply_filter()

    # Render
    done = render.render_function()
