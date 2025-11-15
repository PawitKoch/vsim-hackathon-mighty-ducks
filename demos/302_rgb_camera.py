import vlearn as v

import torch
import matplotlib.pyplot as plt

# == Initialize Gym == #
gym = v.create_gym(with_render=True, enable_scene_query=True)

# == Environment definition == #
env_def_handle = gym.create_environment_def()
env_def = gym.get_environment_def(env_def_handle)

# Import cartpole definition
env_def.import_definitions("assets/vsim/cartpole_rgb/cartpole.vsim", fixed=True)

cartpole_def_handle = env_def.get_articulation_def_handle_by_name("cartpole")

cartpole_rot = v.shortest_rotation(v.Vec3(0, 0, 1), v.Vec3(0, 1, 0))
cartpole_pos = v.Vec3(0, 1, 0)

cartpole_transform = v.Transform(cartpole_rot, cartpole_pos)
cartpole_handle = env_def.create_articulation(cartpole_def_handle, cartpole_transform, "cartpole")

# Configure RGB camera render
art_def = env_def.get_articulation_def(cartpole_def_handle)

camera_def = art_def.get_rgb_camera_def_by_name("slider_camera_0")
camera = art_def.get_rgb_camera_by_name("slider_camera_0_instance")

rot = v.Quat(v.Vec3(0, 1, 0), torch.pi / 2)
pos = v.Vec3(2, 1.5, 0)
camera.render_relative_transform = v.Transform(rot, pos)

camera.render_width = 2
camera.render_height = 0.5

# Print RGB camera definition
print("> RGB camera definition:")
print(camera_def)

# Print RGB camera
print("> RGB camera:")
print(camera)

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

art_def = env_def.get_articulation_def(cartpole_def_handle)
camera_def = art_def.get_rgb_camera_def_by_name("slider_camera_0")
res_x, res_y = camera_def.resolution_x, camera_def.resolution_y

# == Getting RGB camera image == #
# RGB image buffer: shape is (num_envs, resolution_y, resolution_x, 4)
get_image_buf = torch.empty((num_envs, res_y, res_x, 4), dtype=torch.uint8, device=device)

# Create command
articulation = env_def.get_articulation(cartpole_handle)
camera_handle = articulation.get_rgb_camera_handle_by_name("slider_camera_0_instance")
get_image_cmd = env_group.create_rgb_camera_command(
    v.wrap_gpu_buffer(get_image_buf),
    camera_handle)

get_image_cmd_arr = gym.create_rgb_camera_command_gpu_array([get_image_cmd])

# == Set up render == #
render = gym.get_render()

render.reset_camera(v.Vec3(5.0, 4.3, 0), v.Vec3(-0.8, -0.5, 0))
render.capped_step = True
render.set_paused(False)

# == GUI controls == #
grab_box = v.UserCheckbox("Grab image", False)
render.register_menu_item(grab_box)


def grab_image():

    if grab_box.get_value() == False:
        return

    grab_box.set_value(False)

    gym.get_rgb_camera_images(get_image_cmd_arr)

    plt.imshow(get_image_buf[0].cpu())
    plt.gca().invert_yaxis()

    plt.savefig("302_rgb_camera.png")
    print("Image saved in 302_rgb_camera.png")

    plt.close()


# == Simulation loop == #
timestep = 0.01667
gym.set_timestep(timestep)

done = False
while not done:

    # Grab image
    grab_image()

    # Simulation step
    gym.step()

    # Render
    done = render.render_function()
