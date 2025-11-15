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

cartpole_rot = v.shortest_rotation(v.Vec3(0, 0, 1), v.Vec3(0, 1, 0))
cartpole_pos = v.Vec3(0, 1, 0)

cartpole_transform = v.Transform(cartpole_rot, cartpole_pos)
cartpole_handle = env_def.create_articulation(cartpole_def_handle, cartpole_transform, "cartpole")

# Configure RGB camera render
art_def = env_def.get_articulation_def(cartpole_def_handle)

camera = art_def.get_rgb_camera(0)

rot = v.Quat(v.Vec3(0, 1, 0), torch.pi / 2)
pos = v.Vec3(2, 1.5, 0)
camera.render_relative_transform = v.Transform(rot, pos)

camera.render_width = 2
camera.render_height = 0.5

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

# == Getting light transform == #
# Get light transform buffer: shape is (num_envs, 7)
get_light_transform_buf = torch.empty((num_envs, 7), dtype=torch.float32, device=device)

# Create command
articulation = env_def.get_articulation(cartpole_handle)
light_handle = articulation.get_light_handle(0)
get_light_transform_cmd = env_group.create_light_transform_command(
    v.wrap_gpu_buffer(get_light_transform_buf),
    light_handle)

get_light_transform_arr = gym.create_light_transform_command_gpu_array(
    [get_light_transform_cmd])

# == Setting light transform == #
# Set light transform buffer: shape is (num_envs, 7)
set_light_transform_buf = torch.empty((num_envs, 7), dtype=torch.float32, device=device)

# Create command
set_light_transform_cmd = env_group.create_light_transform_command(
    v.wrap_gpu_buffer(set_light_transform_buf),
    light_handle)

set_light_transform_arr = gym.create_light_transform_command_gpu_array(
    [set_light_transform_cmd])

# Get initial light transform
gym.get_light_transforms(get_light_transform_arr)
print("Initial light transform: ", get_light_transform_buf)

# == Set up render == #
render = gym.get_render()

render.reset_camera(v.Vec3(7.7, 4.2, -1.6), v.Vec3(-0.7, -0.5, 0.5))
render.capped_step = True
render.set_paused(False)

# == GUI controls == #
low, high, init = 0, 10, 5
D_sliders = []
for i in range(num_envs):
    D_sliders.append(v.UserSlider(f"Env {i} light distance", low, high, init))

for D_slider in D_sliders:
    render.register_menu_item(D_slider)


def set_distances():
    # Get values from sliders
    D_vals = torch.tensor([slider.get_value() for slider in D_sliders])

    # Copy initial light position
    set_light_transform_buf[:] = get_light_transform_buf

    # Assign distance in X dir
    set_light_transform_buf[:, 4] = D_vals

    # Set distances
    gym.set_light_transforms(set_light_transform_arr)


# == Simulation loop == #
timestep = 0.01667
gym.set_timestep(timestep)

done = False
while not done:

    # Assign light distances
    set_distances()

    # Simulation step
    gym.step()

    # Render
    done = render.render_function()
