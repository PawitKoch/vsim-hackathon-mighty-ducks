import vlearn as v

import torch

# == Initialize Gym == #
gym = v.create_gym(with_render=True, enable_scene_query=True)

# == Environment definition == #
env_def_handle = gym.create_environment_def()
env_def = gym.get_environment_def(env_def_handle)

# Import spring definition
env_def.import_definitions("assets/vsim/spring.vsim", fixed=True)

spring_def_handle = env_def.get_articulation_def_handle_by_name("spring")

spring_rot = v.Quat(0, 0, 0, 1)
spring_pos = v.Vec3(0, 0, 0)

spring_transform = v.Transform(spring_rot, spring_pos)
env_def.create_articulation(spring_def_handle, spring_transform, "spring")

# Print link defs
print("> Link definitions:")
art_def = env_def.get_articulation_def(spring_def_handle)
for link_def in art_def.get_link_defs():
    print(link_def)

# Finalize environment def
env_def.finalize()

# == Environment group == #
num_env_sets = 2
num_envs_per_env_set = 2

env_group_handle = gym.create_environment_group(env_def_handle)
env_group = gym.get_environment_group(env_group_handle)

for i in range(num_env_sets):
    env_group.create_environment_set(num_envs_per_env_set)

env_group.finalize()

# Set environment transforms
env_group.tile_environments(spacing=3)

# == Finalize Gym == #
gym.gym_finalize()

# ================== #
# == GPU commands == #
# ================== #
assert torch.cuda.is_available(), "Could not find GPU"
device = torch.device("cuda:0")

# Change the mass of the weight only, i.e. the second link
link_index_range = (1, 2)
num_links = 1

# == Setting link masses == #
# Link masses buffer: shape is (num_env_sets, num_links)
set_mass_buf = torch.empty((num_env_sets, num_links), dtype=torch.float32, device=device)

# Create command
set_mass_cmd = env_group.create_link_property_command(
    v.LinkProperty.MASS,
    v.wrap_gpu_buffer(set_mass_buf),
    spring_def_handle,
    link_index_range)

set_mass_cmd_arr = gym.create_link_property_command_gpu_array([set_mass_cmd])

# == Set up render == #
render = gym.get_render()

render.reset_camera(v.Vec3(4.8, 0.2, 15.8), v.Vec3(-0.1, -0.2, -1.0))
render.capped_step = True
render.set_paused(False)

# == GUI controls == #
low, high = 0.001, 10
mass_sliders = []
for i in range(num_env_sets):
    mass_sliders.append(v.UserSlider(f"Mass in environment set {i}", low, high, 1))

for slider in mass_sliders:
    render.register_menu_item(slider)


def set_link_masses():
    # Get values from sliders
    slider_vals = torch.tensor([[slider.get_value()] for slider in mass_sliders])

    # Write to link mass buffer
    set_mass_buf[:, :] = slider_vals

    # Set link masses
    gym.set_link_properties(set_mass_cmd_arr)


# == Simulation loop == #
timestep = 0.01667
gym.set_timestep(timestep)

done = False
while not done:

    # Set link masses
    set_link_masses()

    # Simulation step
    gym.step()

    # Render
    done = render.render_function()
