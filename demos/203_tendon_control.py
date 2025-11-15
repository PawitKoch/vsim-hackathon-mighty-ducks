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

# Import TendonRig4.vsim
filename = "assets/vsim/Tendon/TendonRig4.vsim"
env_def.import_definitions(filename, fixed=True)

art_def_handle = env_def.get_articulation_def_handle_by_name("TendonRig")
art_def = env_def.get_articulation_def(art_def_handle)

print("> Spatial tendon definitions:")
for tendon_def in art_def.get_spatial_tendon_defs():
    tendon_def.rest_offset = 0.1
    print(tendon_def)

# Instantiate TendonRig4.vsim
rot = v.shortest_rotation(v.Vec3(0, 0, 1), v.Vec3(0, 1, 0))
pos = v.Vec3(0, 0, 0)
transform = v.Transform(rot, pos)

art_handle = env_def.create_articulation(art_def_handle, transform)

# Finalize environment def
env_def.finalize()

# == Environment group == #
num_envs = 2

env_group_handle = gym.create_environment_group(env_def_handle)
env_group = gym.get_environment_group(env_group_handle)
env_group.create_environment_set(num_envs)
env_group.finalize()

# Set environment transforms
env_group.tile_environments(spacing=1)

# == Finalize Gym == #
gym.gym_finalize()

# ================== #
# == GPU commands == #
# ================== #
assert torch.cuda.is_available(), "Could not find GPU"
device = torch.device("cuda:0")

art_def = env_def.get_articulation_def(art_def_handle)
num_tendons = art_def.get_num_spatial_tendon_defs()

# == Setting control offset == #
# Control offsets buffer: shape is (num_envs, num_tendons)
set_offset_buf = torch.zeros((num_envs, num_tendons), dtype=torch.float32, device=device)

# Create command
set_offset_cmd = env_group.create_spatial_tendon_control_command(
    v.wrap_gpu_buffer(set_offset_buf),
    art_handle,
    (0, num_tendons))

set_offset_cmd_arr = gym.create_spatial_tendon_control_command_gpu_array([set_offset_cmd])

# == Getting tendon state == #
# = Tendon length = #
# Tendon length buffer: shape is (num_envs, num_tendons)
get_length_buf = torch.zeros((num_envs, num_tendons), dtype=torch.float32, device=device)

# Create command
get_length_cmd = env_group.create_spatial_tendon_state_command(
    v.SpatialTendonState.LENGTH,
    v.wrap_gpu_buffer(get_length_buf),
    art_handle,
    (0, num_tendons))

# = Tendon velocity = #
# Tendon velocity buffer: shape is (num_envs, num_tendons)
get_velocity_buf = torch.zeros((num_envs, num_tendons), dtype=torch.float32, device=device)

# Create command
get_velocity_cmd = env_group.create_spatial_tendon_state_command(
    v.SpatialTendonState.VELOCITY,
    v.wrap_gpu_buffer(get_velocity_buf),
    art_handle,
    (0, num_tendons))

# = Tendon force = #
# Tendon force buffer: shape is (num_envs, num_tendons)
get_force_buf = torch.zeros((num_envs, num_tendons), dtype=torch.float32, device=device)

# Create command
get_force_cmd = env_group.create_spatial_tendon_state_command(
    v.SpatialTendonState.FORCE,
    v.wrap_gpu_buffer(get_force_buf),
    art_handle,
    (0, num_tendons))

get_tendon_state_cmd_arr = gym.create_spatial_tendon_state_command_gpu_array(
    [get_length_cmd, get_velocity_cmd, get_force_cmd])

# == Set up render == #
render = gym.get_render()

render.reset_camera(v.Vec3(-3.2, 3.9, 0.6), v.Vec3(0.6, -0.8, 0))
render.capped_step = True
render.set_paused(False)

# == GUI controls == #
print_box = v.UserCheckbox("Print tendon state of environment 0", False)

render.register_menu_item(print_box)

offset_sliders = []
for tendon_def in art_def.get_spatial_tendon_defs():
    name = tendon_def.name
    low = -0.2
    high = 0.2

    offset_sliders.append(v.UserSlider(name, low, high, 0))

for slider in offset_sliders:
    render.register_menu_item(slider)


def print_tendon_states():

    if print_box.get_value() == False:
        return

    gym.get_spatial_tendon_states(get_tendon_state_cmd_arr)

    length = get_length_buf.cpu()[0]
    velocity = get_velocity_buf.cpu()[0]
    force = get_force_buf.cpu()[0]

    print("Tendon lengths: " + ", ".join(["{:.4f}".format(num) for num in length]))
    print("Tendon velocities: " + ", ".join(["{:.4f}".format(num) for num in velocity]))
    print("Tendon forces: " + ", ".join(["{:.4f}".format(num) for num in force]))


def set_control_offsets():
    # Get values from sliders
    slider_vals = torch.tensor([slider.get_value() for slider in offset_sliders])

    # Write to control offsets buffer (values are duplicated across all environments)
    set_offset_buf[:, :] = slider_vals

    # Set control offset
    gym.set_spatial_tendon_controls(set_offset_cmd_arr)


# == Simulation loop == #
timestep = 0.01667
gym.set_timestep(timestep)

done = False
while not done:

    # Set control offsets
    set_control_offsets()

    # Simulation step
    gym.step()

    # Print tendon states
    print_tendon_states()

    # Render
    done = render.render_function()
