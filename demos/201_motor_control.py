import vlearn as v
import common

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
ant_def_handle, ant_handle = common.import_ant_definition(env_def)

# Enable motor control
art_def = env_def.get_articulation_def(ant_def_handle)
art_def.enable_control_type(v.ArticulationControlType.MOTOR, True)
assert art_def.has_control_type(v.ArticulationControlType.MOTOR), "Ant does not have motor control"

# Print motor definitions
print("> Motor definitions:")
for motor_def in art_def.get_motor_defs():
    print(motor_def)

# Finalize environment def
env_def.finalize()

# == Environment group == #
num_envs = 2

env_group_handle = gym.create_environment_group(env_def_handle)
env_group = gym.get_environment_group(env_group_handle)
env_group.create_environment_set(num_envs)
env_group.finalize()

# Set environment transforms
env_group.tile_environments(spacing=2)

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
num_motors = art_def.get_num_motor_defs()

# == Setting motor forces == #
# Motor forces buffer: shape is (num_envs, num_motors)
set_motor_buf = torch.zeros((num_envs, num_motors), dtype=torch.float32, device=device)

# Create command
set_motor_cmd = env_group.create_motor_control_command(
    v.wrap_gpu_buffer(set_motor_buf),
    ant_handle,
    (0, num_motors))

set_motor_cmd_arr = gym.create_motor_control_command_gpu_array([set_motor_cmd])

# == Set up render == #
render = gym.get_render()

render.reset_camera(v.Vec3(-0.5, 6.4, -2.9), v.Vec3(0.2, -0.9, 0.4))
render.capped_step = True
render.set_paused(False)

# == GUI controls == #
motor_sliders = []
for motor_def in art_def.get_motor_defs():
    name = motor_def.name
    low = motor_def.low_limit
    high = motor_def.high_limit

    motor_sliders.append(v.UserSlider(name, low, high, 0))

for slider in motor_sliders:
    render.register_menu_item(slider)


def set_motor_forces():
    # Get values from sliders
    slider_vals = torch.tensor([slider.get_value() for slider in motor_sliders])

    # Write to motor forces buffer (values are duplicated across all environments)
    set_motor_buf[:, :] = slider_vals

    # Set motor forces
    gym.set_motor_forces(set_motor_cmd_arr)


# == Simulation loop == #
timestep = 0.01667
gym.set_timestep(timestep)

done = False
while not done:

    # Set motor forces
    set_motor_forces()

    # Simulation step
    gym.step()

    # Render
    done = render.render_function()
