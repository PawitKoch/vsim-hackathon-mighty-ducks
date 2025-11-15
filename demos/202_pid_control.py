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

# Print PID definitions
art_def = env_def.get_articulation_def(ant_def_handle)
print("> PID definitions:")
for pid_def in art_def.get_pid_defs():
    print(pid_def)

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
num_pids = art_def.get_num_pid_defs()

# == Setting PID targets == #
# PID targets buffer: shape is (num_envs, num_pids)
set_pid_buf = torch.zeros((num_envs, num_pids), dtype=torch.float32, device=device)

# Create command
set_pid_cmd = env_group.create_pid_control_command(
    v.wrap_gpu_buffer(set_pid_buf),
    ant_handle,
    (0, num_pids))

set_pid_cmd_arr = gym.create_pid_control_command_gpu_array([set_pid_cmd])

# == Set up render == #
render = gym.get_render()

render.reset_camera(v.Vec3(-0.5, 6.4, -2.9), v.Vec3(0.2, -0.9, 0.4))
render.capped_step = True
render.set_paused(False)

# == GUI controls == #
pid_sliders = []
for pid_def in art_def.get_pid_defs():
    name = pid_def.name

    # Get joint dof definition for PID definition
    dof_idx = art_def.get_joint_dof_index(pid_def.link_index, pid_def.local_dof_index)
    dof_def = art_def.get_joint_dof_def(dof_idx)

    low = dof_def.low_limit
    high = dof_def.high_limit
    init = max(low, min(0, high))

    pid_sliders.append(v.UserSlider(name, low, high, init))

for slider in pid_sliders:
    render.register_menu_item(slider)


def set_pid_targets():
    # Get values from sliders
    slider_vals = torch.tensor([slider.get_value() for slider in pid_sliders])

    # Write to pid targets buffer (values are duplicated across all environments)
    set_pid_buf[:, :] = slider_vals

    # Set joint target positions
    gym.set_joint_target_positions(set_pid_cmd_arr)


# == Simulation loop == #
timestep = 0.01667
gym.set_timestep(timestep)

done = False
while not done:

    # Set PID targets
    set_pid_targets()

    # Simulation step
    gym.step()

    # Render
    done = render.render_function()
