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
art_def_handle, art_handle = common.import_ant_definition(env_def)

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

art_def = env_def.get_articulation_def(art_def_handle)
num_pids = art_def.get_num_pid_defs()
num_dofs = art_def.get_num_joint_dof_defs()

# == Setting PID targets == #
# PID targets buffer: shape is (num_envs, num_pids)
set_pid_buf = torch.zeros((num_envs, num_pids), dtype=torch.float32, device=device)

# Create command
set_pid_cmd = env_group.create_pid_control_command(
    v.wrap_gpu_buffer(set_pid_buf),
    art_handle,
    (0, num_pids))

set_pid_cmd_arr = gym.create_pid_control_command_gpu_array([set_pid_cmd])

# == Setting joint active masks == #
# Joint active masks buffer: shape is (num_envs, num_dofs)
set_joint_active_buf = torch.zeros((num_envs, num_dofs), dtype=torch.bool, device=device)

# Create command
set_joint_active_cmd = env_group.create_joint_active_mask_command(
    v.wrap_gpu_buffer(set_joint_active_buf),
    art_handle,
    (0, num_dofs))

set_joint_active_cmd_arr = gym.create_joint_active_mask_command_gpu_array([set_joint_active_cmd])

# == Set up render == #
render = gym.get_render()

render.reset_camera(v.Vec3(-0.5, 6.4, -2.9), v.Vec3(0.2, -0.9, 0.4))
render.capped_step = True
render.set_paused(False)

# == GUI controls == #
dof_checkboxes = []
for dof_def in art_def.get_joint_dof_defs():
    name = dof_def.name

    dof_checkboxes.append(v.UserCheckbox(name + " active mask", True))

pid_sliders = []
for pid_def in art_def.get_pid_defs():
    name = pid_def.name

    # Get joint dof definition for PID definition
    dof_idx = art_def.get_joint_dof_index(pid_def.link_index, pid_def.local_dof_index)
    dof_def = art_def.get_joint_dof_def(dof_idx)

    low = dof_def.low_limit
    high = dof_def.high_limit
    init = max(low, min(0, high))

    pid_sliders.append(v.UserSlider(name + " PID target", low, high, init))

for checkbox in dof_checkboxes:
    render.register_menu_item(checkbox)

for slider in pid_sliders:
    render.register_menu_item(slider)


def set_active_joints():
    # Get values from checkboxes
    checkbox_vals = torch.tensor([box.get_value() for box in dof_checkboxes])

    # Write to joint active masks buffer (values are duplicated across all environments)
    set_joint_active_buf[:, :] = checkbox_vals

    # Set joint active masks
    gym.set_joint_active_masks(set_joint_active_cmd_arr)


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

    # Set active joints
    set_active_joints()

    # Set PID targets
    set_pid_targets()

    # Simulation step
    gym.step()

    # Render
    done = render.render_function()
