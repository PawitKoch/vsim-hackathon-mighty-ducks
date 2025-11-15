import vlearn as v
import torch

# == Initialize Gym == #
gym = v.create_gym(with_render=True, enable_scene_query=True)

# == Environment definition == #
env_def_handle = gym.create_environment_def()
env_def = gym.get_environment_def(env_def_handle)

# Import ant definition
env_def.import_definitions("assets/vsim/ant.vsim", fixed=True, scale=1, alias="ant")
ant_def_handle = env_def.get_articulation_def_handle_by_name("ant.torso")

# Instantiate ant
rot = v.shortest_rotation(v.Vec3(0, 0, 1), v.Vec3(0, 1, 0))
pos = v.Vec3(0.0, 0.7, 0)

ant_transform = v.Transform(rot, pos)
ant_handle = env_def.create_articulation(ant_def_handle, ant_transform)

# Finalize environment def
env_def.finalize()

# == Environment group == #
num_env_sets = 3
num_envs_per_env_set = 2
total_num_envs = num_env_sets * num_envs_per_env_set

env_group_handle = gym.create_environment_group(env_def_handle)
env_group = gym.get_environment_group(env_group_handle)

for i in range(num_env_sets):
    env_group.create_environment_set(num_envs_per_env_set)

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
assert torch.cuda.is_available(), "GPU required"
device = torch.device("cuda:0")

# == Setting child frame == #
masks_buffer = torch.tensor([1, 0, 1], dtype=torch.bool, device=device)

# Set child frame buffer: shape is (num_env_sets, 7)
set_child_frame_buf = torch.empty((num_env_sets, 7), dtype=torch.float32, device=device)

# Create command
set_child_frame_cmd = env_group.create_joint_property_command(
    v.JointProperty.CHILD_FRAME,
    v.wrap_gpu_buffer(set_child_frame_buf),
    ant_def_handle,
    [0, 1],
    masks_buffer=v.wrap_gpu_buffer(masks_buffer),
    )

set_child_frame_cmd_arr = gym.create_joint_property_command_gpu_array([set_child_frame_cmd])

# == Getting child frame == #
# Get child frame buffer: shape is (num_env_sets, 7)
get_child_frame_buf = torch.empty((num_env_sets, 7), dtype=torch.float32, device=device)

# Create command
get_child_frame_cmd = env_group.create_joint_property_command(
    v.JointProperty.CHILD_FRAME,
    v.wrap_gpu_buffer(get_child_frame_buf),
    ant_def_handle,
    [0, 1],
    )

get_child_frame_cmd_arr = gym.create_joint_property_command_gpu_array([get_child_frame_cmd])

# == Set up render == #
render = gym.get_render()
render.reset_camera(v.Vec3(-1.8, 4.9, 9.7), v.Vec3(0.3, -0.4, -0.8))
render.capped_step = True
render.set_paused(False)

# == GUI controls == #
angle_limits = (-torch.pi, torch.pi)
pos_limits = (-0.5, 0.5)

rpy_sliders = [
    v.UserSlider("roll", *angle_limits, 0.0),
    v.UserSlider("pitch", *angle_limits, 0.0),
    v.UserSlider("yaw", *angle_limits, 0.0),
    ]
pos_sliders = [
    v.UserSlider("x_offset", *pos_limits, 0.0),
    v.UserSlider("y_offset", *pos_limits, 0.0),
    v.UserSlider("z_offset", *pos_limits, 0.0),
    ]

for s in rpy_sliders + pos_sliders:
    render.register_menu_item(s)

prev_rpy = torch.tensor([s.get_value() for s in rpy_sliders], device=device)
prev_pos = torch.tensor([s.get_value() for s in pos_sliders], device=device)


def set_child_frames():
    global prev_rpy, prev_pos

    # Get values from sliders
    rpy = torch.tensor([s.get_value() for s in rpy_sliders], device=device)
    pos = torch.tensor([s.get_value() for s in pos_sliders], device=device)

    if not torch.allclose(rpy, prev_rpy) or not torch.allclose(pos, prev_pos):

        prev_rpy = rpy
        prev_pos = pos

        # Write to buffers
        quat = v.quat_from_rpy(v.Vec3(*rpy[0:3]))
        set_child_frame_buf[:, 0:4] = torch.tensor([quat.x, quat.y, quat.z, quat.w], device=device)
        set_child_frame_buf[:, 4:] = pos

        # Set child frame transforms
        gym.set_joint_properties(set_child_frame_cmd_arr)


def print_child_frames():

    gym.get_joint_properties(get_child_frame_cmd_arr)

    print("get_child_frame_buf", get_child_frame_buf)


# == Simulation loop == #
timestep = 0.01667
gym.set_timestep(timestep)

done = False
while not done:

    # Set child frames
    set_child_frames()

    # Simulation step
    gym.step()

    # Render
    done = render.render_function()

    # Print child frames
    print_child_frames()
