import vlearn as v
import torch

# == Initialize Gym == #
gym = v.create_gym(with_render=True, enable_scene_query=True)

# == Environment definition == #
env_def_handle = gym.create_environment_def()
env_def = gym.get_environment_def(env_def_handle)

# Import box definition
filename = "assets/vsim/box_distance_constraint.vsim"
env_def.import_definitions(filename)

box_def_handle = env_def.get_rigid_body_def_handle_by_name("box")

# Instantiate box
rot = v.Quat(0, 0, 0, 1)
pos = v.Vec3(0)
box_transform = v.Transform(rot, pos)

box_handle = env_def.create_rigid_body(box_def_handle, box_transform)

# Finalize environment def
env_def.finalize()

# == Environment group == #
num_envs = 3

env_group_handle = gym.create_environment_group(env_def_handle)
env_group = gym.get_environment_group(env_group_handle)
env_group.create_environment_set(num_envs)
env_group.finalize()

# Randomize the distance joint anchors
assert torch.cuda.is_available(), "Could not find GPU"
device = torch.device("cuda:0")

# Change the anchor positions of the 2nd and 3rd envs, so the cubes will start swinging
# default anchor positions in the vsim file: <anchor1 pos="0 1 0"/> and <anchor2 pos="0 5 0"/>
anchor1_buffer = torch.tensor([[0, 1, 0], [0.5, 1, 0.5], [-1, 1, -1]],
                              dtype=torch.float32, device=device)
anchor2_buffer = torch.tensor([[0, 5, 0], [0.5, 5, 0.5], [-1, 5, -1]],
                              dtype=torch.float32, device=device)

distance_joint_handle = env_def.get_rigid_distance_joint_handle(0)

cmd_1 = env_group.create_rigid_distance_joint_property_command(
    v.RigidDistanceJointProperty.ANCHOR1,
    v.wrap_gpu_buffer(anchor1_buffer),
    distance_joint_handle
    )

cmd_2 = env_group.create_rigid_distance_joint_property_command(
    v.RigidDistanceJointProperty.ANCHOR2,
    v.wrap_gpu_buffer(anchor2_buffer),
    distance_joint_handle
    )

distance_joint_property_cmd_arr = gym.create_rigid_distance_joint_property_command_gpu_array([
                                                                                             cmd_1, cmd_2])

# Set environment transforms
env_group.tile_environments(spacing=4)

# == Finalize Gym == #
gym.gym_finalize()

# == Set up render == #
render = gym.get_render()

render.reset_camera(v.Vec3(10.6, 8.2, -6.7), v.Vec3(-0.6, -0.6, 0.6))
render.capped_step = True
render.set_paused(True)

# == Simulation loop == #
timestep = 0.01667
gym.set_timestep(timestep)

# Execute the domain randomization
gym.set_rigid_distance_joint_properties(distance_joint_property_cmd_arr)

done = False
while not done:

    # Simulation step
    gym.step()

    # Render
    done = render.render_function()
