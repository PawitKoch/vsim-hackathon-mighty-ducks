import vlearn as v

import torch

# == Initialize Gym == #
gym = v.create_gym(with_render=True,
                   enable_scene_query=True,
                   max_contact_pairs=512 * 1024,
                   max_patches=512 * 1024,
                   max_contacts=4 * 512 * 1024)

# == Environment definition 1 == #
env_def_handle1 = gym.create_environment_def("ant on a box")
env_def1 = gym.get_environment_def(env_def_handle1)

# Create box definition
box_def_handle = env_def1.create_box_def(v.Vec3(1, 0.5, 1))

# Import ant definition
filename = "assets/vsim/ant.vsim"
env_def1.import_definitions(filename, fixed=False)

ant_def_handle = env_def1.get_articulation_def_handle_by_name("torso")

# Instantiate environment
# Box
rot = v.Quat(0, 0, 0, 1)
pos = v.Vec3(0, 0.5, 0)
box_transform = v.Transform(rot, pos)

env_def1.create_rigid_body(box_def_handle, box_transform)

# Ant
ant_rot = v.shortest_rotation(v.Vec3(0, 0, 1), v.Vec3(0, 1, 0))
ant_pos = v.Vec3(0, 2, 0)
ant_transform = v.Transform(ant_rot, ant_pos)

ant_handle = env_def1.create_articulation(ant_def_handle, ant_transform)

# Finalize environment def
env_def1.finalize()

# == Environment definition 2 == #
env_def_handle2 = gym.create_environment_def("humanoid on a box")
env_def2 = gym.get_environment_def(env_def_handle2)

# Create box definition
box_def_handle = env_def2.create_box_def(v.Vec3(1, 0.5, 1))

# Import ant definition
filename = "assets/vsim/humanoid.vsim"
env_def2.import_definitions(filename, fixed=False)

humanoid_def_handle = env_def2.get_articulation_def_handle_by_name("torso")

# Instantiate environment
# Box
rot = v.Quat(0, 0, 0, 1)
pos = v.Vec3(0, 0.5, 0)
box_transform = v.Transform(rot, pos)

env_def2.create_rigid_body(box_def_handle, box_transform)

# Humanoid
humanoid_rot = v.shortest_rotation(v.Vec3(0, 0, 1), v.Vec3(0, 1, 0))
humanoid_pos = v.Vec3(0, 3, 0)
humanoid_transform = v.Transform(humanoid_rot, humanoid_pos)

humanoid_handle = env_def2.create_articulation(humanoid_def_handle, humanoid_transform)

# Finalize environment def
env_def2.finalize()

# == Environment group 1 == #
num_envs1 = 2

env_group_handle1 = gym.create_environment_group(env_def_handle1)
env_group1 = gym.get_environment_group(env_group_handle1)
env_group1.create_environment_set(num_envs1)
env_group1.finalize()

# Set environment transforms
env_group1.tile_environments(spacing=3)

# == Environment group 2 == #
num_envs2 = 3

env_group_handle2 = gym.create_environment_group(env_def_handle2)
env_group2 = gym.get_environment_group(env_group_handle2)
env_group2.create_environment_set(num_envs2)
env_group2.finalize()

# Set environment transforms
env_group2.tile_environments(spacing=3, offset=v.Vec3(0, 0, 3))

# == Create plane == #
gym.create_plane()

# == Finalize Gym == #
gym.gym_finalize()

# ================== #
# == GPU commands == #
# ================== #
assert torch.cuda.is_available(), "Could not find GPU"
device = torch.device("cuda:0")

# == Setting kinematic state for environment group 1 == #
art_def = env_def1.get_articulation_def(ant_def_handle)
num_dofs = art_def.get_num_joint_dof_defs()

# Compute initial joint positions: 0 clamped by joint limits
dof_pos_init = []
for dof_def in art_def.get_joint_dof_defs():
    low, high = dof_def.get_limits()
    init = max(low, min(0, high))
    dof_pos_init.append(init)

# Joint positions buffer: shape is (num_envs, num_dofs)
ant_set_dof_pos_buf = torch.tensor([dof_pos_init] * num_envs1, dtype=torch.float32, device=device)

# Joint velocities buffer: shape is (num_envs, num_dofs)
ant_set_dof_vel_buf = torch.zeros_like(ant_set_dof_pos_buf)

# Root positions buffer: shape is (num_envs, 7)
ant_set_root_pos_buf = torch.tensor([[ant_rot.x,
                                      ant_rot.y,
                                      ant_rot.z,
                                      ant_rot.w,
                                      ant_pos.x,
                                      ant_pos.y,
                                      ant_pos.z]] * num_envs1,
                                    dtype=torch.float32,
                                    device=device)

# Root velocities buffer: shape is (num_envs, 6)
ant_set_root_vel_buf = torch.zeros((num_envs1, 6), dtype=torch.float32, device=device)

# Reset buffer: shape is (num_envs,)
ant_reset_buf = torch.zeros(num_envs1, dtype=torch.bool, device=device)

# Create command
ant_set_kine_cmd = env_group1.create_articulation_kinematic_state_command(
    v.wrap_gpu_buffer(ant_set_dof_pos_buf),
    v.wrap_gpu_buffer(ant_set_dof_vel_buf),
    v.wrap_gpu_buffer(ant_set_root_pos_buf),
    v.wrap_gpu_buffer(ant_set_root_vel_buf),
    ant_handle,
    (0, num_dofs),
    (0, 1),
    masks_buffer=v.wrap_gpu_buffer(ant_reset_buf))

# == Setting kinematic state for environment group 2 == #
art_def = env_def2.get_articulation_def(humanoid_def_handle)
num_dofs = art_def.get_num_joint_dof_defs()

# Compute initial joint positions: 0 clamped by joint limits
dof_pos_init = []
for dof_def in art_def.get_joint_dof_defs():
    low, high = dof_def.get_limits()
    init = max(low, min(0, high))
    dof_pos_init.append(init)

# Joint positions buffer: shape is (num_envs, num_dofs)
humanoid_set_dof_pos_buf = torch.tensor([dof_pos_init] * num_envs2, dtype=torch.float32,
                                        device=device)

# Joint velocities buffer: shape is (num_envs, num_dofs)
humanoid_set_dof_vel_buf = torch.zeros_like(humanoid_set_dof_pos_buf)

# Root positions buffer: shape is (num_envs, 7)
humanoid_set_root_pos_buf = torch.tensor([[humanoid_rot.x,
                                           humanoid_rot.y,
                                           humanoid_rot.z,
                                           humanoid_rot.w,
                                           humanoid_pos.x,
                                           humanoid_pos.y,
                                           humanoid_pos.z]] * num_envs2,
                                         dtype=torch.float32,
                                         device=device)

# Root velocities buffer: shape is (num_envs, 6)
humanoid_set_root_vel_buf = torch.zeros((num_envs2, 6), dtype=torch.float32, device=device)

# Reset buffer: shape is (num_envs,)
humanoid_reset_buf = torch.zeros(num_envs2, dtype=torch.bool, device=device)

# Create command
humanoid_set_kine_cmd = env_group2.create_articulation_kinematic_state_command(
    v.wrap_gpu_buffer(humanoid_set_dof_pos_buf),
    v.wrap_gpu_buffer(humanoid_set_dof_vel_buf),
    v.wrap_gpu_buffer(humanoid_set_root_pos_buf),
    v.wrap_gpu_buffer(humanoid_set_root_vel_buf),
    humanoid_handle,
    (0, num_dofs),
    (0, 1),
    masks_buffer=v.wrap_gpu_buffer(humanoid_reset_buf))

# == Create command GPU array == #
set_kine_cmd_arr = gym.create_articulation_kinematic_state_command_gpu_array(
    [ant_set_kine_cmd, humanoid_set_kine_cmd])

# == Set up render == #
render = gym.get_render()

render.reset_camera(v.Vec3(-0.7, 6.8, -3.2), v.Vec3(0.4, -0.7, 0.5))
render.capped_step = True
render.set_paused(False)

# == GUI controls == #
reset_box1 = v.UserCheckbox("Reset environment group 1", False)
reset_box2 = v.UserCheckbox("Reset environment group 2", False)

render.register_menu_item(reset_box1)
render.register_menu_item(reset_box2)


def reset_environment_groups():

    # Set reset buffers
    ant_reset_buf[:] = reset_box1.get_value()
    humanoid_reset_buf[:] = reset_box2.get_value()

    # Reset tick box
    reset_box1.set_value(False)
    reset_box2.set_value(False)

    # Set kinematic states
    gym.set_articulation_kinematic_states(set_kine_cmd_arr)


# == Simulation loop == #
timestep = 0.01667
gym.set_timestep(timestep)

done = False
while not done:

    # Reset environment groups
    reset_environment_groups()

    # Simulation step
    gym.step()

    # Render
    done = render.render_function()
