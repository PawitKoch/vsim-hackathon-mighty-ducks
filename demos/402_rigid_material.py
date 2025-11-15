import vlearn as v

import torch

# == Initialize Gym == #
gym = v.create_gym(with_render=True, enable_scene_query=True)

# == Plane incline == #
plane_rot = v.Quat(v.Vec3(1, 0, 0), - 35 / 180 * torch.pi)

# == Environment definition == #
env_def_handle = gym.create_environment_def()
env_def = gym.get_environment_def(env_def_handle)

# Create rigid material
rigid_mat = v.RigidMaterial()
rigid_mat.dynamic_friction = 1
rigid_mat.static_friction = 1

rigid_mat_handle = env_def.create_rigid_material(rigid_mat)

# Create box definition
box_def_handle = env_def.create_box_def(v.Vec3(1), rigid_material_handle=rigid_mat_handle)

# Instantiate box
rot = plane_rot
pos = plane_rot.rotate(v.Vec3(0, 1, 0))
box_transform = v.Transform(rot, pos)

env_def.create_rigid_body(box_def_handle, box_transform)

# Import ant definition
filename = "assets/vsim/ant.vsim"
env_def.import_definitions(filename)

ant_def_handle = env_def.get_articulation_def_handle_by_name("torso")
art_def = env_def.get_articulation_def(ant_def_handle)

# Instantiate ant
ant_rot = plane_rot * v.shortest_rotation(v.Vec3(0, 0, 1), v.Vec3(0, 1, 0))
ant_pos = ant_rot.rotate(v.Vec3(0, -5, 0.75))

ant_transform = v.Transform(ant_rot, ant_pos)
ant_handle = env_def.create_articulation(ant_def_handle, ant_transform)

for i in range(art_def.get_num_link_defs()):
    env_def.assign_rigid_material_to_articulation_link(ant_def_handle, rigid_mat_handle, i)

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

# == Create plane == #
gym.create_plane(plane_transform=v.Transform(plane_rot, v.Vec3(0)))

# == Finalize Gym == #
gym.gym_finalize()

# ================== #
# == GPU commands == #
# ================== #
assert torch.cuda.is_available(), "Could not find GPU"
device = torch.device("cuda:0")

# == Setting dynamic frictions == #
# Dynamic friction buffer: shape is (num_env_sets,)
set_dynamic_buf = torch.empty((num_env_sets,), dtype=torch.float32, device=device)

# Create command
set_dynamic_cmd = env_group.create_rigid_material_property_command(
    v.RigidMaterialProperty.DYNAMIC_FRICTION,
    v.wrap_gpu_buffer(set_dynamic_buf),
    rigid_mat_handle)

# == Setting static frictions == #
# Static friction buffer: shape is (num_env_sets,)
set_static_buf = torch.empty((num_env_sets,), dtype=torch.float32, device=device)

# Create command
set_static_cmd = env_group.create_rigid_material_property_command(
    v.RigidMaterialProperty.STATIC_FRICTION,
    v.wrap_gpu_buffer(set_static_buf),
    rigid_mat_handle)

set_friction_cmd_arr = gym.create_rigid_material_property_command_gpu_array([set_static_cmd,
                                                                             set_dynamic_cmd])

# == Set up render == #
render = gym.get_render()

render.reset_camera(v.Vec3(0.9, 22.3, -1.1), v.Vec3(0.0, -1.0, -0.2))
render.capped_step = True
render.set_paused(False)

# == GUI controls == #
low, high, init = 0.001, 2, 1
static_sliders = []
dynamic_sliders = []
for i in range(num_env_sets):
    static_sliders.append(v.UserSlider(f"Env set {i} static friction", low, high, init))
    dynamic_sliders.append(v.UserSlider(f"Env set {i} dynamic friction", low, high, init))

for static_slider, dynamic_slider in zip(static_sliders, dynamic_sliders):
    render.register_menu_item(static_slider)
    render.register_menu_item(dynamic_slider)


def set_frictions():
    # Get values from sliders
    static_vals = torch.tensor([slider.get_value() for slider in static_sliders])
    dynamic_vals = torch.tensor([slider.get_value() for slider in dynamic_sliders])

    # Write to buffers
    set_static_buf[:] = static_vals
    set_dynamic_buf[:] = dynamic_vals

    # Set frictions
    gym.set_rigid_material_properties(set_friction_cmd_arr)


# == Simulation loop == #
timestep = 0.01667
gym.set_timestep(timestep)

done = False
while not done:

    # Set frictions
    set_frictions()

    # Simulation step
    gym.step()

    # Render
    done = render.render_function()
