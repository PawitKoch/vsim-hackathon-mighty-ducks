import vlearn as v
from math import pi

# == Initialize Gym == #
gym = v.create_gym(with_render=True,
                   enable_scene_query=True,
                   broad_phase_type=v.BroadPhaseType.ENVIRONMENT,
                   # broad_phase_type = v.BroadPhaseType.SCENE,
                   max_contact_pairs=512 * 1024,
                   max_patches=512 * 1024,
                   max_contacts=4 * 512 * 1024)

# == Environment definition == #
env_def_handle = gym.create_environment_def()
env_def = gym.get_environment_def(env_def_handle)

# Import ant definition
filename = "assets/vsim/ant.vsim"
env_def.import_definitions(filename, fixed=False, alias="ant")

ant_def_handle = env_def.get_articulation_def_handle_by_name("ant.torso")

# Import humanoid definition
filename = "assets/vsim/humanoid.vsim"
env_def.import_definitions(filename, fixed=False, alias="humanoid")

humanoid_def_handle = env_def.get_articulation_def_handle_by_name("humanoid.torso")

humanoid_def = env_def.get_articulation_def(humanoid_def_handle)
humanoid_def.has_self_collisions = True

# Create box definition
box_def_handle = env_def.create_box_def(v.Vec3(1, 0.5, 1))

# Instantiate environment
# Box
rot = v.Quat(0, 0, 0, 1)
pos = v.Vec3(0, 0.5, 0)
box_transform = v.Transform(rot, pos)

env_def.create_rigid_body(box_def_handle, box_transform)

# Ant
rot = v.shortest_rotation(v.Vec3(0, 0, 1), v.Vec3(0, 1, 0))
pos = v.Vec3(0, 30, 0)
ant_transform = v.Transform(rot, pos)

env_def.create_articulation(ant_def_handle, ant_transform)

# Humanoid 1
rot = v.Quat(v.Vec3(0, 1, 0), 0) * v.shortest_rotation(v.Vec3(0, 0, 1), v.Vec3(0, 1, 0))
pos = v.Vec3(-2, 1.5, 0)
humanoid_transform = v.Transform(rot, pos)

env_def.create_articulation(humanoid_def_handle, humanoid_transform)

# Humanoid 2
rot = v.Quat(v.Vec3(0, 1, 0), pi) * v.shortest_rotation(v.Vec3(0, 0, 1), v.Vec3(0, 1, 0))
pos = v.Vec3(2, 1.5, 0)
humanoid_transform = v.Transform(rot, pos)

env_def.create_articulation(humanoid_def_handle, humanoid_transform)

# Humanoid 3
rot = v.Quat(v.Vec3(0, 1, 0), - pi / 2) * v.shortest_rotation(v.Vec3(0, 0, 1), v.Vec3(0, 1, 0))
pos = v.Vec3(0, 1.5, -2)
humanoid_transform = v.Transform(rot, pos)

env_def.create_articulation(humanoid_def_handle, humanoid_transform)

# Humanoid 4
rot = v.Quat(v.Vec3(0, 1, 0), pi / 2) * v.shortest_rotation(v.Vec3(0, 0, 1), v.Vec3(0, 1, 0))
pos = v.Vec3(0, 1.5, 2)
humanoid_transform = v.Transform(rot, pos)

env_def.create_articulation(humanoid_def_handle, humanoid_transform)

# Finalize environment def
env_def.finalize()

# == Environment group == #
num_envs = 1024

env_group_handle = gym.create_environment_group(env_def_handle)
env_group = gym.get_environment_group(env_group_handle)
env_group.create_environment_set(num_envs)
env_group.finalize()

# Set environment transforms
env_rot = v.Quat(0, 0, 0, 1)

row_size = 32
spacing = 5

for idx, env in enumerate(env_group.get_environments()):

    x = idx % row_size
    z = idx // row_size

    env_pos = spacing * v.Vec3(x, 0, z)

    env_transform = v.Transform(env_rot, env_pos)
    env.set_transform(env_transform)

# == Create plane == #
gym.create_plane()

# == Finalize Gym == #
gym.gym_finalize()

# == Set up render == #
render = gym.get_render()

render.reset_camera(v.Vec3(-15.5, 12, -13.5), v.Vec3(0.7, -0.2, 0.7))
render.capped_step = True
render.set_paused(False)

# == Simulation loop == #
timestep = 0.01667
gym.set_timestep(timestep)

done = False
while not done:

    # Simulation step
    gym.step()

    # Render
    done = render.render_function()
