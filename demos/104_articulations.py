import vlearn as v

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
filename = "assets/vsim/ant.vsim"
env_def.import_definitions(filename, fixed=True, scale=20, alias="ant")

ant_def_handle = env_def.get_articulation_def_handle_by_name("ant.torso")

# Print articulation definition
art_def = env_def.get_articulation_def(ant_def_handle)
print("> Articulation definition")
print(art_def)

# Instantiate ant
rot = v.shortest_rotation(v.Vec3(0, 0, 1), v.Vec3(0, 1, 0))
pos = v.Vec3(0, 10, 0)

ant_transform = v.Transform(rot, pos)
ant_handle = env_def.create_articulation(ant_def_handle, ant_transform)

# Print articulation
articulation = env_def.get_articulation(ant_handle)
print("> Articulation")
print(articulation)

# Import humanoid definition
filename = "assets/vsim/humanoid.vsim"
env_def.import_definitions(filename, fixed=False, alias="humanoid")

humanoid_def_handle = env_def.get_articulation_def_handle_by_name("humanoid.torso")

humanoid_def = env_def.get_articulation_def(humanoid_def_handle)
humanoid_def.has_self_collisions = True

# Instantiate humanoid
rot = v.shortest_rotation(v.Vec3(0, 0, 1), v.Vec3(0, 1, 0))
base_pos = v.Vec3(-10, 20, -10)

for i in range(10):
    for j in range(10):
        for k in range(10):
            pos = base_pos + i * v.Vec3(2, 0, 0) + j * v.Vec3(0, 2, 0) + k * v.Vec3(0, 0, 2)
            humanoid_transform = v.Transform(rot, pos)
            env_def.create_articulation(humanoid_def_handle, humanoid_transform)

# Finalize environment def
env_def.finalize()

# == Environment group == #
env_group_handle = gym.create_environment_group(env_def_handle)
env_group = gym.get_environment_group(env_group_handle)
env_group.create_environment_set(1)
env_group.finalize()

# == Create plane == #
gym.create_plane()

# == Finalize Gym == #
gym.gym_finalize()

# == Set up render == #
render = gym.get_render()

render.reset_camera(v.Vec3(30, 30, 50), v.Vec3(-0.6, -0.2, -0.8))
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
