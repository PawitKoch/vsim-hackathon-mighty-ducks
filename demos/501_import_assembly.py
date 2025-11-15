import vlearn as v

# == Initialize Gym == #
gym = v.create_gym(with_render=True, enable_scene_query=True)

# == Environment definition == #
env_def_handle = gym.create_environment_def()
env_def = gym.get_environment_def(env_def_handle)

# Import environment
env_def.import_assembly("assets/vsim/demo/demo.vasm")

# Finalize environment def
env_def.finalize()

# == Environment group == #
num_envs = 1

env_group_handle = gym.create_environment_group(env_def_handle)
env_group = gym.get_environment_group(env_group_handle)
env_group.create_environment_set(num_envs)
env_group.finalize()

# == Create plane == #
gym.create_plane()

# == Finalize Gym == #
gym.gym_finalize()

# == Set up render == #
render = gym.get_render()

render.reset_camera(v.Vec3(-3, 4.1, -6), v.Vec3(0.3, -0.5, 0.8))
render.capped_step = True
render.set_paused(True)

# == Simulation loop == #
timestep = 0.01667
gym.set_timestep(timestep)

done = False
while not done:

    # Simulation step
    gym.step()

    # Render
    done = render.render_function()
