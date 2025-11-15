import vlearn as v

# == Initialize Gym == #
gym = v.create_gym(with_render=True)

# == Finalize Gym == #
gym.gym_finalize()

# == Set up render == #
render = gym.get_render()

# == Simulation loop == #
timestep = 0.01667
gym.set_timestep(timestep)

done = False
while not done:

    # Simulation step
    gym.step()

    # Render
    done = render.render_function()
