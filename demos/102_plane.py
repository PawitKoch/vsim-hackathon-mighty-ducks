import vlearn as v
from vlearn.constant import GLOBAL_ENV_DEF_HANDLE

# == Initialize Gym == #
gym = v.create_gym(with_render=True)

# == Create plane == #
global_env_def = gym.get_environment_def(GLOBAL_ENV_DEF_HANDLE)

plane_def_handle = global_env_def.create_plane_def()

plane_pos = v.Vec3(0, 0, 0)
plane_quat = v.Quat(0, 0, 0, 1)
plane_transform = v.Transform(plane_quat, plane_pos)

global_env_def.create_rigid_body(plane_def_handle, plane_transform)

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
