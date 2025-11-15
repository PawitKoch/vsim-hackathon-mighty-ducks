import vlearn as v
from vlearn.constant import GLOBAL_ENV_DEF_HANDLE

# == Initialize Gym == #
gym = v.create_gym(with_render=True)

# == Import height field == #
global_env_def = gym.get_environment_def(GLOBAL_ENV_DEF_HANDLE)

tri_def_handle = global_env_def.import_triangle_mesh_def("assets/snow-mountain.obj")

tri_pos = v.Vec3(0, 0, 0)
tri_quat = v.Quat(0, 0, 0, 1)
tri_transform = v.Transform(tri_quat, tri_pos)

global_env_def.create_rigid_body(tri_def_handle, tri_transform)

# == Finalize Gym == #
gym.gym_finalize()

# == Set up render == #
render = gym.get_render()
render.reset_camera(v.Vec3(8.2, 383.7, 57.7), v.Vec3(-0.7, -0.4, 0.5))

# == Simulation loop == #
timestep = 0.01667
gym.set_timestep(timestep)

done = False
while not done:

    # Simulation step
    gym.step()

    # Render
    done = render.render_function()
