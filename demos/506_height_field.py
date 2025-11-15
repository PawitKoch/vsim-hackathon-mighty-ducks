import vlearn as v
from vlearn.constant import GLOBAL_ENV_DEF_HANDLE

# == Initialize Gym == #
gym = v.create_gym(with_render=True)

# == Import height field == #
global_env_def = gym.get_environment_def(GLOBAL_ENV_DEF_HANDLE)

filename = "assets/vsim/TerrainExportTest/heightfields/Landscape_HeightField.png"

min_height = 0
max_height = 25

hf_def_handle = global_env_def.import_height_field_def(filename, min_height, max_height)

hf_pos = v.Vec3(0, 0, 0)
hf_quat = v.Quat(0, 0, 0, 1)
hf_transform = v.Transform(hf_quat, hf_pos)

global_env_def.create_rigid_body(hf_def_handle, hf_transform)

# == Finalize Gym == #
gym.gym_finalize()

# == Set up render == #
render = gym.get_render()
render.reset_camera(v.Vec3(4.7, 115.8, 55.4), v.Vec3(0.3, -1, 0))

# == Simulation loop == #
timestep = 0.01667
gym.set_timestep(timestep)

done = False
while not done:

    # Simulation step
    gym.step()

    # Render
    done = render.render_function()
