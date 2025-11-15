import vlearn as v
import numpy as np

from vlearn.utils.joint_monkey import joint_monkey
from vlearn.utils import get_VL_VISUAL_TESTS
from sys import argv

with_window = get_VL_VISUAL_TESTS()
with_render = True
enable_scene_query = True

mode = 'position'
num_iter = np.inf
init_pause = True
interval = 0.1

if len(argv) >= 2:
    if argv[1] == "--help" or argv[1] == "-h":
        print("Usage: {} [mode] [num_iter] [init_pause] [interval]".format(argv[0]))
        exit(0)
    mode = argv[1]
if len(argv) >= 3:
    num_iter = int(argv[2])
if len(argv) >= 4:
    if argv[3] == "True":
        init_pause = True
    else:
        init_pause = False
if len(argv) >= 5:
    interval = float(argv[4])

assert mode in ['position', 'velocity', 'motor', 'target_velocity', 'target_position']

v.create_gym(with_render=with_render, enable_scene_query=enable_scene_query,
             with_window=with_window, max_contact_pairs=1024, max_patches=1024, max_contacts=2048)
gym = v.get_gym()

# Environment def
env_def_handle = gym.create_environment_def("ant and humanoid")
env_def = gym.get_environment_def(env_def_handle)

ant_filename = "assets/mjcf/ant.xml"
env_def.import_definitions(ant_filename, True)
ant_def_handle = env_def.get_articulation_def_handle_by_name("torso")

ant_rot = v.shortest_rotation(v.Vec3(0, 0, 1), v.Vec3(0, 1, 0))
ant_pos = v.Vec3(0, 2, 0)
ant_transform = v.Transform(ant_rot, ant_pos)

env_def.create_articulation(ant_def_handle, ant_transform, 'crawler')

humanoid_filename = "assets/mjcf/humanoid.xml"
env_def.import_definitions(humanoid_filename, True, alias="humanoid")
humanoid_def_handle = env_def.get_articulation_def_handle_by_name("humanoid.torso")

humanoid_rot = v.shortest_rotation(v.Vec3(0, 0, 1), v.Vec3(0, 1, 0))
humanoid_pos = v.Vec3(2, 2, 0)
humanoid_transform = v.Transform(humanoid_rot, humanoid_pos)

env_def.create_articulation(humanoid_def_handle, humanoid_transform, 'runner')

# Set drive damping and stiffness
if mode == 'target_position' or mode == 'target_velocity':
    ant_def = env_def.get_articulation_def(ant_def_handle)
    humanoid_def = env_def.get_articulation_def(humanoid_def_handle)

    if mode == 'target_position':
        for i in range(ant_def.get_num_pid_defs()):
            pid_data = ant_def.get_pid_def(i)
            pid_data.stiffness = 1e+10
        for i in range(humanoid_def.get_num_pid_defs()):
            pid_data = humanoid_def.get_pid_def(i)
            pid_data.stiffness = 1e+10
    else:
        for i in range(ant_def.get_num_pid_defs()):
            pid_data = ant_def.get_pid_def(i)
            pid_data.damping = 1e+10
        for i in range(humanoid_def.get_num_pid_defs()):
            pid_data = humanoid_def.get_pid_def(i)
            pid_data.damping = 1e+10

# Set motor
if mode == "motor":
    ant_def = env_def.get_articulation_def(ant_def_handle)
    humanoid_def = env_def.get_articulation_def(humanoid_def_handle)

    ant_def.enable_control_type(v.ArticulationControlType.MOTOR, True)
    humanoid_def.enable_control_type(v.ArticulationControlType.MOTOR, True)

# Environment transforms
env_pos = v.Vec3(0, 0, 0)
env_rot = v.Quat(0, 0, 0, 1)
env_transform = v.Transform(env_rot, env_pos)

# Gravity
gravity = v.Vec3(0, -9.81, 0)
if mode == 'target_position' or mode == 'target_velocity':
    gravity = v.Vec3(0, 0, 0)

# Finalize environment def
env_def.finalize()

# Set camera
render = gym.get_render()
render.reset_camera(v.Vec3(6, 4, 0), v.Vec3(-1, -0.5, 0))
render.capped_step = True

# Start joint monkey loop
joint_monkey(env_def_handle, mode, env_transform=env_transform, gravity=gravity,
             num_iterations=num_iter, initial_pause=init_pause, interval=interval)
