import vlearn as v
import numpy as np

from sys import argv
from vlearn.utils.joint_monkey import joint_monkey_vectorized
from vlearn.utils import get_VL_VISUAL_TESTS

with_window = get_VL_VISUAL_TESTS()
with_render = True
enable_scene_query = True

num_envs = 2

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
env_def_handle = gym.create_environment_def("panda")
env_def = gym.get_environment_def(env_def_handle)

filename = "assets/urdf/franka_description/robots/franka_panda.urdf"
env_def.import_definitions(filename, True)
def_handle = env_def.get_articulation_def_handle_by_name("panda")

rot = v.shortest_rotation(v.Vec3(0, 0, 1), v.Vec3(0, 1, 0))
pos = v.Vec3(0, 0, 0)
transform = v.Transform(rot, pos)

env_def.create_articulation(def_handle, transform, 'robot-arm')

# Set drive damping and stiffness
if mode == 'target_position' or mode == 'target_velocity':
    artDef = env_def.get_articulation_def(def_handle)
    if mode == 'target_position':

        for i in range(artDef.get_num_pid_defs()):
            pid_data = artDef.get_pid_def(i)
            pid_data.stiffness = 1e+10
    else:
        for i in range(artDef.get_num_pid_defs()):
            pid_data = artDef.get_pid_def(i)
            pid_data.damping = 1e+10

# Environment transforms
env_pos = v.Vec3(0, 0, 0)
env_rot = v.Quat(0, 0, 0, 1)
env_transforms = []
for i in range(num_envs):
    env_transforms.append(v.Transform(env_rot, env_pos))
    env_pos += v.Vec3(2, 0, 0)

# Gravity
gravity = v.Vec3(0, -9.81, 0)
if mode == 'target_position' or mode == 'target_velocity':
    gravity = v.Vec3(0, 0, 0)

# Finalize environment def
env_def.finalize()

# Start joint monkey loop
joint_monkey_vectorized(
    env_def_handle,
    mode,
    num_envs=num_envs,
    env_transforms=env_transforms,
    gravity=gravity,
    num_iterations=num_iter,
    initial_pause=init_pause,
    interval=interval)
