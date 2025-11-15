import vlearn as v
import numpy as np

from vlearn.utils.joint_monkey import joint_monkey
from vlearn.utils import get_VL_VISUAL_TESTS
from sys import argv

with_render = get_VL_VISUAL_TESTS()
enable_scene_query = get_VL_VISUAL_TESTS()

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

v.create_gym(
    with_render=with_render,
    enable_scene_query=enable_scene_query,
    up_axis=v.Vec3(
        0,
        0,
        1))
gym = v.get_gym()

# Environment def
env_def_handle = gym.create_environment_def("allegro_hand")
env_def = gym.get_environment_def(env_def_handle)

filename = "assets/kuka_allegro_description/allegro_touch_sensor.vsim"
env_def.import_definitions(filename, fixed=True, use_visual_mesh=True)
def_handle = env_def.get_articulation_def_handle_by_name("kuka_allegro")

art_def = env_def.get_articulation_def(def_handle)
art_def.has_self_collisions = True

rot = v.Quat(0.283045, 0.683330, -0.621782, 0.257551)
pos = v.Vec3(0, 0, 0.5)
transform = v.Transform(rot, pos)

env_def.create_articulation(def_handle, transform, 'allegro-hand')

# Environment transform
env_pos = v.Vec3(0, 0, 0)
# env_rot = v.Quat(v.Vec3(1, 0, 0), -np.pi*0.5)
env_rot = v.Quat(0, 0, 0, 1)
env_transform = v.Transform(env_rot, env_pos)

# Gravity
gravity = v.Vec3(0, 0, -9.81)

# Finalize environment def
env_def.finalize()

# Start joint monkey loop
joint_monkey(env_def_handle, mode, env_transform=env_transform, gravity=gravity,
             num_iterations=num_iter, initial_pause=init_pause, interval=interval)
