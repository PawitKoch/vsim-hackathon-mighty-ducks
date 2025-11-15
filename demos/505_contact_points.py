import vlearn as v
from vlearn.constant import GLOBAL_ENV_DEF_HANDLE

import torch


def print_result(label, data, prec=4):
    print(label)
    for row in data.cpu():
        vals = row.tolist()
        if torch.is_floating_point(row):
            elem_str = ", ".join(f"{v:.{prec}f}" for v in vals)
        else:                                    # int / uint
            elem_str = ", ".join(str(int(v)) for v in vals)
        print(f"({elem_str})")


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
env_def.import_definitions(filename)

ant_def_handle = env_def.get_articulation_def_handle_by_name("torso")

# Instantiate ant
ant_rot = v.shortest_rotation(v.Vec3(0, 0, 1), v.Vec3(0, 1, 0))
ant_pos = v.Vec3(0, 0.75, 0)

ant_transform = v.Transform(ant_rot, ant_pos)
ant_handle = env_def.create_articulation(ant_def_handle, ant_transform)

# Finalize environment def
env_def.finalize()

# == Environment group == #
num_envs = 2

env_group_handle = gym.create_environment_group(env_def_handle)
env_group = gym.get_environment_group(env_group_handle)
env_group.create_environment_set(num_envs)
env_group.finalize()

# Set environment transforms
env_group.tile_environments(spacing=2)

# == Create plane == #
gym.create_plane()

# ================== #
# == GPU commands == #
# ================== #
assert torch.cuda.is_available(), "Could not find GPU"
device = torch.device("cuda:0")

# Here, `num_output` refers to the size of the contact buffer that you specify.
# This value can be greater or smaller than the actual number of contacts.

# * When the actual number of contacts is **less than** the buffer size,
#   it returns only the actual number of contacts
#   (e.g., if the buffer size is 10 but there are only 8 contacts, it returns 8 values).

# * When the actual number of contacts is **greater than** the buffer size,
#   it returns only as many as the buffer can hold
#   (e.g., if the buffer size is 3 but there are 8 contacts, it returns just 3 values).

num_output = 6

# Normals buffer: shape is (num_output, 3)
normals_buf = torch.zeros((num_output, 3), dtype=torch.float32, device=device)

# Point separations buffer: shape is (num_output, 4)
point_seps_buf = torch.zeros((num_output, 4), dtype=torch.float32, device=device)

# shape identify A buffer: shape is (num_output, 4) 0 = environemnt group index, 1 = environment set index,
# 2 = environment index, 3 = environment transform handle
id_a_buf = torch.zeros((num_output, 4), dtype=torch.uint32, device=device)

# shape identify B buffer: shape is (num_output, 4) 0 = environemnt group index, 1 = environment set index,
# 2 = environment index, 3 = environment transform handle
id_b_buf = torch.zeros((num_output, 4), dtype=torch.uint32, device=device)

# == Finalize Gym == #
gym.gym_finalize()

# == Set up render == #
render = gym.get_render()

render.reset_camera(v.Vec3(-0.5, 6.4, -2.9), v.Vec3(0.2, -0.9, 0.4))
render.capped_step = True
render.set_paused(False)

# == GUI controls == #
# print_box = v.UserCheckbox("Print contacts", False)
print_box = v.UserCheckbox("Print contacts", False)

render.register_menu_item(print_box)


def print_contacts():

    if print_box.get_value() == False:
        return

    print_box.set_value(False)

    # num_contacts
    num_contacts = gym.get_rigid_contacts(
        v.wrap_gpu_buffer(normals_buf),
        v.wrap_gpu_buffer(point_seps_buf),
        v.wrap_gpu_buffer(id_a_buf),
        v.wrap_gpu_buffer(id_b_buf),
        num_output
        )

    # Print result
    num_stored = min(num_contacts, num_output)
    print("Number of contacts counted: {}".format(num_contacts))
    print("Number of contacts stored: {}".format(num_stored))
    print_result("Normal:", normals_buf[:num_stored])
    print_result("Point separation (x, y, z, w):", point_seps_buf[0:num_stored])
    print_result("Rigid body A:", id_a_buf[0:num_stored])
    print_result("Rigid body B:", id_b_buf[0:num_stored])


# == Simulation loop == #
timestep = 0.01667
gym.set_timestep(timestep)

done = False
while not done:

    # Print contacts
    print_contacts()

    # Simulation step
    gym.step()

    # Render
    done = render.render_function()
