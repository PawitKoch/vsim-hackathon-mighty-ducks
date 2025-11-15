import vlearn as v

import torch

# == Initialize Gym == #
gym = v.create_gym(with_render=True,
                   enable_scene_query=True,
                   max_contact_pairs=512 * 1024,
                   max_patches=512 * 1024,
                   max_contacts=4 * 512 * 1024)

# == Environment definition == #
env_def_handle = gym.create_environment_def()
env_def = gym.get_environment_def(env_def_handle)

# Import dummy RGB camera
filename = "assets/vsim/DummyRgbCamera.vsim"
num_art, num_rigid = env_def.import_definitions(filename, fixed=True, merge_fixed_joints=False)

dummy_def_handle = env_def.get_articulation_def_handle_by_name("DummyRgbCamera")

art_def = env_def.get_articulation_def(dummy_def_handle)

camera = art_def.get_rgb_camera(0)
camera.render_relative_transform = v.Transform(v.Vec3(0, 3, 0))
camera.render_width = 1
camera.render_height = 1

# Instantiate camera
rot = v.Quat(v.Vec3(1, 0, 0), -3.14159265 / 2)
pos = v.Vec3(0, 0.5, 1)

dummy_transform = v.Transform(rot, pos)
dummy_handle = env_def.create_articulation(dummy_def_handle, dummy_transform)

# Import link6 definition
filename = "assets/vsim/Link6_Blue/Link2_Blue_query_geometry_test.vsim"
link6_def_handle = env_def.import_definitions(filename, fixed=False,
                                              query_mode=v.QueryMode.USE_VISUALS, import_extra_mesh_data=True)

link6_def_handle = env_def.get_rigid_body_def_handle_by_name("Link6_Blue")

# Import stage definition
filename = "assets/trifinger/robot_properties_fingers/urdf/high_table_boundary.urdf"
num_arti, num_rigid = env_def.import_definitions(filename, fixed=False,
                                                 query_mode=v.QueryMode.USE_VISUALS)

stage_def_handle = env_def.get_rigid_body_def_handle_by_name("stage")

# Instantiate rigid body 1
def_handles1 = [stage_def_handle, link6_def_handle]
def_names1 = ["Stage", "Link 6"]

rot = v.Quat(0, 0, 0, 1)
pos = v.Vec3(0.5, 1, 0)

rigid_transform = v.Transform(rot, pos)

rigid_handle1 = env_def.create_rigid_body(def_handles1, rigid_transform)

# Instantiate rigid body 2
def_handles2 = [link6_def_handle, stage_def_handle]
def_names2 = ["Link 6", "Stage"]

rot = v.Quat(0, 0, 0, 1)
pos = v.Vec3(-0.5, 1, 0)

rigid_transform = v.Transform(rot, pos)

rigid_handle2 = env_def.create_rigid_body(def_handles2, rigid_transform)

# Finalize environment def
env_def.finalize()

# == Environment group == #
num_envs_per_env_set = 2
num_env_sets = 2
num_envs = num_envs_per_env_set * num_env_sets

env_group_handle = gym.create_environment_group(env_def_handle)
env_group = gym.get_environment_group(env_group_handle)

for i in range(num_env_sets):
    env_group.create_environment_set(num_envs_per_env_set)

env_group.finalize()

env_group.tile_environments()

# == Create plane == #
gym.create_plane()

# == Finalize Gym == #
gym.gym_finalize()

# ================== #
# == GPU commands == #
# ================== #
assert torch.cuda.is_available(), "Could not find GPU"
device = torch.device("cuda:0")

# Common mask
masks = torch.zeros(num_envs, dtype=torch.bool, device=device)

# == Setting rigid body def of rigid body 1 == #
# Definition indices buffer: shape is (num_envs,)
set_def_idx_buf1 = torch.zeros(num_envs, dtype=torch.uint32, device=device)

# Create command
set_def_idx_cmd1 = env_group.create_rigid_body_def_command(
    v.wrap_gpu_buffer(set_def_idx_buf1),
    rigid_handle1,
    masks_buffer=v.wrap_gpu_buffer(masks)
    )

set_def_idx_cmd_arr1 = gym.create_rigid_body_def_command_gpu_array([set_def_idx_cmd1])

# == Setting rigid body def of rigid body 2 == #
# Definition indices buffer: shape is (num_envs,)
set_def_idx_buf2 = torch.zeros(num_envs, dtype=torch.uint32, device=device)

# Create command
set_def_idx_cmd2 = env_group.create_rigid_body_def_command(
    v.wrap_gpu_buffer(set_def_idx_buf2),
    rigid_handle2,
    masks_buffer=v.wrap_gpu_buffer(masks)
    )

set_def_idx_cmd_arr2 = gym.create_rigid_body_def_command_gpu_array([set_def_idx_cmd2])

# == Set up render == #
render = gym.get_render()

render.reset_camera(v.Vec3(0.8, 10.7, -11.6), v.Vec3(0.5, -0.6, 0.6))
render.capped_step = True
render.set_paused(False)

# == GUI controls == #
mask_combo = v.UserCombo("Active environment", ["All"] + ["Env {}".format(i) for i in
                                                          range(num_envs)], 0)

def_combo1 = v.UserCombo("Rigid body 1", def_names1, 0)
def_combo2 = v.UserCombo("Rigid body 2", def_names2, 0)

render.register_menu_item(mask_combo)
render.register_menu_item(def_combo1)
render.register_menu_item(def_combo2)


def set_rigid_body_defs():

    if mask_combo.get_current_index() == 0:
        masks[:] = True
    else:
        masks[:] = False
        masks[mask_combo.get_current_index() - 1] = True

    if set_def_idx_buf1[0] != def_combo1.get_current_index():
        set_def_idx_buf1[:] = def_combo1.get_current_index()
        gym.set_rigid_body_defs(set_def_idx_cmd_arr1)

    if set_def_idx_buf2[0] != def_combo2.get_current_index():
        set_def_idx_buf2[:] = def_combo2.get_current_index()
        gym.set_rigid_body_defs(set_def_idx_cmd_arr2)


# == Simulation loop == #
timestep = 0.01667
gym.set_timestep(timestep)

done = False
while not done:

    # Set rigid body defs
    set_rigid_body_defs()

    # Simulation step
    gym.step()

    # Render
    done = render.render_function()
