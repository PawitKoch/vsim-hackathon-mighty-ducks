import vlearn as v

# == Initialize Gym == #
gym = v.create_gym(with_render=True,
                   enable_scene_query=True,
                   max_contact_pairs=512 * 1024,
                   max_patches=512 * 1024,
                   max_contacts=4 * 512 * 1024)

# == Environment definition == #
env_def_handle = gym.create_environment_def()
env_def = gym.get_environment_def(env_def_handle)

# Import capsule definition
filename = "assets/mjcf/capsule.xml"
env_def.import_definitions(filename, fixed=True, scale=100)

capsule_def_handle = env_def.get_rigid_body_def_handle_by_name("capsule")

# Print rigid body definition
rigid_body_def = env_def.get_rigid_body_def(capsule_def_handle)
print("> Rigid body definition")
print(rigid_body_def)

# Instantiate capsules
rot = v.Quat(0, 0, 0, 1)
base_pos = v.Vec3(5, 25, 0)

capsule_handles = []
for i in range(6):
    for j in range(3):
        pos = base_pos + i * v.Vec3(10, 0, 0) + j * v.Vec3(0, 0, 30)
        capsule_transform = v.Transform(rot, pos)
        capsule_handles.append(env_def.create_rigid_body(capsule_def_handle, capsule_transform))

# Print rigid bodies
print("> Rigid bodies")
for capsule_handle in capsule_handles:
    rigid_body = env_def.get_rigid_body(capsule_handle)
    print(rigid_body)

# Create box definition
box_def_handle = env_def.create_box_def(v.Vec3(0.4), fixed=False)

# Instantiate boxes
rot = v.Quat(0, 0, 0, 1)
base_pos = v.Vec3(0, 32, 0)

for i in range(32):
    for j in range(32):
        for k in range(32):
            pos = base_pos + i * v.Vec3(2, 0, 0) + j * v.Vec3(0, 2, 0) + k * v.Vec3(0, 0, 2)
            box_transform = v.Transform(rot, pos)
            env_def.create_rigid_body(box_def_handle, box_transform)

# Finalize environment def
env_def.finalize()

# == Environment group == #
env_group_handle = gym.create_environment_group(env_def_handle)
env_group = gym.get_environment_group(env_group_handle)
env_group.create_environment_set(1)
env_group.finalize()

# == Create plane == #
gym.create_plane()

# == Finalize Gym == #
gym.gym_finalize()

# == Set up render == #
render = gym.get_render()

render.reset_camera(v.Vec3(128, 11, 180), v.Vec3(-0.6, 0.3, -0.8))
render.capped_step = True
render.set_paused(False)

# == Simulation loop == #
timestep = 0.01667
gym.set_timestep(timestep)

done = False
while not done:

    # Simulation step
    gym.step()

    # Render
    done = render.render_function()
