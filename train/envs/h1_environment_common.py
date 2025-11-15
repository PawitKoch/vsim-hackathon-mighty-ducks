import numpy as np
import torch

import vlearn as v


def create_envs_helper(gym, transform=None, fixed=False, use_visual_mesh=False,
                       enable_motor_control=True, enable_self_collisions=True, use_xbot=False,
                       fix_upper_body=False):

    # Environment def
    env_def_handle = gym.create_environment_def("h1")
    env_def = gym.get_environment_def(env_def_handle)

    filename, name = "assets/vsim/H1/h1.vsim", "pelvis"
    if fix_upper_body:
        filename, name = "assets/vsim/H1/h1_fix_upper_body.vsim", "pelvis"
    if use_xbot:
        filename, name = "assets/vsim/XBot-L/XBot-L.vsim", "XBot-L"
    num_arti, num_rigid = env_def.import_definitions(filename, fixed=fixed,
                                                     use_visual_mesh=use_visual_mesh)
    assert num_arti == 1
    assert num_rigid == 0
    arti_model_def_handle = env_def.get_articulation_def_handle_by_name(name)

    if transform is None:
        rot = v.Quat(0, 0, 0, 1)
        pos = v.Vec3(0, 0, 1.4)
        transform = v.Transform(rot, pos)

    arti_handle = env_def.create_articulation(arti_model_def_handle, transform, 'runner')

    art_def = env_def.get_articulation_def(arti_model_def_handle)
    art_def.enable_control_type(v.ArticulationControlType.MOTOR, enable_motor_control)

    art_def.has_self_collisions = enable_self_collisions

    return env_def_handle, art_def, arti_handle


def store_initial_conditions_helper(art_def, device):

    # Store dof limits and initial values
    dof_pos_init = []
    for dofdef in art_def.get_joint_dof_defs():
        low, high = dofdef.get_limits()
        init = np.clip(0, low, high)

        dof_pos_init.append(init)

    dof_pos_init = torch.tensor(dof_pos_init, dtype=torch.float32, device=device)

    # Store initial root transform
    rot = v.shortest_rotation(v.Vec3(0, 0, 1), v.Vec3(0, 1, 0))
    pos = v.Vec3(0, 1.4, 0)
    root_trans_init = v.Transform(rot, pos)

    # Store initial root velocity
    root_vel_init = v.SpatialVector(0)

    return dof_pos_init, root_trans_init, root_vel_init


def allocate_gpu_buffers(envs):

    # Root transforms buffers
    envs.root_pos_buf = torch.zeros((envs.num_envs, 7), device=envs.device, dtype=torch.float32)

    # Root velocities buffer
    envs.root_vel_buf = torch.zeros((envs.num_envs, 6), device=envs.device, dtype=torch.float32)

    # Reset state
    envs.gpu_init_dof_pos = torch.tile(envs.dof_pos_init, (envs.num_envs, 1))
    envs.gpu_init_dof_vel = torch.zeros_like(envs.gpu_init_dof_pos)

    envs.gpu_init_root_velocities = torch.empty(6, dtype=torch.float32, device=envs.device)
    envs.gpu_init_root_velocities[0] = envs.root_vel_init.top.x
    envs.gpu_init_root_velocities[1] = envs.root_vel_init.top.y
    envs.gpu_init_root_velocities[2] = envs.root_vel_init.top.z
    envs.gpu_init_root_velocities[3] = envs.root_vel_init.bottom.x
    envs.gpu_init_root_velocities[4] = envs.root_vel_init.bottom.y
    envs.gpu_init_root_velocities[5] = envs.root_vel_init.bottom.z
    envs.gpu_init_root_velocities = torch.tile(envs.gpu_init_root_velocities, (envs.num_envs, 1))

    envs.gpu_init_root_transforms = torch.empty(7, dtype=torch.float32, device=envs.device)
    envs.gpu_init_root_transforms[0] = envs.root_trans_init.q.x
    envs.gpu_init_root_transforms[1] = envs.root_trans_init.q.y
    envs.gpu_init_root_transforms[2] = envs.root_trans_init.q.z
    envs.gpu_init_root_transforms[3] = envs.root_trans_init.q.w
    envs.gpu_init_root_transforms[4] = envs.root_trans_init.p.x
    envs.gpu_init_root_transforms[5] = envs.root_trans_init.p.y
    envs.gpu_init_root_transforms[6] = envs.root_trans_init.p.z
    envs.gpu_init_root_transforms = torch.tile(envs.gpu_init_root_transforms, (envs.num_envs, 1))

    # Set motor GPU command
    try:
        num_motors = envs.num_motors
        set_motor_data = envs.env_group.create_motor_control_command(
            v.wrap_gpu_buffer(envs.act_buf), envs.arti_handle)

        envs.gpu_set_motor_data_array = envs.gym.create_motor_control_command_gpu_array(
            [set_motor_data])
    except AttributeError:
        pass

    # Set kinematic state GPU command
    set_kinematic_state_command = envs.env_group.create_articulation_kinematic_state_command(
        v.wrap_gpu_buffer(envs.set_dof_pos_buf),
        v.wrap_gpu_buffer(envs.set_dof_vel_buf),
        v.wrap_gpu_buffer(envs.gpu_init_root_transforms),
        v.wrap_gpu_buffer(envs.gpu_init_root_velocities),
        envs.arti_handle,
        link_index_range=(0, 1),
        masks_buffer=v.wrap_gpu_buffer(envs.reset_buf))

    envs.gpu_set_kinematic_state_command_array = \
        envs.gym.create_articulation_kinematic_state_command_gpu_array(
            [set_kinematic_state_command])

    # Get kinematic state GPU command
    get_kinematic_state_command = envs.env_group.create_articulation_kinematic_state_command(
        v.wrap_gpu_buffer(envs.get_dof_pos_buf),
        v.wrap_gpu_buffer(envs.get_dof_vel_buf),
        v.wrap_gpu_buffer(envs.root_pos_buf),
        v.wrap_gpu_buffer(envs.root_vel_buf),
        envs.arti_handle,
        link_index_range=(0, 1))

    envs.gpu_get_kinematic_state_command_array = \
        envs.gym.create_articulation_kinematic_state_command_gpu_array(
            [get_kinematic_state_command])

    # Sensor forces buffer
    env_def = envs.gym.get_environment_def(envs.env_def_handle)
    articulation = env_def.get_articulation(envs.arti_handle)

    envs.force_sensor_handles = []
    envs.force_sensor_buffers = []
    envs.force_sensor_cmds = []
    for i in range(envs.num_sensors):
        envs.force_sensor_handles.append(articulation.get_force_sensor_handle(i))
        envs.force_sensor_buffers.append(torch.zeros((envs.num_envs, 6), device=envs.device,
                                                     dtype=torch.float32))
        envs.force_sensor_cmds.append(envs.env_group.create_force_sensor_command(
            v.wrap_gpu_buffer(envs.force_sensor_buffers[-1]), envs.force_sensor_handles[-1]))

    envs.gpu_get_sensor_forces_command_array = envs.gym.create_force_sensor_command_gpu_array(
        envs.force_sensor_cmds)
