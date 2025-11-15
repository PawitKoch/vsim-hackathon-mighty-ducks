import numpy as np
import torch
import vlearn as v


def create_envs_helper(gym, control_mode='motor', env_def_name="ant", up_axis=v.Vec3(0, 1, 0)):

    # Environment def
    env_def_handle = gym.create_environment_def(env_def_name)
    env_def = gym.get_environment_def(env_def_handle)

    filename = "assets/vsim/ant.vsim"
    num_arti, num_rigid = env_def.import_definitions(filename, False)
    assert num_arti == 1
    assert num_rigid == 0

    arti_def_handle = env_def.get_articulation_def_handle_by_name("torso")
    assert env_def.get_articulation_def(arti_def_handle).get_num_joint_dof_defs() == 8

    rot = v.shortest_rotation(v.Vec3(0, 0, 1), up_axis)
    pos = 0.75 * up_axis
    transform = v.Transform(rot, pos)

    arti_handle = env_def.create_articulation(arti_def_handle, transform, 'crawler')

    art_def = env_def.get_articulation_def(arti_def_handle)
    if control_mode == 'motor':
        art_def.enable_control_type(v.ArticulationControlType.MOTOR, True)
        assert art_def.has_control_type(v.ArticulationControlType.MOTOR)
    elif control_mode == 'pid':
        pass

    return env_def_handle, arti_def_handle, arti_handle


def store_initial_conditions_helper(art_def, device, up_axis=v.Vec3(0, 1, 0)):

    # Store dof limits and initial values
    dof_pos_init = []
    for dofdef in art_def.get_joint_dof_defs():
        low, high = dofdef.get_limits()
        init = np.clip(0, low, high)

        dof_pos_init.append(init)

    dof_pos_init = torch.tensor(dof_pos_init, dtype=torch.float32, device=device)

    # Store initial root transform
    rot = v.shortest_rotation(v.Vec3(0, 0, 1), up_axis)
    pos = 0.75 * up_axis
    root_trans_init = v.Transform(rot, pos)

    # Store initial root velocity
    root_vel_init = v.SpatialVector(0)

    return dof_pos_init, root_trans_init, root_vel_init


@torch.jit.script
def compute_observations_helper(root_pos, root_vel, dof_pos, dof_vel, actions, sensor_forces=None):

    torso_pos = root_pos[:, 5].unsqueeze(-1)    # y position
    torso_rot = root_pos[:, 0:4]                # rotation quaternion
    velocity = root_vel[:, 3:6]                 # linear velocity
    ang_velocity = root_vel[:, 0:3]             # angular velocity

    obs = torch.cat(
        (torso_pos,
         torso_rot,
         velocity,
         ang_velocity,
         dof_pos,
         dof_vel,
         actions),
        dim=-1)

    if sensor_forces is not None:
        obs = torch.cat((obs, sensor_forces), dim=-1)

    return obs


@torch.jit.script
def compute_reward_termination_truncation_helper(actions,
                                                 obs,
                                                 root_pos,
                                                 old_root_pos,
                                                 progress,
                                                 healthy_y_range: tuple[float,
                                                                        float],
                                                 healthy_reward_default: float,
                                                 dt: float,
                                                 ctrl_cost_weight: float,
                                                 max_episode_length: int):

    # Termination
    healthy = torch.logical_and(
        healthy_y_range[0] <= obs[:, 0],
        obs[:, 0] <= healthy_y_range[1])

    term = torch.logical_not(healthy)

    # Healthy reward
    healthy_reward = torch.where(healthy, healthy_reward_default, 0)

    # Forward reward
    x_new = root_pos[:, 4]
    x_old = old_root_pos[:, 4]

    x_vel = (x_new - x_old) / dt

    forward_reward = x_vel

    # Action cost
    action_cost = torch.sum(actions * actions, dim=1)

    # Total reward
    reward = healthy_reward + forward_reward - ctrl_cost_weight * action_cost

    # Truncation
    trunc = torch.where(progress >= max_episode_length, 1, 0)

    return reward, term, trunc


def allocate_gpu_buffers(envs):

    # Root transforms buffer
    envs.root_pos_buf = torch.zeros((envs.num_envs, 7), device=envs.device, dtype=torch.float32)

    # Root velocities buffer
    envs.root_vel_buf = torch.zeros((envs.num_envs, 6), device=envs.device, dtype=torch.float32)

    # Sensor forces buffer
    envs.sensor_force_buf = torch.zeros((envs.num_envs, envs.num_sensors * 6), device=envs.device,
                                        dtype=torch.float32)

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
    if hasattr(envs, "control_mode"):
        control_mode = envs.control_mode
    else:
        control_mode = "motor"

    if control_mode == 'motor':
        set_motor_cmd = envs.env_group.create_motor_control_command(
            v.wrap_gpu_buffer(envs.act_buf), envs.arti_handle)

        envs.gpu_set_motor_data_array = envs.gym.create_motor_control_command_gpu_array([
                                                                                        set_motor_cmd])

    elif control_mode == 'pid':
        set_pid_cmd = envs.env_group.create_pid_control_command(v.wrap_gpu_buffer(envs.act_buf),
                                                                envs.arti_handle)
        envs.gpu_set_pid_cmd_arr = envs.gym.create_pid_control_command_gpu_array([set_pid_cmd])

    # Set kinematic state GPU command
    set_kinematic_state_command = envs.env_group.create_articulation_kinematic_state_command(
        v.wrap_gpu_buffer(
            envs.set_dof_pos_buf), v.wrap_gpu_buffer(
            envs.set_dof_vel_buf), v.wrap_gpu_buffer(
                envs.gpu_init_root_transforms), v.wrap_gpu_buffer(
                    envs.gpu_init_root_velocities), envs.arti_handle, link_index_range=(
                        0, 1), masks_buffer=v.wrap_gpu_buffer(
                            envs.reset_buf))

    envs.gpu_set_kinematic_state_command_array = envs.gym.create_articulation_kinematic_state_command_gpu_array(
        [set_kinematic_state_command])

    # Get kinematic state GPU command
    get_kinematic_state_command = envs.env_group.create_articulation_kinematic_state_command(
        v.wrap_gpu_buffer(
            envs.get_dof_pos_buf), v.wrap_gpu_buffer(
            envs.get_dof_vel_buf), v.wrap_gpu_buffer(
                envs.root_pos_buf), v.wrap_gpu_buffer(
                    envs.root_vel_buf), envs.arti_handle, link_index_range=(
                        0, 1))

    envs.gpu_get_kinematic_state_command_array = envs.gym.create_articulation_kinematic_state_command_gpu_array(
        [get_kinematic_state_command])

    # Get sensor forces GPU command
    num_force_sensors = envs.art_def.get_num_force_sensor_defs()
    env_def = envs.gym.get_environment_def(envs.env_def_handle)
    articulation = env_def.get_articulation(envs.arti_handle)

    envs.force_sensor_handles = []
    envs.force_sensor_buffers = []
    envs.force_sensor_cmds = []
    for i in range(num_force_sensors):
        envs.force_sensor_handles.append(articulation.get_force_sensor_handle(i))
        envs.force_sensor_buffers.append(torch.zeros((envs.num_envs, 6), device=envs.device,
                                                     dtype=torch.float32))
        envs.force_sensor_cmds.append(envs.env_group.create_force_sensor_command(
            v.wrap_gpu_buffer(envs.force_sensor_buffers[-1]), envs.force_sensor_handles[-1]))

    envs.gpu_get_sensor_forces_command_array = envs.gym.create_force_sensor_command_gpu_array(
        envs.force_sensor_cmds)
