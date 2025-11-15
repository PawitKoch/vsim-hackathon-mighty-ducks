import vlearn as v
import os
import torch
from torchvision.utils import save_image


def debug_save_images(envs):
    if envs.num_envs_int == 1:
        os.makedirs("./runs/sim_images_segment", exist_ok=True)
        img = envs.rgb_camera_handler.segmented_images[0][0].float().permute(2, 0, 1)
        img = torch.flip(img, dims=[1])
        save_image(img, "./runs/sim_images_segment/latest_segmented.jpg")
    if envs.use_segmentation_for_inference and envs.num_envs_int == 1:
        os.makedirs("./runs/sim_images_rgb", exist_ok=True)
        img = envs.rgb_camera_handler.rgb_images[..., :3].permute(2, 0, 1) / 255.0
        img = torch.flip(img, dims=[1])
        save_image(img, "./runs/sim_images_rgb/latest_rgb.jpg")
                    
        os.makedirs("./runs/sim_images_sam2", exist_ok=True)
        img = envs.info["image"][0] * 255.0
        img = torch.flip(img, dims=[2])
        save_image(img, "./runs/sim_images_sam2/latest_sam2.jpg")

def print_step_debug(envs,):
    assert envs.num_envs_int == 1

    # Safe CPU copies
    act = envs.act_buf[0].detach().float().cpu()
    cur = envs.joint_handler.dof_pos_value[0].detach().float().cpu()
    target = envs.joint_handler.dof_pos_target[0].detach().float().cpu()
    ob  = envs.obs_buf[0].detach().float().cpu()
    # box1_pos = envs.box1_handler.pose_value[0, 4:7].detach().float().cpu()
    ik_target_pose = envs.ik_handler.ik_target_pose[0].detach().float().cpu()
    ee_pose_value = envs.kinematic_sensor_handler.ee_pose_value[0].detach().float().cpu()
    jaw_force_value = envs.jaw_force_sensor_handler.force_sensor_val[0].detach().float().cpu()
    gripper_force_value = envs.gripper_force_sensor_handler.force_sensor_val[0].detach().float().cpu()
    wrist_force_value = envs.wrist_force_sensor_handler.force_sensor_val[0].detach().float().cpu()

    img_range = None

    # SAM2 mask quick stats (fraction of foreground pixels)
    mask_fg_frac = None
    if getattr(envs, "use_segmentation_for_inference", False) and "image" in envs.info and envs.info["image"]:
        # info["image"][0] is (B=1, C=1, H, W) with 0/1 values
        mask = envs.info["image"][0][0, 0].detach().float().cpu()
        
        total = mask.numel()
        if total > 0:
            mask_fg_frac = float(mask.sum().item() / total)
            img_range = (float(mask.min().item()), float(mask.max().item()))

    # Rounders
    def rlist(x):
        return [round(float(v), 4) for v in x]

    # Print
    print(f"\n[Step {envs.progress_buf[0]}] ----------------------------------------")
    print(f"Action ({act.numel()}): {rlist(act.tolist())}")
    print(f"Current ({cur.numel()}): {rlist(cur.tolist())}")
    print(f"Target ({target.numel()}): {rlist(target.tolist())}")
    print(f"Obs ({ob.numel()}):    {rlist(ob.tolist())}")
    # print(f"box1 pos (xyz): {rlist(box1_pos.tolist())}")
    print(f"Ee pose ({ee_pose_value.numel()}): {rlist(ee_pose_value.tolist())}")
    print(f"Ee target ({ik_target_pose.numel()}): {rlist(ik_target_pose.tolist())}")
    print(f"Jaw->Table forces ({jaw_force_value.numel()}): {rlist(jaw_force_value.tolist())}")
    print(f"Gripper->Table forces ({gripper_force_value.numel()}): {rlist(gripper_force_value.tolist())}")
    print(f"Wrist->Table forces ({wrist_force_value.numel()}): {rlist(wrist_force_value.tolist())}")
    if mask_fg_frac is not None:
        print(f"SAM2 mask — foreground fraction: {round(mask_fg_frac, 4)}")
    if img_range is not None:
        print(f"Image-to-model range (min,max): ({round(img_range[0], 6)}, {round(img_range[1], 6)})")
    print("-------------------------------------------------------")

def init_debug_draw(envs):
    render = envs.gym.get_render()
    if render is None:
        return
    
    box_size_real = 0.03

    box_size_end_eff = 1.01 * box_size_real
    box_size_target = 1.01 * box_size_real
    box_size_actual = 1.01 * box_size_real

    envs.end_eff_boxes = []
    envs.target_boxes = []
    envs.box_goal_poses = []
    envs.reach_space_targets = []

    envs.end_eff_axis_lines = []
    envs.target_axis_lines = []
    envs.goal_axis_lines = []
    envs.reach_axis_lines = []

    for i in range(envs.num_env_sets):
        env_set_handle = envs.env_group.get_environment_set_handle(i)
        env_set = envs.env_group.get_environment_set(env_set_handle)

        num_envs = envs.num_envs[i]

        # Draw local offsets
        line_box_color = v.Vec3(0.17, 0.98, 0.12)
        for i in range(num_envs):
            env_handle = env_set.get_environment_handle(i)

            envs.end_eff_boxes.append(
                render.create_user_line_cube(
                    box_size_end_eff, v.Transform(
                        v.Quat(
                            0, 0, 0, 1), v.Vec3(
                            0, 0, 0)), line_box_color, env_handle=env_handle))

            render.register_line_shape(envs.end_eff_boxes[-1])
            axes = add_local_frame(render, env_handle)
            envs.end_eff_axis_lines.append(axes)

        # Draw target cubes
        line_box_color = v.Vec3(0.63, 0.13, 0.94)
        for i in range(num_envs):
            env_handle = env_set.get_environment_handle(i)

            envs.target_boxes.append(
                render.create_user_line_cube(
                    box_size_target, v.Transform(
                        v.Quat(
                            0, 0, 0, 1), v.Vec3(
                            0, 0, 0)), line_box_color, env_handle=env_handle))

            render.register_line_shape(envs.target_boxes[-1])
            axes = add_local_frame(render, env_handle)
            envs.target_axis_lines.append(axes)

        # # Draw actual goal location
        # line_box_color = v.Vec3(0.9, 0.5, 0.5)
        # for i in range(num_envs):
        #     env_handle = env_set.get_environment_handle(i)

        #     envs.box_goal_poses.append(
        #         render.create_user_line_cube(
        #             box_size_actual, v.Transform(
        #                 v.Quat(
        #                     0, 0, 0, 1), v.Vec3(
        #                     0, 0, 0)), line_box_color, env_handle=env_handle))

        #     render.register_line_shape(envs.box_goal_poses[-1])
        #     axes = add_local_frame(render, env_handle)
        #     envs.goal_axis_lines.append(axes)

        # # Draw location of reach target in space
        # line_box_color = v.Vec3(0.9, 0.5, 0.5)
        # for i in range(num_envs):
        #     env_handle = env_set.get_environment_handle(i)

        #     envs.reach_space_targets.append(
        #         render.create_user_line_cube(
        #             box_size_actual, v.Transform(
        #                 v.Quat(
        #                     0, 0, 0, 1), v.Vec3(
        #                     0, 0, 0)), line_box_color, env_handle=env_handle))

        #     render.register_line_shape(envs.reach_space_targets[-1])
        #     axes = add_local_frame(render, env_handle)
        #     envs.reach_axis_lines.append(axes)

def update_draw_targets(envs):
    render = envs.gym.get_render()
    if render is None:
        return

    # End effector offset pose
    # print("end_eff_offset_pose_cpu ", end_eff_offset_pose_cpu[0])
    for i in range(envs.num_envs_int):
        pose_tensor = envs.kinematic_sensor_handler.ee_pose_value[i].cpu()
        rot = v.Quat(*pose_tensor[:4])
        pos = v.Vec3(*pose_tensor[4:])
        tf = v.Transform(rot, pos)

        envs.end_eff_boxes[i].set_transform(tf)
        
        update_local_frame_lines(pose_tensor, envs.end_eff_axis_lines[i])

    # Target pose
    target_pos = envs.ik_handler.ik_target_pose.cpu()
    for i in range(envs.num_envs_int):
        rot = v.Quat(*target_pos[i][:4])
        pos = v.Vec3(*target_pos[i][4:])
        tf = v.Transform(rot, pos)

        envs.target_boxes[i].set_transform(tf)

        update_local_frame_lines(target_pos[i], envs.target_axis_lines[i])


    # # Goal box pose
    # box_goal_pose = envs.goal_pose.cpu()
    # for i in range(envs.num_envs_int):
    #     rot = v.Quat(*box_goal_pose[0][:4])
    #     pos = v.Vec3(*box_goal_pose[0][4:])
    #     tf = v.Transform(rot, pos)

    #     envs.box_goal_poses[i].set_transform(tf)

    #     update_local_frame_lines(box_goal_pose[i], envs.goal_axis_lines[i])

    # box1_pose_value = envs.box1_handler.pose_value.cpu()
    # for i in range(envs.num_envs_int):
    #     rot = v.Quat(*box1_pose_value[i][:4])
    #     pos = v.Vec3(*box1_pose_value[i][4:])
    #     tf = v.Transform(rot, pos)

    #     envs.reach_space_targets[i].set_transform(tf)

    #     update_local_frame_lines(box1_pose_value[i], envs.reach_axis_lines[i])



def add_local_frame(render, env_handle, axis_len=0.05, line_width=2.0):
    """Create and register local XYZ axis lines for visualization."""
    axes = []

    axes.append(render.create_user_line(
        [v.Vec3(0, 0, 0), v.Vec3(axis_len, 0, 0)], v.Vec3(0, 1, 0), line_width, env_handle))  # X
    axes.append(render.create_user_line(
        [v.Vec3(0, 0, 0), v.Vec3(0, axis_len, 0)], v.Vec3(1, 0, 0), line_width, env_handle))  # Y
    axes.append(render.create_user_line(
        [v.Vec3(0, 0, 0), v.Vec3(0, 0, axis_len)], v.Vec3(0, 0, 1), line_width, env_handle))  # Z

    for a in axes:
        render.register_line_shape(a)

    return axes

# code taken out of v_rpy_from_quat to get rotation matrix for drawing lines
@torch.jit.script
def rot_matrix_from_quat(q: torch.Tensor) -> torch.Tensor:
    # Normalise quaternion
    q = q / torch.clamp(torch.linalg.norm(q, dim=-1, keepdim=True), min=1e-12)
    x, y, z, w = q.unbind(dim=-1)

    x2 = x + x
    y2 = y + y
    z2 = z + z
    xx = x2 * x
    yy = y2 * y
    zz = z2 * z
    xy = x2 * y
    xz = x2 * z
    xw = x2 * w
    yz = y2 * z
    yw = y2 * w
    zw = z2 * w

    # Build rotation matrix columns
    m00 = 1.0 - (yy + zz)
    m01 = xy - zw
    m02 = xz + yw

    m10 = xy + zw
    m11 = 1.0 - (xx + zz)
    m12 = yz - xw

    m20 = xz - yw
    m21 = yz + xw
    m22 = 1.0 - (xx + yy)

    R = torch.stack([
        torch.stack([m00, m01, m02], dim=-1),
        torch.stack([m10, m11, m12], dim=-1),
        torch.stack([m20, m21, m22], dim=-1),
    ], dim=-2)

    return R

def update_local_frame_lines(pose_tensor, axis_lines, axis_len=0.05):

    rot = pose_tensor[:4]
    pos = pose_tensor[4:]

    R = rot_matrix_from_quat(rot.unsqueeze(0))[0]

    origin = pos
    x_end = origin + R[:, 0] * axis_len
    y_end = origin + R[:, 1] * axis_len
    z_end = origin + R[:, 2] * axis_len

    x_line, y_line, z_line = axis_lines

    x_line.set_points([v.Vec3(*origin.tolist()), v.Vec3(*x_end.tolist())])
    y_line.set_points([v.Vec3(*origin.tolist()), v.Vec3(*y_end.tolist())])
    z_line.set_points([v.Vec3(*origin.tolist()), v.Vec3(*z_end.tolist())])