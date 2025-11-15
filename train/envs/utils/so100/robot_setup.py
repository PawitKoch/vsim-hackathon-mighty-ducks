import os
import sys
import math
import argparse
import yaml
import numpy as np
import torch
import vlearn as v

current_dir = os.path.dirname(os.path.abspath(__file__))
train_root = os.path.abspath(os.path.join(current_dir, "../../../"))
if train_root not in sys.path:
    sys.path.insert(0, train_root)

from envs.so100_parking import SO100Parking
from envs.so100_tower import SO100Tower

from vlearn.torch_utils.torch_jit_utils import v_quat_from_rpy, v_rpy_from_quat



def get_params_from_yaml(yml_path: str | None):
    class_kwargs = {}
    # Only need one env for this configuration
    num_envs = 1
    try:
        # Get file
        if os.path.exists(yml_path):
            resolved = yml_path
        else:
            resolved = os.path.join("../../../train_config", yml_path)
        with open(resolved, "r") as stream:
            config = yaml.safe_load(stream)

        # Load params
        env_params = config.get("envs", {})
        for k, v_ in env_params.items():
            if k in ("num_envs", "_target_"):
                continue
            # Need debug for rendering the target cubes
            if k == "debug":
                class_kwargs[k] = True
                continue
            # We need an effectively infinite episode length
            if k == "max_episode_length":
                class_kwargs[k] = 400_000
                continue
            # Make sure we're not using sam2
            if k == "use_segmentation_for_inference":
                class_kwargs[k] = False
                continue
            class_kwargs[k] = v_
        # if "num_envs" in env_params:
        #     num_envs = env_params["num_envs"]
    except yml_path is None:
        print("[Warning] No YAML path provided; using default env params.")

    if isinstance(num_envs, (list, tuple)):
        num_envs = int(num_envs[0]) if len(num_envs) > 0 else 1

    return int(num_envs), class_kwargs

def canonicalize_quat(q: torch.Tensor) -> torch.Tensor:
    flip = (q[..., 3] < 0).unsqueeze(-1)
    return torch.where(flip, -q, q)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yml", default="so100_parking.yaml", help="env config YAML")
    parser.add_argument("--ik", dest="control_ik", action=argparse.BooleanOptionalAction, default=True,
                        help="Use IK pose sliders when true; joint sliders when false.")
    parser.add_argument("--no_write", action="store_true")
    args = parser.parse_args()

    # Load env from YAML
    num_envs, class_kwargs = get_params_from_yaml(args.yml)

    # Get env
    # env = SO100Tower(
    #     num_envs=num_envs,
    #     **class_kwargs,
    # )
    env = SO100Parking(
        num_envs=num_envs,
        **class_kwargs,
    )

    render = env.gym_render
    render.capped_step = True
    render.set_paused(True)

    env.reset()
    env.approach_quat = env.approach_quat.repeat(env.num_envs_int, 1)

    sliders = []

    if args.control_ik:
        # Seed from env.ee_pose_value (SO100 naming)
        env.kinematic_sensor_handler.get_state()
        init_pos = env.kinematic_sensor_handler.ee_pose_value[:, 4:][0]
        init_rpy = v_rpy_from_quat(env.kinematic_sensor_handler.ee_pose_value[:, 0:4])[0]

        sliders.append(v.UserSlider("IK target x",   -1.5,   1.5, init_pos[0]))
        sliders.append(v.UserSlider("IK target y",   -1.5,   1.5, init_pos[1]))
        sliders.append(v.UserSlider("IK target z",   -0.10,  1.5, init_pos[2]))
        sliders.append(v.UserSlider("IK roll",       -np.pi, np.pi, init_rpy[0]))
        sliders.append(v.UserSlider("IK pitch",      -np.pi, np.pi, init_rpy[1]))
        sliders.append(v.UserSlider("IK yaw NOT USED",        -np.pi, np.pi, init_rpy[2]))
    else:
        # Use env.angle_limits_* and env.dof_pos_value
        angle_limits_low = env.angle_limits_low[0].detach().cpu().numpy()
        angle_limits_high = env.angle_limits_high[0].detach().cpu().numpy()
        dof_pos_value = env.joint_handler.dof_pos_value[0].detach().cpu().numpy()
        for j in range(env.num_dofs):
            sliders.append(
                v.UserSlider(f"q[{j}]", float(angle_limits_low[j]), float(angle_limits_high[j]), float(dof_pos_value[j]))
            )

    for s in sliders:
        render.register_menu_item(s)

    # GUI checkboxes for printing
    chk_print_ik_quat = v.UserCheckbox("Print IK target quaternion", False)
    chk_print_joint_pos = v.UserCheckbox("Print joint positions", False)
    render.register_menu_item(chk_print_ik_quat)
    render.register_menu_item(chk_print_joint_pos)

    def handle_print_checkboxes():
        # Print IK target quaternion if requested
        if chk_print_ik_quat.get_value():
            if args.control_ik:
                pos = v.Vec3(sliders[0].get_value(), sliders[1].get_value(), sliders[2].get_value())
                rot = v.quat_from_rpy(v.Vec3(sliders[3].get_value(), sliders[4].get_value(), sliders[5].get_value()))
                print(
                    f"[IK target quaternion] "
                    f"x:{rot.x:.4f}, y:{rot.y:.4f}, z:{rot.z:.4f}, w:{rot.w:.4f} | "
                    f"pos:({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})"
                )
            else:
                print("[Info] IK mode is off — sliders control joint angles.")
            chk_print_ik_quat.set_value(False)

        # Print joint positions if requested
        if chk_print_joint_pos.get_value():
            env.joint_handler.get_positions()
            joints = np.array(env.joint_handler.dof_pos_value[0].tolist(), dtype=np.float32)
            print("[Joint positions] (rad):", np.round(joints, 4).tolist())
            chk_print_joint_pos.set_value(False)

    while True:

        if args.control_ik:
            # Turn euler into quat for env
            rot_rpy = torch.tensor(
                [sliders[3].get_value(),
                sliders[4].get_value(),
                sliders[5].get_value()],
                dtype=torch.float32,
                device=env.device)
            rot = v_quat_from_rpy(rot_rpy).unsqueeze(0)
            rot = canonicalize_quat(rot)

            ee_rpy = v_rpy_from_quat(env.kinematic_sensor_handler.ee_pose_value[:, :4])
            rot_rpy = v_rpy_from_quat(rot)
            print("rot_rpy", rot_rpy.shape)
            print("ee_rpy", ee_rpy.shape)
            rot_rpy[:, 2] = ee_rpy[:, 2] # allow swinging yaw
            rot = v_quat_from_rpy(rot_rpy) 

            # Directly update env target using slider values
            env.ik_handler.ik_target_pose[:, 0] = rot[:, 0]
            env.ik_handler.ik_target_pose[:, 1] = rot[:, 1]
            env.ik_handler.ik_target_pose[:, 2] = rot[:, 2]
            env.ik_handler.ik_target_pose[:, 3] = rot[:, 3]
            env.ik_handler.ik_target_pose[:, 4] = sliders[0].get_value()
            env.ik_handler.ik_target_pose[:, 5] = sliders[1].get_value()
            env.ik_handler.ik_target_pose[:, 6] = sliders[2].get_value()

            # Compute dof pos
            env.ik_handler.compute_inverse_kinematics()
            delta_action = env.ik_handler.ik_dof_pos_target - env.joint_handler.dof_pos_value
            env.step(delta_action)
        else:
            dof_slider_vals = torch.tensor([s.get_value() for s in sliders] * num_envs, dtype=torch.float32, device=env.device)
            delta_action = dof_slider_vals - env.joint_handler.dof_pos_value
            env.step(delta_action)

        # Handle checkbox-triggered prints each frame
        handle_print_checkboxes()

if __name__ == "__main__":
    main()
