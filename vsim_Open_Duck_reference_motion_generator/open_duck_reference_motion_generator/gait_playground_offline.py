import numpy as np
import json
import os
from scipy.spatial.transform import Rotation as R
from placo_walk_engine import PlacoWalkEngine
from scipy.stats import qmc
import matplotlib.pyplot as plt

# Define the parameters class to hold the variables
class GaitParameters:
    def __init__(self, duck = "open_duck_mini_v2"):
        script_path = os.path.dirname(os.path.abspath(__file__))
        if duck == "open_duck_mini":
            self.robot = 'open_duck_mini'
            self.robot_urdf = "open_duck_mini.urdf"
            self.asset_path = os.path.join(script_path, "../open_duck_reference_motion_generator/robots/open_duck_mini/")
        elif duck == "open_duck_mini_v2":
            self.robot = 'open_duck_mini_v2'
            self.robot_urdf = "open_duck_mini_v2.urdf"
            self.asset_path = os.path.join(script_path, "../open_duck_reference_motion_generator/robots/open_duck_mini_v2/")
        elif duck == "go_bdx":
            self.robot = 'go_bdx'
            self.robot_urdf = "go_bdx.urdf"
            self.asset_path = os.path.join(script_path, "../open_duck_reference_motion_generator/robots/go_bdx/")
        self.dx = 0.1
        self.dy = 0.0
        self.dtheta = 0.0
        self.duration = 5
        self.hardware = True

    def reset(self, pwe):
        pwe.parameters.double_support_ratio = self.double_support_ratio
        pwe.parameters.startend_double_support_ratio = self.startend_double_support_ratio
        pwe.parameters.planned_timesteps = self.planned_timesteps
        pwe.parameters.replan_timesteps = self.replan_timesteps
        pwe.parameters.walk_com_height = self.walk_com_height
        pwe.parameters.walk_foot_height = self.walk_foot_height
        pwe.parameters.walk_trunk_pitch = np.deg2rad(self.walk_trunk_pitch)
        pwe.parameters.walk_foot_rise_ratio = self.walk_foot_rise_ratio
        pwe.parameters.single_support_duration = self.single_support_duration
        pwe.parameters.single_support_timesteps = self.single_support_timesteps
        pwe.parameters.foot_length = self.foot_length
        pwe.parameters.feet_spacing = self.feet_spacing
        pwe.parameters.zmp_margin = self.zmp_margin
        pwe.parameters.foot_zmp_target_x = self.foot_zmp_target_x
        pwe.parameters.foot_zmp_target_y = self.foot_zmp_target_y
        pwe.parameters.walk_max_dtheta = self.walk_max_dtheta
        pwe.parameters.walk_max_dy = self.walk_max_dy
        pwe.parameters.walk_max_dx_forward = self.walk_max_dx_forward
        pwe.parameters.walk_max_dx_backward = self.walk_max_dx_backward

    def save_to_json(self, filename):
        data = {
            'dx': self.dx,
            'dy': self.dy,
            'dtheta': self.dtheta,
            'duration': self.duration,
            'hardware': self.hardware,
            'double_support_ratio': self.double_support_ratio,
            'startend_double_support_ratio': self.startend_double_support_ratio,
            'planned_timesteps': self.planned_timesteps,
            'replan_timesteps': self.replan_timesteps,
            'walk_com_height': self.walk_com_height,
            'walk_foot_height': self.walk_foot_height,
            'walk_trunk_pitch': self.walk_trunk_pitch,
            'walk_foot_rise_ratio': self.walk_foot_rise_ratio,
            'single_support_duration': self.single_support_duration,
            'single_support_timesteps': self.single_support_timesteps,
            'foot_length': self.foot_length,
            'feet_spacing': self.feet_spacing,
            'zmp_margin': self.zmp_margin,
            'foot_zmp_target_x': self.foot_zmp_target_x,
            'foot_zmp_target_y': self.foot_zmp_target_y,
            'walk_max_dtheta': self.walk_max_dtheta,
            'walk_max_dy': self.walk_max_dy,
            'walk_max_dx_forward': self.walk_max_dx_forward,
            'walk_max_dx_backward': self.walk_max_dx_backward,
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)

    def create_pwe(self, parameters=None):
        pwe = PlacoWalkEngine(self.asset_path, self.robot_urdf, parameters)
        self.reset(pwe)
        pwe.set_traj(0, 0, 0)
        return pwe

    def custom_preset_name(self):
        return f"placo_{gait.robot}_defaults.json"

    def save_custom_presets(self):
        filename = self.custom_preset_name()
        self.save_to_json(filename)

    def load_defaults(self, pwe):
        self.load_from_json(os.path.join(pwe.asset_path, "placo_defaults.json"))

    def load_from_json(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        self.load_from_data(data)

    def load_from_data(self, data):
        self.dx = data.get('dx')
        self.dy = data.get('dy')
        self.dtheta = data.get('dtheta')
        self.duration = data.get('duration')
        self.hardware = data.get('hardware')
        self.double_support_ratio = data.get('double_support_ratio')
        self.startend_double_support_ratio = data.get('startend_double_support_ratio')
        self.planned_timesteps = data.get('planned_timesteps')
        self.replan_timesteps = data.get('replan_timesteps')
        self.walk_com_height = data.get('walk_com_height')
        self.walk_foot_height = data.get('walk_foot_height')
        self.walk_trunk_pitch = data.get('walk_trunk_pitch')
        self.walk_foot_rise_ratio = data.get('walk_foot_rise_ratio')
        self.single_support_duration = data.get('single_support_duration')
        self.single_support_timesteps = data.get('single_support_timesteps')
        self.foot_length = data.get('foot_length')
        self.feet_spacing = data.get('feet_spacing')
        self.zmp_margin = data.get('zmp_margin')
        self.foot_zmp_target_x = data.get('foot_zmp_target_x')
        self.foot_zmp_target_y = data.get('foot_zmp_target_y')
        self.walk_max_dtheta = data.get('walk_max_dtheta')
        self.walk_max_dy = data.get('walk_max_dy')
        self.walk_max_dx_forward = data.get('walk_max_dx_forward')
        self.walk_max_dx_backward = data.get('walk_max_dx_backward')

def do_sample():
    # # Latin Hypercube Sampling: dx and dtheta parameter ranges
    # Generating a lot of samples is quite slow, so start with a low number
    n_samples = 2
    linear_velocity_x_range = (0.0, 0.15)
    angular_velocity_z_range = (-0.5, 0.5)
    
    # Create LHS sampler for 2 parameters
    sampler = qmc.LatinHypercube(d=2, optimization="random-cd")
    lhs_samples = sampler.random(n=n_samples)

    # Scale samples to specified parameter ranges
    scaled_samples = qmc.scale(lhs_samples,
                            l_bounds=[linear_velocity_x_range[0], angular_velocity_z_range[0]],
                            u_bounds=[linear_velocity_x_range[1], angular_velocity_z_range[1]])
    
    # # Override samples manually
    # scaled_samples = np.array([[0.1, 0.0]])

    # Plot the samples
    plt.figure(figsize=(6, 5))
    plt.scatter(scaled_samples[:, 0], scaled_samples[:, 1], c='blue', s=40, edgecolor='k')
    plt.xlabel('Lin vel x')
    plt.ylabel('Ang vel z')
    plt.title('Gait samples')
    plt.grid(True)
    plt.tight_layout()

    # Save the plot to file
    output_filename = 'gait_samples.png'
    plt.savefig(output_filename, dpi=300)
    plt.close()  # Close the figure after saving

    print(f"LHS sampling complete. Scatter plot saved as '{output_filename}'.")
    print("Samples:")
    print(scaled_samples)
    
    return scaled_samples

def process_episode(episode, dt, steps_per_period):
    N = steps_per_period
    
    def extract_reference_slice(data):
        start_slice = 3
        num_slices = 10
        avg_data = data[start_slice * N: (start_slice + 1) * N, :]
        for i in range(start_slice + 1, start_slice + num_slices):
            avg_data += data[i * N: (i + 1) * N, :]
        avg_data /= float(num_slices)
        return avg_data
    
    joint_pos = np.array(episode["joint_pos"], dtype=np.float32)
    # joint_vel = np.array(episode["joint_vel"], dtype=np.float32)
    # world_linear_vel = np.array(episode["world_linear_vel"], dtype=np.float32)
    # world_angular_vel = np.array(episode["world_angular_vel"], dtype=np.float32)
    foot_contacts = np.array(episode["foot_contacts"], dtype=np.float32) > 0.5
    
    joint_pos = extract_reference_slice(joint_pos)
    # joint_vel = extract_reference_slice(joint_vel)
    # world_linear_vel = extract_reference_slice(world_linear_vel)
    # world_angular_vel = extract_reference_slice(world_angular_vel)
    foot_contacts = extract_reference_slice(np.float32(foot_contacts))
    foot_contacts = np.int32(foot_contacts > 0.5)
    
    assert len(joint_pos) == N
    # assert len(joint_vel) == N
    # assert len(world_linear_vel) == N
    # assert len(world_angular_vel) == N
    assert len(foot_contacts) == N
    
    def fit_polynomial(data, dt):
        T, J = data.shape # shape (T, J)
        time = np.arange(T) * dt

        degree = 15  # polynomial degree
        polys = []  # will hold coefficients for each joint
        for j in range(J):
            y = data[:, j]
            coeffs = np.polyfit(time, y, degree)
            polys.append(coeffs)

        return polys
    
    def eval_polynomial(polys, data, dt):
        T, J = data.shape # shape (T, J)
        time = np.arange(T) * dt

        # Evaluate fitted curves
        fitted = []
        for j, coeffs in enumerate(polys):
            p = np.poly1d(coeffs)
            fitted.append(p(time))

        return np.array(fitted).T  # shape (T, J)

    # Helps to smooth the signal a bit
    polys = fit_polynomial(joint_pos, dt)
    joint_pos = eval_polynomial(polys, joint_pos, dt)
    assert len(joint_pos) == N
    
    # polys = fit_polynomial(world_linear_vel, dt)
    # world_linear_vel = eval_polynomial(polys, world_linear_vel, dt)
    # assert len(world_linear_vel) == N
    
    # polys = fit_polynomial(world_angular_vel, dt)
    # world_angular_vel = eval_polynomial(polys, world_angular_vel, dt)
    # assert len(world_angular_vel) == N
    
    return joint_pos, foot_contacts

def main():
    FPS = 50
    MESHCAT_FPS = 50
    DT = 0.001
    episode = {
        # "LoopMode": "Wrap",
        "FrameDuration": np.around(1 / FPS, 4),
        # "EnableCycleOffsetPosition": True,
        # "EnableCycleOffsetRotation": False,
        # "Debug_info": [],
        # "Frames": [],
        # "MotionWeight": 1,
        "root_position": [],
        "root_orientation_quat": [],
        "joint_pos": [],
        "left_toe_pos": [],
        "right_toe_pos": [],
        "world_linear_vel": [],
        "world_angular_vel": [],
        "joint_vel": [],
        "left_toe_vel": [],
        "right_toe_vel": [],
        "foot_contacts": [],
    }
    
    dt = 0.02
    
    # # Latin Hypercube Sampling: dx and dtheta parameter ranges
    samples = do_sample()
    n_samples = len(samples)
    
    for idx, sample in enumerate(samples):
        print("Generating sample no. ", idx)
        gait = GaitParameters()    
        filename = os.path.join(gait.asset_path, "placo_defaults.json")
        print("gait config file: ", filename)
        with open(filename, 'r') as f:
            gait_parameters = json.load(f)
            print(f"gait_parameters {gait_parameters}")
            gait.load_from_data(gait_parameters)
            
        # adjust gait params here
        print("next sample: ", sample)
        gait.dx = sample[0]
        gait.dtheta = sample[1]
        
        # create pwe every time, as I noticed an issue with reset in the browser
        print("Gait generator started...")
        pwe = gait.create_pwe(gait_parameters)
        gait.reset(pwe)
        pwe.set_traj(gait.dx, gait.dy, gait.dtheta + 0.001)
        start = pwe.t
        
        episode["joint_names"] = pwe.joints
        episode["gait_params"] = {
            'dx': gait.dx,
            'dy': gait.dy,
            'dtheta': gait.dtheta,
            'duration': gait.duration,
            'hardware': gait.hardware,
            'double_support_ratio': gait.double_support_ratio,
            'startend_double_support_ratio': gait.startend_double_support_ratio,
            'planned_timesteps': gait.planned_timesteps,
            'replan_timesteps': gait.replan_timesteps,
            'walk_com_height': gait.walk_com_height,
            'walk_foot_height': gait.walk_foot_height,
            'walk_trunk_pitch': gait.walk_trunk_pitch,
            'walk_foot_rise_ratio': gait.walk_foot_rise_ratio,
            'single_support_duration': gait.single_support_duration,
            'single_support_timesteps': gait.single_support_timesteps,
            'foot_length': gait.foot_length,
            'feet_spacing': gait.feet_spacing,
            'zmp_margin': gait.zmp_margin,
            'foot_zmp_target_x': gait.foot_zmp_target_x,
            'foot_zmp_target_y': gait.foot_zmp_target_y,
            'walk_max_dtheta': gait.walk_max_dtheta,
            'walk_max_dy': gait.walk_max_dy,
            'walk_max_dx_forward': gait.walk_max_dx_forward,
            'walk_max_dx_backward': gait.walk_max_dx_backward,
            "period": pwe.period
        }
        assert dt == episode["FrameDuration"]
        period = episode["gait_params"]["period"]
        from math import floor
        steps_per_period = floor(period / dt)
        print("period: ", period)
        print("steps_per_period: ", steps_per_period)

        last_record = 0
        prev_root_position = [0, 0, 0]
        prev_root_orientation_euler = [0, 0, 0]
        prev_left_toe_pos = [0, 0, 0]
        prev_right_toe_pos = [0, 0, 0]
        prev_joints_positions = None
        prev_initialized = False
        while True:
            pwe.tick(DT)
            if pwe.t <= 0:
                # print("waiting ")
                start = pwe.t
                last_record = pwe.t + 1 / FPS
                last_meshcat_display = pwe.t + 1 / MESHCAT_FPS
                continue

            # print(np.around(pwe.robot.get_T_world_fbase()[:3, 3], 3))

            if pwe.t - last_record >= 1 / FPS:
                # before
                # T_world_fbase = pwe.robot.get_T_world_fbase()
                # after
                T_world_fbase = pwe.robot.get_T_world_trunk()
                # fv.pushFrame(T_world_fbase, "trunk")
                root_position = list(T_world_fbase[:3, 3])
                root_orientation_quat = list(R.from_matrix(T_world_fbase[:3, :3]).as_quat())
                joints_positions = list(pwe.get_angles().values())

                T_world_leftFoot = pwe.robot.get_T_world_left()
                T_world_rightFoot = pwe.robot.get_T_world_right()

                # fv.pushFrame(T_world_leftFoot, "left")
                # fv.pushFrame(T_world_rightFoot, "right")

                T_body_leftFoot = np.linalg.inv(T_world_fbase) @ T_world_leftFoot
                T_body_rightFoot = np.linalg.inv(T_world_fbase) @ T_world_rightFoot

                # left_foot_pose = pwe.robot.get_T_world_left()
                # right_foot_pose = pwe.robot.get_T_world_right()

                left_toe_pos = list(T_body_leftFoot[:3, 3])
                right_toe_pos = list(T_body_rightFoot[:3, 3])

                world_linear_vel = list(
                    (np.array(root_position) - np.array(prev_root_position)) / (1 / FPS)
                )
                body_rot_mat = T_world_fbase[:3, :3]
                body_linear_vel = list(body_rot_mat.T @ world_linear_vel)
                # print("world linear vel", world_linear_vel)
                # print("body linear vel", body_linear_vel)

                world_angular_vel = list(
                    (
                        R.from_quat(root_orientation_quat).as_euler("xyz")
                        - prev_root_orientation_euler
                    )
                    / (1 / FPS)
                )
                body_angular_vel = list(body_rot_mat.T @ world_angular_vel)
                # print("world angular vel", world_angular_vel)
                # print("body angular vel", body_angular_vel)

                if prev_joints_positions == None:
                    prev_joints_positions = [0] * len(joints_positions)

                joints_vel = list(
                    (np.array(joints_positions) - np.array(prev_joints_positions)) / (1 / FPS)
                )
                left_toe_vel = list(
                    (np.array(left_toe_pos) - np.array(prev_left_toe_pos)) / (1 / FPS)
                )
                right_toe_vel = list(
                    (np.array(right_toe_pos) - np.array(prev_right_toe_pos)) / (1 / FPS)
                )

                foot_contacts = pwe.get_current_support_phase()

                if prev_initialized:
                    if gait.hardware:
                        episode["root_position"].append(root_position)
                        episode["root_orientation_quat"].append(root_orientation_quat)
                        episode["joint_pos"].append(joints_positions)
                        episode["left_toe_pos"].append(left_toe_pos)
                        episode["right_toe_pos"].append(right_toe_pos)
                        episode["world_linear_vel"].append(world_linear_vel)
                        episode["world_angular_vel"].append(world_angular_vel)
                        episode["joint_vel"].append(joints_vel)
                        episode["left_toe_vel"].append(left_toe_vel)
                        episode["right_toe_vel"].append(right_toe_vel)
                        episode["foot_contacts"].append(foot_contacts)
                    else:
                        episode["Frames"].append(
                            root_position + root_orientation_quat + joints_positions
                        )

                prev_root_position = root_position.copy()
                prev_root_orientation_euler = (
                    R.from_quat(root_orientation_quat).as_euler("xyz").copy()
                )
                prev_left_toe_pos = left_toe_pos.copy()
                prev_right_toe_pos = right_toe_pos.copy()
                prev_joints_positions = joints_positions.copy()
                prev_initialized = True

                last_record = pwe.t
                # print("saved frame")

            if pwe.t - start > gait.duration:
                # joint_pos, world_linear_vel, world_angular_vel, foot_contacts = process_episode(episode, dt=dt, steps_per_period=steps_per_period)
                joint_pos, foot_contacts = process_episode(episode, dt=dt, steps_per_period=steps_per_period)
                
                
                joint_pos = np.expand_dims(joint_pos, axis=0)
                # world_linear_vel = np.expand_dims(world_linear_vel, axis=0)
                # world_angular_vel = np.expand_dims(world_angular_vel, axis=0)
                foot_contacts = np.expand_dims(foot_contacts, axis=0)
                
                if idx == 0:
                    all_joint_pos = joint_pos
                    # all_world_linear_vel = world_linear_vel
                    # all_world_angular_vel = world_angular_vel
                    all_foot_contacts = foot_contacts
                else:
                    all_joint_pos = np.concatenate((all_joint_pos, joint_pos), axis=0)
                    # all_world_linear_vel = np.concatenate((all_world_linear_vel, world_linear_vel), axis=0)
                    # all_world_angular_vel = np.concatenate((all_world_angular_vel, world_angular_vel), axis=0)
                    all_foot_contacts = np.concatenate((all_foot_contacts, foot_contacts), axis=0)
                
                break

    # dump all processed episodes
    assert len(all_joint_pos) == n_samples
    # assert len(all_world_linear_vel) == n_samples
    # assert len(all_world_angular_vel) == n_samples
    assert len(all_foot_contacts) == n_samples
    all_episodes = {
        "joint_names": episode['joint_names'],
        "dt": dt,
        "period": period,
        "steps_per_period": steps_per_period,
        "n_samples": n_samples,
        "dx_dtheta": samples.tolist(),
        "joint_pos": all_joint_pos.tolist(),
        # "world_linear_vel": all_world_linear_vel.tolist(),
        # "world_angular_vel": all_world_angular_vel.tolist(),
        "foot_contacts": all_foot_contacts.tolist(),
        }
    with open("all_episodes.json", "w") as f:
        json.dump(all_episodes, f, indent=4, sort_keys=True)
    print("Saving gait done.")
        

if __name__ == '__main__':
    main()
