from time import perf_counter, sleep
from math import cos, sin, pi
import onnxruntime as ort
import numpy as np

from rustypot_position_hwi import HWI
from raw_imu import Imu
from feet_contacts import FeetContacts
from duck_config import DuckConfig
import time
import json
import os

"""
This is a standalone script that runs on the hardware.
"""


class LowPassActionFilter:
    def __init__(self, control_freq, cutoff_frequency):
        self.last_action = 0
        self.current_action = 0
        self.control_freq = float(control_freq)
        self.cutoff_frequency = float(cutoff_frequency)
        self.alpha = self.compute_alpha()
        print("LowPassActionFilter alpha: ", self.alpha)

    def compute_alpha(self):
        return (1.0 / self.cutoff_frequency) / (
            1.0 / self.control_freq + 1.0 / self.cutoff_frequency
            )

    def push(self, action):
        self.current_action = action

    def get_filtered_action(self):
        self.last_action = (
            self.alpha * self.last_action + (1.0 - self.alpha) * self.current_action
            )
        return self.last_action


class DuckSim2Real():
    def __init__(self, duck_id):
        self.control_freq = 50.0
        # self.action_filter = LowPassActionFilter(control_freq=self.control_freq, cutoff_frequency=30.0)
        
        self.period = 0.72
        self.vel_x = 0.1
        self.yaw_rate = 0.0
        
        self.configs = {
            0: {
                "Kp": 17.0,
                "Kd": 0.0,
                "neck_pitch":0.0 / 180.0 * pi,
                "head_pitch":0.0 / 180.0 * pi,
            }, # default
            2: {
                "Kp": 17.0,
                "neck_pitch":5.0 / 180.0 * pi,
                "head_pitch":0.0 / 180.0 * pi,
            },
            3: {
                "Kp": 19.0,
                "neck_pitch":0.0 / 180.0 * pi,
                "head_pitch":-0.0 / 180.0 * pi,
            },
            4: {
                "Kp": 17.0,
                "neck_pitch":20.0 / 180.0 * pi,
                "head_pitch":-26.0 / 180.0 * pi,
            },
            5: {
                "Kp": 17.0,
                "neck_pitch":10.0 / 180.0 * pi,
                "head_pitch":-13.0 / 180.0 * pi,
            },
            6: {
                "Kp": 17.0,
                "neck_pitch":10.0 / 180.0 * pi,
                "head_pitch":-13.0 / 180.0 * pi,
            },
            7: {
                "Kp": 17.0,
                "neck_pitch":20.0 / 180.0 * pi,
                "head_pitch":-26.0 / 180.0 * pi,
            },
        }
        
        self.max_motor_velocity = 5.24  # rad/s
        self.serial_port = "/dev/ttyACM0"
        self.pitch_bias = 0

        # use zero offsets externally
        self.duck_config = DuckConfig(config_json_path="config.json")
        
        # take the actual offset
        home_dir = os.path.expanduser("~")
        calibrated_config = json.load(open(f"{home_dir}/duck_config.json", "r"))
        self.joints_offsets = calibrated_config["joints_offsets"]
        self.hwi = HWI(self.duck_config, self.serial_port)

        def start():
            if "Kp" in self.configs[duck_id]:
                Kp = self.configs[duck_id]["Kp"]
            else:
                Kp = 17.0
            if "Kd" in self.configs[duck_id]:
                Kd = self.configs[duck_id]["Kd"]
            else:
                Kd = 0.0
            self.hwi.set_kps([Kp] * 14)
            self.hwi.set_kds([Kd] * 14)
            
            # self.hwi.init_pos["neck_pitch"] += self.configs[duck_id]["neck_pitch"]
            # self.hwi.init_pos["head_pitch"] += self.configs[duck_id]["head_pitch"]
            self.hwi.turn_on()
            sleep(2)
        start()

        self.imu = Imu(
            sampling_freq=int(self.control_freq),
            user_pitch_bias=self.pitch_bias,
            upside_down=self.duck_config.imu_upside_down,
            )
        self.feet_contacts = FeetContacts()

        keys_legs = [
            "left_hip_yaw",
            "left_hip_roll",
            "left_hip_pitch",
            "left_knee",
            "left_ankle",
            "right_hip_yaw",
            "right_hip_roll",
            "right_hip_pitch",
            "right_knee",
            "right_ankle",
        ]
        keys_head = [
            "neck_pitch",
            "head_pitch",
            "head_yaw",
            "head_roll",
        ]
        
        offsets_legs = {k: self.joints_offsets[k] for k in keys_legs}
        offsets_head = {k: self.joints_offsets[k] for k in keys_head}
        # push the head back for balance        
        offsets_head["neck_pitch"] += self.configs[duck_id]["neck_pitch"]
        offsets_head["head_pitch"] += self.configs[duck_id]["head_pitch"]
        
        print("offsets_legs: ", offsets_legs)
        print("offsets_head: ", offsets_head)
                
        self.offsets_legs = np.array(list(offsets_legs.values()), dtype=np.float32)
        # self.offsets_legs[:] = 0.0
        # self.home_pos_head = [0.078, -0.036, 0.16, 0.053]
        self.home_pos_head = list(offsets_head.values())
        self.temp_pos_head = [0.0] * 4
        # self.home_pos_head = [0.0] * 4
        
        # self.offsets_legs[:] = 0.0
        # self.home_pos_head = [0.0] * 4
        
        home_pos = [0.002, 0.053, -0.63, 1.368, -0.784,
                    -0.003, -0.065, 0.635, 1.379, -0.796]
        self.home_pos = np.array(home_pos, dtype=np.float32) + self.offsets_legs

        self.dof_pos = home_pos
        self.dof_vel = [0.0] * len(home_pos)

        self.q_curr = np.copy(self.home_pos)
        self.q_last = np.copy(self.home_pos)
        self.q_targets = np.copy(self.home_pos)

        self.action_min_delay = 0
        self.action_max_delay = 4
        self.action_hist = np.zeros((self.action_max_delay - self.action_min_delay, len(self.home_pos)), dtype=np.float32)

        # From vsim env
        self.dof_dict = {'left_hip_yaw': 4,
                         'left_hip_roll': 6,
                         'left_hip_pitch': 8,
                         'left_knee': 10,
                         'left_ankle': 12,
                         'right_hip_yaw': 14,
                         'right_hip_roll': 16,
                         'right_hip_pitch': 18,
                         'right_knee': 20,
                         'right_ankle': 22}

        # Hardware
        hw_joints = {'left_hip_yaw': 20,
                     'left_hip_roll': 21,
                     'left_hip_pitch': 22,
                     'left_knee': 23,
                     'left_ankle': 24,
                     'neck_pitch': 30,
                     'head_pitch': 31,
                     'head_yaw': 32,
                     'head_roll': 33,
                     'right_hip_yaw': 10,
                     'right_hip_roll': 11,
                     'right_hip_pitch': 12,
                     'right_knee': 13,
                     'right_ankle': 14}
        self.hw_joint_ids = list(hw_joints.values())

        # import hardware dof ranges
        self.limits = {
            "left_hip_yaw": {"zero": 0.004601942363656963, "high": 0.45866025557780654, "low": -0.6642136811544828}, "left_hip_roll": {"zero": -0.004601942363656963, "high": 0.6734175658817967, "low": -0.5123495831538043}, "left_hip_pitch": {"zero": -0.010737865515199285, "high": 0.47093210188089163, "low": -1.4419419406125027}, "left_knee": {"zero": -0.006135923151542766, "high": 1.9251458887964796, "low": -0.9418642037617837}, "left_ankle": {"zero": -0.0015339807878858025, "high": 0.9495341077012123, "low": -1.2701360923693108}, "right_hip_yaw": {"zero": -0.003067961575771161, "high": 0.6672816427302539, "low": -0.5338253141842033}, "right_hip_roll": {"zero": -0.0015339807878858025, "high": 0.636602026972541, "low": -0.5921165841238576}, "right_hip_pitch": {"zero": 0.010737865515199285, "high": 1.4940972874006144, "low": -0.38196121618352485}, "right_knee": {"zero": -0.003067961575771161, "high": 1.9680973508572777, "low": -0.8989127417009857}, "right_ankle": {"zero": 0.0, "high": 0.9710098387316108, "low": -1.2241166687327416}
            }

        self.joints = []
        self.pid_low = []
        self.pid_high = []

        for joint, lim in self.limits.items():
            self.joints.append(joint)
            self.pid_low.append(lim["low"])
            self.pid_high.append(lim["high"])

        self.pid_low = np.array(self.pid_low, dtype=np.float32)
        self.pid_high = np.array(self.pid_high, dtype=np.float32)
        deltas = self.pid_high - self.pid_low
        self.pid_low = self.pid_low + 0.1 * deltas
        self.pid_high = self.pid_high - 0.1 * deltas
        print("pid_low 10%", self.pid_low.tolist())
        print("home_pos ", self.home_pos.tolist())
        print("pid_high 10%", self.pid_high.tolist())

        self.ignore = [
            "left_antenna",
            "right_antenna",
            "neck_pitch",
            "head_pitch",
            "head_yaw",
            "head_roll",
            ]

    def get_obs(self, total_time: float, use_control_policy:bool) -> bool:
        t_start = perf_counter()
        print("Getting observations from hardware")

        # HWI
        dof_pos = self.hwi.get_present_positions(ignore=self.ignore)  # rad
        print("dof_pos read time ", perf_counter() - t_start)
        dof_vel = self.hwi.get_present_velocities(ignore=self.ignore)  # rad/s
        print("dof_vel read time ", perf_counter() - t_start)
        if dof_pos is not None:
            self.dof_pos = dof_pos
        if dof_vel is not None:
            self.dof_vel = dof_vel
        print("hwi read time ", perf_counter() - t_start)

        # IMU + feet contacts
        imu_data = self.imu.get_data()
        print("imu read time ", perf_counter() - t_start)
        
        gyro = imu_data["gyro"]
        accelero = imu_data["accelero"]
        # print("gyro ", gyro)
        # print("accelero ", accelero)
        
        if not isinstance(gyro, list):
            gyro = gyro.tolist()
        if not isinstance(accelero, list):
            accelero = accelero.tolist()
            
        gyro = np.array(gyro, dtype=np.float32)
        accelero = np.array(accelero, dtype=np.float32)
        
        # empirical ranges to avoid outlier noise
        gyro = np.clip(gyro, a_min=-2.0, a_max=2.0)
        accelero[0:2] = np.clip(accelero[0:2], a_min=-15.0, a_max=15.0)
        accelero[2:3] = np.clip(accelero[2:3], a_min=-5.0, a_max=25.0)
        print("imu process time ", perf_counter() - t_start)
        
        feet_contacts = self.feet_contacts.get()
        print("feet_contacts read time ", perf_counter() - t_start)

        self.q_last = np.copy(self.q_curr)
        self.q_curr[:] = np.array(self.dof_pos, dtype=np.float32)

        vel_xy_cmd = np.array([self.vel_x, 0.0], dtype=np.float32)
        yaw_rate_cmd = np.array([self.yaw_rate], dtype=np.float32)

        phase = (total_time % self.period) / self.period  # range [0 - 1)
        phase_cos = cos(phase * 2.0 * pi)
        phase_sin = sin(phase * 2.0 * pi)
        # print("phase ", phase)

        # assemble the observations
        lin_vel_x_range = [0.0, 0.15]
        mean = 0.5 * (lin_vel_x_range[0] + lin_vel_x_range[1])
        vel_xy_cmd[0] = (vel_xy_cmd[0] - mean) * 10.0
        # print("vel_xy_cmd ", vel_xy_cmd)
        print("obs preproc time ", perf_counter() - t_start)
        
        dof_vel_scale = 0.05
        obs = np.concatenate(
            (
                gyro,  # gyroscope
                accelero,  # accelerometer
                vel_xy_cmd,
                yaw_rate_cmd,
                self.q_curr - self.home_pos,  # deltas from home, no need to offset
                dof_vel_scale * np.array(self.dof_vel, dtype=np.float32),  # velocity, no offset
                np.reshape(self.action_hist, (-1)),  # deltas from home, no need to offset
                self.q_targets - self.offsets_legs,  # absolute pid targets, offset real to sim
                np.array(feet_contacts, dtype=np.float32),  # order: left, right
                np.array([phase_cos, phase_sin], dtype=np.float32)
                )
            )
        
        ####################################################################################################
        # HACKATHON TODO
        ####################################################################################################
        obs_ctrl = None
        if use_control_policy:
            obs_ctrl = np.concatenate(
                (
                    [0.0] * 9
                ))
        print("obs concat time ", perf_counter() - t_start)

        # print("Getting observations from hardware DONE, dof_pos: ", self.q_curr.tolist())
        print("Getting observations from hardware DONE")
        return obs, obs_ctrl

    def publish(self, q_delta_relative: np.array, time: float = 0.0) -> None:
        self.q_targets[:] = 1.0 * q_delta_relative + self.home_pos
        self.q_targets[:] = np.clip(
            self.q_targets,
            a_min=self.pid_low,
            a_max=self.pid_high)

        # save last action for observations
        self.action_hist[:] = np.roll(self.action_hist, (-1, 0))  # roll left along dim 0
        self.action_hist[-1, :] = q_delta_relative  # assign last

        # self.action_filter.push(np.copy(self.q_targets))
        # if time > 1.0:  # give time to the filter to stabilize
        #     self.q_targets[:] = self.action_filter.get_filtered_action()

        # self.q_targets[:] = self.offsets_legs

        q_targets = self.q_targets.tolist()
        # print("\nQ_TARGETS: ", q_targets)
        # add zeros for head and neck joints
        q_targets = q_targets[0:5] + self.temp_pos_head + q_targets[5:10]
        self.hwi.io.write_goal_position(self.hw_joint_ids, q_targets)


class Moderator():
    def __init__(self, duck_id):
        self.use_control_policy = False
        
        self.ort_gait = ort.InferenceSession("duck_gait.onnx")
        if self.use_control_policy:
            self.ort_ctrl = ort.InferenceSession("duck_ctrl.onnx")

        self.hardware = DuckSim2Real(duck_id)

        self.obs = None
        self.obs_ctrl = None
        self.dt = 1.0 / 50.0
        self.time = 0.0

    def send_home(self) -> None:
        print("sending home: ", self.hardware.home_pos.tolist())
        self.hardware.publish(np.array([0.0] * 10))
        sleep(2.0)
        print("sending home done")

    def step(self, warmup_time):
        t_start = perf_counter()
        
        # Read obs
        obs, obs_ctrl = self.hardware.get_obs(total_time=self.time, use_control_policy=self.use_control_policy)
        if obs is not None:
            print("new obs dof pos: ", obs[9:19].tolist())
            self.obs = obs
            self.obs_ctrl = obs_ctrl

        if self.obs is None:
            print("WARN: no observations, skipping step")
            step_time = perf_counter() - t_start
            print("step_time ", step_time)
            self.time += step_time
            return None

        # Infer
        ts = perf_counter()
        
        if self.use_control_policy:
            ctrl_out = self.ort_ctrl.run(None, {"input": np.expand_dims(self.obs_ctrl, 0)})[0][0]
            cmd_x = ctrl_out[0]
            cmd_yaw = ctrl_out[1]
            # override commands in obs
            self.obs[6] = cmd_x
            self.obs[8] = cmd_yaw
        
        q_delta_relative = self.ort_gait.run(None, {"input": np.expand_dims(self.obs, 0)})[0][0]
        print("inference time ", perf_counter() - ts)

        print("q_delta_relative ", q_delta_relative)

        # slow start to avoid falling
        if self.time < warmup_time:
            ratio = self.time / warmup_time
            q_delta_relative = q_delta_relative * ratio
            
            self.hardware.temp_pos_head = [pos * ratio for pos in self.hardware.home_pos_head]
        # q_delta_relative = q_delta_relative * 0.0

        print("new actions ", q_delta_relative.tolist())
        self.hardware.publish(q_delta_relative, self.time)

        step_time = perf_counter() - t_start
        to_sleep_time = max(0.0, self.dt - step_time)
        print("step_time ", step_time)
        print("to_sleep_time ", to_sleep_time)
        sleep(to_sleep_time)
        self.time += self.dt

        return obs


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--duck_id", type=int, default=0, help="duck_id")
    parser.add_argument("--warmup", type=float, default=4.0, help="warmup time")
    parser.add_argument("--gait_only", type=int, default=1, help="gait_only")
    parser.add_argument("--vel_x", type=float, default=0.1, help="vel_x")
    parser.add_argument("--yaw_rate", type=float, default=0.0, help="yaw_rate")
    parser.add_argument("--period", type=float, default=0.72, help="period")
    parser.add_argument("--write_obs", type=int, default=0, help="write_obs")

    args = parser.parse_args()
    duck_id = args.duck_id
    warmup_time = args.warmup
    gait_only = args.gait_only
    vel_x = args.vel_x
    yaw_rate = args.yaw_rate
    period = args.period
    write_obs = args.write_obs

    moderator = Moderator(duck_id)
    moderator.hardware.period = period
    print("Moderator created, starting sim2real.")
    
    if gait_only:
        moderator.hardware.vel_x = vel_x
        moderator.hardware.yaw_rate = yaw_rate
        
        print("using vel_x: ", vel_x)
        print("using yaw_rate: ", yaw_rate)

    moderator.send_home()
    
    time_lst = []
    obs_lst = []
    
    obs_written = not write_obs
    t_start = -1.0
    dt = 0.0
    t_max = 10.0

    print("starting inference...")
    t_start = time.perf_counter()
    while True:
        ts = perf_counter()
        obs = moderator.step(warmup_time=warmup_time)
        print("moderator step time: ", perf_counter() - ts)
    
        if obs_written:
            continue
        
        dt = time.perf_counter() - t_start
        if dt > 0.0:
            time_lst.append(dt - 0.0)
            obs_lst.append(obs.tolist())
            
        if dt > t_max:
            with open("obs.json", "w") as f:
                data = {
                    "time": time_lst,
                    "obs": obs_lst
                }
                json.dump(data, f, indent=4)
                
            obs_written = True
            
            input("Observations written, press any key to continue running the policy...")
