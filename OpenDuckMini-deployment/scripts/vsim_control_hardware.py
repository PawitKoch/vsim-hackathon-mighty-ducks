from vsim_endpoint.ops import PubSub, JointStatePublisher, JointStateSubscriber, FloatListPublisher
from vsim_endpoint.ops import JointState as VsimJointState
from vsim_endpoint.ops import FloatList

from rustypot_position_hwi import HWI
from raw_imu import Imu
# from rl_utils import LowPassActionFilter
from feet_contacts import FeetContacts
from duck_config import DuckConfig

import time
import json

"""
This script uses vsim_endpoint to control the duck from another computer by sending actions and receiving joint and kinematic states.
This script runs on the hardware.
"""


class DuckHardwarePubSub():
    def __init__(self):
        self.pubsub = PubSub(subscribe_port=5051, publish_port=5050, ip="*")
        self.joint_sub = JointStateSubscriber(self.pubsub, "joint_targets")
        self.joint_pub = JointStatePublisher(self.pubsub, "joint_states")
        self.kine_pub = FloatListPublisher(self.pubsub, "kinematics")

        self.joint_targets = VsimJointState()
        self.kine_states = FloatList()

    def read(self) -> list[float]:
        self.pubsub.flush()
        joint_states = self.joint_sub.read()
        if joint_states is None:
            return None
        return joint_states.positions

    def publish(self, dof_pos: list[float], dof_vel: list[float], kinematics: list[float]):
        self.joint_targets.positions = dof_pos
        self.joint_targets.velocities = dof_vel
        self.joint_pub.publish(self.joint_targets)

        self.kine_states.data = kinematics
        print("publishing kine states: ", self.kine_states.data)
        self.kine_pub.publish(self.kine_states)


class HardwareIO():
    def __init__(self,
                 duck_config_path
                 ):
        self.control_freq = 50.0
        self.cutoff_frequency = 30.0
        self.pid = [30, 0, 0]
        self.max_motor_velocity = 5.24  # rad/s
        self.serial_port = "/dev/ttyACM0"
        self.pitch_bias = 0

        self.duck_config = DuckConfig(config_json_path=duck_config_path)
        self.hwi = HWI(self.duck_config, self.serial_port)
        self.start()
        self.imu = Imu(
            sampling_freq=int(self.control_freq),
            user_pitch_bias =self.pitch_bias,
            upside_down=self.duck_config.imu_upside_down,
            )
        self.dof_pos = None
        self.dof_vel = None
        self.feet_contacts = FeetContacts()
        # self.action_filter = LowPassActionFilter(
        #     control_freq=self.control_freq, cutoff_frequency=self.cutoff_frequency)

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

        self.pubsub = DuckHardwarePubSub()

    def start(self):
        kps = [self.pid[0]] * 14
        kds = [self.pid[2]] * 14

        # # lower head kps
        # kps[5:9] = [8, 8, 8, 8]

        self.hwi.set_kps(kps)
        self.hwi.set_kds(kds)
        self.hwi.turn_on()

        time.sleep(2)

    def send_observations_to_policy(self) -> None:
        t_start = time.perf_counter()
        
        # HWI
        dof_pos = self.hwi.get_present_positions(
            ignore=[
                "left_antenna",
                "right_antenna",
                "neck_pitch",
                "head_pitch",
                "head_yaw",
                "head_roll",
                ]
            )  # rad
        dof_vel = self.hwi.get_present_velocities(
            ignore=[
                "left_antenna",
                "right_antenna",
                "neck_pitch",
                "head_pitch",
                "head_yaw",
                "head_roll",
                ]
            )  # rad/s
        if dof_pos is not None:
            self.dof_pos = dof_pos
        if dof_vel is not None:
            self.dof_vel = dof_vel
        print("hwi ", time.perf_counter() - t_start)
        

        # IMU + feet contacts
        imu_data = self.imu.get_data()
        gyro = imu_data["gyro"]
        accelero = imu_data["accelero"]
        if not isinstance(gyro, list):
            gyro = gyro.tolist()
        if not isinstance(accelero, list):
            accelero = accelero.tolist()
        feet_contacts = self.feet_contacts.get()
        kinematics = gyro + accelero + [float(c) for c in feet_contacts]
        # assert len(kinematics) == 8, kinematics
        print("imu ", time.perf_counter() - t_start)

        # send to policy
        if dof_pos is None or dof_vel is None:
            return
        print("publishing dof pos ", dof_pos)
        self.pubsub.publish(dof_pos, dof_vel, kinematics)
        
        print("step_time ", time.perf_counter() - t_start)

    def forward_targets_to_hw_from_policy(self) -> None:
        # read from policy
        q_targets = self.pubsub.read()

        if q_targets:
            print("\nQ_TARGETS: ", q_targets)
            # add zeros for head and neck joints
            q_targets = q_targets[0:5] + [0.0] * 4 + q_targets[5:10]
            self.hwi.io.write_goal_position(
                self.hw_joint_ids, q_targets
                )
        else:
            print("\nNO Q_TARGETS RECEIVED")
        
        return q_targets

if __name__ == "__main__":
    hardware_io = HardwareIO(duck_config_path="config.json")
    
    time_lst = []
    target_lst = []
    
    first_received = False
    t_start = -1.0
    dt = 0.0
    t_max = 20.0

    while True:
        hardware_io.send_observations_to_policy()
        
        q_targets = hardware_io.forward_targets_to_hw_from_policy()
        
        # if q_targets:
        #     if not first_received:
        #         first_received = True
        #         t_start = time.perf_counter()
            
        #     if dt > 5.0: # warmup time
        #         time_lst.append(dt)
        #         target_lst.append(q_targets)
                
        # if first_received:
        #     dt = time.perf_counter() - t_start
        
        # if dt > t_max:
        #     with open("targets.json", "w") as f:
        #         data = {
        #             "time": time_lst,
        #             "targets": target_lst
        #         }
        #         json.dump(data, f, indent=4)
                
        #     break

        print("loop_time ", time.perf_counter() - t_start)