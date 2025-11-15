"""
Get the pid motor limits by moving each joint.
Verifies each motor is accessible and allows testing movement.
"""

from rustypot_position_hwi import HWI
from duck_config import DuckConfig
import time
import traceback

def main():
    # ⚠️ WARN: max motor stiffness
    kps = 30.0
    kds = 0.0
        
    print("Initializing hardware interface...")
    try:
        # Initialize with default USB port - you might need to modify this
        print("Attempting to connect to motor controller...")
        duck_config = DuckConfig()  # Create default configuration
        
        # Initialize with duck_config
        print("Attempting to connect to motor controller...")
        hwi = HWI(duck_config=duck_config)
        print("Successfully connected to hardware!")
    except Exception as e:
        print(f"Error connecting to hardware: {e}")
        print(f"Error details: {traceback.format_exc()}")
        print("Check that the robot is powered on and USB connection is correct.")
        return

    # Turn on with low torque for safety - ONE BY ONE
    print("\nTurning on motors with low torque (one by one)...")
    unresponsive_motors = []
    
    for joint_name, joint_id in hwi.joints.items():
        try:
            print(f"Setting low torque for motor '{joint_name}' (ID: {joint_id})...")
            hwi.io.set_kps([joint_id], [kps])
            hwi.io.set_kds([joint_id], [kds])
            print(f"✓ Low torque set successfully for motor '{joint_name}' (ID: {joint_id}).")
        except Exception as e:
            print(f"✗ Error setting low torque for motor '{joint_name}' (ID: {joint_id}): {e}")
            print(f"Error details: {traceback.format_exc()}")
            unresponsive_motors.append((joint_name, joint_id))
    
    # Check if all motors are responsive
    print("\nChecking if all motors are responsive...")
    
    for joint_name, joint_id in hwi.joints.items():
        # Skip motors that already failed
        if (joint_name, joint_id) in unresponsive_motors:
            print(f"Skipping previously unresponsive motor: '{joint_name}' (ID: {joint_id})")
            continue
            
        print(f"Attempting to read position from motor '{joint_name}' (ID: {joint_id})...")
        try:
            # Try to read the position to check if motor is responsive
            position = hwi.io.read_present_position([joint_id])
            print(f"✓ Motor '{joint_name}' (ID: {joint_id}) is responsive. Position: {position[0]:.3f}")
        except Exception as e:
            print(f"✗ Error accessing motor '{joint_name}' (ID: {joint_id}): {e}")
            print(f"Error details for motor {joint_id}: {traceback.format_exc()}")
            unresponsive_motors.append((joint_name, joint_id))
    
    if unresponsive_motors:
        print("\nWARNING: Some motors are not responsive!")
        print("Unresponsive motors:", unresponsive_motors)
        continue_anyway = input("Do you want to continue anyway? (y/n): ").lower()
        if continue_anyway != 'y':
            print("Exiting...")
            try:
                print("Attempting to turn off responsive motors before exiting...")
                for joint_name, joint_id in hwi.joints.items():
                    if (joint_name, joint_id) not in unresponsive_motors:
                        try:
                            hwi.io.disable_torque([joint_id])
                            print(f"Disabled torque for motor '{joint_name}' (ID: {joint_id})")
                        except:
                            pass
            except:
                pass
            return
    
    print("\n--- Joint limit test ---")
    input("Press Enter to begin...")
    def send_to_zero():
        for joint_name, joint_id in hwi.joints.items():
            hwi.io.write_goal_position([joint_id], [0.0])
        time.sleep(2)
    send_to_zero()
    limits = {}
    for joint_name, joint_id in hwi.joints.items():
        if "neck" in joint_name or "head" in joint_name or "antenna" in joint_name:
            continue
        
        # if "pitch" not in joint_name:
        #     continue
        
        limits[joint_name] = {}
        
        hwi.io.write_goal_position([joint_id], [0.0])
        time.sleep(1)
        limits[joint_name]["zero"] = hwi.io.read_present_position([joint_id])[0]
        
        hwi.io.write_goal_position([joint_id], [3.0])
        time.sleep(1)
        limits[joint_name]["high"] = hwi.io.read_present_position([joint_id])[0]
        
        hwi.io.write_goal_position([joint_id], [0.0])
        time.sleep(1)
        
        hwi.io.write_goal_position([joint_id], [-3.0])
        time.sleep(1)
        limits[joint_name]["low"] = hwi.io.read_present_position([joint_id])[0]
        
        send_to_zero()
        
    print("limits ", limits)
    

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nScript interrupted by user. Attempting to turn off motors...")
        try:
            print("Initializing HWI to turn off motors...")
            hwi = HWI()
            for joint_name, joint_id in hwi.joints.items():
                try:
                    print(f"Turning off motor '{joint_name}' (ID: {joint_id})...")
                    hwi.io.disable_torque([joint_id])
                    print(f"✓ Motor '{joint_name}' (ID: {joint_id}) turned off successfully.")
                except Exception as e:
                    print(f"✗ Error turning off motor '{joint_name}' (ID: {joint_id}): {e}")
        except Exception as e:
            print(f"Error initializing HWI to turn off motors: {e}")
            print(f"Error details: {traceback.format_exc()}")