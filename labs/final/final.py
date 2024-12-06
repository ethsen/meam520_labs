import numpy as np
from copy import deepcopy
from math import pi
import rospy
from core.interfaces import ArmController, ObjectDetector
from lib.IK_position_null import IK
from lib.calculateFK import FK
from lib.finalAssist import FinalAssist
if __name__ == "__main__":
    try:
        team = rospy.get_param("team")  # 'red' or 'blue'
    except KeyError:
        print('Team must be red or blue - make sure you are running final.launch!')
        exit()

    rospy.init_node("team_script")
    arm_controller = ArmController()
    object_detector = ObjectDetector()
    initial_arm_position = np.array([0, 0, 0, -3.1415 / 2, 0, 3.1415 / 2, 3.1415 / 4])
    arm_controller.safe_move_to_position(initial_arm_position)  # On your mark!

    print("\n****************")
    if team == 'blue':
        print("** BLUE TEAM  **")
    else:
        print("**  RED TEAM  **")
    print("****************")
    input("\nWaiting for start... Press ENTER to begin!\n")  # Get set!
    print("Go!\n")  # Go!

    fa = FinalAssist(arm_controller,object_detector)
    fa.start()
    blockPoses = fa.detectBlocks()
    for pose in blockPoses:
        fa.pickUp(pose)


    """
    for block_pose in block_poses:
        arm_controller.open_gripper()
        block_pose[2,3] += 0.225
        joint_positions, _, success, message = inverse_kinematics.inverse(block_pose, arm_controller.get_positions(), "J_pseudo", 0.5)
        if not success:
            print(f"Failed to compute IK for block pose: {message}")
            continue
        arm_controller.safe_move_to_position(joint_positions)
        block_pose[2,3] -= 0.225
        joint_positions_lower, _, success, message = inverse_kinematics.inverse(block_pose, joint_positions, "J_pseudo", 0.5)
        if not success:
            print(f"Failed to compute IK for lower block pose: {message}")
            continue

        arm_controller.safe_move_to_position(joint_positions_lower)

        arm_controller.exec_gripper_cmd(0.03, 60)

        dropoff_transform = np.eye(4)
        dropoff_transform[:3, 3] = dropoff_position
        dropoff_transform[:3, 0] = np.array([1, 0, 0])
        dropoff_transform[:3, 1] = np.array([0, 1, 0])
        dropoff_transform[:3, 2] = np.array([0, 0, 1])

        dropoff_joint_positions, _, success, message = inverse_kinematics.inverse(
            dropoff_transform, arm_controller.get_positions(), "J_pseudo", 0.5
        )
        if not success:
            print("Failed to compute IK for drop-off position:", message)

        print("Moving to drop-off position")
        arm_controller.safe_move_to_position(
            np.array([0.35, 0, 0, -3.1415 / 2, 0, 3.1415 / 2, 3.1415 / 4])
        )
        print("Drop-off position reached")
        arm_controller.open_gripper()
    """