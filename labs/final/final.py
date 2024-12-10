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

    fa = FinalAssist(arm_controller,object_detector, team)
    fa.start()
    blockPoses = fa.detectBlocks()
    for pose in blockPoses:
        fa.pickUp(pose)
        fa.dropOff()


   