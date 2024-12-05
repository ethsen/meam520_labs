from lib.IK_position_null import IK
from lib.calculateFK import FK
import numpy as np
from math import pi

class FinalAssist:
    def __init__(self) :
        self.dropT = np.array([[1,0,0,0.56],
                                   [0,1,0,0.15],
                                   [0,0,0,0.24],
                                   [0,0,0,1]])
    @staticmethod
    def start(arm):
        """
        Sets the arm in the neutral position
        """
        neutralPos = np.array([0,0,0,-pi/2,0,pi/2,pi/4])
        arm.safe_move_to_position(neutralPos)

    @staticmethod
    def detectBlocks(arm, detector):
        """
        Block detection in order to find and transform 
        block's position into world frame.

        INPUTS:
        arm - Arm controller object
        detector - Detector object

        OUTPUTS:
        poses - Array of poses for each block in world frame
        """

        blocks = detector.get_detections()
        print(arm.get_positions())
        currT0e = FK.forward(arm.get_positions())[1]
        cameraToWorld = currT0e @ detector.get_H_ee_camera()
        return [cameraToWorld @ pose for _,pose in blocks]
    
    @staticmethod
    def getJointConfig(transformation):
        """
        Uses IK class to find and return the joint configuration
        for block pose

        INPUTS:
        transformation - 4x4 transformation matrix of a desired 
        position in the world frame

        OUTPUTS:
        jointConfig - 1x7 array of the joint configurations
        """
        neutralPos = np.array([0,0,0,-pi/2,0,pi/2,pi/4])
        
        
        jointConfig,_,success,_ =   IK.inverse(transformation,neutralPos, 'J_pseudo', 0.3)

        if success:
            return jointConfig
        else:
            return neutralPos

    def getDropoffPos(self):
        """
        Gets the dropoff position based on previous
        dropoffs
        
        INPUTS:

        OUTPUTS:
        jointConfig - 1x7 array of the joint configurations
        """
        neutralPos = np.array([0,0,0,-pi/2,0,pi/2,pi/4])
        jointConfig,_,success,_ =   IK.inverse(self.dropT,neutralPos, 'J_pseudo', 0.3)

        if success:
            return jointConfig
        else:
            return neutralPos

    @staticmethod
    def pickUp(arm, blockPose):
        """
        Pickup function for static blocks. The arm
        positions itself above the block and then lowers down
        to pick it up.

        INPUTS:
        arm - Arm controller object
        blockPose - 4x4 pose of block in world frame

        OUTPUTS:
        success - Boolean representing if pickup was successful or not
        """
        neutralPos = np.array([0,0,0,-pi/2,0,pi/2,pi/4])
        arm.open_gripper()
        overBlockPose = np.copy(blockPose)
        overBlockPose[2,3] += 0.225
        jointConfig = FinalAssist.getJointPos(overBlockPose)
        arm.safe_move_to_position(jointConfig)

        jointConfig = FinalAssist.getJointPos(blockPose)
        arm.safe_move_to_position(jointConfig)
        arm.exec_gripper_cmd(0.03,60)

        arm.safe_move_to_position(neutralPos)
        