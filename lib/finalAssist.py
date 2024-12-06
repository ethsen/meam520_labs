from lib.IK_position_null import IK
from lib.calculateFK import FK
import numpy as np
from math import pi

class FinalAssist:
    def __init__(self) :
        self.ik = IK()
        self.fk =FK()
        self.dropT = np.array([[1,0,0,0.56],
                                   [0,1,0,0.15],
                                   [0,0,0,0.24],
                                   [0,0,0,1]])
        self.neutralPos = np.array([-pi/8,0,0,-pi/2,0,pi/2,pi/4])
    def start(self,arm):
        """
        Sets the arm in the neutral position
        """
        arm.safe_move_to_position(self.neutralPos)

    def detectBlocks(self, arm, detector):
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
        currT0e = self.fk.forward(arm.get_positions())[1]
        cameraToWorld = currT0e @ detector.get_H_ee_camera()
        return [cameraToWorld @ pose for _,pose in blocks]
    
    def getJointConfig(self,transformation):
        """
        Uses IK class to find and return the joint configuration
        for block pose

        INPUTS:
        transformation - 4x4 transformation matrix of a desired 
        position in the world frame

        OUTPUTS:
        jointConfig - 1x7 array of the joint configurations
        """
        
        
        jointConfig,_,success,_ = self.ik.inverse(transformation,self.neutralPos, 'J_pseudo', 0.3)

        if success:
            return jointConfig
        else:
            return self.neutralPos

    def getDropoffPos(self):
        """
        Gets the dropoff position based on previous
        dropoffs
        
        INPUTS:

        OUTPUTS:
        jointConfig - 1x7 array of the joint configurations
        """
        jointConfig,_,success,_ =   self.ik.inverse(self.dropT,self.neutralPos, 'J_pseudo', 0.3)

        if success:
            return jointConfig
        else:
            return self.neutralPos

    def pickUp(self, arm, blockPose):
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
        print(np.round(blockPose,4))
        blockPose = np.array([[1,0,0,blockPose[0,3]],
                                [0,-1,0,blockPose[1,3]],
                                [0,0,-1,blockPose[2,3]],
                                [0,0,0,1]])
        arm.open_gripper()
        overBlockPose = np.copy(blockPose)
        overBlockPose[2,3] += 0.225
        jointConfig = self.getJointConfig(overBlockPose)
        arm.safe_move_to_position(jointConfig)

        jointConfig = self.getJointConfig(blockPose)
        arm.safe_move_to_position(jointConfig)
        arm.exec_gripper_cmd(0.03,60)

        arm.safe_move_to_position(self.neutralPos)
        

"""
Position camera over the pick up area, detect blocks,
Transform their orientation and center position into the world
frame, pick up, place, reset over the neutral position.
"""