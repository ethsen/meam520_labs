from lib.IK_position_null import IK
from lib.calculateFK import FK
import numpy as np
from math import pi

class FinalAssist:
    def __init__(self,arm, detector) :
        self.ik = IK()
        self.fk =FK()
        self.arm = arm
        self.detector = detector
        self.dropT = np.array([[1,0,0,0.56],
                                   [0,1,0,0.15],
                                   [0,0,-1,0.24],
                                   [0,0,0,1]])
        self.neutralPos = np.array([-pi/8,0,0,-pi/2,0,pi/2,pi/4])

    def start(self):
        """
        Sets the arm in the neutral position
        """
        self.arm.safe_move_to_position(self.neutralPos)

    def detectBlocks(self):
        """
        Block detection in order to find and transform 
        block's position into world frame.

        INPUTS:
        arm - Arm controller object
        detector - Detector object

        OUTPUTS:
        poses - Array of poses for each block in world frame
        """
        invalidBlocks = True
        currT0e = self.fk.forward(self.arm.get_positions())[1]
        #print("currT0e: ", np.round(currT0e))

        cameraToWorld = currT0e @ self.detector.get_H_ee_camera()
        #print("Cam2World: ",np.round(cameraToWorld))
        while invalidBlocks:
            blocks = self.detector.get_detections()

            poses = []
            for _,pose in blocks:
                #print("Pose: ",np.round(pose,4))
                #input("wait")
                orientation = pose[:3,:3] * -1
                pose = cameraToWorld @ pose
                pose[:3,:3] = orientation
                #print("Pose: ",np.round(pose,4))
                #input("wait")
                poses.append(pose)

            invalidBlocks = False
        return poses
    
    def getJointConfig(self,transformation, guess = np.array([-pi/8,0,0,-pi/2,0,pi/2,pi/4])):
        """
        Uses IK class to find and return the joint configuration
        for block pose

        INPUTS:
        transformation - 4x4 transformation matrix of a desired 
        position in the world frame
        guess - best guess for inverse solver to use

        OUTPUTS:
        jointConfig - 1x7 array of the joint configurations
        """
        
        
        jointConfig,_,success,_ = self.ik.inverse(transformation,guess, 'J_pseudo', 0.3)

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

    def pickUp(self, blockPose):
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
        #print(np.round(blockPose,4))
        blockPose = np.array([[1,0,0,blockPose[0,3]],
                                [0,-1,0,blockPose[1,3]],
                                [0,0,-1,blockPose[2,3]],
                                [0,0,0,1]])
        self.arm.open_gripper()
        orientation, bestGuess = self.approach(blockPose)
        blockPose[:3,:3] = orientation
        blockPose[:3,2] = np.array([0,0,-1])
        print("Gripping Block: ", blockPose)
        jointConfig = self.getJointConfig(blockPose,bestGuess)
        print("Picking up block...")
        input("test")
        self.arm.safe_move_to_position(jointConfig)
        self.arm.exec_gripper_cmd(0.03,60)

        #self.arm.safe_move_to_position(self.neutralPos)

    def approach(self, blockPose):
        """
        Approach a blcok and rescan the fov to get 
        clearer AprilTag detection

        INPUTS:
        arm - Arm object
        blockPose - Pose of position right above
        block in world frame

        OUPUTS:
        orientation - 3x3 array of rotation matrix
        with respect to world frame
        jointConfig - 1x7 array of joint configuration 
        right above the block
        """
        print("Approaching Block...")
        blockPose[0,3] -= 0.025
        blockPose[2,3] += 0.075
        jointConfig = self.getJointConfig(blockPose)
        self.arm.safe_move_to_position(jointConfig)
        blocks = self.detectBlocks()
        orientation = blocks[0][:3,:3]*-1
        print("Updated Pose: ", orientation)

        return orientation, jointConfig
        

"""
Position camera over the pick up area, detect blocks,
Transform their orientation and center position into the world
frame, pick up, place, reset over the neutral position.
"""