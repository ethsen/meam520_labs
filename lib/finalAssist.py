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
        self.neutralPos = np.array([-pi/8,0,0,-pi/2,0,pi/2,pi/4])
        self.neutralDrop = np.array([pi/8,0,0,-pi/2,0,pi/2,pi/4])
        self.dropOffPos = np.array([[1,0,0,0.56],
                                    [0,-1,0,0.15],
                                    [0,0,-1,0.24],
                                    [0,0,0,1]])

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
        blockDict = {}
        currT0e = self.fk.forward(self.arm.get_positions())[1]
        #print("currT0e: ", np.round(currT0e))

        cameraToWorld = currT0e @ self.detector.get_H_ee_camera()
        #print("Cam2World: ",np.round(cameraToWorld))
        for _ in range(50):
            blocks = self.detector.get_detections()
            for id, pose in blocks:
                world_pose = cameraToWorld @ pose
                if id not in blockDict:
                    blockDict[id] = np.zeros_like(world_pose)  # Initialize to a zero array
                blockDict[id] += world_pose

        # Compute the average pose for each block
        poses = {id: blockDict[id] / 50 for id in blockDict}
        print(poses.values())
        """
        #print("Pose in camera frame: ",np.round(pose,4))
        pose = cameraToWorld @ pose
        #print("Pose in world frame: ",np.round(pose,4))
        poses.append(pose)
        """
        return poses.values()
    
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
        
        
        jointConfig,_,success,message = self.ik.inverse(transformation,guess, 'J_pseudo', 0.3)

        if success:
            print(message)
            return jointConfig
        else:
            print(message)
            return self.neutralPos

    def pickUp(self, blockPose):
        """
        Pickup function for static blocks. The arm
        positions itself above the block and then lowers down
        to pick it up.

        INPUTS:
        blockPose - 4x4 pose of block in world frame

        OUTPUTS:
        success - Boolean representing if pickup was successful or not
        """
        self.arm.open_gripper()

        blockPose, bestGuess = self.approach(blockPose)
        
        jointConfig = self.getJointConfig(blockPose, bestGuess)
        bestGuess[4:] = jointConfig[4:]
        self.arm.safe_move_to_position(bestGuess)
        print("Picking up block...")
        self.arm.safe_move_to_position(jointConfig)
        self.arm.exec_gripper_cmd(0.03,60)
        self.arm.safe_move_to_position(bestGuess)

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
        blockPose = np.array([[1,0,0,blockPose[0,3]-0.025],
                                [0,-1,0,blockPose[1,3]],
                                [0,0,-1,blockPose[2,3]+0.075],
                                [0,0,0,1]])

        jointConfig = self.getJointConfig(blockPose)
        self.arm.safe_move_to_position(jointConfig)
        pose = self.detectBlocks()[0]
        angle = np.arccos((np.trace(pose[:3,:3]) -1)/2) #+ pi/4
        print("Old Pose: ", np.round(pose,4))

        pose[:3,:3] = np.array([[np.cos(angle),-np.sin(angle),0],
                                [np.sin(angle),np.cos(angle),0],
                                [0,0,1]])

        pose = pose @ np.array([[-1,0,0,0],
                                [0,1,0,0],
                                [0,0,-1,0],
                                [0,0,0,1]])
        print("Updated Pose: ", np.round(pose,4))        
        return pose, jointConfig
    
    def dropOff(self):
        """
        Drop off function. The arm first places itself above
        the drop off point and then lowers the block to its
        final position.
        """
        drop = self.ik.inverse(self.dropOffPos,self.neutralDrop, 'J_pseudo', 0.3)[0]
        self.arm.safe_move_to_position(self.neutralDrop)
        self.arm.safe_move_to_position(drop)
        self.arm.open_gripper()
        self.arm.safe_move_to_position(self.neutralDrop)
        self.dropOffPos[2,3] += 0.05


        

"""

Position camera over the pick up area, detect blocks,
Transform their orientation and center position into the world
frame, pick up, place, reset over the neutral position.
"""