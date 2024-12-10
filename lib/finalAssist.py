from lib.IK_position_null import IK
from lib.calculateFK import FK
import numpy as np
from math import pi
from scipy.spatial.transform import Rotation

class FinalAssist:
    def __init__(self,arm, detector) :
        self.ik = IK()
        self.fk =FK()
        self.arm = arm
        self.detector = detector
        self.neutralPos = np.array([-pi/8,0,0,-pi/2,0,pi/2,pi/4])
        self.dropOffPos = np.array([[1,0,0,0.56],
                                    [0,-1,0,0.15],
                                    [0,0,-1,0.24],
                                    [0,0,0,1]])
        self.neutralDropPos = np.array([[1,0,0,0.56],
                                    [0,-1,0,0.15],
                                    [0,0,-1,0.8],
                                    [0,0,0,1]])
        self.neutralDrop = self.ik.inverse(self.neutralDropPos,np.array([pi/8,0,0,-pi/2,0,pi/2,pi/4]), 'J_pseudo', 0.3)


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
        for _ in range(1):
            blocks = self.detector.get_detections()
            for id, pose in blocks:
                pose = self.adjustRotation(pose)
                world_pose = cameraToWorld @ pose
                if id not in blockDict:
                    blockDict[id] = np.zeros_like(world_pose)  # Initialize to a zero array
                blockDict[id] += world_pose

        # Compute the average pose for each block
        poses = [blockDict[id] / 1 for id in blockDict]

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

        aboveBlock = self.getJointConfig(blockPose)
        self.arm.safe_move_to_position(aboveBlock)
        pose = self.detectBlocks()[0]
        """
        if self.checkAxisofRot(pose) != 2:
            blockPose[:3,:3] = np.array([[-1,0,0],
                                        [0,1,0],
                                        [0,0,-1]])
         
            adjPos = self.getJointConfig(blockPose)
            self.arm.safe_move_to_position(adjPos)
            pose = self.detectBlocks()[0]
        
        angle = np.arccos((np.trace(pose[:3,:3]) -1)/2) #+ pi/4
        print("Old Pose: ", np.round(pose,4))
        
        pose[:3,:3] = np.array([[np.cos(angle),-np.sin(angle),0],
                                [np.sin(angle),np.cos(angle),0],
                                [0,0,1]])
        
        pose = pose @ np.array([[1,0,0,0],
                                [0,-1,0,0],
                                [0,0,-1,0],
                                [0,0,0,1]])
        """  
        #print("Updated Pose: ", np.round(pose,4))        
        
        return pose, aboveBlock
    
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

    @staticmethod
    def adjustRotation(pose):
        """
        Adjusts the pose of the detected block in order
        for the end-effector to easily grasp it. 
        
        INPUTS:
        pose - 4x4 matrix of a pose 

        OUPUTS:
        adjPose - 4x4 matrix after adjusting pose 
        """
        rotDetected= pose[:3, :3]
        tDetected = pose[:3, 3]
        for i in range(3):
            col = rotDetected[:, i]
            if np.allclose(col, [0, 0, 1], atol=1e-3):
                top_face_col = i
                flip = -1
                break
            elif np.allclose(col, [0, 0, -1], atol=1e-3):
                top_face_col = i
                flip = 1
                break
        else:
            raise ValueError("No column aligns with the top face direction [0, 0, ±1].")

        
        if top_face_col == 0:
            angle = pi/2 * flip
            rotY = np.array([[np.cos(angle),0,np.sin(angle)],
                             [0,1,0],
                             [-np.sin(angle),0,np.cos(angle)]])
            rotDetected = rotDetected @ rotY

        elif top_face_col == 1:
            angle = pi/2 * flip
            rotX= np.array([[np.cos(angle),-np.sin(angle),0]
                             [np.sin(angle),1,0],
                             [0,0,1]])
            rotDetected = rotDetected @ rotX

        elif flip == 1:
            rotDetected = rotDetected @ np.array([[1,0,0],
                                                  [0,-1,0],
                                                  [0,0,-1]])

        """
        # Construct a permutation matrix to swap columns
        R_swap = np.eye(3)
        R_swap[:, [2, top_face_col]] = R_swap[:, [top_face_col, 2]]  # Swap the third column with the top_face_col

        # If the top face points to -z, flip the orientation
        if flip:
           R_swap = R_swap @ np.array([[1,0,0,],
                                [0,-1,0,],
                                [0,0,-1,]])

        # Adjust the rotation matrix
        R_corrected = np.dot(rotDetected, R_swap)
        """
        # Construct the corrected pose
        pose_corrected = np.eye(4)
        pose_corrected[:3, :3] = rotDetected
        pose_corrected[:3, 3] = tDetected  # Keep the translation unchanged
        return pose_corrected
