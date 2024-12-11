from lib.IK_position_null import IK
from lib.calculateFK import FK
import numpy as np
from math import pi

class FinalAssist:
    def __init__(self,arm, detector,team) :
        self.team = team
        self.ik = IK()
        self.fk =FK()
        self.arm = arm
        self.detector = detector

    def start(self):
        """
        Sets the arm in the neutral position
        """
        if self.team == 'blue':
            self.neutralPos = np.array([pi/10,0,0,-pi/2,0,pi/2,pi/4])
            self.dropOffPos = np.array([[1,0,0,0.56],
                                        [0,-1,0,-0.15],
                                        [0,0,-1,0.24],
                                        [0,0,0,1]])
            self.neutralDrop = np.array([-0.1286, 0.07215, -0.13995, -1.53771, 0.01006, 1.60915, 0.51682])

        else:
            self.neutralPos = np.array([-pi/10,0,0,-pi/2,0,pi/2,pi/4])
            self.dropOffPos = np.array([[1,0,0,0.56],
                                        [0,-1,0,0.15],
                                        [0,0,-1,0.24],
                                        [0,0,0,1]])
            self.neutralDrop = np.array([0.15668, 0.07189, 0.11041,-1.53771, -0.00792, 1.60917, 1.05251])

        self.arm.safe_move_to_position(self.neutralPos)
        self.goStatic()

    def goStatic(self):
        """
        Start function for static block pick ups
        """

        blockPoses = self.detectBlocks(1)
        for id in blockPoses:
            pose = blockPoses[id]
            self.pickUp(id,pose)
            self.dropOff(id)
    
    def detectBlocks(self, iters):
        """
        Block detection in order to find and transform
        block's position into world frame.

        INPUTS:
        iters - amount of scans desired

        OUTPUTS:
        poses - Array of poses for each block in world frame
        """
        blockDict = {}
        currT0e = self.fk.forward(self.arm.get_positions())[1]
        #print("currT0e: ", np.round(currT0e))

        cameraToWorld = currT0e @ self.detector.get_H_ee_camera()
        #print("Cam2World: ",np.round(cameraToWorld))
        for _ in range(iters):
            blocks = self.detector.get_detections()
            for id, pose in blocks:
                pose = self.adjustRotation(pose, cameraToWorld)
                world_pose = cameraToWorld @ pose
                if id not in blockDict:
                    blockDict[id] = np.zeros_like(world_pose)  # Initialize to a zero array
                blockDict[id] += world_pose

        for id in blockDict:
            blockDict[id] /= iters
        
        return blockDict

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
            return jointConfig, success
        else:
            print(message)
            return self.neutralPos, success

    def pickUp(self,id, blockPose):
        """
        Pickup function for static blocks. The arm
        positions itself above the block and then lowers down
        to pick it up.

        INPUTS:
        id -  Block id to ensure correct block picked up
        blockPose - 4x4 pose of block in world frame

        OUTPUTS:
        success - Boolean representing if pickup was successful or not
        """
        self.arm.open_gripper()

        blockPose, bestGuess = self.approach(id,blockPose)

        jointConfig,_ = self.getJointConfig(blockPose, bestGuess)
        bestGuess[4:] = jointConfig[4:]
        self.arm.safe_move_to_position(bestGuess)
        print("Picking up block...")
        self.arm.safe_move_to_position(jointConfig)
        self.arm.exec_gripper_cmd(0.03,60)
        self.arm.safe_move_to_position(bestGuess)

    def approach(self,id, blockPose):
        """
        Approach a blcok and rescan the fov to get
        clearer AprilTag detection

        INPUTS:
        id -  Block id to ensure correct block picked up
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

        aboveBlock,_ = self.getJointConfig(blockPose)
        self.arm.safe_move_to_position(aboveBlock)
        pose = self.detectBlocks(1)

        return pose[id], aboveBlock

    def dropOff(self, id):
        """
        Drop off function. The arm first places itself above
        the drop off point and then lowers the block to its
        final position.

        INPUTS:
        id - block id
        """
        drop = self.ik.inverse(self.dropOffPos,self.neutralDrop, 'J_pseudo', 0.3)[0]
        self.arm.safe_move_to_position(self.neutralDrop)
        self.arm.safe_move_to_position(drop)
        self.arm.open_gripper()
        self.arm.safe_move_to_position(self.neutralDrop)
        print(self.checkDrop(id))
        self.dropOffPos[2,3] += 0.05
        self.arm.safe_move_to_position(self.neutralPos)

    def checkDrop(self, id):
        """
        Checks whether successful drop has been made

        INPUTS:
        id - block id

        OUTPUTS:
        success - boolean whether a successful drop was
        made
        """
        blocksDetected = self.detectBlocks(1)

        if id in blocksDetected:
            return True
        else:
            return False

    def adjustRotation(self,pose, cameraToWorld):
        """
        Adjusts the pose of the detected block in order
        for the end-effector to correctly grasp each block.

        INPUTS:
        pose - 4x4 matrix of a pose

        OUPUTS:
        adjPose - 4x4 matrix after adjusting pose
        """
        print("Adjusting Rotation")
        rotDetected= pose[:3, :3]
        tDetected = pose[:3, 3]
        for i in range(3):
            col = np.round(rotDetected[:, i],0)
            if np.allclose(col, [0, 0, 1], atol=1e-3):
                top_face_col = i
                flip = 1
                break
            elif np.allclose(col, [0, 0, -1], atol=1e-3):
                top_face_col = i
                flip = -1
                break
        else:
            raise ValueError("No column aligns with the top face directon [0, 0, ±1].")

        if top_face_col == 0:
            angle = pi/2 * flip
            rotY = np.array([[np.cos(angle),0,np.sin(angle)],
                             [0,1,0],
                             [-np.sin(angle),0,np.cos(angle)]])
            rotDetected = rotDetected @ rotY

        elif top_face_col == 1:
            angle = pi/2 * -flip
            rotX = np.array([[1,0,0],
                             [0,np.cos(angle),-np.sin(angle)],
                             [0,np.sin(angle),np.cos(angle)]])
            rotDetected = rotDetected @ rotX
            rotDetected = rotDetected @ np.array([[-1,0,0],
                                                  [0,-1,0],
                                                  [0,0,1]])
        elif flip == -1:
            rotDetected = rotDetected @ np.array([[1,0,0],
                                                  [0,-1,0],
                                                  [0,0,-1]])

        pose_corrected = np.eye(4)
        pose_corrected[:3, :3] = rotDetected
        pose_corrected[:3, 3] = tDetected
        print(top_face_col)
        print(flip)
        print("fucked:", np.round(rotDetected,4))
        #q,_,success,message = self.ik.inverse(cameraToWorld @ pose_corrected, self.neutralDrop, 'J_pseudo', 0.3)
        _, success = self.getJointConfig(cameraToWorld @ pose_corrected,self.neutralDrop)
        print(success)
        while not success:
            print("Init:", np.round(rotDetected,4))
            rotDetected = rotDetected @ np.array([[0,-1,0],
                                                [1,0,0],
                                                [0,0,1]])
            print("Fixed:", np.round(rotDetected,4))
            pose_corrected[:3, :3] = rotDetected
            #success = self.ik.inverse(cameraToWorld @pose_corrected, self.neutralPos, 'J_pseudo', 0.3)[2]
            _, success = self.getJointConfig(cameraToWorld @ pose_corrected,self.neutralDrop)
            print(success)

            
        return pose_corrected
        #print(top_face_col)
        #print(flip)
        #print("FInal:", np.round(rotDetected,4))
