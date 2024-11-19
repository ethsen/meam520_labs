import numpy as np
from math import pi
import matplotlib as plt
from potentialFieldTester import * 


class FK_Jac():

    def __init__(self):

        # TODO: you may want to define geometric parameters here that will be
        # useful in computing the forward kinematics. The data you will need
        # is provided in the lab 1 and 4 handout
        self.xDisplacement =[0,0,0.0825,-0.0825,0,0.088,0]
        self.zDisplacement = [0.333,0,0.316,0,.384,0,0.21] 
        self.angleDisplacement = [-pi/2,pi/2,pi/2,-pi/2,pi/2,pi/2,0]
        self.jointOffsets = np.stack(([0,0,.141,1], [0,0,0,1], [0,0,.195,1],
                                      [0,0,0,1],[0,0,0.125,1],[0,0,-.015,1],
                                      [0,0,.051,1],[0,0,0,1]),axis= 0)
        

    def forward_expanded(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        jointPositions -10 x 3 matrix, where each row corresponds to a physical or virtual joint of the robot or end effector
                  Each row contains the [x,y,z] coordinates in the world frame of the respective joint's center in meters.
                  The base of the robot is located at [0,0,0].
        T0e       - a 10 x 4 x 4 homogeneous transformation matrix,
                  representing the each joint/end effector frame expressed in the
                  world frame
        """

        # Your code starts here
        jointPositions = np.zeros((10,3))
        jointPositions[0,:] = [0,0,.141]
        T0eCol = []
        T0e = np.identity(4)
        T0eCol.append(T0e)
        a4 = [0, 0, 0, 1]
        
        for i in range(len(q)):
            angle = q[i]
            if i ==6:
                angle -= pi/4
            
            a1 = [np.cos(angle), -np.sin(angle)*np.cos(self.angleDisplacement[i]), np.sin(angle)*np.sin(self.angleDisplacement[i]), self.xDisplacement[i]*np.cos(angle)] 
            a2 = [np.sin(angle), np.cos(angle)*np.cos(self.angleDisplacement[i]), -np.cos(angle)*np.sin(self.angleDisplacement[i]), self.xDisplacement[i]*np.sin(angle)] 
            a3 = [0, np.sin(self.angleDisplacement[i]), np.cos(self.angleDisplacement[i]), self.zDisplacement[i]] 
            A = np.array([a1,a2,a3,a4])
            T0e = T0e @ A
            T0eCol.append(T0e)
            jointPositions[i+1,:] = (T0e @ self.jointOffsets[i+1,:])[:3]

        vJoint1T = T0e @ np.array([[1,0,0,0],
                                   [0,1,0,0.1],
                                   [0,0,1,-.105],
                                   [0,0,0,1]])

        vJoint2T = T0e @ np.array([[1,0,0,0],
                                   [0,1,0,-0.1],
                                   [0,0,1,-.105],
                                   [0,0,0,1]])
        #T0eCol.insert(len(T0eCol)-3, vJoint1T)
        #T0eCol.insert(len(T0eCol)-2, vJoint1T)
        #jointPositions[-1] = jointPositions[len(jointPositions)-3]
        #jointPositions[len(jointPositions)-3] = vJoint1T[:3,3]
        #jointPositions[len(jointPositions)-2] = vJoint2T[:3,3]
        T0eCol.append(vJoint1T)
        T0eCol.append(vJoint2T)
        jointPositions[len(jointPositions)-2,: ] = vJoint1T[:3,3]
        jointPositions[-1,:] = vJoint2T[:3,3]


        # Your code ends here

        return jointPositions, np.array(T0eCol)

    # feel free to define additional helper methods to modularize your solution for lab 1

    
    def compute_Ai(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        Ai: - 4x4 list of np array of homogenous transformations describing the FK of the robot. Transformations are not
              necessarily located at the joint locations
        """
        
        a4 = [0, 0, 0, 1]
        transformationList = []
        A = np.identity(4)
        for i in range(len(q)):
            angle = q[i]
            if i ==6:
                angle -= pi/4
            a1 = [np.cos(angle), -np.sin(angle)*np.cos(self.angleDisplacement[i]), np.sin(angle)*np.sin(self.angleDisplacement[i]), self.xDisplacement[i]*np.cos(angle)] 
            a2 = [np.sin(angle), np.cos(angle)*np.cos(self.angleDisplacement[i]), -np.cos(angle)*np.sin(self.angleDisplacement[i]), self.xDisplacement[i]*np.sin(angle)] 
            a3 = [0, np.sin(self.angleDisplacement[i]), np.cos(self.angleDisplacement[i]), self.zDisplacement[i]] 
            A = A @ np.array([a1,a2,a3,a4])
            transformationList.append(A)

        vJoint1T = A @ np.array([[1,0,0,0],
                                   [0,1,0,0.1],
                                   [0,0,1,-.105],
                                   [0,0,0,1]])

        vJoint2T = A @ np.array([[1,0,0,0],
                                   [0,1,0,-0.1],
                                   [0,0,1,-.105],
                                   [0,0,0,1]])
        transformationList.append(vJoint1T)
        transformationList.append(vJoint2T)
        
        return np.array(transformationList)

    def calcLinJacobian(self, q,i):


        jointPos, T0eCol = self.forward_expanded(q)
        jv = np.zeros((3,9))
        j = 0
        while j != i:
            jw = T0eCol[j][:3,2]
            originDiff = jointPos[i] - jointPos[j]
            jv[:,j]= np.cross(jw,originDiff).flatten()
            j+=1
            #print(np.round(jv,4))
        return jv
    
    
if __name__ == "__main__":

    fk = FK_Jac()
    
    # matches figure in the handout
    q = np.array([0,0,0,-pi/2,0,pi/2,pi/4])
    #q = np.array([0,0,0,0,0,0,0])
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    jointPos, T0eCol =  fk.forward_expanded(q)

    for i in range(9):
        jv = fk.calcLinJacobian(q,i+1)
        print(np.round(jv,3))
        plotJacobianCalculation(ax, jointPos, T0eCol,i+1)
    #print("Joint Positions:\n",joint_positions)
    #print("End Effector Pose:\n",T0e)
    #print(np.round(joint_positions,4))
    #print(np.round(fk.calcLinJacobian(q,8),4))