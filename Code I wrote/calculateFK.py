import numpy as np
import matplotlib.pyplot as plt
from math import pi

class FK():

    def __init__(self):

        self.xDisplacement =[0,0,0.0825,-0.0825,0,0.088,0]
        self.zDisplacement = [0.333,0,0.316,0,.384,0,0.21]
        self.angleDisplacement = [-pi/2,pi/2,pi/2,-pi/2,pi/2,pi/2,0]
        self.jointOffsets = np.stack(([0,0,.141,1], [0,0,0,1], [0,0,.195,1],
                                      [0,0,0,1],[0,0,0.125,1],[0,0,-.015,1],
                                      [0,0,.051,1],[0,0,0,1]),axis= 0)
    def forward(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        jointPositions -8 x 3 matrix, where each row corresponds to a rotational joint of the robot or end effector
                  Each row contains the [x,y,z] coordinates in the world frame of the respective joint's center in meters.
                  The base of the robot is located at [0,0,0].
        T0e       - a 4 x 4 homogeneous transformation matrix,
                  representing the end effector frame expressed in the
                  world frame
        """

        jointPositions = np.zeros((8,3))
        jointPositions[0,:] = [0,0,.141]
        T0e = np.identity(4)
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
            jointPositions[i+1,:] = (T0e @ self.jointOffsets[i+1,:])[:3]
        return np.round(jointPositions,decimals =6), T0e


    def testPlot(self,jointPositions):
        ax = plt.figure().add_subplot(projection='3d')
        ax.plot(jointPositions[:,0], jointPositions[:,1], jointPositions[:,2])
        ax.scatter(jointPositions[:,0], jointPositions[:,1], jointPositions[:,2], c= 'red')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(0, 2)
        plt.show()
    
    def get_axis_of_rotation(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        axis_of_rotation_list: - 3x7 np array of unit vectors describing the axis of rotation for each joint in the
                                 world frame

        """
        # STUDENT CODE HERE: This is a function needed by lab 2
        T0e = np.identity(4)
        a4 = [0, 0, 0, 1]
        self.axisOfRot = [np.array([0,0,1])]

        for i in range(len(q)-1):
            angle = q[i]
            if i ==6:
                angle -= pi/4
            
            a1 = [np.cos(angle), -np.sin(angle)*np.cos(self.angleDisplacement[i]), np.sin(angle)*np.sin(self.angleDisplacement[i]), self.xDisplacement[i]*np.cos(angle)] 
            a2 = [np.sin(angle), np.cos(angle)*np.cos(self.angleDisplacement[i]), -np.cos(angle)*np.sin(self.angleDisplacement[i]), self.xDisplacement[i]*np.sin(angle)] 
            a3 = [0, np.sin(self.angleDisplacement[i]), np.cos(self.angleDisplacement[i]), self.zDisplacement[i]] 
            A = np.array([a1,a2,a3,a4])
            T0e = T0e @ A
            self.axisOfRot += [T0e[:3,2]]

        return np.array(self.axisOfRot)
    
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
        
        return np.array(transformationList)
    
if __name__ == "__main__":

    fk = FK()

    # matches figure in the handout
    
    q = np.array([0,0,0,-pi/2,0,pi/2,pi/4])
    #q = np.array([0,0,0,0,0,0,0])
    joint_positions, T0e = fk.forward(q)
    #fk.testPlot(joint_positions)
    #print(fk.get_axis_of_rotation(q))
    #print(fk.compute_Ai(q))
    print(np.round(T0e,4))
    print("Joint Positions:\n",joint_positions)
    #print("End Effector Pose:\n",T0e)
