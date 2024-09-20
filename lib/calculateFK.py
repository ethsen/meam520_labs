import numpy as np
import matplotlib.pyplot as plt
from math import pi

class FK():

    def __init__(self):

        # TODO: you may want to define geometric parameters here that will be
        # useful in computing the forward kinematics. The data you will need
        # is provided in the lab handout

        self.xDisplacement =[0,0,0.0825,0.0825,0,0.088,0]
        self.zDisplacement = [0.192+0.141,0,0.195+0.121,0,0.125+0.259,0,0.051+0.159]
        self.angleDisplacement = [-np.pi/2,np.pi/2,np.pi/2,np.pi/2,-np.pi/2,np.pi/2,0]
        self.jointOffsets = np.stack(([0,0,.141], [0,0,0], [0,0,.195],
                                      [0,0,0],[0,0,0.125],[0,0,-0.015],
                                      [0,0,.051],[0,0,0]),axis= 0)

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

        # Your Lab 1 code starts here

        jointPositions = np.zeros((8,3))
        T0e = np.identity(4)
        a4 = [0, 0, 0, 1]
        q[3] += np.pi
        q[5] -= np.pi
        q[6] -= np.pi/2
        for i in range(len(q)):
            angle = q[i]
            a1 = [np.cos(angle), -np.sin(angle)*np.cos(self.angleDisplacement[i]), np.sin(angle)*np.sin(self.angleDisplacement[i]), self.xDisplacement[i]*np.cos(angle)] 
            a2 = [np.sin(angle), np.cos(angle)*np.cos(self.angleDisplacement[i]), -np.cos(angle)*np.sin(self.angleDisplacement[i]), self.xDisplacement[i]*np.sin(angle)] 
            a3 = [0, np.sin(self.angleDisplacement[i]), np.cos(self.angleDisplacement[i]), self.zDisplacement[i]] 
            A = np.stack((a1,a2,a3,a4), axis = 0)
            T0e = np.matmul(T0e,A)
            if i ==1:
                print(A)
            #print(i)
            jointPositions[i+1] = T0e[:3,3]
            #self.jointOffsets[i+1] = np.matmul(T0e[:3,:3], self.jointOffsets[i+1,:])
    
        # Your code ends here
        #print("Joint Positions:\n",jointPositions)
        jointPositions += self.jointOffsets
        #print("Joint Positions:\n",jointPositions)
        return jointPositions, T0e

    # feel free to define additional helper methods to modularize your solution for lab 1

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
    
    # This code is for Lab 2, you can ignore it ofr Lab 1
    def get_axis_of_rotation(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        axis_of_rotation_list: - 3x7 np array of unit vectors describing the axis of rotation for each joint in the
                                 world frame

        """
        # STUDENT CODE HERE: This is a function needed by lab 2

        return()
    
    def compute_Ai(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        Ai: - 4x4 list of np array of homogenous transformations describing the FK of the robot. Transformations are not
              necessarily located at the joint locations
        """
        # STUDENT CODE HERE: This is a function needed by lab 2

        return()
    
if __name__ == "__main__":

    fk = FK()

    # matches figure in the handout
    for i in range(1,15):
        q = np.array([0,0,np.pi/2,-pi/4,np.pi/2,pi,pi/4])

        joint_positions, T0e = fk.forward(q)
        print(joint_positions)
        print(i)
    fk.testPlot(joint_positions)
    #print("Joint Positions:\n",joint_positions)
    #print("End Effector Pose:\n",T0e)
