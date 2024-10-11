import numpy as np


def calcAngDiff(R_des, R_curr):
    """
    Helper function for the End Effector Orientation Task. Computes the axis of rotation 
    from the current orientation to the target orientation

    This data can also be interpreted as an end effector velocity which will
    bring the end effector closer to the target orientation.

    INPUTS:
    R_des - 3x3 numpy array representing the desired orientation from
    end effector to world
    R_curr - 3x3 numpy array representing the "current" end effector orientation

    OUTPUTS:
    omega - 0x3 a 3-element numpy array containing the axis of the rotation from
    the current frame to the end effector frame. The magnitude of this vector
    must be sin(angle), where angle is the angle of rotation around this axis
    """
    ## STUDENT CODE STARTS HERE

    relativeR = R_curr.T @ R_des
    skewSymR = (1/2) * (relativeR - relativeR.T)

    omega= np.array([skewSymR[2,1], skewSymR[0,2], skewSymR[1,0]])
    

    #theta = np.arccos((relativeR[0,0] + relativeR[1,1] + relativeR[2,2] - 1)/2)
    #omega = (1/ 2* np.sin(theta)) * np.array([relativeR[2,1] -relativeR[1,2], relativeR[0,2] -relativeR[2,0], relativeR[1,0] -relativeR[0,1]])
    omega = R_curr @ omega

    return omega


if __name__ == "__main__":

    R_curr_2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    R_des_2 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    print(calcAngDiff(R_des_2, R_curr_2))