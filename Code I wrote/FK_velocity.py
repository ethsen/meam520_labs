import numpy as np 
from lib.calcJacobian import calcJacobian

def FK_velocity(q_in, dq):
    """
    :param q_in: 1 x 7 vector corresponding to the robot's current configuration.
    :param dq: 1 x 7 vector corresponding to the joint velocities.
    :return:
    velocity - 6 x 1 vector corresponding to the end effector velocities.    
    """

    j = calcJacobian(q_in)
    print(np.linalg.det(j))

    velo = j @ dq
    return velo

if __name__ == '__main__':
    q= np.array([0, 0, 0, 0, 0, 0, 0])
    dq = np.array([10,0,0,0,0,0,0])
    print(np.round(FK_velocity(q,dq),3))
