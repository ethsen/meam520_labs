import numpy as np
from lib.IK_velocity import IK_velocity
from lib.calcJacobian import calcJacobian

#from IK_velocity import IK_velocity
#from calcJacobian import calcJacobian

"""
Lab 3
"""

def IK_velocity_null(q_in, v_in, omega_in, b):
    """
    :param q_in: 1 x 7 vector corresponding to the robot's current configuration.
    :param v_in: The desired linear velocity in the world frame. If any element is
    Nan, then that velocity can be anything
    :param omega_in: The desired angular velocity in the world frame. If any
    element is Nan, then that velocity is unconstrained i.e. it can be anything
    :param b: 7 x 1 Secondary task joint velocity vector
    :return:
    dq + null - 1 x 7 vector corresponding to the joint velocities + secondary task null velocities
    """
    v_mask = ~np.isnan(v_in)
    omega_mask = ~np.isnan(omega_in)
    j = calcJacobian(q_in)
    JMasked = j[np.hstack((v_mask, omega_mask)).flatten(), :]
    jPinv = np.linalg.pinv(JMasked)
    dq = IK_velocity(q_in, v_in, omega_in)
    b = b.reshape((7, 1))
    null = (np.eye(7) -  (jPinv @ JMasked))  @ b

    return dq + null.flatten()


if __name__ == '__main__':
    #q= np.array([0, 0, 0, -np.pi/2, 0, np.pi/2, np.pi/4])
    q_in= np.array([-2.7, 0, 0, 0, 0, 0, 0])
    v_in = np.array([np.nan,2,0.088])
    omega_in = np.array([-35,0,1])
    b = np.array([45,125,23,1,2,0,1])
    #print(IK_velocity_null(q_in,v_in,omega_in,b))
    IK_velocity_null(q_in,v_in,omega_in,b)
