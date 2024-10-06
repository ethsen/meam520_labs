import numpy as np 
from lib.calcJacobian import calcJacobian



def IK_velocity(q_in, v_in, omega_in):
     """
     :param q_in: 1 x 7 vector corresponding to the robot's current configuration.
     :param v_in: The desired linear velocity in the world frame. If any element is
     Nan, then that velocity can be anything
     :param omega_in: The desired angular velocity in the world frame. If any
     element is Nan, then that velocity is unconstrained i.e. it can be anything
     :return:
     dq - 1 x 7 vector corresponding to the joint velocities. If v_in and omega_in
          are infeasible, then dq should minimize the least squares error. If v_in
          and omega_in have multiple solutions, then you should select the solution
          that minimizes the l2 norm of dq
     """

     ## STUDENT CODE GOES HERE
     v_in = np.where(np.isnan(v_in), 0, v_in)
     omega_in = np.where(np.isnan(omega_in), 0, omega_in)
     v_in = v_in.reshape((3,1))
     omega_in = omega_in.reshape((3,1))

     j = np.round(calcJacobian(q_in),3)
     xi = np.vstack((v_in, omega_in))
     pinvJ = np.linalg.pinv(j)
     augJ = np.hstack(j, xi)

     
     augJ = np.hstack([j, xi])
     
     if np.linalg.matrix_rank(j) == np.linalg.matrix_rank(augJ):
          return pinvJ @ xi
     else:
          dqMinNorm, residuals, rank, s = np.linalg.lstsq(j, xi, rcond=None)
          b = np.zeros((j.shape[1], 1))  # Placeholder for null space vector
          I = np.eye(j.shape[1])

          dq_null_space = (I - np.linalg.pinv(j) @ j) @ b
          dq = dqMinNorm + dq_null_space
          return dq.flatten()

     
     

if __name__ == '__main__':
    #q= np.array([0, 0, 0, -np.pi/2, 0, np.pi/2, np.pi/4])
    q_in= np.array([0, 0, 0, 0, 0, 0, 0])
    v_in = np.array([np.nan,0,0.088])
    omega_in = np.array([0,0,1])
    print(IK_velocity(q_in,v_in,omega_in))
