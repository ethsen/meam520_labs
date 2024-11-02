import sys
from math import pi, sin, cos
import numpy as np
from time import perf_counter

import rospy
import roslib
import tf
import geometry_msgs.msg
import visualization_msgs
from tf.transformations import quaternion_from_matrix

from core.interfaces import ArmController

from lib.IK_position_null import IK
from lib.calcManipulability import calcManipulability

rospy.init_node("visualizer")

# Using your solution code
ik = IK()

# Turn on/off Manipulability Ellipsoid
visulaize_mani_ellipsoid = False

#########################
##  RViz Communication ##
#########################

tf_broad  = tf.TransformBroadcaster()
ellipsoid_pub = rospy.Publisher('/vis/ellip', visualization_msgs.msg.Marker, queue_size=10)

# Broadcasts a frame using the transform from given frame to world frame
def show_pose(H,frame):
    tf_broad.sendTransform(
        tf.transformations.translation_from_matrix(H),
        tf.transformations.quaternion_from_matrix(H),
        rospy.Time.now(),
        frame,
        "world"
    )

def show_manipulability_ellipsoid(M):
    eigenvalues, eigenvectors = np.linalg.eig(M)

    marker = visualization_msgs.msg.Marker()
    marker.header.frame_id = "endeffector"
    marker.header.stamp = rospy.Time.now()
    marker.type = visualization_msgs.msg.Marker.SPHERE
    marker.action = visualization_msgs.msg.Marker.ADD

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    #axes_len = np.sqrt(eigenvalues)

    marker.scale.x = eigenvalues[0]
    marker.scale.y = eigenvalues[1]
    marker.scale.z = eigenvalues[2]

    R = np.vstack((np.hstack((eigenvectors, np.zeros((3,1)))), \
                    np.array([0.0, 0.0, 0.0, 1.0])))
    q = quaternion_from_matrix(R)
    q = q / np.linalg.norm(q)
    marker.pose.orientation.x = q[0]
    marker.pose.orientation.y = q[1]
    marker.pose.orientation.z = q[2]
    marker.pose.orientation.w = q[3]

    marker.color.a = 0.5
    marker.color.r = 0.0
    marker.color.g = 0.0
    marker.color.b = 1.0

    ellipsoid_pub.publish(marker)

#############################
##  Transformation Helpers ##
#############################

def trans(d):
    """
    Compute pure translation homogenous transformation
    """
    return np.array([
        [ 1, 0, 0, d[0] ],
        [ 0, 1, 0, d[1] ],
        [ 0, 0, 1, d[2] ],
        [ 0, 0, 0, 1    ],
    ])

def roll(a):
    """
    Compute homogenous transformation for rotation around x axis by angle a
    """
    return np.array([
        [ 1,     0,       0,  0 ],
        [ 0, cos(a), -sin(a), 0 ],
        [ 0, sin(a),  cos(a), 0 ],
        [ 0,      0,       0, 1 ],
    ])

def pitch(a):
    """
    Compute homogenous transformation for rotation around y axis by angle a
    """
    return np.array([
        [ cos(a), 0, -sin(a), 0 ],
        [      0, 1,       0, 0 ],
        [ sin(a), 0,  cos(a), 0 ],
        [ 0,      0,       0, 1 ],
    ])

def yaw(a):
    """
    Compute homogenous transformation for rotation around z axis by angle a
    """
    return np.array([
        [ cos(a), -sin(a), 0, 0 ],
        [ sin(a),  cos(a), 0, 0 ],
        [      0,       0, 1, 0 ],
        [      0,       0, 0, 1 ],
    ])

def transform(d,rpy):
    """
    Helper function to compute a homogenous transform of a translation by d and
    rotation corresponding to roll-pitch-yaw euler angles
    """
    return trans(d) @ roll(rpy[0]) @ pitch(rpy[1]) @ yaw(rpy[2])

#################
##  IK Targets ##
#################

# TODO: Try testing your own targets!

# Note: below we are using some helper functions which make it easier to generate
# valid transformation matrices from a translation vector and Euler angles, or a
# sequence of successive rotations around z, y, and x. You are free to use these
# to generate your own tests, or directly write out transforms you wish to test.
"""
targets = [
    transform( np.array([-.2, -.3, .5]), np.array([0,pi,pi])            ),
    transform( np.array([-.2, .3, .5]),  np.array([pi/6,5/6*pi,7/6*pi]) ),
    transform( np.array([.5, 0, .5]),    np.array([0,pi,pi])            ),
    transform( np.array([.7, 0, .5]),    np.array([0,pi,pi])            ),
    transform( np.array([.2, .6, 0.5]),  np.array([0,pi,pi])            ),
    transform( np.array([.2, .6, 0.5]),  np.array([0,pi,pi-pi/2])       ),
    transform( np.array([.2, -.6, 0.5]), np.array([0,pi-pi/2,pi])       ),
    transform( np.array([.2, -.6, 0.5]), np.array([pi/4,pi-pi/2,pi])    ),
    transform( np.array([.5, 0, 0.2]),   np.array([0,pi-pi/2,pi])       ),
    transform( np.array([.4, 0, 0.2]),   np.array([pi/2,pi-pi/2,pi])    ),
]
"""
targets = [
    transform( np.array([.1, .2, .5]), np.array([0,pi,pi])            ),
    transform( np.array([.2, -.3, .5]),  np.array([pi/6,5/6*pi,7/6*pi]) ),
    transform( np.array([.5, 0, .5]),    np.array([0,pi,pi])            ),
    transform( np.array([.2, -.3, .5]),    np.array([0,pi,pi])            ),
    transform( np.array([.5, 0, .5]),    np.array([pi,0,pi])            )
]

simTestTargets = np.array([
    [-1.7121,  0.5267, -2.4947, -1.0443, -0.3604,  1.9798, -1.1319],
    [-1.3826,  1.4994,  1.4938, -0.4640,  1.6740,  2.4560, -0.0729],
    [ 0.6375,  1.2705,  1.2596, -1.2483, -0.5908,  2.5139, -2.1915],
    [-1.6529,  0.9070, -0.4124, -0.5422, -1.5658,  0.4518,  1.1462],
    [-1.6626,  0.5305, -1.3992, -2.5199,  2.7170,  0.2320,  0.7246],
    [-1.1988,  0.7238, -0.4017, -0.3353, -0.3793,  2.3673, -1.8063],
    [ 2.6187, -0.7386, -2.7455, -2.9360, -1.3760,  1.9037, -0.5194],
    [-2.2727,  1.7062,  2.7379, -2.1329,  0.1109,  0.6922, -1.5960],
    [-1.4313, -0.6537,  2.1048, -2.7960,  0.4988,  0.7104,  1.0473],
    [-0.0952,  1.4196, -1.7893, -0.4457, -2.6308,  1.4369,  2.7665],
    [ 1.0860, -1.2851,  0.9301, -2.6200,  0.1468,  1.5680, -0.4623],
    [-0.7082, -0.2991, -0.1736, -2.4335, -0.6911,  0.6751,  0.9907],
    [ 1.7338, -0.6660,  0.6619, -2.9930,  2.8965,  3.1692,  1.1194],
    [-1.8296, -0.3801,  0.3682, -2.9050, -2.0198,  3.7421,  0.5146],
    [-1.1312, -1.1193,  0.8083, -1.2311, -2.6133,  0.0638,  1.9411],
    [-0.1042,  0.9489,  1.6279, -1.3732,  2.8001,  1.4044, -2.0496],
    [-2.6854,  0.8650,  0.7439, -1.0431,  2.4277,  2.3908,  1.6472],
    [-1.7046,  0.1825, -2.1625, -1.8383,  2.4817,  1.2434,  1.4371],
    [ 1.1383,  1.0600, -2.7997, -2.8323,  1.4094,  0.6223, -0.5693],
    [-2.2643, -1.4824, -0.3950, -2.0862, -0.8309,  0.8589,  2.4947],
    [ 1.1949,  1.5313,  1.6968, -2.6528,  2.5777,  3.4867,  1.1606],
    [-0.2896,  0.9123,  1.3243, -1.2042, -1.2219,  0.2824, -1.7369],
    [ 0.8668, -0.8352, -0.2682, -0.1609, -0.6870,  3.4901, -2.7311],
    [ 0.0105,  1.5884, -1.7391, -2.5940, -2.8651,  3.3509, -1.3968],
    [ 1.3190, -1.0026,  2.3758, -1.3101,  1.0519,  0.3108,  1.8206]
])

####################
## Test Execution ##
####################

np.set_printoptions(suppress=True)

if __name__ == "__main__":

    arm = ArmController()
    seed = arm.neutral_position()
    arm.safe_move_to_position(seed)

    # Iterates through the given targets, using your IK solution
    # Try editing the targets list above to do more testing!
    for i, target in enumerate(targets):
        print("Target " + str(i) + " located at:")
        print(target)
        print("Solving... ")
        show_pose(target,"target")

        seed = arm.neutral_position() # use neutral configuration as seed
        #seed = np.array([0,0,0,0,pi/2,pi/4, pi/4])

        start = perf_counter()
        q, rollout, success, message = ik.inverse(target, seed, method='J_pseudo', alpha=.53)  #try both methods
        stop = perf_counter()
        dt = stop - start

        if success:
            print("Solution found in {time:2.2f} seconds ({it} iterations).".format(time=dt,it=len(rollout)))
            arm.safe_move_to_position(q)

            # Visualize
            if visulaize_mani_ellipsoid:
                mu, M = calcManipulability(q)
                show_manipulability_ellipsoid(M)
                print('Manipulability Index',mu)
        else:
            print('IK Failed for this target using this seed.')


        if i < len(targets) - 1:
            input("Press Enter to move to next target...")
