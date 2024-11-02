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
from lib.calculateFK import FK

rospy.init_node("visualizer")

# Using your solution code
ik = IK()
fk = FK()

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

targetConfigs = np.array([
    [-1.4887,  1.1504,  1.2076, -2.1524,  2.3081,  1.6036,  2.0963],
    [ 0.8507, -0.1507,  0.4648, -2.4340, -0.8489,  1.5249,  0.2696],
    [-1.2735, -0.8899,  1.7616, -2.1059,  0.9283,  1.8044,  0.6954],
    [-2.0396, -1.5320,  1.6340, -0.6818, -2.1843,  0.8750,  1.1682],
    [ 0.6771, -0.5774, -0.8771, -0.0914, -1.2338,  3.3333,  1.8801],
    [-0.6320, -1.1459,  1.0572, -0.9284,  2.2563,  0.2906,  1.9642],
    [ 0.2892, -0.8065, -1.4352, -0.3636,  1.5782,  3.4578,  2.3652],
    [-1.4517,  0.4870,  1.1732, -2.3066, -1.0701,  1.2054, -0.6441],
    [-0.5029,  1.3613, -1.9751, -0.2524,  1.2572,  0.4109, -2.2458],
    [ 2.7390, -1.2510,  2.6872, -1.0706, -0.3875,  2.6189,  2.0046],
    [-2.1547,  0.0937, -0.4679, -2.6991,  0.9519,  3.5556, -1.4124],
    [ 2.3915,  1.1441, -0.4278, -0.6986, -1.5379,  2.1299, -2.2882],
    [ 0.9748, -0.4224, -0.7921, -0.3643,  0.0756,  1.8126,  0.1755],
    [ 1.3139, -1.2134,  0.7492, -2.5810, -2.7968,  1.7046, -1.7470],
    [-0.6886, -0.4837, -0.0006, -0.9558, -2.6126,  2.8717,  0.2098],
    [ 1.5295,  0.2061,  1.3047, -1.6133, -2.4031,  2.1307, -2.2484],
    [ 0.8194,  0.1352,  2.4559, -1.5007, -0.3279,  3.5329, -0.9941],
    [ 2.5906, -1.1500,  0.9102, -1.9801, -1.1071,  0.8587, -1.2524],
    [ 1.4886,  1.4329,  0.4216, -1.1348,  2.2625,  0.7308,  1.5978],
    [ 0.3768,  1.0200, -1.0203, -1.5041,  2.4230,  1.6443,  1.3482],
    [-1.8583, -1.6610, -0.5348, -0.1849, -2.7778,  3.3491, -0.6375],
    [-1.1826,  0.4100, -1.3554, -2.0102,  1.9231,  2.1356,  0.1593],
    [-2.3778, -1.1081,  0.4795, -3.0265,  1.5041,  0.8729, -1.8017],
    [-0.8231,  0.1034,  0.7360, -3.0673,  2.0668,  2.8108, -1.1185],
    [-1.8238,  0.7635, -0.7901, -1.9685,  1.1382,  3.3400, -0.7217]
])

####################
## Test Execution ##
####################

np.set_printoptions(suppress=True)

if __name__ == "__main__":

    arm = ArmController()
    seed = arm.neutral_position()
    arm.safe_move_to_position(seed)
    timetaken= []
    itTaken = []
    successCount = 0
    lower = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
    upper = np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973])  
    center = lower + (upper - lower) / 2 # compute middle of range of motion of each joint
    # Iterates through the given targets, using your IK solution
    # Try editing the targets list above to do more testing!
    for i, target in enumerate(targetConfigs):
        _,target = fk.forward(target)
        print("Target " + str(i) + " located at:")
        print(target)
        print("Solving... ")
        show_pose(target,"target")

        #seed = arm.neutral_position() # use neutral configuration as seed
        seed = center
        #seed = np.array([0,0,0,0,pi/2,pi/4, pi/4])

        start = perf_counter()
        q, rollout, success, message = ik.inverse(target, seed, method='J_pseudo', alpha=.53)  #try both methods
        stop = perf_counter()
        dt = stop - start
        timetaken.append(dt)
        itTaken.append(len(rollout))
        if success:
            successCount+=1
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

    print("avg Time: ",np.mean(timetaken))
    print("median Time: ",np.median(timetaken))
    print("max Time: ",np.max(timetaken))
    
    print("avg IT: ",np.mean(itTaken))
    print("median IT: ",np.median(itTaken))
    print("max IT: ",np.max(itTaken))

    print("Success Rate: ", (successCount/25))