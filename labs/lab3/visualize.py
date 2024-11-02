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
    [ 0.1523,  0.2314, -0.5231, -1.5708,  0.4231,  1.8675,  0.4231],
    [-1.2341,  0.8234,  1.2341, -1.5708, -1.2341,  1.8675, -0.8234],
    [ 0.8234, -0.5231,  0.8234, -1.5708,  0.8234,  1.8675,  1.2341],
    [-0.5231,  0.4231, -1.2341, -1.5708, -0.5231,  1.8675, -1.2341],
    [ 1.2341, -0.8234,  0.4231, -1.5708,  1.2341,  1.8675,  0.8234],
    [-0.8234,  1.2341, -0.8234, -1.5708, -0.8234,  1.8675, -0.4231],
    [ 0.4231, -1.2341,  1.2341, -1.5708,  0.4231,  1.8675,  0.5231],
    [-1.2341,  0.4231, -0.4231, -1.5708, -1.2341,  1.8675, -1.2341],
    [ 0.8234, -0.8234,  0.8234, -1.5708,  0.8234,  1.8675,  0.8234],
    [-0.4231,  1.2341, -1.2341, -1.5708, -0.4231,  1.8675, -0.8234],
    [ 1.2341, -0.4231,  0.4231, -1.5708,  1.2341,  1.8675,  1.2341],
    [-0.8234,  0.8234, -0.8234, -1.5708, -0.8234,  1.8675, -0.4231],
    [ 0.4231, -1.2341,  1.2341, -1.5708,  0.4231,  1.8675,  0.8234],
    [-1.2341,  0.4231, -0.4231, -1.5708, -1.2341,  1.8675, -1.2341],
    [ 0.8234, -0.8234,  0.8234, -1.5708,  0.8234,  1.8675,  0.4231],
    [-0.4231,  1.2341, -1.2341, -1.5708, -0.4231,  1.8675, -0.8234],
    [ 1.2341, -0.4231,  0.4231, -1.5708,  1.2341,  1.8675,  1.2341],
    [-0.8234,  0.8234, -0.8234, -1.5708, -0.8234,  1.8675, -0.4231],
    [ 0.4231, -1.2341,  1.2341, -1.5708,  0.4231,  1.8675,  0.8234],
    [-1.2341,  0.4231, -0.4231, -1.5708, -1.2341,  1.8675, -1.2341],
    [ 0.8234, -0.8234,  0.8234, -1.5708,  0.8234,  1.8675,  0.4231],
    [-0.4231,  1.2341, -1.2341, -1.5708, -0.4231,  1.8675, -0.8234],
    [ 1.2341, -0.4231,  0.4231, -1.5708,  1.2341,  1.8675,  1.2341],
    [-0.8234,  0.8234, -0.8234, -1.5708, -0.8234,  1.8675, -0.4231],
    [ 0.4231, -1.2341,  1.2341, -1.5708,  0.4231,  1.8675,  0.8234]
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
    q = arm.neutral_position()
    #seed = center
    for i, target in enumerate(targetConfigs):
        _,target = fk.forward(target)
        print("Target " + str(i) + " located at:")
        print(target)
        print("Solving... ")
        show_pose(target,"target")

        seed = arm.neutral_position() # use neutral configuration as seed
        #seed = np.array([0,0,0,0,pi/2,pi/4, pi/4])

        start = perf_counter()
        q, rollout, success, message = ik.inverse(target, seed, method='J_pseuo', alpha=.6)  #try both methods
        stop = perf_counter()
        dt = stop - start
        timetaken.append(dt)
        itTaken.append(len(rollout))
        if success:
            successCount+=1
            print("Solution found in {time:2.2f} seconds ({it} iterations).".format(time=dt,it=len(rollout)))
            arm.safe_move_to_position(q)
            #seed = q

            # Visualize
            if visulaize_mani_ellipsoid:
                mu, M = calcManipulability(q)
                show_manipulability_ellipsoid(M)
                print('Manipulability Index',mu)
        else:
            print('IK Failed for this target using this seed.')

    print("avg Time: ",np.mean(timetaken))
    print("median Time: ",np.median(timetaken))
    print("max Time: ",np.max(timetaken))
    
    print("avg IT: ",np.mean(itTaken))
    print("median IT: ",np.median(itTaken))
    print("max IT: ",np.max(itTaken))

    print("Success Rate: ", (successCount/25))