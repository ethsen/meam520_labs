import numpy as np
from math import pi, acos
from scipy.linalg import null_space
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


#from lib.calculateFKJac import FK_Jac
#from lib.detectCollision import detectCollision
#from lib.loadmap import loadmap

from loadmap import loadmap
from calculateFKJac import FK_Jac
from detectCollision import detectCollision
from potentialFieldTester import *

class PotentialFieldPlanner:

    # JOINT LIMITS
    lower = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
    upper = np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973])

    center = lower + (upper - lower) / 2 # compute middle of range of motion of each joint
    fk = FK_Jac()
    plt.ion()  # Turn on interactive mode
    fig = plt.figure()  
    ax = fig.add_subplot(111, projection='3d')

    def __init__(self, tol=1e-4, max_steps=500, min_step_size=1e-5):
        """
        Constructs a potential field planner with solver parameters.

        PARAMETERS:
        tol - the maximum distance between two joint sets
        max_steps - number of iterations before the algorithm must terminate
        min_step_size - the minimum step size before concluding that the
        optimizer has converged
        """

        # YOU MAY NEED TO CHANGE THESE PARAMETERS

        # solver parameters
        self.tol = tol
        self.max_steps = max_steps
        self.min_step_size = min_step_size
        

    ######################
    ## Helper Functions ##
    ######################
    # The following functions are provided to you to help you to better structure your code
    # You don't necessarily have to use them. You can also edit them to fit your own situation 

    @staticmethod
    def attractive_force(target, current, joint):
        """
        Helper function for computing the attactive force between the current position and
        the target position for one joint. Computes the attractive force vector between the 
        target joint position and the current joint position 

        INPUTS:
        target - 3x1 numpy array representing the desired joint position in the world frame
        current - 3x1 numpy array representing the current joint position in the world frame
        joint - joint number 
        OUTPUTS:
        att_f - 3x1 numpy array representing the force vector that pulls the joint 
        from the current position to the target position 
        """
        ## STUDENT CODE STARTS HERE
        d = 0.12
        diff = current - target
        if np.linalg.norm(diff)**2 > d:
            att_f = -(diff /np.linalg.norm(diff))

        else:
            if joint > 5:
                xi = 15
            else: 
                xi = 30 #attractive field strength
            att_f = -xi* (diff)

        ## END STUDENT CODE

        return att_f.reshape(-1,1)

    @staticmethod
    def repulsive_force(distance, current, unitvec=np.zeros((3,1))):
        """
        Helper function for computing the repulsive force between the current position
        of one joint and one obstacle. Computes the repulsive force vector between the 
        obstacle and the current joint position 

        INPUTS:
        obstacle - 1x6 numpy array representing the an obstacle box in the world frame
        distance - distance of joint to the obstacle
        current - 3x1 numpy array representing the current joint position in the world frame
        unitvec - 3x1 numpy array representing the unit vector from the current joint position 
        to the closest point on the obstacle box 

        OUTPUTS:
        rep_f - 3x1 numpy array representing the force vector that pushes the joint 
        from the obstacle
        """

        ## STUDENT CODE STARTS HERE

        eta = -.001 # repulsive field strength
        d0 = 0.25
        rep_f = np.zeros((3, 1))

        if distance < d0:
            rep_f = eta * ((1 / distance) - (1 / d0)) * (1 / distance**2) * unitvec.reshape(-1,1)

        ## END STUDENT CODE
        return rep_f

    @staticmethod
    def dist_point2box(p, box):
        """
        Helper function for the computation of repulsive forces. Computes the closest point
        on the box to a given point 
    
        INPUTS:
        p - nx3 numpy array of points [x,y,z]
        box - 1x6 numpy array of minimum and maximum points of box

        OUTPUTS:
        dist - nx1 numpy array of distance between the points and the box
                dist > 0 point outside
                dist = 0 point is on or inside box
        unit - nx3 numpy array where each row is the corresponding unit vector 
        from the point to the closest spot on the box
            norm(unit) = 1 point is outside the box
            norm(unit)= 0 point is on/inside the box

         Method from MultiRRomero
         @ https://stackoverflow.com/questions/5254838/
         calculating-distance-between-a-point-and-a-rectangular-box-nearest-point
        """
        # THIS FUNCTION HAS BEEN FULLY IMPLEMENTED FOR YOU

        # Get box info
        boxMin = np.array([box[0], box[1], box[2]])
        boxMax = np.array([box[3], box[4], box[5]])
        boxCenter = boxMin*0.5 + boxMax*0.5
        p = np.array(p)

        # Get distance info from point to box boundary
        dx = np.amax(np.vstack([boxMin[0] - p[:, 0], p[:, 0] - boxMax[0], np.zeros(p[:, 0].shape)]).T, 1)
        dy = np.amax(np.vstack([boxMin[1] - p[:, 1], p[:, 1] - boxMax[1], np.zeros(p[:, 1].shape)]).T, 1)
        dz = np.amax(np.vstack([boxMin[2] - p[:, 2], p[:, 2] - boxMax[2], np.zeros(p[:, 2].shape)]).T, 1)

        # convert to distance
        distances = np.vstack([dx, dy, dz]).T
        dist = np.linalg.norm(distances, axis=1)

        # Figure out the signs
        signs = np.sign(boxCenter-p)

        # Calculate unit vector and replace with
        unit = distances / dist[:, np.newaxis] * signs
        unit[np.isnan(unit)] = 0
        unit[np.isinf(unit)] = 0
        return dist, unit

    @staticmethod
    def compute_forces(target, obstacle, current):
        """
        Helper function for the computation of forces on every joints. Computes the sum 
        of forces (attactive, repulsive) on each joint. 

        INPUTS:
        target - 3x9 numpy array representing the desired joint/end effector positions 
        in the world frame
        obstacle - nx6 numpy array representing the obstacle box min and max positions
        in the world frame
        current- 3x9 numpy array representing the current joint/end effector positions 
        in the world frame

        OUTPUTS:
        joint_forces - 3x9 numpy array representing the force vectors on each 
        joint/end effector
        """
        joint_forces = np.zeros((3, 9))
        
        ## STUDENT CODE STARTS HERE
        attForces = np.zeros((3,9))

        for i in range(len(target)):
            attForce = PotentialFieldPlanner.attractive_force(target[i],current[i],i)
            attForces[:,i] = attForce.flatten()
        repForces = np.zeros((3,9))
        
        for obs in obstacle:
            dist,unit = PotentialFieldPlanner.dist_point2box(current, obs)
            for i in range(1,len(target)):
                repForce = PotentialFieldPlanner.repulsive_force(dist[i],current[i], unit[i])
                repForces[:,i-1] = repForce.flatten()

        joint_forces = attForces+repForces
        plotAttractiveVector(PotentialFieldPlanner.ax,target, current, (joint_forces).T,obstacle)
        return joint_forces
                    

            
        ## END STUDENT CODE

    @staticmethod
    def compute_torques(joint_forces, q):
        """
        Helper function for converting joint forces to joint torques. Computes the sum 
        of torques on each joint.

        INPUTS:
        joint_forces - 3x9 numpy array representing the force vectors on each 
        joint/end effector
        q - 1x7 numpy array representing the current joint angles

        OUTPUTS:
        joint_torques - 1x9 numpy array representing the torques on each joint 
        """

        ## STUDENT CODE STARTS HERE
        joint_torques = np.zeros((9,9))
        for i in range(len(joint_torques)):
            linJac = PotentialFieldPlanner.fk.calcLinJacobian(q,i+1)
            joint_torques[i]= (linJac.T @ joint_forces[:,i])

        ## END STUDENT CODE
        joint_torques =np.sum(joint_torques, axis = 0).flatten()
        return joint_torques

    @staticmethod
    def q_distance(target, current):
        """
        Helper function which computes the distance between any two
        vectors.

        This data can be used to decide whether two joint sets can be
        considered equal within a certain tolerance.

        INPUTS:
        target - 1x7 numpy array representing some joint angles
        current - 1x7 numpy array representing some joint angles

        OUTPUTS:
        distance - the distance between the target and the current joint sets 

        """

        ## STUDENT CODE STARTS HERE

        distance = np.linalg.norm(target - current)

        ## END STUDENT CODE

        return distance
    
    @staticmethod
    def compute_gradient(q, target, map_struct):
        """
        Computes the joint gradient step to move the current joint positions to the
        next set of joint positions which leads to a closer configuration to the goal 
        configuration 

        INPUTS:
        q - 1x7 numpy array. the current joint configuration, a "best guess" so far for the final answer
        target - 1x7 numpy array containing the desired joint angles
        map_struct - a map struct containing the obstacle box min and max positions

        OUTPUTS:
        dq - 1x7 numpy array. a desired joint velocity to perform this task. 
        """

        ## STUDENT CODE STARTS HERE
        targetJointPos, _ = PotentialFieldPlanner.fk.forward_expanded(target)
        currJointPos, _ =  PotentialFieldPlanner.fk.forward_expanded(q)
        forces = PotentialFieldPlanner.compute_forces(targetJointPos[1:], map_struct.obstacles, currJointPos[1:])
        torques = PotentialFieldPlanner.compute_torques(forces, q)
        dq = torques[:7]
        #dq =  np.concatenate((torques[:6], [torques[-1]]))

       #print(np.linalg.norm(dq)) 
        dq = dq /np.linalg.norm(dq)


        ## END STUDENT CODE

        return dq
    
    def testPlot(jointPositions1,jointPositions2):
        ax = plt.figure().add_subplot(projection='3d')
        ax.plot(jointPositions1[:,0], jointPositions1[:,1], jointPositions1[:,2])
        ax.scatter(jointPositions1[:,0], jointPositions1[:,1], jointPositions1[:,2], c= 'red')
        ax.plot(jointPositions2[:,0], jointPositions2[:,1], jointPositions2[:,2])
        ax.scatter(jointPositions2[:,0], jointPositions2[:,1], jointPositions2[:,2], c= 'blue')
        #prism = Poly3DCollection([box],edgecolor='g',facecolor='g',alpha=0.5)
        #ax.add_collection3d(prism)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(0, 2)
        plt.show()
        
    ###############################
    ### Potential Feild Solver  ###
    ###############################

    def plan(self, map_struct, start, goal):
        """
        Uses potential field to move the Panda robot arm from the startng configuration to
        the goal configuration.

        INPUTS:
        map_struct - a map struct containing min and max positions of obstacle boxes 
        start - 1x7 numpy array representing the starting joint angles for a configuration 
        goal - 1x7 numpy array representing the desired joint angles for a configuration

        OUTPUTS:
        q - nx7 numpy array of joint angles [q0, q1, q2, q3, q4, q5, q6]. This should contain
        all the joint angles throughout the path of the planner. The first row of q should be
        the starting joint angles and the last row of q should be the goal joint angles. 
        """
        
        q_path = np.array([]).reshape(0,7)
        q_path = np.vstack([q_path,start])
        q = start
        jointPosOld, _ = PotentialFieldPlanner.fk.forward_expanded(q)
        goalJointPos, _ = PotentialFieldPlanner.fk.forward_expanded(goal)
        steps = 0            
        alpha = 0.02
        while steps!= self.max_steps:
            steps+=1
            ## STUDENT CODE STARTS HERE
            
            # The following comments are hints to help you to implement the planner
            # You don't necessarily have to follow these steps to complete your code 
            
            # Compute gradient 
            # TODO: this is how to change your joint angles 
            
            dq = PotentialFieldPlanner.compute_gradient(q,goal,map_struct)
            qNew = q + alpha*dq
            jointPosNew,  _ = PotentialFieldPlanner.fk.forward_expanded(qNew)
            #plotTorqueVectors(PotentialFieldPlanner.ax, jointPosOld, goalJointPos,dq, map_struct.obstacles)

            collision = False
            for obs in map_struct.obstacles:
                test = detectCollision(jointPosOld, jointPosNew,obs)
                if any(test):
                    indices = np.where(test)[0]
                    collision = True
                    modified_dq = dq.copy()
                    for idx in indices:
                        modified_dq[:idx+1] = -dq[:idx+1]
                    reduced_alpha = alpha * 0.5
                    #random_perturbation = np.random.uniform(-.05, 0.05, size=q.shape)
                    qNew = q + reduced_alpha * modified_dq #+ random_perturbation            
                    break
            # Termination Conditions
            if self.q_distance(goal,qNew) < self.tol: # TODO: check termination conditions
                q_path = np.vstack([q_path,qNew])
                break # exit the while loop if conditions are met!
            
            elif np.linalg.norm(dq) < self.min_step_size:
                random_perturbation = np.random.uniform(-0.5, 0.5, size=q.shape)
                q = q + random_perturbation
            elif len(q_path) > 20 and np.linalg.norm(qNew - q_path[len(q_path)-20]) < 0.1:
                random_perturbation = np.random.uniform(-0.5, 0.5, size=q.shape)
                q = q + random_perturbation
            else:
                    #print(np.linalg.norm(qNew - q_path[len(q_path)-20]))
                q_path = np.vstack([q_path,qNew])
                q = qNew.copy()
                jointPosOld = jointPosNew



        return q_path
    
    
################################
## Simple Testing Environment ##
################################

if __name__ == "__main__":

    np.set_printoptions(suppress=True,precision=5)

    planner = PotentialFieldPlanner()
    
    # inputs 
    map_struct = loadmap("maps/emptyMap.txt")
    #map_struct = loadmap("maps/map1.txt")
    start = np.array([0,-1,0,-2,0,1.57,0])
    goal =  np.array([-1.2, 1.57 , 1.57, -2.07, -1.57, 1.57, 0.7])

    #start = np.array([0,0,0,0,0,0,0])
    #goal =  np.array([0,0,0,0,0,0,1])
    # potential field planning
    q_path = planner.plan(deepcopy(map_struct), deepcopy(start), deepcopy(goal))
    
    # show results
    for i in range(q_path.shape[0]):
        error = PotentialFieldPlanner.q_distance(q_path[i, :], goal)
        print('iteration:',i,' q =', q_path[i, :], ' error={error}'.format(error=error))

    print("q path: ", q_path)
