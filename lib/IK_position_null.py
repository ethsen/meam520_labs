import numpy as np
from math import pi, acos
from scipy.linalg import null_space
from time import perf_counter


from lib.calcJacobian import calcJacobian
from lib.calculateFK import FK
from lib.calcAngDiff import calcAngDiff
#from lib.IK_velocity import IK_velocity  #optional

#from calcJacobian import calcJacobian
#from calculateFK import FK
#from calcAngDiff import calcAngDiff
#from IK_velocity import IK_velocity

class IK:

    #JOINT LIMITS
    lower = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
    upper = np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973])

    center = lower + (upper - lower) / 2 # compute middle of range of motion of each joint
    fk = FK()

    def __init__(self,linear_tol=1e-4, angular_tol=1e-3, max_steps=1000, min_step_size=1e-5):
        """
        Constructs an optimization-based IK solver with given solver parameters.
        Default parameters are tuned to reasonable values.

        PARAMETERS:
        linear_tol - the maximum distance in meters between the target end
        effector origin and actual end effector origin for a solution to be
        considered successful
        angular_tol - the maximum angle of rotation in radians between the target
        end effector frame and actual end effector frame for a solution to be
        considered successful
        max_steps - number of iterations before the algorithm must terminate
        min_step_size - the minimum step size before concluding that the
        optimizer has converged
        """

        # solver parameters
        self.linear_tol = linear_tol
        self.angular_tol = angular_tol
        self.max_steps = max_steps
        self.min_step_size = min_step_size


    ######################
    ## Helper Functions ##
    ######################

    @staticmethod
    def displacement_and_axis(target, current):
        """
        Helper function for the End Effector Task. Computes the displacement
        vector and axis of rotation from the current frame to the target frame

        This data can also be interpreted as an end effector velocity which will
        bring the end effector closer to the target position and orientation.

        INPUTS:
        target - 4x4 numpy array representing the desired transformation from
        end effector to world
        current - 4x4 numpy array representing the "current" end effector orientation

        OUTPUTS:
        displacement - a 3-element numpy array containing the displacement from
        the current frame to the target frame, expressed in the world frame
        axis - a 3-element numpy array containing the axis of the rotation from
        the current frame to the end effector frame. The magnitude of this vector
        must be sin(angle), where angle is the angle of rotation around this axis
        """

        ## STUDENT CODE STARTS HERE
        displacement = target[:-1,-1] -  current[:-1,-1]
        axis = calcAngDiff(target[:-1,:-1], current[:-1,:-1])

        ## END STUDENT CODE
        return displacement, axis

    @staticmethod
    def distance_and_angle(G, H):
        """
        Helper function which computes the distance and angle between any two
        transforms.

        This data can be used to decide whether two transforms can be
        considered equal within a certain linear and angular tolerance.

        Be careful! Using the axis output of displacement_and_axis to compute
        the angle will result in incorrect results when |angle| > pi/2

        INPUTS:
        G - a 4x4 numpy array representing some homogenous transformation
        H - a 4x4 numpy array representing some homogenous transformation

        OUTPUTS:
        distance - the distance in meters between the origins of G & H
        angle - the angle in radians between the orientations of G & H
        """
        
        ## STUDENT CODE STARTS HERE
        distance, _ = IK.displacement_and_axis(G,H)
        distance = np.linalg.norm(distance, ord=2)
        rot = G[:-1,:-1] @ H[:-1,:-1].T
        inner = np.clip((np.trace(rot)-1)/2, -1.0, 1.0)
        angle = np.arccos(inner)
        ## END STUDENT CODE
        return distance, angle

    def is_valid_solution(self,q,target):
        """
        Given a candidate solution, determine if it achieves the primary task
        and also respects the joint limits.

        INPUTS
        q - the candidate solution, namely the joint angles
        target - 4x4 numpy array representing the desired transformation from
        end effector to world

        OUTPUTS:
        success - a Boolean which is True if and only if the candidate solution
        produces an end effector pose which is within the given linear and
        angular tolerances of the target pose, and also respects the joint
        limits.
        """

        ## STUDENT CODE STARTS HERE
        success = False
        if np.all(q >= IK.lower) and np.all(q <= IK.upper):
    
            _,T0e  = IK.fk.forward(q)
            displacement, angle  = IK.distance_and_angle(T0e,target)
            
            if displacement > self.linear_tol:
                message = "Solution not found: Displacement greater than linear tolerance"
                return success, message
            elif angle > self.angular_tol:
                message = "Solution not found: Angle greater than angular tolerance"
                return success, message
            else:
                success = True
                message = "Solution found"
                return success, message
        else:
            message = "Solution not found: At least one joint exceeds the joint limits"
            return success, message

    ####################
    ## Task Functions ##
    ####################

    @staticmethod
    def end_effector_task(q,target,method):
        """
        Primary task for IK solver. Computes a joint velocity which will reduce
        the error between the target end effector pose and the current end
        effector pose (corresponding to configuration q).

        INPUTS:
        q - the current joint configuration, a "best guess" so far for the final answer
        target - a 4x4 numpy array containing the desired end effector pose
        method - a boolean variable that determines to use either 'J_pseudo' or 'J_trans' 
        (J pseudo-inverse or J transpose) in your algorithm
        
        OUTPUTS:
        dq - a desired joint velocity to perform this task, which will smoothly
        decay to zero magnitude as the task is achieved
        """
    
        _,T0e  = IK.fk.forward(q)
        displacement, axis = IK.displacement_and_axis(target, T0e)
        des = np.hstack((displacement,axis))
        j = calcJacobian(q)
        if method == 'J_pseudo':
            jPinv = np.linalg.pinv(j)
            dq = jPinv @ des.T
            
        else:
            dq = j.T @ des

        return dq

    @staticmethod
    def joint_centering_task(q,rate=5e-1): 
        """
        Secondary task for IK solver. Computes a joint velocity which will
        reduce the offset between each joint's angle and the center of its range
        of motion. This secondary task acts as a "soft constraint" which
        encourages the solver to choose solutions within the allowed range of
        motion for the joints.

        INPUTS:
        q - the joint angles
        rate - a tunable parameter dictating how quickly to try to center the
        joints. Turning this parameter improves convergence behavior for the
        primary task, but also requires more solver iterations.

        OUTPUTS:
        dq - a desired joint velocity to perform this task, which will smoothly
        decay to zero magnitude as the task is achieved
        """

        # normalize the offsets of all joints to range from -1 to 1 within the allowed range
        offset = 2 * (q - IK.center) / (IK.upper - IK.lower)
        dq = rate * -offset # proportional term (implied quadratic cost)

        return dq
        
    ###############################
    ## Inverse Kinematics Solver ##
    ###############################

    def inverse(self, target, seed, method, alpha):
        """
        Uses gradient descent to solve the full inverse kinematics of the Panda robot.

        INPUTS:
        target - 4x4 numpy array representing the desired transformation from
        end effector to world
        seed - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6], which
        is the "initial guess" from which to proceed with optimization
        method - a boolean variable that determines to use either 'J_pseudo' or 'J_trans' 
        (J pseudo-inverse or J transpose) in your algorithm

        OUTPUTS:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6], giving the
        solution if success is True or the closest guess if success is False.
        success - True if the IK algorithm successfully found a configuration
        which achieves the target within the given tolerance. Otherwise False
        rollout - a list containing the guess for q at each iteration of the algorithm
        """

        q = seed
        rollout = []
      
        ## gradient descent:
        while True:
            rollout.append(q.copy())

            # Primary Task - Achieve End Effector Pose
            dq_ik = IK.end_effector_task(q,target, method)

            # Secondary Task - Center Joints
            dq_center = IK.joint_centering_task(q)

            ## Task Prioritization
            j  = calcJacobian(q)
            jPinv = np.linalg.pinv(j)
            null = (np.eye(7) - (jPinv @ j)) @ dq_center
            dq = alpha*(dq_ik+null)
            # Check termination conditions
            if len(rollout) > self.max_steps or  np.linalg.norm(dq_ik) < self.min_step_size:
                break

            # update q
            q = q + dq


        success, message = self.is_valid_solution(q,target)
        return q, rollout, success, message
    
    @staticmethod
    def generateTarget(amt,margin = 0.25): 
        configs =np.zeros((amt,7))
        for i in range(amt):
            configs[i] = np.random.uniform(IK.lower + margin,IK.upper - margin)
        return configs
    
    @staticmethod
    def test():
        fk = FK()
        alphas = [0.30, 0.50]
        methods = ['J_pseudo', 'J_trans']
        targets = IK.generateTarget(50)
        seed = np.array([0,0,0,-pi/2,0,pi/2,pi/4])
        #seed = IK.center
        for i in range(len(alphas)):
            timeTaken =[]
            itTaken = []
            alpha = alphas[i]
            method = methods[i]
            successCount = 0
            for targ in targets:
                _,target = fk.forward(targ)
                start = perf_counter()
                q, rollout, success, message = ik.inverse(target, seed, method, alpha)  #try both methods
                stop = perf_counter()
                dt = stop - start
                
                if success:
                    #seed = q
                    successCount+=1
                    timeTaken.append(dt)
                    itTaken.append(len(rollout))

            print(method)
            print("avg Time: ",np.mean(timeTaken))
            print("median Time: ",np.median(timeTaken))
            print("max Time: ",np.max(timeTaken))
            
            print("avg IT: ",np.mean(itTaken))
            print("median IT: ",np.median(itTaken))
            print("max IT: ",np.max(itTaken))

            print("Success Rate: ", (successCount/50))
        
    @staticmethod
    def singleTest():
        seed = np.array([0,0,0,-pi/2,0,pi/2,pi/4])
        target = np.array([[1,0,0,0.56],
                            [0,-1,0,0.15],
                            [0,0,-1,0.5],
                            [0,0,0,1]])
        q, rollout, success, message = ik.inverse(target, seed, 'J_pseudo', 0.3)  #try both methods

        print(q)




################################
## Simple Testing Environment ##
################################



if __name__ == "__main__":

    np.set_printoptions(suppress=True,precision=5)

    ik = IK()
    # matches figure in the handout
    #seed = np.array([0,0,0,-pi/2,0,pi/2,pi/4])
    ik.singleTest()

    """
    #Testing for best Alpha
    alpha = 0
    bestSuccess = 0
    bestScore = -1000
    bestIter = ik.max_steps
    targets = ik.generateTarget(15)
    while alpha < 1:
        iterCount = []
        successCount = 0
        alpha += 0.1
        for target in targets:
            #print(j/50)
            _, target = ik.fk.forward(target)
            # Using pseudo-inverse 
            q, rollout, success, message = ik.inverse(target, seed, method='J_psedo',alpha = alpha)
            if success:
                iters = len(rollout)
                successCount += 1
                iterCount.append(iters)

        if iterCount:
            if successCount > bestSuccess-1 and np.mean(iterCount) < bestIter:
                bestSuccess = successCount
                bestAlpha = alpha
                bestIter = np.mean(iterCount)

    #print("Most Successful Alpha: ", bestAlphaSuccess, " With: ",bestSuccess)
    #print("Most Successful Alpha: ", bestAlphaIter, " With: ",bestIter)
    print("Best Alpha Value: ", bestAlpha, "With Success Rate of: ", bestSuccess/len(targets), "And avg Iterations of: ", bestIter)       
            

        
    
    for i, q_pseudo in enumerate(rollout_pseudo):
        joints, pose = ik.fk.forward(q_pseudo)
        d, ang = IK.distance_and_angle(target,pose)
        print('iteration:',i,' q =',q_pseudo, ' d={d:3.4f}  ang={ang:3.3f}'.format(d=d,ang=ang))

    # Using pseudo-inverse 
    
    rans, rollout_trans, success_trans, message_trans = ik.inverse(target, seed, method='J_trans', alpha=.60)

    for i, q_trans in enumerate(rollout_trans):
        joints, pose = ik.fk.forward(q_trans)
        d, ang = IK.distance_and_angle(target,pose)
        print('iteration:',i,' q =',q_trans, ' d={d:3.4f}  ang={ang:3.3f}'.format(d=d,ang=ang))
    

    # compare
    print("\n method: J_pseudo-inverse")
    print("   Success: ",success_pseudo, ":  ", message_pseudo)
    print("   Solution: ",q_pseudo)
    print("   #Iterations : ", len(rollout_pseudo))
    print("\n method: J_transpose")
    print("   Success: ",success_trans, ":  ", message_trans)
    print("   Solution: ",q_trans)
    print("   #Iterations :", len(rollout_trans),'\n')
    """