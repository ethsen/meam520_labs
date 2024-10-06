import numpy as np
import sympy as sp
from math import pi

# Input parameters
xDisplacement = [0, 0, 0.0825, -0.0825, 0, 0.088, 0]
zDisplacement = [0.333, 0, 0.316, 0, .384, 0, 0.21]
angleDisplacement = [-pi/2, pi/2, pi/2, -pi/2, pi/2, pi/2, 0]
jointOffsets = np.stack(([0, 0, .141, 1], [0, 0, 0, 1], [0, 0, .195, 1],
                         [0, 0, 0, 1], [0, 0, 0.125, 1], [0, 0, -.015, 1],
                         [0, 0, .051, 1], [0, 0, 0, 1]), axis=0)

def angularJacobian(q):
    rot= sp.eye(3)
    for i in range(len(q)):
        currTheta = 'theta_' + str(i+1)
        theta = sp.symbols(currTheta)
        
        # Define rotation matrix rows
        r1 = [sp.cos(theta), -sp.sin(theta) * int(sp.cos(angleDisplacement[i])), sp.sin(theta) * int(sp.sin(angleDisplacement[i]))]
        r2 = [sp.sin(theta), sp.cos(theta) * int(sp.cos(angleDisplacement[i])), -sp.cos(theta) * int(sp.sin(angleDisplacement[i]))]
        r3 = [0, int(sp.sin(angleDisplacement[i])), int(sp.cos(angleDisplacement[i]))]
        
        # Create a rotation matrix from the rows
        rot = rot * sp.Matrix([r1, r2, r3])  # Pass the rows as a list of lists
        
        # Print the symbolic rotation matrix
        print(f"Rotation matrix for link {i}:")
        sp.pprint(rot[:,-1])
        print("\n")
    return

def linearJacobian(q):
    T0e= calcT0e(q)
    prev = sp.eye(4)
    for i in range(len(q)):
        if i  == 0:
            currTheta = 'theta_' + str(i+1)
            theta = sp.symbols(currTheta)
            a1 = [sp.cos(theta), -sp.sin(theta) * int(sp.cos(angleDisplacement[i])), sp.sin(theta) * int(sp.sin(angleDisplacement[i])), xDisplacement[i]*sp.cos(theta)]
            a2 = [sp.sin(theta), sp.cos(theta) * int(sp.cos(angleDisplacement[i])), -sp.cos(theta) * int(sp.sin(angleDisplacement[i])), xDisplacement[i]*sp.sin(theta)]
            a3 = [0, int(sp.sin(angleDisplacement[i])), int(sp.cos(angleDisplacement[i])),zDisplacement[i]]
            a4 = [0,0,0,1]
            ai = sp.Matrix([a1,a2,a3,a4])
            zi = sp.Matrix([[0],[0],[1]])
            jvi = zi.cross(T0e[:3,-1] - jointOffsets[i])
            prev = ai

        else:
            currTheta = 'theta_' + str(i+1)
            theta = sp.symbols(currTheta)
            a1 = [sp.cos(theta), -sp.sin(theta) * int(sp.cos(angleDisplacement[i])), sp.sin(theta) * int(sp.sin(angleDisplacement[i])), xDisplacement[i]*sp.cos(theta)]
            a2 = [sp.sin(theta), sp.cos(theta) * int(sp.cos(angleDisplacement[i])), -sp.cos(theta) * int(sp.sin(angleDisplacement[i])), xDisplacement[i]*sp.sin(theta)]
            a3 = [0, int(sp.sin(angleDisplacement[i])), int(sp.cos(angleDisplacement[i])),zDisplacement[i]]
            a4 = [0,0,0,1]
            ai = sp.Matrix([a1,a2,a3,a4])
            zi = prev[:3,2]
            jvi = zi.cross((T0e[:3,-1] - (ai[:3,-1] + jointOffsets[i])))
            prev = ai


        print(f"Translation matrix for joint {i}:")
        sp.pprint(jvi)
        print("\n")


def calcT0e(q):
    T0e= sp.eye(4)
    for i in range(len(q)):
        currTheta = 'theta_' + str(i+1)
        theta = sp.symbols(currTheta)
        a1 = [sp.cos(theta), -sp.sin(theta) * int(sp.cos(angleDisplacement[i])), sp.sin(theta) * int(sp.sin(angleDisplacement[i])), xDisplacement[i]*sp.cos(theta)]
        a2 = [sp.sin(theta), sp.cos(theta) * int(sp.cos(angleDisplacement[i])), -sp.cos(theta) * int(sp.sin(angleDisplacement[i])), xDisplacement[i]*sp.sin(theta)]
        a3 = [0, int(sp.sin(angleDisplacement[i])), int(sp.cos(angleDisplacement[i])),zDisplacement[i]]
        a4 = [0,0,0,1]
        ai = sp.Matrix([a1,a2,a3,a4])
        T0e = T0e * ai

    return T0e


if __name__ == "__main__":

    q = np.array([0, 0, 0, 0, 0, 0, 0])
    #angularJacobian(q)
    linearJacobian(q)
