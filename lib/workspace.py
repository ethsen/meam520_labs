from calculateFK import FK
#from core.interfaces import ArmController
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fk = FK()

# the dictionary below contains the data returned by calling arm.joint_limits()
limits = [
    {'lower': -2.8973, 'upper': 2.8973},
    {'lower': -1.7628, 'upper': 1.7628},
    {'lower': -2.8973, 'upper': 2.8973},
    {'lower': -3.0718, 'upper': -0.0698},
    {'lower': -2.8973, 'upper': 2.8973},
    {'lower': -0.0175, 'upper': 3.7525},
    {'lower': -2.8973, 'upper': 2.8973}
 ]
# Number of samples for the workspace visualization
num_samples = 20000

# Create arrays to store the x, y, z coordinates of the end-effector
x_vals = []
y_vals = []
z_vals = []
# Sample random joint angles within the limits
for _ in range(num_samples):
    # Randomly sample each joint angle within its limits
    joint_angles = np.array([np.random.uniform(limit['lower'], limit['upper']) for limit in limits])
    
    # Use the FK solver to get the end-effector position (for a given set of joint angles)
    pos,_ = fk.forward(joint_angles)  # pos should be a 3D position (x, y, z)
        # Append the x, y, z coordinates

    x_vals.append(pos[-1][0])
    y_vals.append(pos[-1][1])
    z_vals.append(pos[-1][2])
        

# Create a 3D plot of the reachable workspace
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the sampled end-effector positions
ax.scatter(x_vals, y_vals, z_vals, s=1, c='blue', alpha=0.1)

# Label axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set equal aspect ratio for better visualization
ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1

plt.show()
