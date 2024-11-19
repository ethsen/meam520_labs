
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection



import numpy as np



def plotAttractiveVector(ax, target, current, force, obstacles):
    """
    Plots the attractive force vector between the current joint position and
    the target joint position, along with obstacles.
    
    INPUTS:
    target - nx3 numpy array representing the desired joint/end effector positions 
             in the world frame (n joints or end-effectors)
    current - nx3 numpy array representing the current joint/end effector positions 
              in the world frame
    force - nx3 numpy array representing the force vectors on each joint/end effector
    obstacles - mx6 numpy array representing the obstacle box min and max positions
                in the world frame (m obstacles)
    """
    ax.cla()

    
    # Plot each obstacle
    for obs in obstacles:
        x_min, y_min, z_min, x_max, y_max, z_max = obs
        vertices = np.array([
            [x_min, y_min, z_min],
            [x_max, y_min, z_min],
            [x_max, y_max, z_min],
            [x_min, y_max, z_min],
            [x_min, y_min, z_max],
            [x_max, y_min, z_max],
            [x_max, y_max, z_max],
            [x_min, y_max, z_max]
        ])
        faces = [
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # Bottom
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # Top
            [vertices[0], vertices[3], vertices[7], vertices[4]],  # Left
            [vertices[1], vertices[2], vertices[6], vertices[5]],  # Right
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # Front
            [vertices[4], vertices[5], vertices[6], vertices[7]]   # Back
        ]
        poly3d = [[tuple(vertex) for vertex in face] for face in faces]
        ax.add_collection3d(Poly3DCollection(poly3d, facecolors='lightblue', linewidths=1, edgecolors='black', alpha=0.6))
    
    # Plot the target and current positions
    ax.plot(target[:, 0], target[:, 1], target[:, 2], c='red', label='Target Path')
    ax.scatter(target[:, 0], target[:, 1], target[:, 2], c='red', s=50, label='Target Positions')
    
    ax.plot(current[:, 0], current[:, 1], current[:, 2], c='blue', label='Current Path')
    ax.scatter(current[:, 0], current[:, 1], current[:, 2], c='blue', s=50, label='Current Positions')
    
    # Plot the force vectors
    ax.quiver(
        current[:, 0], current[:, 1], current[:, 2],  # Start point of the vector
        force[:, 0], force[:, 1], force[:, 2],       # Components of the vector
        color='green', label='Force Vectors'
    )
    
    # Set labels and legend
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    
    # Add a grid and set the aspect ratio
    ax.grid(True)
    ax.set_box_aspect([1, 1, 1])  # Equal scaling
    
    plt.draw()
    plt.pause(0.1)

def plotTorqueVectors(ax, current_joints, goal_joints, torques, obstacles):
    """
    Plots the torque/movement vectors at each joint frame along with the goal configuration.
    
    INPUTS:
    ax - matplotlib 3D axis
    current_joints - nx3 numpy array of current joint positions
    goal_joints - nx3 numpy array of goal joint positions
    torques - 1x7 numpy array of joint torques
    obstacles - mx6 numpy array of obstacle positions
    """
    ax.cla()
    
    # Plot obstacles
    for obs in obstacles:
        x_min, y_min, z_min, x_max, y_max, z_max = obs
        vertices = np.array([
            [x_min, y_min, z_min], [x_max, y_min, z_min],
            [x_max, y_max, z_min], [x_min, y_max, z_min],
            [x_min, y_min, z_max], [x_max, y_min, z_max],
            [x_max, y_max, z_max], [x_min, y_max, z_max]
        ])
        
        faces = [
            [vertices[0], vertices[1], vertices[5], vertices[4]],
            [vertices[2], vertices[3], vertices[7], vertices[6]],
            [vertices[0], vertices[3], vertices[7], vertices[4]],
            [vertices[1], vertices[2], vertices[6], vertices[5]],
            [vertices[0], vertices[1], vertices[2], vertices[3]],
            [vertices[4], vertices[5], vertices[6], vertices[7]]
        ]
        
        poly3d = [[tuple(vertex) for vertex in face] for face in faces]
        ax.add_collection3d(Poly3DCollection(poly3d, facecolors='lightblue', 
                                           linewidths=1, edgecolors='black', alpha=0.3))
    
    # Plot current robot configuration (solid blue)
    ax.plot(current_joints[:, 0], current_joints[:, 1], current_joints[:, 2], 
            'b-', linewidth=2, label='Current Config')
    ax.scatter(current_joints[:, 0], current_joints[:, 1], current_joints[:, 2], 
              c='blue', s=50)
    
    # Plot goal configuration (dashed red)
    ax.plot(goal_joints[:, 0], goal_joints[:, 1], goal_joints[:, 2], 
            'r--', linewidth=1, label='Goal Config')
    ax.scatter(goal_joints[:, 0], goal_joints[:, 1], goal_joints[:, 2], 
              c='red', s=30, alpha=0.5)
    
    # Scale factor for torque vectors
    max_torque = np.max(np.abs(torques))
    scale = 0.1 / max_torque if max_torque > 0 else 0.1
    
    # Plot torque vectors at each joint
    for i in range(len(torques)):
        # Skip end effector joints
        if i >= 7:
            continue
            
        # Get current joint position
        pos = current_joints[i+1]
        
        # Calculate vector direction based on torque
        # We'll use the torque magnitude to scale the vector
        torque_mag = torques[i] * scale
        
        """
        # Draw coordinate frame at joint
        frame_size = 0.25
        # X axis (red)
        ax.quiver(pos[0], pos[1], pos[2], 
                 frame_size, 0, 0, 
                 color='red', alpha=0.5)
        # Y axis (green)
        ax.quiver(pos[0], pos[1], pos[2], 
                 0, frame_size, 0, 
                 color='green', alpha=0.5)
        # Z axis (blue) - rotation axis
        ax.quiver(pos[0], pos[1], pos[2], 
                 0, 0, frame_size, 
                 color='blue', alpha=0.5)
        """
        # Draw torque vector (purple)
        if abs(torque_mag) > 1e-6:  # Only draw if torque is non-zero
            ax.quiver(pos[0], pos[1], pos[2],
                     0, 0, torque_mag,  # Assuming rotation around z-axis
                     color='purple', alpha=0.8,
                     linewidth=2)
            
            
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Joint Torques and Goal Configuration')
    
    # Add legend
    ax.legend()
    
    # Set consistent view limits
    #ax.set_xlim([-1, 1])
    #ax.set_ylim([-1, 1])
    #ax.set_zlim([0, 2])
    
    # Equal aspect ratio
    ax.set_box_aspect([1,1,1])
    
    plt.draw()
    plt.pause(0.1)


def plotJacobianCalculation(ax, current_joints, T0eCol, joint_index):
    """
    Visualizes the Linear Velocity Jacobian calculation process for a specific joint.
    Shows the z-axis used (jw) and the origin differences being calculated.
    
    INPUTS:
    ax - matplotlib 3D axis
    current_joints - nx3 numpy array of current joint positions
    T0eCol - list of transformation matrices for each joint
    joint_index - the joint we're currently calculating the Jacobian for
    """
    ax.cla()
    
    # Plot robot arm configuration
    ax.plot(current_joints[:, 0], current_joints[:, 1], current_joints[:, 2], 
            'b-', linewidth=2, label='Robot Arm')
    ax.scatter(current_joints[:, 0], current_joints[:, 1], current_joints[:, 2], 
              c='blue', s=50)
    
    # Plot frame for joint we're calculating Jacobian for (joint_index + 1)
    target_pos = current_joints[joint_index]
    ax.scatter(target_pos[0], target_pos[1], target_pos[2], 
              c='red', s=100, label=f'Target Joint {joint_index}')
    
    # For each joint up to joint_index, show its contribution to the Jacobian
    j= 0
    while j != joint_index:
        # Get current joint position
        joint_pos = current_joints[j]

        # Get z-axis (jw) from transformation matrix
        jw = T0eCol[j][:3, 2]  # z-axis of current joint frame
        
        # Calculate origin difference
        origin_diff = current_joints[joint_index] - current_joints[j]
        
        # Plot z-axis (jw) at joint j
        z_axis_length = 0.1
        ax.quiver(joint_pos[0], joint_pos[1], joint_pos[2],
                 jw[0] * z_axis_length, jw[1] * z_axis_length, jw[2] * z_axis_length,
                 color='black', alpha=0.7, linewidth=2,
                 label=f'z{j} axis' if j == 0 else f'z{j} axis')
        
        # Plot origin difference vector
        ax.quiver(joint_pos[0], joint_pos[1], joint_pos[2],
                 origin_diff[0], origin_diff[1], origin_diff[2],
                 color='green', alpha=0.5, linewidth=1,
                 label=f'Origin diff {j}->{joint_index}' if j == 0 else f'Origin diff {j}->{joint_index}')
        
        # Plot resulting cross product (Jacobian column)
        """
        cross_product = np.cross(jw, origin_diff)
        cross_product_normalized = cross_product / np.linalg.norm(cross_product) * 0.1
        ax.quiver(joint_pos[0], joint_pos[1], joint_pos[2],
                 cross_product_normalized[0], cross_product_normalized[1], cross_product_normalized[2],
                 color='red', alpha=0.8, linewidth=2,
                 label=f'J{j} column' if j == 0 else f'J{j} column')
        """
        # Add text annotations
        ax.text(joint_pos[0], joint_pos[1], joint_pos[2] + 0.1,
                f'Joint {j}', color='black')
        j+=1
    # Set labels and title
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title(f'Linear Velocity Jacobian Calculation for Joint {joint_index + 1}')
    
    # Set consistent view limits
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([0, 2])
    
    # Add legend
    ax.legend()
    
    # Equal aspect ratio
    ax.set_box_aspect([1,1,1])
    
    plt.draw()
    plt.pause(1.0)  # Pause to see the calculation for each joint
    while True:
        if plt.waitforbuttonpress():
            break