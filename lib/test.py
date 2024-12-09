import numpy as np

def reorient_top_face(pose):
    """
    Adjust a pose so that the "top face" of the cube points in the +z direction.
    
    Parameters:
        pose (numpy.ndarray): A 4x4 homogeneous transformation matrix.
        
    Returns:
        numpy.ndarray: A 4x4 homogeneous transformation matrix with the top face aligned to +z.
    """
    # Extract rotation matrix and translation vector
    R_detected = pose[:3, :3]
    t_detected = pose[:3, 3]

    # Check which column is [0, 0, 1] or [0, 0, -1]
    for i in range(3):
        col = R_detected[:, i]
        if np.allclose(col, [0, 0, 1], atol=1e-6):
            top_face_col = i
            flip = False
            break
        elif np.allclose(col, [0, 0, -1], atol=1e-6):
            top_face_col = i
            flip = True
            break
    else:
        raise ValueError("No column aligns with the top face direction [0, 0, ±1].")

    # If the top face is already in the third column and correctly oriented, return the pose
    if top_face_col == 2 and not flip:
        return pose

    # Construct a permutation matrix to swap columns
    R_swap = np.eye(3)
    R_swap[:, [2, top_face_col]] = R_swap[:, [top_face_col, 2]]  # Swap the third column with the top_face_col

    # If the top face points to -z, flip the orientation
    if flip:
        R_swap[:, 2] *= -1

    # Adjust the rotation matrix
    R_corrected = np.dot(R_detected, R_swap)

    # Construct the corrected pose
    pose_corrected = np.eye(4)
    pose_corrected[:3, :3] = R_corrected
    pose_corrected[:3, 3] = t_detected  # Keep the translation unchanged

    return pose_corrected

# Example usage
pose_detected = np.array([
    [0, 1, 0, 1],  # Top face points to +z (already aligned)
    [0, 0, 1, 2],
    [1, 0, 0, 3],
    [0, 0, 0, 1]
])

pose_corrected = reorient_top_face(pose_detected)

print("Corrected Pose:")
print(pose_corrected)
