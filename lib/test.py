import numpy as np
from math import pi
def adjustRotation(pose):
        """
        Adjusts the pose of the detected block in order
        for the end-effector to easily grasp it. 
        
        INPUTS:
        pose - 4x4 matrix of a pose 

        OUPUTS:
        adjPose - 4x4 matrix after adjusting pose 
        """
        rotDetected= pose[:3, :3]
        #print(np.round(rotDetected,4))
        tDetected = pose[:3, 3]
        for i in range(3):
            col = rotDetected[:, i]
            if np.allclose(col, [0, 0, 1], atol=1e-3):
                top_face_col = i
                flip = 1
                break
            elif np.allclose(col, [0, 0, -1], atol=1e-3):
                top_face_col = i
                flip = -1
                break
        else:
            raise ValueError("No column aligns with the top face direction [0, 0, ±1].")

        
        if top_face_col == 0:

            angle = pi/2 * flip
            rotY = np.array([[np.cos(angle),0,np.sin(angle)],
                             [0,1,0],
                             [-np.sin(angle),0,np.cos(angle)]])
            if flip == 1:
                rotY = rotY @ np.array([[-1,0,0],
                                    [0,-1,0],
                                    [0,0,1]])
            rotDetected = rotDetected @ rotY

        elif top_face_col == 1:
            angle = pi/2 * -flip
            rotX = np.array([[1,0,0],
                             [0,np.cos(angle),-np.sin(angle)],
                             [0,np.sin(angle),np.cos(angle)]])
            rotDetected = rotDetected @ rotX

        elif flip == -1:
            rotDetected = rotDetected @ np.array([[1,0,0],
                                                  [0,-1,0],
                                                  [0,0,-1]])
        #print(np.round(rotDetected,4))

        # Construct the corrected pose
        pose_corrected = np.eye(4)
        pose_corrected[:3, :3] = rotDetected
        pose_corrected[:3, 3] = tDetected  # Keep the translation unchanged
        return pose_corrected
def test_adjustRotation():
    # Helper function to create a 4x4 pose matrix
    def create_pose(rotation, translation):
        pose = np.eye(4)
        pose[:3, :3] = rotation
        pose[:3, 3] = translation
        return pose

    # Test cases
    test_cases = [
        {
            "description": "Top face aligned with +Z",
            "input": create_pose(np.array([[1, 0, 0],
                                           [0, 1, 0],
                                           [0, 0, 1]]), [0.1, 0.2, 0.3]),
            "expected_rotation": np.array([[1, 0, 0],
                                           [0, 1, 0],
                                           [0, 0, 1]])
        },
        {
            "description": "Top face aligned with -Z",
            "input": create_pose(np.array([[1, 0, 0],
                                           [0, -1, 0],
                                           [0, 0, -1]]), [0.1, 0.2, 0.3]),
            "expected_rotation": np.array([[1, 0, 0],
                                           [0, 1, 0],
                                           [0, 0, 1]])
        },
        {
            "description": "Top face aligned with -X",
            "input": create_pose(np.array([[0, 0, 1],
                                           [0, 1, 0],
                                           [-1, 0, 0]]), [0.1, 0.2, 0.3]),
            "expected_rotation": np.array([[1, 0, 0],
                                           [0, 1, 0],
                                           [0, 0, 1]])
        },
        {
            "description": "Top face aligned with +X",
            "input": create_pose(np.array([[0, 0, 1],
                                           [0, -1, 0],
                                           [1, 0, 0]]), [0.1, 0.2, 0.3]),
            "expected_rotation": np.array([[1, 0, 0],
                                           [0, 1, 0],
                                           [0, 0, 1]])
        },
        {
            "description": "Top face aligned with -Y",
            "input": create_pose(np.array([[1, 0, 0],
                                           [0, 0, 1],
                                           [0, -1, 0]]), [0.1, 0.2, 0.3]),
            "expected_rotation": np.array([[1, 0, 0],
                                           [0, 1, 0],
                                           [0, 0, 1]])
        },
        {
            "description": "Top face aligned with +Y",
            "input": create_pose(np.array([[1, 0, 0],
                                           [0, 0, -1],
                                           [0, 1, 0]]), [0.1, 0.2, 0.3]),
            "expected_rotation": np.array([[1, 0, 0],
                                           [0, 1, 0],
                                           [0, 0, 1]])
        },
        
        
    ]

    # Run tests
    for i, test in enumerate(test_cases):
        print(f"Running Test {i+1}: {test['description']}")
        adjusted_pose = adjustRotation(test["input"])
        
        # Extract the adjusted rotation matrix
        adjusted_rotation = adjusted_pose[:3,:3]
        
        # Check if the rotation matches the expected rotation
        if np.allclose(adjusted_rotation,
                       test["expected_rotation"], atol=1e-3):
            print(f"Test {i+1} PASSED!")
        else:
            print(f"Test {i+1} FAILED!")
            print("Expected Rotation:")
            print(test["expected_rotation"])
            print("Adjusted Rotation:")
            print(adjusted_rotation)

# Run the test function
test_adjustRotation()