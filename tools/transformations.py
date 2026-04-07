import numpy as np
import open3d as o3d

def validate_rotation(rotation): 
    # Check if the rotation matrix is valid (orthogonal and determinant = 1)
    is_orthogonal = np.allclose(np.dot(rotation[:3, :3], rotation[:3, :3].T), np.eye(3))
    is_determinant_one = np.isclose(np.linalg.det(rotation[:3, :3]), 1)
    return is_orthogonal and is_determinant_one

def random_translation(translation_range):
    return np.random.uniform(translation_range[0], translation_range[1], size=3)


def create_transformation(x_angle, y_angle, z_angle, translation_range):
    '''
    Generate a random transformation matrix with given rotation angles and translation range using open3d methods.

    Args:
        x_angle (float): Rotation angle around the x-axis in degrees.
        y_angle (float): Rotation angle around the y-axis in degrees.
        z_angle (float): Rotation angle around the z-axis in degrees.
        translation_range (list): Range for random translation [min, max].

    Returns:
        np.ndarray: A 4x4 transformation matrix.
    '''
    x_angle = np.deg2rad(x_angle)
    y_angle = np.deg2rad(y_angle)
    z_angle = np.deg2rad(z_angle)
    
    rotation = o3d.geometry.get_rotation_matrix_from_zxy([z_angle, y_angle, x_angle])
    translation = random_translation(translation_range)

    # Create a 4x4 transformation matrix
    random_transformation = np.eye(4) 
    random_transformation[:3, :3] = rotation  
    random_transformation[:3, 3] = translation  
    
    return random_transformation