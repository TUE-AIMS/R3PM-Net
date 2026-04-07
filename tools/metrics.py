
import open3d as o3d
import torch
import numpy as np
import matplotlib.pyplot as plt
from math import atan2
from scipy.spatial.transform import Rotation as R
import warnings


def get_rotaion(est):
    # Extract the rotation matrices
    try:
        estimated_rotation = est[:3, :3]
    except TypeError:
        try:
            estimation_transformation = est.transformation # for open3d
        except AttributeError:
            try: 
                estimation_transformation = est.rot  # for Probreg
            except: # for learning 3d
                detached_est = est['est_T'].detach().cpu().numpy()[0]
                estimation_transformation = detached_est.reshape(4,4)
                
        estimated_rotation = estimation_transformation[:3, :3]

    return estimated_rotation

def get_translation(est):
    # Extract the translation vectors
    try:
        estimated_translation = est[:3, 3]
    except TypeError:
        try:
            estimated_translation = est.t  # for Probreg
        except AttributeError:
            try: 
                estimation_transformation = est.transformation # for open3d
            except: # for learning 3d
                detached_est = est['est_T'].detach().cpu().numpy()[0]
                estimation_transformation = detached_est.reshape(4,4)
            
            estimated_translation = estimation_transformation[:3, 3]

    return estimated_translation

def compute_rmse(result, target, corres):
    '''
    This function computes the root-mean-square error (RMSE) between the (transforemed) source and target point clouds based on the correspondences.
    Based on C++ code in Open3D https://github.com/isl-org/Open3D/blob/c8856fc0d4ec89f8d53591db245fd29ad946f9cb/cpp/open3d/pipelines/registration/TransformationEstimation.cpp#L20

    args:
        result (point cloud data): the transformed point cloud
        target (point cloud data): the target point cloud
        corres (Vector2iVector): the point-to-point correspondences between the source and target point clouds
    returns:
        rmse: the root-mean-square error (centimeters or meters based on the source and target point clouds)
    '''
    err = 0.0
    for c in corres:
        diff = np.asarray(result.points)[c[0]] - np.asarray(target.points)[c[1]] # Euclidean distance
        err += np.sum(diff**2)  # sum of squared distances
    rmse = np.sqrt(err / len(corres))  
    
    return rmse

def rotation_error(gt_transformation, est):
    ''' 
    This function computes the rotation error as the Geodesic distance between the estimated and the ground truth rotation matrices in 3D space.
    Based on formula on this page http://www.boris-belousov.net/2016/12/01/quat-dist/

    args:
        gt_transformation: the ground truth transformation (rotation matrix will be extracted from this)
        est: the estimated transformation (rotation matrix will be extracted from this)

    returns:
        rotation_error_deg: the rotation error (degrees)
    '''
    estimated_rotation = get_rotaion(est)
    ground_truth_rotation = gt_transformation[:3, :3]

    # Compute the angular distance between the two rotation matrices
    R = np.dot(ground_truth_rotation, estimated_rotation.T)  # Compute the relative rotation matrix (np.matmul(gt.T, est))       
    trace = np.trace(R)
    normalized_trace = min(max(((trace - 1) / 2), -1.0), 1.0) # Normalize and clamp the trace to avoid numerical errors
    theta = np.arccos(normalized_trace) 
    rotation_error = abs(theta)
    rotation_error_deg = np.rad2deg(rotation_error) # Convert to degrees

    return rotation_error_deg

def translation_error(gt_transformation, est):
    ''' 
    This function computes the translation error as the Euclidean distance between Ground Truth & Estimated translation.
    Based on definitions in this paper https://arxiv.org/pdf/2103.02690

    args:
        gt_transformation: the ground truth transformation (translation vector will be extracted from this)
        est: the estimated transformation (translation vector will be extracted from this)

    returns:
        abs(translation_error): the absolute translation error (centimeters)
    '''
    estimated_translation = get_translation(est)
    ground_truth_translation = gt_transformation[:3, 3]

    # Compute the translation error 
    translation_error = np.linalg.norm(estimated_translation - ground_truth_translation)

    return abs(translation_error)

def residual_error(cloud1: o3d.geometry.PointCloud, cloud2: o3d.geometry.PointCloud) -> float:
    '''
    This metric combines the rotation and translation errors so that different approaches can be compared.
    It uses root mean squared distance between homologous points of the source point cloud, after the execution of an algorithm, and the same point cloud at the ground truth pose.
    Based on papers: 
        https://www.sciencedirect.com/science/article/pii/S0921889021000191?via=ihub (3.3. Error metric) (main paper) -> https://github.com/iralabdisco/point_clouds_registration_benchmark/tree/master
        https://link.springer.com/article/10.1007/s11370-024-00562-1 [14]
    '''
    if len(cloud1.points) != len(cloud2.points):
        if len(cloud1.points) > len(cloud2.points):
            cloud1.points = cloud1.points[:len(cloud2.points)]
        else:
            cloud2.points = cloud2.points[:len(cloud1.points)]
    
    assert len(cloud1.points) == len(cloud2.points), "len(cloud1.points) != len(cloud2.points)"
    
    centroid, _ = cloud1.compute_mean_and_covariance()
    weights = np.linalg.norm(np.asarray(cloud1.points) - centroid, 2, axis=1)
    distances = np.linalg.norm(np.asarray(cloud1.points) - np.asarray(cloud2.points), 2, axis=1)/len(weights)
    return np.sum(distances/weights)

def chamfer_distance(pcd1, pcd2):
    '''
    Computes the Chamfer Distance between two point clouds using Open3D and PyTorch.

    The Chamfer Distance is a metric that measures the similarity between two point clouds.
    It is calculated as the sum of the mean squared distances from each point in one point cloud
    to its nearest neighbor in the other point cloud, in both directions.

    Args:
        pcd1 (o3d.geometry.PointCloud): The first point cloud.
        pcd2 (o3d.geometry.PointCloud): The second point cloud.

    Returns:
        chamfer_distance (float): The Chamfer distance between the two point clouds.
    '''
    dist1 = pcd1.compute_point_cloud_distance(pcd2)
    dist2 = pcd2.compute_point_cloud_distance(pcd1)
    dist1 = torch.tensor(np.asarray(dist1), dtype=torch.float32)
    dist2 = torch.tensor(np.asarray(dist2), dtype=torch.float32)
    
    chamfer_distance = torch.mean(dist1) + torch.mean(dist2)
    chamfer_distance = chamfer_distance.item()

    return chamfer_distance


def all_evaluations(source, target, result, time, gt_transformation = None, est_transformation = None, corres = None):

    cd = chamfer_distance(result, target)
    error = residual_error(target, result)
    computation_time = time

    max_treshold = 0.5
    inlier_rmse = o3d.pipelines.registration.evaluate_registration(result, target, max_treshold, np.eye(4)).inlier_rmse
    fitness = o3d.pipelines.registration.evaluate_registration(result, target, max_treshold, np.eye(4)).fitness

    if gt_transformation is not None:
        rmse = 1
        rotation_err = rotation_error(gt_transformation, est_transformation)
        translation_err = translation_error(gt_transformation, est_transformation)

        return rmse, rotation_err, translation_err, computation_time, cd, error, fitness, inlier_rmse #8
        
    else:
        return cd, fitness, inlier_rmse, computation_time #4
    

def summerize_results(results: np.ndarray) -> dict:
    if results.shape[2] == 8:
        mean_rmse = np.round(np.mean(results[:, :, 0]), 4)
        mean_rotation_error = np.round(np.mean(results[:, :, 1]), 4)
        mean_translation_error = np.round(np.mean(results[:, :, 2]), 4)
        mean_computation_time = np.round(np.mean(results[:, :, 3]), 4)
        mean_cd = np.round(np.mean(results[:, :, 4]), 4)
        mean_error = np.round(np.mean(results[:, :, 5]), 4)
        mean_fitness = np.round(np.mean(results[:, :, 6]), 4)
        mean_inlier_rmse = np.round(np.mean(results[:, :, 7]), 4)
        return {
            'mean_rmse': mean_rmse,
            'mean_rotation_error': mean_rotation_error,
            'mean_translation_error': mean_translation_error,
            'mean_computation_time': mean_computation_time,
            'mean_cd': mean_cd,
            'mean_error': mean_error,
            'mean_fitness': mean_fitness,
            'mean_inlier_rmse': mean_inlier_rmse
        }
    elif results.shape[2] == 5:
        mean_cd = np.round(np.mean(results[:, :, 0]), 4)
        mean_error = np.round(np.mean(results[:, :, 1]), 4)
        mean_fitness = np.round(np.mean(results[:, :, 2]), 4)
        mean_inlier_rmse = np.round(np.mean(results[:, :, 3]), 4)
        mean_computation_time = np.round(np.mean(results[:, :, 4]), 4)
        return {
            'mean_cd': mean_cd,
            'mean_error': mean_error,
            'mean_fitness': mean_fitness,
            'mean_inlier_rmse': mean_inlier_rmse,
            'mean_computation_time': mean_computation_time
        }
    else:   
        raise ValueError('Invalid results shape. Expected shape (N, M, 8) or (N, M, 5).')


def inlier_ratio(pcd, all_errors, threshold = 5):
    '''
    This function calculates the inlier ratio based on a given threshold.
    
    args:
        pcd (point cloud data): the point cloud
        all_errors (list): the errors between the two point clouds (from error_histogram)
        threshold (float): the threshold for inliers
        
    returns:
        inlier_ratio (float): the ratio of inliers (set at 5 cm)
        '''
    inliers = []
    for error in all_errors:
        # if the error is below the threshold, the pair is an inlier
        if error < threshold:
            inliers.append(error)
    inlier_ratio = len(inliers) / len(pcd.points)

    return inlier_ratio


def calculate_snr(clean_pcd, noisy_pcd):
    '''
    This function calculates the signal-to-noise ratio (SNR) between two point clouds.
    SNR is defined as the ratio of the RMS power of the signal (clean point cloud) to the RMS power of the noise (difference between clean and noisy point clouds).

    args:
        clean_pcd (point cloud): the clean point cloud
        noisy_pcd (point cloud): the noisy point cloud

    returns:
        snr_db (float): the signal-to-noise ratio in decibels
    '''
    # Convert point clouds to numpy arrays
    clean_points = np.asarray(clean_pcd.points)
    noisy_points = np.asarray(noisy_pcd.points)

    # Calculate the RMS power of the signal (clean point cloud)
    signal_amplitude = np.sqrt(np.mean(np.sum(clean_points**2, axis=1)))

    # Calculate the RMS power of the noise (difference between clean and noisy point clouds)
    noise_amplitude = np.sqrt(np.mean(np.sum((clean_points - noisy_points)**2, axis=1)))

    # Compute the SNR
    snr = (signal_amplitude / noise_amplitude)**2
    snr_db = 10 * np.log10(snr)

    return snr_db

def rotation_error_along_axis(gt_transformation, est_transformation, convention = 'zyx', verbose = False):
    '''
    This function calculates the rotation error along the each axis (Euler angle) between the estimated and ground truth rotation matrices.
    It gives a warning if gimbal lock is detected (90 or 270 degrees).

    Based on formulas and discussions in:
        https://www.youtube.com/watch?v=wg9bI8-Qx2Q (10:29)
    Args:
        gt_transformation: the ground truth transformation
        est_transformation: the estimated transformation
        convention: the convention for the Euler angles (default and only one supported is 'zyx')
        verbose: if True, the Euler angles are printed

    Returns:
        theta_x_deg: the rotation error along the x-axis (degrees) rounded to 3 decimal places
        theta_y_deg: the rotation error along the y-axis (degrees) rounded to 3 decimal places
        theta_z_deg: the rotation error along the z-axis (degrees) rounded to 3 decimal places
    '''
    if convention != 'zyx':
        raise ValueError("Invalid convention. Only 'zyx' is supported.")
    
    estimated_rotation = get_rotaion(est_transformation)
    ground_truth_rotation = gt_transformation[:3, :3]

    R_relative = np.dot(ground_truth_rotation, estimated_rotation.T) 

    theta_z = atan2(R_relative[1, 0], R_relative[0, 0]) # Euler angles = atan2(r21, r11)
    theta_x = atan2(R_relative[2, 1], R_relative[2, 2]) # Euler angles = atan2(r32, r33)

    if np.isclose(np.cos(theta_z), 0.0, atol=1e-6):
        second_term = R_relative[1, 0]/np.sin(theta_z)
        theta_y = atan2(-R_relative[2, 0], second_term)
    else:
        second_term = R_relative[0, 0]/np.cos(theta_z) 
        theta_y = atan2(-R_relative[2, 0], second_term)

    # Convert to degrees
    theta_x_deg = np.round(abs(np.rad2deg(theta_x)), 3)
    theta_y_deg = np.round(abs(np.rad2deg(theta_y)), 3)
    theta_z_deg = np.round(abs(np.rad2deg(theta_z)), 3)

    if verbose:
        if np.round(theta_y_deg, 3) == 90 or np.round(theta_y_deg, 3) == 270:
            print("Warning: Gimbal lock detected! It might not be possible to uniquely and accurately determine all angles.")

        print(f'{theta_x_deg}° error along x-axis, {theta_y_deg}° error along y-axis and {theta_z_deg}° error along z-axis.')

    return theta_x_deg, theta_y_deg, theta_z_deg

def angle_diff(a, b):
    '''
    This function calculates the smallest unsigned angle difference in degrees.
    '''
    diff = abs(a - b) % 360
    return min(diff, 360 - diff)

def signed_angle_diff(a, b):
    '''
    This function calculates the signed angle difference in degrees.
    '''
    diff = (b - a + 180) % 360 - 180
    return diff

def decompose_rotation_error(gt_transformation, est_transformation, signed = True, convention = 'zyx', verbose = False):
    '''
    This fuction calculates Euler angles (rotation) differences between the estimated and ground truth transformation matrices.
    It uses the scipy.spatial.transform.Rotation class to extract the Euler angles from the rotation matrices.
    If Gimbal lock is detected, it uses a manual calculation of the angles.

    Args:
        gt_transformation: the ground truth transformation matrix
        est: the estimated transformation matrix
        signed: if True, the signed angle difference is calculated
        convention: the convention for the Euler angles (default and only supported is 'zyx')
                    This restriction is because the rotation matrix applied to simulate data is 'xyz' order which is the inverse of 'zyx' - because initrinsic and extrinsic rotations are inverted. Source: https://dominicplein.medium.com/extrinsic-intrinsic-rotation-do-i-multiply-from-right-or-left-357c38c1abfd
        verbose: if True, the Euler angles are printed
    Returns:
        x_diff: the difference in the x-axis rotation (degrees)
        y_diff: the difference in the y-axis rotation (degrees)
        z_diff: the difference in the z-axis rotation (degrees)

        OR if Gimbal lock is detected:
        re_along_x: the rotation error along the x-axis (degrees)
        re_along_y: the rotation error along the y-axis (degrees)
        re_along_z: the rotation error along the z-axis (degrees)
    
    Raises:
        RuntimeError: if Gimbal lock is detected.
    '''
    if convention != 'zyx':
        raise ValueError("Invalid convention. Only 'zyx' is supported.")
    
    # Get rotation matrices
    estimated_rotation = get_rotaion(est_transformation)
    gt_rotation = gt_transformation[:3, :3]

    # Get Euler degrees using scipy
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            gt_euler = R.from_matrix(gt_rotation).as_euler(convention, degrees=True)
            gt_z, gt_y, gt_x = np.round(gt_euler, 3)
            est_euler = R.from_matrix(estimated_rotation).as_euler(convention, degrees=True)
            est_z, est_y, est_x = np.round(est_euler, 3)

            # Check for Gimbal lock warning
            gimal_lock_warning = any("Gimbal lock detected" in str(warn.message) for warn in w)
            if gimal_lock_warning:
                raise RuntimeError("Gimbal lock detected!")

        if verbose:
            print(f"Extracted Ground Truth Euler angles (deg): x={gt_x}, y={gt_y}, z={gt_z}")
            print(f"Extracted Estimated Euler angles (deg): x={est_x}, y={est_y}, z={est_z}")
    
        # compute differences
        diff_fn = signed_angle_diff if signed else angle_diff
        x_diff = diff_fn(gt_x, est_x)  
        y_diff = diff_fn(gt_y, est_y)  
        z_diff = diff_fn(gt_z, est_z)

    # Use manual calculation of angles if Gimbal lock is detected
    except RuntimeError as e:
        print(f'{e} - Trying manual calculation of angles.') if verbose else None
        x_diff, y_diff, z_diff = rotation_error_along_axis(gt_transformation, est_transformation, convention)

    if verbose:
        print(f'{x_diff}° error along x-axis, {y_diff}° error along y-axis and {z_diff}° error along z-axis.')

    return x_diff, y_diff, z_diff


# --------- UNUSED FUNCTIONS ---------
def error_histogram(source, target, result, dis_type ='mse', bins = 200):
    '''
    This function creates a histogram of the errors between the target and result point clouds.

    args:
        target: the target point cloud
        result: the result point cloud
        dis_type: the type of distance to calculate the error (mse or mae)
        bins: the number of bins for the histogram
    
    returns:
        histogram (Plot): the histogram plot of the errors 
        all_errors (list): the errors between the source and target point clouds (for inlier ratio calculation)
    '''
    # Calcualte error between the correspondences
    all_errors = [] 

    for i in range(len(source.points)):
        if dis_type =='mse':
            # Error as root mean square error
            error = np.sqrt(np.sum((np.asarray(result.points[i]) - np.asarray(target.points[i]))**2))
        elif dis_type == 'mae':
            # Error as mean absolute error
            error = np.mean(np.abs(np.asarray(result.points[i]) - np.asarray(target.points[i])))
                
        all_errors.append(error)

    # Make a histogram of the errors
    plt.hist(all_errors, bins = bins)
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.show()

    return all_errors


def rotation_error_along_z (gt_transformation, est, symmetry = None):
    '''
    This function calculates the rotation error along the z-axis (Euler angle) between the estimated and ground truth rotation matrices.
    Based on formulas and discussions in:
        https://math.stackexchange.com/questions/31001/finding-the-cos-angle-between-two-matrices-using-the-euclidean-inner-product
        https://stackoverflow.com/questions/15022630/how-to-calculate-the-angle-from-rotation-matrix
        https://www.youtube.com/watch?v=wg9bI8-Qx2Q (10:29)

    For C2 and C4 symmetries, the explanation is here:
        https://www2.math.upenn.edu/~mlazar/math170/notes07-4.pdf
        https://web.stanford.edu/~kaleeg/chem32/groupT/
    '''
    estimated_rotation = get_rotaion(est)
    ground_truth_rotation = gt_transformation[:3, :3]

    R = np.dot(ground_truth_rotation, estimated_rotation.T) 
    theta_z = atan2(R[1, 0], R[0, 0]) # Euler angles 
    theta_z_deg = abs(np.rad2deg(theta_z))

    if symmetry == 'C2':
        theta_z_deg = min(theta_z_deg, abs(180 - theta_z_deg))
    elif symmetry == 'C4':
        theta_z_deg = min(theta_z_deg, abs(90 - theta_z_deg), abs(180 - theta_z_deg), abs(270 - theta_z_deg))

    return np.round(theta_z_deg, 3)

def rotation_error_along_x (gt_transformation, est):
    '''
    This function calculates the rotation error along the x-axis (Euler angle) between the estimated and ground truth rotation matrices.
    Based on formulas and discussions in:
        https://www.youtube.com/watch?v=wg9bI8-Qx2Q (10:29)

    Args:
        est: the estimated transformation
        gt_transformation: the ground truth transformation
    Returns:
        theta_x_deg: the rotation error along the x-axis (degrees) rounded to 3 decimal places
    '''
    estimated_rotation = get_rotaion(est)
    ground_truth_rotation = gt_transformation[:3, :3]

    R = np.dot(ground_truth_rotation, estimated_rotation.T) 
    theta_x = atan2(R[2, 1], R[2, 2]) # Euler angles = atan2(r32, r33)
    theta_x_deg = abs(np.rad2deg(theta_x))

    return np.round(theta_x_deg, 3)

def rotation_error_along_y (gt_transformation, est, theta_z_deg):
    '''
    This function calculates the rotation error along the y-axis (Euler angle) between the estimated and ground truth rotation matrices.
    It gives a warning if gimbal lock is detected (90 or 270 degrees).
    Based on formulas and discussions in:
        https://www.youtube.com/watch?v=wg9bI8-Qx2Q
    
    Args:
        est: the estimated transformation
        gt_transformation: the ground truth transformation
        theta_z_deg: the rotation error along the z-axis (degrees)
    Returns:
        theta_y_deg: the rotation error along the y-axis (degrees) rounded to 3 decimal places
    '''
   
    estimated_rotation = get_rotaion(est)
    ground_truth_rotation = gt_transformation[:3, :3]

    R = np.dot(ground_truth_rotation, estimated_rotation.T) 
    if np.cos(np.deg2rad(theta_z_deg)) == 0:
        second_term = R[1, 0]/np.sin(np.deg2rad(theta_z_deg))
        theta_y = atan2(-R[2, 0], second_term)
    else:
        second_term = R[0, 0]/np.cos(np.deg2rad(theta_z_deg)) 
        theta_y = atan2(-R[2, 0], second_term)

    theta_y_deg = abs(np.rad2deg(theta_y))

    if np.round(theta_y_deg, 3) == 90 or np.round(theta_y_deg, 3) == 270:
        print("Warning: Gimbal lock detected! It might not be possible to uniquely and accurately determine all angles.")

    return np.round(theta_y_deg, 3)

# Construct the transformation matrix from the rotation matrix, translation vector, and scale (for Probreg)
def reconstruct_transformation_propreg(rot, t, scale):
    scaled_rotation = scale * rot
    T = np.eye(4)
    T[:3, :3] = scaled_rotation
    T[:3, 3] = t   
    return T

def get_transformation(est):
    # Extract the transformation matrices
    try:
        estimation_transformation = est.transformation # for open3d
    except AttributeError:
        try: 
            estimation_transformation = reconstruct_transformation_propreg(est.rot, est.t, est.scale) # for Probreg
        except: # for learning 3d
            estimation_transformation = est['est_T'].detach().cpu().numpy()[0]
            estimation_transformation = estimation_transformation.reshape(4,4)

    return estimation_transformation

def transformation_error(est, gt_transformation):  # ATTENTION: not a good metric because transformations consist of rotation and translation, which have different metrics (one is radian, the other centimeters).
    ''' 
    This function calculates transformation error as the root-mean-square error between estimated transformation and ground truth transformation 
    Based on definitions in this paper https://arxiv.org/pdf/2103.02690
    
    args:
        est: the estimation object
        gt_transformation: the ground truth transformation

    returns:
        transformation_error: the transformation error
    '''
    estimation_transformation = get_transformation(est)
    rmseT = np.sqrt(np.mean(np.square(estimation_transformation - gt_transformation)))
    
    return rmseT


def remove_outliers(data):
    '''
    This function removes outliers from the data based on the interquartile range (IQR).
    Based on https://medium.com/@davidnh8/outlier-detection-101-median-and-interquartile-range-cc9dde94c0ac

    Args:
        data (np.array): The data to remove outliers from.
    Returns:
        clear_data (np.array): The data without outliers.
        outlier_index (np.array): The indices of the outliers.
    '''
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outlier_index = np.where((data <= lower_bound) | (data >= upper_bound))
    clear_data = np.delete(data, outlier_index)
    return clear_data, outlier_index


def get_overlap_ratio(source,target,threshold):
    """
    - Overlap is defined as the ratio of the number of points in each point cloud that cover a region of the scene, 
    which is also covered by the other point cloud, to the total number of points in the point cloud.
    
    - Overlap is computed as the ratio of the number of points in the source point cloud that are within a distance 
    threshold to the target point cloud to the total number of points in the source point cloud.
    
    Taken from https://github.com/prs-eth/OverlapPredator/blob/main/scripts/cal_overlap.py
    Based on https://www.open3d.org/docs/latest/tutorial/Basic/kdtree.html
    """
    pcd_tree = o3d.geometry.KDTreeFlann(target)
    
    match_count=0
    for i, point in enumerate(source.points):
        [count, _, _] = pcd_tree.search_radius_vector_3d(point, threshold)
        if(count!=0):
            match_count+=1

    overlap_ratio = match_count / len(source.points) *100
    return overlap_ratio