import time
import copy
import open3d as o3d

from tools import metrics

def icp_reg_and_eval(source, target, method, max_correspondence_distance, init_transformation, gt_transformation):
    """
    Perform registration and evaluation for a given ICP-based method.

    Args:
        source (open3d.geometry.PointCloud): The source point cloud.
        target (open3d.geometry.PointCloud): The target point cloud.
        method (str): The registration method ('p2point', 'p2plane', 'robust', 'gicp', 'fgr').
        max_correspondence_distance (float): The maximum correspondence distance.
        init_transformation (np.ndarray): The initial transformation matrix.
        gt_transformation (np.ndarray): The ground truth transformation matrix.
        
    Returns:
        result (np.ndarray): The evaluation results (rmse, rotation_error, translation_error, computation_time).
    """
    if method == 'p2point':
        start_time = time.time()
        result = o3d.pipelines.registration.registration_icp(
            source, target, max_correspondence_distance, init_transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        end_time = time.time()
        computation_time = end_time - start_time

    elif method == 'p2plane':
        source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        start_time = time.time()
        result = o3d.pipelines.registration.registration_icp(
            source, target, max_correspondence_distance, init_transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        end_time = time.time()
        computation_time = end_time - start_time
    elif method == 'robust':
        loss = o3d.pipelines.registration.TukeyLoss(k=0.5)
        start_time = time.time()
        result = o3d.pipelines.registration.registration_icp(
            source, target, max_correspondence_distance, init_transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(loss))
        end_time = time.time()
        computation_time = end_time - start_time
    elif method == 'gicp':
        # Define convergence criteria
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=50)  #default: relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=30
        estimation_method = o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(epsilon=0.001)
        
        start_time = time.time()
        result = o3d.pipelines.registration.registration_generalized_icp(
            source, target, max_correspondence_distance, init_transformation,estimation_method, criteria)
        end_time = time.time()
        computation_time = end_time - start_time
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Apply transformation
    pc_result = copy.deepcopy(source).transform(result.transformation)

    # Evaluation
    evaluation_results = metrics.all_evaluations(source, target, pc_result, computation_time, gt_transformation, result, corres= None)
    
    return pc_result, evaluation_results