import open3d as o3
import math
import numpy as np
import torch

def load_point_cloud(file_path: str, as_mesh: bool = False) -> o3.geometry.PointCloud:
    '''
    This function loads a point cloud from a file and returns it as a point cloud object.
    If the as_mesh parameter is set to True, the point cloud is loaded as a mesh object.

    Args:
        file_path (str): The path to the point cloud file
        as_mesh (bool): If True, the point cloud is loaded as a mesh object
    
    Returns:
        pcd (open3d.geometry.PointCloud): The point cloud object
    '''
    if as_mesh:
        # Read the mesh and convert to point cloud
        mesh = o3.io.read_triangle_mesh(file_path)
        if file_path.endswith('.off'):
            pcd = mesh.sample_points_poisson_disk(number_of_points=10000)  
        else:
            pcd = o3.geometry.PointCloud()
            pcd.points = mesh.vertices
    else:
        # Read the point cloud
        pcd = o3.io.read_point_cloud(file_path)

    return pcd

def normalize_pc(pcd: o3.geometry.PointCloud, return_as_np: bool = False):
    '''
    This fuction normalizes the point cloud to the range [-1, 1] (to enclose all points within a unit sphere) by centering it at the origin and scaling it.
    Taken from: https://soulhackerslabs.com/normalizing-feature-scaling-point-clouds-for-machine-learning-8138c6e69f5

    Args:
        pcd (open3d.geometry.PointCloud): The input point cloud
        return_as_np (bool): If True, the function returns the normalized point cloud as a numpy array

    Returns:
        normalized_pcd (open3d.geometry.PointCloud): The normalized point cloud
    '''
    # Convert the Open3D PointCloud to a numpy array
    points = np.asarray(pcd.points)

    # Normalize the points to the range [-1, 1]
    centroid = np.mean(points, axis=0)  # centroid of the point cloud
    points -= centroid   # move the pcd to the origin
    furthest_distance = np.max(np.sqrt(np.sum(abs(points)**2,axis=-1))) # furthest distance from the origin
    points /= furthest_distance  # scale the pcd to fit in a unit sphere

    # Covert the normalized points back to Open3D PointCloud
    normalized_pcd = o3.geometry.PointCloud()
    normalized_pcd.points = o3.utility.Vector3dVector(points)
    
    if return_as_np:
        return points
    else:
        return normalized_pcd

def compute_distances(pcd: o3.geometry.PointCloud) -> torch.Tensor:
    '''
    This function computes the distance for tasks such as normal estimation or voxel size in voxel downsampling.

    Args:
        pcd (open3d.geometry.PointCloud): The input point cloud
    
    Returns:
        distances_tensor (torch.Tensor): The distances between the points in the point cloud
    '''
    distances = pcd.compute_nearest_neighbor_distance()
    distances_tensor = torch.tensor(distances, dtype=torch.float32)

    return distances_tensor

def get_normal(pcd: o3.geometry.PointCloud) -> np.ndarray:
    '''
    This fucntion computes the normals of the point cloud using the KDTreeSearchParamHybrid search parameter.

    Args:
        pcd (open3d.geometry.PointCloud): The input point cloud
    
    Returns:
        normals (numpy.ndarray): The normals of the point cloud as a numpy array
    '''
    max_distance = float(compute_distances(pcd).max())
    radius = max(max_distance, 0.3) # to avoid too samll or zero radius
    pcd.estimate_normals(search_param=o3.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    normals = np.asarray(pcd.normals)
    return normals

def uniform_downsample(pcd: o3.geometry.PointCloud, every_k_points: int, keep_indices: bool = False) -> o3.geometry.PointCloud:
    '''
    This function downsamples the point cloud uniformly by selecting every k-th point (not random).

    Args:
        pcd (open3d.geometry.PointCloud): The input point cloud
        every_k_points (int): Sample rate, the selected point indices are [0, k, 2k, …] 
        keep_indices (bool): If True, the function returns the kept indices

    Returns:
        downsampled_pcd (open3d.geometry.PointCloud): The downsampled point cloud
        kept_indices (list): The indices of the kept points (if keep_indices is True)
    '''
    downsampled_pcd = pcd.uniform_down_sample(every_k_points)

    if keep_indices:
        kept_indices = list(range(0, len(pcd.points), every_k_points))
        return downsampled_pcd, kept_indices
    else:
        return downsampled_pcd

def voxel_downsample(pcd: o3.geometry.PointCloud, voxel_size: float = None, compute_normals=False) -> o3.geometry.PointCloud:
    '''
    Function to downsample input pointcloud into output pointcloud with a voxel.
    This is a two step process. First, it creates a voxel grid from min_bound to max_bound (think of an axis-aligned cuboid which can hold the pointcloud) 
    and then maps each point to the voxel that holds it. Next, averages the points belonging to same voxel. (https://github.com/isl-org/Open3D/blob/881ae76500708aec6d7d8ab070a92776334ce0cd/cpp/open3d/geometry/PointCloud.cpp#L354)
    Normals and colors are averaged if they exist. 

    Args:
        pcd (open3d.geometry.PointCloud): The input point cloud
        voxel_size (float):  Voxel size for downsampling in the same unit as the pointcloud; meter, cm, feet, etc.)
        compute_normals (bool): If True, the normals of the point cloud are computed, averaged and returned as numpy arrays
    
    Returns:
        downsampled_pcd (open3d.geometry.PointCloud): The downsampled point cloud
        normals (numpy.ndarray): The averaged normals of the downsampled point cloud (if compute_normals is True)
    ''' 
    if voxel_size is None:
        min_distance = float(compute_distances(pcd).min())
        voxel_size = max(min_distance * 10, 0.8) # to avoid too samll or zero voxel sizes, lower sizes might result in memory errors
    else:
        voxel_size = voxel_size    
    
    if compute_normals:
        max_distance = float(compute_distances(pcd).max())
        radius = max(max_distance, 0.3) # to avoid too samll or zero radius
        pcd.estimate_normals(search_param=o3.geometry.KDTreeSearchParamHybrid(radius = radius, max_nn=30))

    downsampled_pcd = pcd.voxel_down_sample(voxel_size) #averaged the normals if pcd.estimate_normals exists

    if compute_normals: 
        return downsampled_pcd, np.asarray(downsampled_pcd.normals)
    else:
        return downsampled_pcd

def crop_point_cloud(pcd: o3.geometry.PointCloud, min_bound: tuple = (-118, -118, -118) , max_bound: tuple = (118, 118, 118)):
    '''
    This function crops the point cloud to a specified bounding box defined by the minimum and maximum bounds.
    Usful to get rid of obvious outliers at the edges of the point cloud / long distances from the origin.

    Args:
        pcd (open3d.geometry.PointCloud): The input point cloud
        min_bound (tuple): Minimum bounds of the bounding box
        max_bound (tuple): Maximum bounds of the bounding box
    
    Returns:
        cropped_pcd (open3d.geometry.PointCloud): The cropped point cloud
    '''
    bbox = o3.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    cropped_pcd = pcd.crop(bbox)

    return cropped_pcd

def outlier_removal(pcd: o3.geometry.PointCloud, nb_points: int, radius: float) -> o3.geometry.PointCloud:
    '''
    This function removes the outliers from the point cloud using the radius outlier removal method.

    Args:
        pcd (open3d.geometry.PointCloud): The input point cloud
        nb_points (int): Minimum number of points to define a neighborhood
        radius (float): Radius of the sphere that will determine which points are neighbors
    
    Returns:
        cleaned_pcd (open3d.geometry.PointCloud): The point cloud with the outliers removed
    '''
    _, ind = pcd.remove_radius_outlier(nb_points, radius)
    cleaned_pcd = pcd.select_by_index(ind)

    return cleaned_pcd

def match_size(pcd1: o3.geometry.PointCloud, pcd2:o3.geometry.PointCloud) -> tuple:
    '''
    This function ensures that the two point clouds have the same number of points.

    Args:
        pcd1 (open3d.geometry.PointCloud): The first point cloud
        pcd2 (open3d.geometry.PointCloud): The second point cloud
    
    Returns:
        pcd1 (open3d.geometry.PointCloud): The first point cloud with the same number of points as the second point cloud
        pcd2 (open3d.geometry.PointCloud): The second point cloud with the same number of points as the first point cloud
    '''
    # min_len = min(len(pcd1.points), len(pcd2.points)) # change min_len to a specific length if you want to have a fixed number of points
    min_len = 1441
    pcd1.points = pcd1.points[0:min_len]
    pcd2.points = pcd2.points[0:min_len]
    return pcd1, pcd2 

def load_data(source_path, target_path, every_k_points, voxel_size = None , same_length = False, vdownsample=False, remove_outliers=False, compute_normals=False):
    '''
    This function loads and preprocesses point clouds and optionally computers normals.

    Args:
        source_path (str): Path to the source point cloud
        target_path (str): Path to the target point cloud
        every_k_points (int): Sample rate, the selected point indices are [0, k, 2k, …]
        voxel_size (float) [optional, default = None]: Voxel size for downsampling (in the same unit as the pointcloud; meter, cm, feet, etc.). If None, the voxel size is computed as a multiple of the min distance between points in the point cloud.
        same_length (bool) [optional, default = False]: If True, the target point cloud has the same number of points as the source.
        vdownsample (bool) [optional, defualt = False]: If True, the point clouds are downsampled using voxel downsample
        remove_outliers (bool) [optional, default = False]: If True, the outliers are removed from the point cloud
        compute_normals (bool) [optional, default = False]: If True, the normals of the point clouds are computed and returned as numpy arrays

    Returns:
        source (open3d.geometry.PointCloud): Source point cloud
        target (open3d.geometry.PointCloud): Target point cloud
        source_normals (numpy.ndarray): Normals of the source point cloud (if compute_normals is True)
        target_normals (numpy.ndarray): Normals of the target point cloud (if compute_normals is True)
    '''
    source = load_point_cloud(source_path)
    target = load_point_cloud(target_path, as_mesh=True)

    source = crop_point_cloud(source)

    # source = normalize_pc(source)
    # target = normalize_pc(target)


    if compute_normals:
        source_normals = get_normal(source)
        target_normals = get_normal(target)

    if every_k_points is not None:
        if compute_normals:
            target, target_kept_indices = uniform_downsample(target, every_k_points, keep_indices=True)
            source, source_kept_indices = uniform_downsample(source, every_k_points=math.ceil(len(source.points) / len(target.points)), keep_indices=True)
            source_normals = source_normals[source_kept_indices]
            target_normals = target_normals[target_kept_indices]
        else:
            target = uniform_downsample(target, every_k_points)
            source = uniform_downsample(source, every_k_points=math.ceil(len(source.points) / len(target.points)))

    if vdownsample:
        if compute_normals:                        
            source, source_normals = voxel_downsample(source, voxel_size, compute_normals)
            target, target_normals = voxel_downsample(target, voxel_size, compute_normals)
        else:
            source = voxel_downsample(source, voxel_size)
            target = voxel_downsample(target, voxel_size)

    if remove_outliers:
        source = outlier_removal(source, nb_points= 10, radius=1)

    if same_length:
        source, target = match_size(source, target)
        if compute_normals:
            source_normals = source_normals[0:len(source.points)]
            target_normals = target_normals[0:len(target.points)]

    if compute_normals:
        return source, target, source_normals, target_normals
    
    else:
        return source, target
            

def center_mass_alignment(pcd1, pcd2):
    '''
    get_center() a method in Open3D that computes the centroid (geometric center) of a point cloud. 
    This method returns the average position of all the points in the point cloud in a numpy array with shape (3,).
    The first number is the average x-coordinate, the second number is the average y-coordinate, 
    and the third number is the average z-coordinate of all the points in the point cloud.
    '''
    pcd1_center = pcd1.get_center()
    pcd2_center = pcd2.get_center()

    # Compute the translation vector
    translation = pcd1_center - pcd2_center

    # Translate the target point cloud to align with the source point cloud
    pcd2.translate(translation)

    return pcd1, pcd2

def bounding_box_alignment(pcd):
    '''
    This function aligns the bounding box of the point cloud to the origin (0, 0, 0).
    The bounding box is an axis-aligned bounding box (AABB) that is aligned with the x, y, and z axes.
    The bounding box is defined by the minimum and maximum bounds of the point cloud in each dimension.
    The translation needed to move the minimum bound to the origin is computed and applied to the point cloud.

    Args:
        pcd (open3d.geometry.PointCloud): Point cloud

    Returns:
        pcd (open3d.geometry.PointCloud): Point cloud with the bounding box aligned to the origin
    '''
    aabb = pcd.get_axis_aligned_bounding_box()

    # Compute the translation needed to move the min bound to (0, 0, 0)
    translation = -aabb.min_bound    

    pcd.translate(translation)
    
    return pcd