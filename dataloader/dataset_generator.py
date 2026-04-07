import copy
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import open3d as o3

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools import augmentation, data, transformations

_SIM_DATA = _REPO_ROOT / "data" / "simulators"
'''
This module provides functions to generate a dataset of point clouds with random transformations, with options for noise, outliers, and occlusions.
It also includes functions to check the shape of the data and to generate a data dictionary for training and testing,
and a function to combine multiple dataset dictionaries.
'''

def generate_dataset(pcd, pcdPath, cadPath, num_transformation, angles, translation_range, index, noise_level = 0, outlier_level = 0, outlier_bounds = (-10, 10), occ_level = 0, save_dir=None):
    '''
    A function to generate a dataset of point clouds with random transformations.

    Args:
        pcd (open3d.geometry.PointCloud): The source point cloud
        pcdPath (str): The path to the source point cloud
        cadPath (str): The path to the target point cloud
        num_transformation (int): The number of transformations to generate
        angles (numpy.ndarray): The range of angles for the random transformations
        translation_range (tuple): The range of translations for the random transformations
        index (int): The index to start saving the generated dataset
        noise_level (float): The level of noise to add to the point clouds
        outlier_level (float): The level of outliers to add to the point clouds
        occ_level (float): The level of occlusions to add to the point clouds
        save (bool): A flag to save the generated dataset

    Returns:
        None
    '''
    np.random.seed(42)
    target_list = []
    gt_transformation_list = []

    for i in range(num_transformation):
        # Generate random gt transformation
        x_angle= np.random.uniform(angles[0], angles[-1], size=1)
        y_angle= np.random.uniform(angles[0], angles[-1], size=1)
        z_angle= np.random.uniform(angles[0], angles[-1], size=1)
        gt_transformation = transformations.create_transformation(x_angle, y_angle, z_angle, translation_range)

        target = copy.deepcopy(pcd)
        target.transform(gt_transformation)

        if noise_level != 0:
            target = augmentation.apply_noise(target, noise_level)
            print('Noise applied')

        if outlier_level != 0 or occ_level != 0:
            _, another_cad = data.load_data(pcdPath, cadPath, every_k_points=1)
            target = copy.deepcopy(another_cad).transform(gt_transformation)
            if occ_level != 0:
                target, _ = augmentation.apply_occlusion(target, occ_level)
                print('Occlusion applied')
            if outlier_level != 0:
                target = augmentation.add_outliers(target, outlier_level, outlier_lowerbound=outlier_bounds[0], outlier_upperbound=outlier_bounds[1])
                print('Outliers applied')

        # randomly take points away from target to get to same length as source
        if len(target.points) >= len(pcd.points):
            np.random.seed(42)
            target_points = np.asarray(target.points) 
            indices = np.random.choice(len(target_points), 1441, replace=False)  # change len(source.points) to a specific num if you want to have a fixed number of points
            sampled_points = target_points[indices]
            target.points = o3.utility.Vector3dVector(sampled_points)
        else:
            print('Target has fewer points than source and can\'t be downsampled to the same length.')

        print(f'size of source and target: {len(pcd.points)}, {len(target.points)}')
        target_list.append(target)
        gt_transformation_list.append(gt_transformation)

    # Save the generated dataset
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for i, (target, transformation) in enumerate(zip(target_list, gt_transformation_list)):
            target_path = os.path.join(save_dir, f"target_{i+index}.pcd")
            transformation_path = os.path.join(save_dir, f"transformation_{i+index}.npy")
            o3.io.write_point_cloud(target_path, target)
            np.save(transformation_path, transformation)

def check_shape(data, expected_shape_3d, expected_shape_6d):
    return data.shape == expected_shape_3d or data.shape == expected_shape_6d

def generate_dataset_dict(source, dataset_size, index, output_train_file_path, output_test_file_path, source_normals = None):
    '''
    This function shuffles the dataset and generates a data_dict for the training and testing data following the pattern acceptable to Learning3D.
    
    Args:
        source (open3d.geometry.PointCloud): The source point cloud
        dataset_size (int): The size of the dataset

    Returns:
        None
    '''
    np.random.seed(42)
    transformed_pcds = []
    gt_transformations = []

    # Load the transformed point clouds and ground truth transformations
    for i in range(index,index+dataset_size):
        transformed_pcd = o3.io.read_point_cloud(str(_SIM_DATA / f"target_{i}.pcd"))
        gt_transformation = np.load(str(_SIM_DATA / f"transformation_{i}.npy"))

        if source_normals is not None: # we also need target normals 
            M = np.linalg.inv(gt_transformation).T
            target_normals = np.dot(source_normals, M[:3,:3]) # transformed_normals = normals * (transformation)^-1.T
            transformed_points = np.concatenate((np.asarray(transformed_pcd.points), target_normals), axis=1)
        else:
            transformed_points = np.asarray(transformed_pcd.points).astype(np.float32)

        transformed_pcds.append(transformed_points)
        gt_transformations.append(gt_transformation)

    # Shuffle the transformed point clouds and ground truth transformations in the same way
    temp = list(zip(transformed_pcds, gt_transformations))
    random.shuffle(temp) 
    transformed_pcds, gt_transformations = zip(*temp)

    # Convert lists to numpy arrays
    transformed_pcds_np = np.array(transformed_pcds)
    gt_transformations_np = np.array(gt_transformations)

    if source_normals is not None:
        source = np.concatenate((np.asarray(source.points), source_normals), axis=1)
    else:
        source = np.asarray(source.points).astype(np.float32)

    data_dict = {
        'template': np.tile(source, (dataset_size, 1, 1)),
        'source': transformed_pcds_np,
        'transformation': gt_transformations_np
    }

    # Split the data_dict into training and testing data_dict
    train_size = int(0.8 * dataset_size)
    test_size = dataset_size - train_size
    num_points = len(source)

    data_dict_train = {}
    data_dict_test = {}
    for key in data_dict.keys():
        data_dict_train[key] = data_dict[key][0:train_size]
        data_dict_test[key] = data_dict[key][train_size:]
    
    assert set(data_dict_train.keys()) == {'template', 'source', 'transformation'}
    assert set(data_dict_test.keys()) == {'template', 'source', 'transformation'}

    expected_shape_3d_train = (train_size, num_points, 3)
    expected_shape_6d_train = (train_size, num_points, 6)

    assert check_shape(data_dict_train['template'], expected_shape_3d_train, expected_shape_6d_train), f"Expected shape: {expected_shape_3d_train} or {expected_shape_6d_train}, but got {data_dict_train['template'].shape}"
    assert check_shape(data_dict_train['source'], expected_shape_3d_train, expected_shape_6d_train), f"Expected shape: {expected_shape_3d_train} or {expected_shape_6d_train}, but got {data_dict_train['source'].shape}"
    assert data_dict_train['transformation'].shape == (train_size, 4, 4), f"Expected shape: {(train_size, 4, 4)}, but got {data_dict_train['transformation'].shape}"

    expected_shape_3d_test = (test_size, num_points, 3)
    expected_shape_6d_test = (test_size, num_points, 6)

    assert check_shape(data_dict_test['template'], expected_shape_3d_test, expected_shape_6d_test), f"Expected shape: {expected_shape_3d_test} or {expected_shape_6d_test}, but got {data_dict_test['template'].shape}"
    assert check_shape(data_dict_test['source'], expected_shape_3d_test, expected_shape_6d_test), f"Expected shape: {expected_shape_3d_test} or {expected_shape_6d_test}, but got {data_dict_test['source'].shape}"
    assert data_dict_test['transformation'].shape == (test_size, 4, 4), f"Expected shape: {(test_size, 4, 4)}, but got {data_dict_test['transformation'].shape}"
    
    with open(output_train_file_path, 'wb') as f:
        pickle.dump(data_dict_train, f)
    print(f"train_dict saved to {output_train_file_path}")

    with open(output_test_file_path, 'wb') as f:
        pickle.dump(data_dict_test, f)
    print(f"test_dict saved to {output_test_file_path}")


def combine_dataset_dict(train_files, test_files, output_train_file_path, output_test_file_path):
    '''
    Combine and shuffle dictionaries from multiple files.

    Args:
        train_files (list of str): List of file paths to training dictionaries.
        test_files (list of str): List of file paths to testing dictionaries.
        output_train_file (str): Output file path for the combined training dictionary.
        output_test_file (str): Output file path for the combined testing dictionary.
    '''
    
    # Load the dictionaries from the .pkl files
    train_dicts = [pickle.load(open(file, 'rb')) for file in train_files]
    test_dicts = [pickle.load(open(file, 'rb')) for file in test_files]

    # Combine the dictionaries
    combined_train_dict = {}
    combined_test_dict = {}

    for key in train_dicts[0].keys():
        combined_train_dict[key] = np.concatenate([d[key] for d in train_dicts], axis=0)
        combined_test_dict[key] = np.concatenate([d[key] for d in test_dicts], axis=0)

    # Shuffle
    train_combined_list = list(zip(combined_train_dict['template'], combined_train_dict['source'], combined_train_dict['transformation']))
    test_combined_list = list(zip(combined_test_dict['template'], combined_test_dict['source'], combined_test_dict['transformation']))

    random.shuffle(train_combined_list)
    random.shuffle(test_combined_list)

    combined_train_dict['template'], combined_train_dict['source'], combined_train_dict['transformation'] = zip(*train_combined_list)
    combined_test_dict['template'], combined_test_dict['source'], combined_test_dict['transformation'] = zip(*test_combined_list)

    # Convert back to numpy arrays
    combined_train_dict['template'] = np.array(combined_train_dict['template'])
    combined_train_dict['source'] = np.array(combined_train_dict['source'])
    combined_train_dict['transformation'] = np.array(combined_train_dict['transformation'])

    combined_test_dict['template'] = np.array(combined_test_dict['template'])
    combined_test_dict['source'] = np.array(combined_test_dict['source'])
    combined_test_dict['transformation'] = np.array(combined_test_dict['transformation'])

    # Checks 
    train_size = len(combined_train_dict['source'])
    test_size = len(combined_test_dict['source'])
    num_points = combined_train_dict['source'].shape[1]
    
    assert set(combined_train_dict.keys()) == {'template', 'source', 'transformation'}
    assert set(combined_test_dict.keys()) == {'template', 'source', 'transformation'}

    expected_shape_3d_train = (train_size, num_points, 3)
    expected_shape_6d_train = (train_size, num_points, 6)

    assert check_shape(combined_train_dict['template'], expected_shape_3d_train, expected_shape_6d_train), f"Expected shape: {expected_shape_3d_train} or {expected_shape_6d_train}, but got {combined_train_dict['template'].shape}"
    assert check_shape(combined_train_dict['source'], expected_shape_3d_train, expected_shape_6d_train), f"Expected shape: {expected_shape_3d_train} or {expected_shape_6d_train}, but got {combined_train_dict['source'].shape}"
    assert combined_train_dict['transformation'].shape == (train_size, 4, 4), f"Expected shape: {(train_size, 4, 4)}, but got {combined_train_dict['transformation'].shape}"

    expected_shape_3d_test = (test_size, num_points, 3)
    expected_shape_6d_test = (test_size, num_points, 6)

    assert check_shape(combined_test_dict['template'], expected_shape_3d_test, expected_shape_6d_test), f"Expected shape: {expected_shape_3d_test} or {expected_shape_6d_test}, but got {combined_test_dict['template'].shape}"
    assert check_shape(combined_test_dict['source'], expected_shape_3d_test, expected_shape_6d_test), f"Expected shape: {expected_shape_3d_test} or {expected_shape_6d_test}, but got {combined_test_dict['source'].shape}"
    assert combined_test_dict['transformation'].shape == (test_size, 4, 4), f"Expected shape: {(test_size, 4, 4)}, but got {combined_test_dict['transformation'].shape}"
    
    # Save the dictionaries
    with open(output_train_file_path, 'wb') as f:
        pickle.dump(combined_train_dict, f)
    print(f"combined_train_dict saved to {output_train_file_path}")

    with open(output_test_file_path, 'wb') as f:
        pickle.dump(combined_test_dict, f)
    print(f"combined_test_dict saved to {output_train_file_path}")
