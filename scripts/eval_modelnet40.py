import os
import copy
import sys
from pathlib import Path
from typing import Any

# Repository root on PYTHONPATH (run: python scripts/test_modelnet40.py from repo root).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import random

import numpy as np
import open3d as o3d
import torch
from tqdm import tqdm

from tools import augmentation, data, l3d_helper, print_results, transformations
from tools import l3d_registration_and_evaluation, predator_registration_and_evaluation, geotransformer_registration_and_evaluation, logdesc_registration_and_evaluation, regtr_registration_and_evaluation
from r3pm_net.config_loader import get_method_paths,get_modelnet40_paths, get_pretrained_rpmnet_dir

'''
This script evaluates the performance on the ModelNet40 test dataset.
The results are averaged ovet the dataset with 2468 samples.
All the point clouds are normalized to a sphere of radius 1.

Augmentations:
- Transformation = Random rotation (0 - 45) and translation (-0.5 to 0.5)
- Noise = Gaussian noise with mean 0 and std deviation of 0.01 [optional]
- Outliers = with level 1 which means 2% of the points are outliers (PC size = 2040) [optional]
- Occlusion = 90000 radius which means 0.7% of the points are occluded (PC size = 1986) [optional]
'''
def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" 

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

# arguments
parser = argparse.ArgumentParser(description="ModelNet40 R3PM-Net evaluation")
parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")

args = parser.parse_args()
set_seed(args.seed)
method_paths = get_method_paths()

pretrained_base_dir = get_pretrained_rpmnet_dir()
_path_zs = os.path.join(pretrained_base_dir, "clean-trained.pth")
_path_ft = os.path.join(pretrained_base_dir, "best_model_PointNet.t7") #TODO: CHANGE 

def fix_off_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    if lines[0].startswith("OFF") and len(lines[0].strip().split()) > 1:
        header = lines[0].strip()
        new_header = "OFF\n" + header[3:] + "\n"
        lines = [new_header] + lines[1:]
        
        with open(file_path, 'w') as f:
            f.writelines(lines)
        print(f"Fixed: {file_path}")

def load_modelnet40_test_data(dataset_path, num_points=2000):
    test_data = []
    test_labels = []
    categories = os.listdir(dataset_path)
    for label, category in enumerate(tqdm(categories, desc="Loading Data")):
        test_dir = os.path.join(dataset_path, category, 'test')
        if not os.path.exists(test_dir):
            continue
        for file in tqdm(os.listdir(test_dir), desc=f"Processing {category} Category", leave=False):
            if file.endswith('.off'):
                file_path = os.path.join(test_dir, file)
                mesh = o3d.io.read_triangle_mesh(file_path)
                point_cloud = mesh.sample_points_poisson_disk(number_of_points=num_points)
                test_data.append(point_cloud)
                test_labels.append(label)
    
    return test_data, test_labels, categories

# download from http://modelnet.cs.princeton.edu/ModelNet40.zip unzip and put the path in the config/eval.yaml
dataset_path, save_dir = get_modelnet40_paths()
test_data_path = os.path.join(save_dir, "test_data.npy")
test_labels_path = os.path.join(save_dir, "test_labels.npy")
categories_path = os.path.join(save_dir, "categories.npy")

os.makedirs(save_dir, exist_ok=True)

# Check if data already exists
if os.path.exists(test_data_path) and os.path.exists(test_labels_path) and os.path.exists(categories_path):
    print("Loading existing test data...")
    test_data_np = np.load(test_data_path, allow_pickle=True)
    test_labels = np.load(test_labels_path)
    categories = np.load(categories_path)
    print("Done! Testing the models...")
else:
    print("Loading and processing ModelNet40 test data...")
    # Fix all .OFF files in the dataset
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".off"):
                fix_off_file(os.path.join(root, file))

    test_data, test_labels, categories = load_modelnet40_test_data(dataset_path)

    test_data_np = [data.normalize_pc(pc, return_as_np = True) for pc in test_data]

    np.save(test_data_path, test_data_np)
    np.save(test_labels_path, test_labels)
    np.save(categories_path, categories)
    print("Test data saved!")

# Initialize arrays to store results
rpm_results_all = []
predator_results_all = []
geotransformer_results_all = []
logdesc_results_all = []
regtr_results_all = []
r3pm_net_results_all = []
tuned_r3pm_net_results_all = []

rpm_reg_results_all = []
predator_reg_results_all = []
geotransformer_reg_results_all = []
logdesc_reg_results_all = []
regtr_reg_results_all = []
r3pm_net_reg_results_all = []
tuned_r3pm_net_reg_results_all = []

all_sources = []
all_targets = []
all_angles ={}

# Reconstruct Open3D PointCloud objects from saved npy arrays
test_data = [o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)) for points in test_data_np]

noise_level = 0
outlier_level = 0
outlier_lowerbound = -0.5
outlier_upperbound = 0.5
# occlusion_level = 90000 # Higher value means less occlusion
occlusion_level = 0 # Higher value means less occlusion


# set arguments for models
rpm_args = l3d_helper.options(modelName="RPMNet")
rpm_args.pretrained = _path_zs

# OverlapPredator (used by Predator runner)
predator_cfg = method_paths.get("predator", {})
predator_root = predator_cfg.get("root")
predator_config_path = predator_cfg.get("config_path")
predator_weights_path = predator_cfg.get("weights_path")

# GeoTransformer 
geo_cfg = method_paths.get("geotransformer", {})
geotransformer_root = geo_cfg.get("root")
geotransformer_exp_subdir = geo_cfg.get("exp_subdir")
geotransformer_weights_path = geo_cfg.get("weights_path")

# LoGDesc
logdesc_cfg = method_paths.get("logdesc", {})
logdesc_root = logdesc_cfg.get("root")
logdesc_weights_path = logdesc_cfg.get("weights_path")

# RegTR
regtr_cfg = method_paths.get("regtr", {})
regtr_root = regtr_cfg.get("root")
regtr_ckpt_path = regtr_cfg.get("ckpt_path")
regtr_config_path = regtr_cfg.get("config_path")

# R3PM-Net (ours) - ZS - no training
r3pm_net_args = l3d_helper.options(modelName="R3PMNet")
r3pm_net_args.pretrained = _path_zs

# R3PM-Net (ours) - FT
tuned_r3pm_net_args = l3d_helper.options(modelName="R3PMNet")
tuned_r3pm_net_args.pretrained = _path_ft

for i, item in enumerate(tqdm(test_data, desc="Testing methods")):
    
   # Simulate data
    x_angle = int(random.uniform(0, 45))
    y_angle = int(random.uniform(0, 45))
    z_angle = int(random.uniform(0, 45))
    translation_range = (-0.5, 0.5)
    gt_transformation = transformations.create_transformation(x_angle, y_angle, z_angle, translation_range)
    source = copy.deepcopy(item)

    target = copy.deepcopy(item).transform(gt_transformation)

    # Apply augmentations
    noisy_source = copy.deepcopy(source)    
    if noise_level != 0:
        noisy_source = augmentation.apply_noise(noisy_source, noise_level)
    if outlier_level != 0:
        noisy_source = augmentation.add_outliers(noisy_source, outlier_level, outlier_lowerbound, outlier_upperbound)
    if occlusion_level != 0:
        noisy_source, _ = augmentation.apply_occlusion(noisy_source, occlusion_level) 
    if len(noisy_source.points) < 1024: # cannot be smaller than embedding dims in config/default.yaml
        noisy_source = copy.deepcopy(source)
        noisy_source = augmentation.apply_noise(noisy_source, noise_level)
        noisy_source, _ = augmentation.apply_occlusion(noisy_source, occlusion_level * 100) 
    assert len(noisy_source.points) >= 1024, "Noisy source point cloud has less than 1024 points."
    
    # RPMNet
    rpm_results_pc, rpm_results = l3d_registration_and_evaluation.l3d_reg_and_eval(
        noisy_source, target, 'rpmnet', gt_transformation, rpm_args)
    rpm_results_all.append(rpm_results)
    rpm_reg_results_all.append(rpm_results_pc)

    # OverlapPredator
    predator_results_pc, predator_results = predator_registration_and_evaluation.predator_reg_and_eval(
        noisy_source,
        target,
        gt_transformation=gt_transformation,
        predator_root=predator_root,
        config_path=predator_config_path,
        weights_path=predator_weights_path,
        ransac_n_points=1000,
        ransac_distance_threshold=0.05,
        ransac_n=3,
        sampling="prob",
        mutual=False,
        input_num_points=1024,
    )
    predator_results_all.append(predator_results)
    predator_reg_results_all.append(predator_results_pc)

    # GeoTransformer (ModelNet)
    geotransformer_results_pc, geotransformer_results = geotransformer_registration_and_evaluation.geotransformer_reg_and_eval(
        noisy_source,
        target,
        gt_transformation=gt_transformation,
        geotransformer_root=geotransformer_root,
        exp_subdir=geotransformer_exp_subdir,
        weights_path=geotransformer_weights_path,
    )
    geotransformer_results_all.append(geotransformer_results)
    geotransformer_reg_results_all.append(geotransformer_results_pc)

    # LoGDesc
    logdesc_results_pc, logdesc_results = logdesc_registration_and_evaluation.logdesc_reg_and_eval(
        noisy_source,
        target,
        gt_transformation=gt_transformation,
        logdesc_root=logdesc_root,
        weights_path=logdesc_weights_path,
        max_keypoints=768,
        num_points_per_sample=128,
        sample_radius=0.3,
        topk_matches=128,
        use_kpt=False,
    )
    logdesc_results_all.append(logdesc_results)
    logdesc_reg_results_all.append(logdesc_results_pc)

    # RegTR (ModelNet)
    regtr_results_pc, regtr_results = regtr_registration_and_evaluation.regtr_reg_and_eval(
        noisy_source,
        target,
        gt_transformation=gt_transformation,
        regtr_root=regtr_root,
        ckpt_path=regtr_ckpt_path,
        config_path=regtr_config_path,
    )
    regtr_results_all.append(regtr_results)
    regtr_reg_results_all.append(regtr_results_pc)

    # R3PM-Net (ours) - no training
    r3pm_net_results_pc, r3pm_net_results = l3d_registration_and_evaluation.l3d_reg_and_eval(
        noisy_source, target, 'r3pmnet', gt_transformation, r3pm_net_args)
    r3pm_net_results_all.append(r3pm_net_results)
    r3pm_net_reg_results_all.append(r3pm_net_results_pc)

    # R3PM-Net (ours) (Tuned on 4 sioux data)
    tuned_r3pm_net_results_pc, tuned_r3pm_net_results = l3d_registration_and_evaluation.l3d_reg_and_eval(
        noisy_source, target, 'r3pmnet', gt_transformation, tuned_r3pm_net_args)
    tuned_r3pm_net_results_all.append(tuned_r3pm_net_results)
    tuned_r3pm_net_reg_results_all.append(tuned_r3pm_net_results_pc)


    all_sources.append(noisy_source)
    all_targets.append(target)
    all_angles[i] = {
        "x_angle": x_angle,
        "y_angle": y_angle,
        "z_angle": z_angle,
        "translation": gt_transformation[:3, 3]
    }
    
# Convert results to numpy arrays for easier manipulation
rpm_results_all = np.array(rpm_results_all)
predator_results_all = np.array(predator_results_all)
geotransformer_results_all = np.array(geotransformer_results_all)
logdesc_results_all = np.array(logdesc_results_all)
regtr_results_all = np.array(regtr_results_all)
r3pm_net_results_all = np.array(r3pm_net_results_all)
tuned_r3pm_net_results_all = np.array(tuned_r3pm_net_results_all)

rpm_mean_results = np.mean(rpm_results_all, axis=0)
predator_mean_results = np.mean(predator_results_all, axis=0)
geotransformer_mean_results = np.mean(geotransformer_results_all, axis=0)
logdesc_mean_results = np.mean(logdesc_results_all, axis=0)
regtr_mean_results = np.mean(regtr_results_all, axis=0)
r3pm_net_mean_results = np.mean(r3pm_net_results_all, axis=0)
tuned_r3pm_net_mean_results = np.mean(tuned_r3pm_net_results_all, axis=0)

# Print the results
metric_names = ['mean_rmse', 'mean_rotation_error', 'mean_translation_error',
                'mean_computation_time', 'mean_cd', 'mean_error',
                'mean_fitness', 'mean_inlier_rmse']

reports = {
    "RPMNet": dict(zip(metric_names, rpm_mean_results)),
    "Predator": dict(zip(metric_names, predator_mean_results)),
    "GeoTransformer": dict(zip(metric_names, geotransformer_mean_results)),
    "LoGDesc": dict(zip(metric_names, logdesc_mean_results)),
    "RegTR": dict(zip(metric_names, regtr_mean_results)),
    "R3PM-Net (ours) (ZS)": dict(zip(metric_names, r3pm_net_mean_results)),
    "R3PM-Net (ours) (FT)": dict(zip(metric_names, tuned_r3pm_net_mean_results)),
}

# Print the table
print_results.print_table(reports)