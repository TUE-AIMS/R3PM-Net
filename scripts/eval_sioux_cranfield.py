import os
import copy
import open3d as o3d
import numpy as np
from tqdm import tqdm
import sys
from pathlib import Path
import torch
import random
import argparse

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools import augmentation, data, l3d_helper, print_results, transformations
from tools import l3d_registration_and_evaluation, predator_registration_and_evaluation, geotransformer_registration_and_evaluation, logdesc_registration_and_evaluation, regtr_registration_and_evaluation
from r3pm_net.config_loader import get_method_paths, get_pretrained_rpmnet_dir, get_sioux_data_root, get_sioux_paths
'''
This script evaluates the performance on a Sioux-Cranfield dataset 
Cranfield dataset from: https://github.com/Menthy-Denayer/PCR_CAD_Model_Alignment_Comparison/tree/main/datasets
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
parser = argparse.ArgumentParser(description="Sioux-Cranfield R3PM-Net evaluation")
parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
args = parser.parse_args()
set_seed(args.seed)

base_dir = get_sioux_data_root()
sioux_cfg = get_sioux_paths()
method_paths = get_method_paths()

pretrained_base_dir = get_pretrained_rpmnet_dir()
_path_zs = os.path.join(pretrained_base_dir, "clean-trained.pth")
_path_ft = os.path.join(pretrained_base_dir, "best_model_PointNet.t7") #TODO: CHANGE 

# Paths to the CAD models
cad_dir_made = os.path.join(base_dir, 'sioux_cranfield') 

cad_paths = [os.path.join(cad_dir_made, 'Base-Top_Plate.stl'),
             os.path.join(cad_dir_made, 'Pendulum.stl'),
             os.path.join(cad_dir_made, 'Round-Peg.stl'),
             os.path.join(cad_dir_made, 'Separator.stl'),
             os.path.join(cad_dir_made, 'Shaft-New.stl'),
             os.path.join(cad_dir_made, 'Square-Peg.stl'),
             os.path.join(cad_dir_made, 'elephant.stl'),
             os.path.join(cad_dir_made, 'house.stl'),
             os.path.join(cad_dir_made, 'shoe.stl')]

# Test parameters
num_tests = 25
angles = list(range(0, 45))
translation_range = (-0.5, 0.5)
np.random.seed(42)

# Augmentation parameters
noise_level = 0
outlier_level = 0
outlier_lowerbound = -0.5
outlier_upperbound = 0.5
# occlusion_level = 9000 # Higher value means less occlusion
occ_level = 0

# Make dataset
sources = []
targets = []
x_angles = []
y_angles = []
z_angles = []
gt_transformations = []

for cadPath in tqdm (cad_paths, desc="Preparing Sioux-Cranfield Dataset", total=len(cad_paths)):

        num_points = 2000
        # Load the data
        mesh = o3d.io.read_triangle_mesh(cadPath)
        cad = mesh.sample_points_poisson_disk(number_of_points=num_points)  # modify to a suitable number of points
        normalized_point_cloud = data.normalize_pc(cad)
        source = copy.deepcopy(normalized_point_cloud)

        for test in range(num_tests):
            # Data simulation
            x_angle= np.random.uniform(angles[0], angles[-1], size=1)
            y_angle= np.random.uniform(angles[0], angles[-1], size=1)
            z_angle= np.random.uniform(angles[0], angles[-1], size=1)
            gt_transformation = transformations.create_transformation(x_angle, y_angle, z_angle, translation_range)
            target = copy.deepcopy(normalized_point_cloud).transform(gt_transformation)

            # Data augmentation
            if occ_level == 0 and noise_level == 0 and outlier_level == 0:
                noisy_source = copy.deepcopy(source)
               
            # Noise + Occlusion
            elif occ_level != 0 and noise_level != 0:
                noisy_source_noise = augmentation.apply_noise(source, noise_level)
                noisy_source, _ = augmentation.apply_occlusion(noisy_source_noise, occ_level)
                if len(noisy_source.points) < 1024:  # Handle excessive occlusion
                    source = copy.deepcopy(target).transform(gt_transformation)
                    noisy_source_noise = augmentation.apply_noise(source, noise_level)
                    noisy_source, _ = augmentation.apply_occlusion(noisy_source_noise, occ_level * 1.5)

            # Noise + Outlier
            elif noise_level != 0 and outlier_level != 0:
                noisy_source_noise = augmentation.apply_noise(source, noise_level)
                noisy_source = augmentation.add_outliers(noisy_source_noise, outlier_level, outlier_lowerbound=-0.5, outlier_upperbound=0.5)

            # Noise + Outlier + Occlusion
            elif occ_level != 0 and noise_level != 0 and outlier_level != 0:       
                noisy_source_noise = augmentation.apply_noise(source, noise_level)
                noisy_source, _ = augmentation.apply_occlusion(noisy_source_noise, occ_level)
                if len(noisy_source.points) < 1024:  # Handle excessive occlusion
                    source = copy.deepcopy(target).transform(gt_transformation)
                    noisy_source_noise = augmentation.apply_noise(source, noise_level)
                    noisy_source, _ = augmentation.apply_occlusion(noisy_source_noise, occ_level * 1.5)
                noisy_source = augmentation.add_outliers(noisy_source, outlier_level, outlier_lowerbound=-0.5, outlier_upperbound=0.5)

            # collect dataset in lists
            sources.append(noisy_source)
            targets.append(target)
            x_angles.append(x_angle)
            y_angles.append(y_angle)
            z_angles.append(z_angle)
            gt_transformations.append(gt_transformation)

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


for i, item in enumerate(tqdm(zip(sources, targets, gt_transformations), desc="Testing methods", total=len(sources))):
    
    # RPMNet
    rpm_results_pc, rpm_results = l3d_registration_and_evaluation.l3d_reg_and_eval(
        sources[i], targets[i], 'rpmnet', gt_transformations[i], rpm_args)
    rpm_results_all.append(rpm_results)
    rpm_reg_results_all.append(rpm_results_pc)

    # OverlapPredator
    predator_results_pc, predator_results = predator_registration_and_evaluation.predator_reg_and_eval(
        sources[i],
        targets[i],
        gt_transformation=gt_transformations[i],
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
        sources[i],
        targets[i],
        gt_transformation=gt_transformations[i],
        geotransformer_root=geotransformer_root,
        exp_subdir=geotransformer_exp_subdir,
        weights_path=geotransformer_weights_path,
    )
    geotransformer_results_all.append(geotransformer_results)
    geotransformer_reg_results_all.append(geotransformer_results_pc)

        # LoGDesc
    logdesc_results_pc, logdesc_results = logdesc_registration_and_evaluation.logdesc_reg_and_eval(
        sources[i],
        targets[i],
        gt_transformation=gt_transformations[i],
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
        sources[i],
        targets[i],
        gt_transformation=gt_transformations[i],
        regtr_root=regtr_root,
        ckpt_path=regtr_ckpt_path,
        config_path=regtr_config_path,
    )
    regtr_results_all.append(regtr_results)
    regtr_reg_results_all.append(regtr_results_pc)

    # R3PM-Net (ours) - ZS - no training
    r3pm_net_results_pc, r3pm_net_results = l3d_registration_and_evaluation.l3d_reg_and_eval(
        sources[i], targets[i], 'r3pmnet', gt_transformations[i], r3pm_net_args)
    r3pm_net_results_all.append(r3pm_net_results)
    r3pm_net_reg_results_all.append(r3pm_net_results_pc)

    # R3PM-Net (ours) - FT
    tuned_r3pm_net_results_pc, tuned_r3pm_net_results = l3d_registration_and_evaluation.l3d_reg_and_eval(
        sources[i], targets[i], 'r3pmnet', gt_transformations[i], tuned_r3pm_net_args)
    tuned_r3pm_net_results_all.append(tuned_r3pm_net_results)
    tuned_r3pm_net_reg_results_all.append(tuned_r3pm_net_results_pc)

 
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
    "R3PM-Net (ours) (FT)": dict(zip(metric_names, tuned_r3pm_net_mean_results)),}

# Print the table
print_results.print_table(reports)
