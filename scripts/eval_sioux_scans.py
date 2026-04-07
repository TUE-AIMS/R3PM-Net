import os
import copy
import argparse
import numpy as np
import random
import torch
from tabulate import tabulate
from tqdm import tqdm
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools import data, l3d_helper, visualization
from tools import icp_registration_and_evaluation, l3d_registration_and_evaluation, predator_registration_and_evaluation, geotransformer_registration_and_evaluation, logdesc_registration_and_evaluation, regtr_registration_and_evaluation
from r3pm_net.config_loader import get_pretrained_rpmnet_dir, get_sioux_data_root, get_method_paths

'''
This script is used to evaluate the performance of the pipeline with R3PM-Net as global and GICP as local registeration.

The script takes the following arguments:
--local_reg: the local registration method to be used.  
--seed: random seed for python/numpy/torch. The default is 42.
--verbose: if set to True, the results will be printed in a table format. The default is False.
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
parser = argparse.ArgumentParser(description="Choosing local registration method")
parser.add_argument(
    "--local_reg", type=str, default="gicp", help="local registration: gicp or freg"
)
parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")

args = parser.parse_args()
set_seed(args.seed)
print(f"Using {args.local_reg} for local registration")

def analyze_results(results: dict, recall_threshold = 1, rmse_threshold = 0.053, verbose = False):  # change the default values to your needs
    table = []
    fail_count = 0
    success_count = 0
    for object, values in results.items():
        row = [object] + list(values)
        if round(row[2], 3) < recall_threshold or round(row[3], 3) > rmse_threshold:
            status = 'failed'
            fail_count += 1
            print(f'No match for {object}! Try a different method. If the issue persists, please check the data.')
        else:
            status = 'success'
            success_count += 1
            print(f'Found match for {object}!')
        row.append(status)
        table.append(row)

    if verbose:
        print(tabulate(table, headers=['Object', 'Chamfer Distance', 'Reg. Recall', 'Inlier RMSE', 'Computation Time', 'Status'], tablefmt='grid'))
        print(f"Success rate: {success_count / (success_count + fail_count) * 100:.2f}%")

    return table

def show_successful_resutls(table, sources, targets, pc_results, method_name = None):
    for i in range (len(table)):
        if table[i][-1] == 'success':
            # visualization.plot_point_cloud(sources[i], targets[i], list(pc_results.values())[i])   # uncomment if below visualization does not work 
            visualization.draw_registration_result(targets[i], list(pc_results.values())[i], np.eye(4), method_name)

def main():
    base_dir = get_sioux_data_root()
    scan_dir = os.path.join(base_dir, 'sioux_scans')
    cad_dir = os.path.join(base_dir, 'sioux_cranfield')

    pcd_paths = [ os.path.join(scan_dir,'teeth_clean.ply'),
                    os.path.join(scan_dir,'lime_clean.ply'),
                    os.path.join(scan_dir,'cube_clean.ply'),  
                    os.path.join(scan_dir,'lego_clean.ply'),
                    os.path.join(scan_dir,'elephant_clean.ply'), 
                    os.path.join(scan_dir,'house_clean.ply'),
                    os.path.join(scan_dir,'shoe_clean.ply')]

    cad_paths = [ os.path.join(cad_dir,'teeth.stl'),
                    os.path.join(cad_dir,'lime.stl'),
                    os.path.join(cad_dir,'cube.stl'),  
                    os.path.join(cad_dir,'lego.stl'),
                    os.path.join(cad_dir,'elephant.stl'), 
                    os.path.join(cad_dir,'house.stl'),
                    os.path.join(cad_dir,'shoe.stl')]

    # Initialize lists and dictionaries to store results
    rpm_net_results = {}
    rpm_net_pc_results = {}
    predator_results = {}
    predator_pc_results = {}
    geotransformer_results = {}
    geotransformer_pc_results = {}
    logdesc_results = {}
    logdesc_pc_results = {}
    regtr_results = {}
    regtr_pc_results = {}
    r3pm_net_results = {}
    r3pm_net_pc_results ={}
    tuned_r3pm_net_results = {}
    tuned_r3pm_net_pc_results = {}
    subset_tuned_r3pm_net_results = {}
    subset_tuned_r3pm_net_pc_results = {}

    sources = []
    targets = []

    pretrained_base_dir = get_pretrained_rpmnet_dir()
    method_paths = get_method_paths()
    _path_zs = os.path.join(pretrained_base_dir, "clean-trained.pth")
    _path_ft = os.path.join(pretrained_base_dir, "best_model_PointNet2.t7") #TODO: CHANGE 
    _path_ft_sub = os.path.join(pretrained_base_dir, "best_model_PointNet_subset.t7") #TODO: CHANGE 

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

    # R3PM-Net (ours) - no training
    r3pm_net_args = l3d_helper.options(modelName="R3PMNet")
    r3pm_net_args.pretrained = _path_zs

    # R3PM-Net (ours) (FT)
    tuned_r3pm_net_args = l3d_helper.options(modelName="R3PMNet")
    tuned_r3pm_net_args.pretrained = _path_ft

    # R3PM-Net (ours) (FT) (Subset)
    subset_tuned_r3pm_net_args = l3d_helper.options(modelName="R3PMNet")
    subset_tuned_r3pm_net_args.pretrained = _path_ft_sub

    for pcdPath, cadPath in tqdm(zip(pcd_paths, cad_paths), desc="Registering objects", total=len(pcd_paths)):
        # Define the number of points to sample from the CAD model (change this based on your data)
        if 'teeth' in pcdPath:
            every_k_points = 100
            key = 'teeth'
        elif'lime' in pcdPath:
            every_k_points = 100
            key = 'lime'
        elif 'cube' in pcdPath:
            every_k_points = 1
            key = 'cube'
        elif 'lego' in pcdPath:
            every_k_points = 10
            key = 'lego'
        elif 'elephant' in pcdPath:
            every_k_points = 30
            key = 'elephant'
        elif 'house' in pcdPath:
            every_k_points = 25
            key = 'house'
        elif 'shoe' in pcdPath:
            every_k_points = 15
            key = 'shoe'
        else:
            print("Unknown object type, using default every_k_points = 1")
            every_k_points = 1
        
        # Load the data
        pcd, cad = data.load_data(pcdPath, cadPath, every_k_points=every_k_points)
        source = copy.deepcopy(pcd)
        target = copy.deepcopy(cad)
        
        # Normalize the point clouds
        source = data.normalize_pc(source)
        target = data.normalize_pc(target)

        sources.append(source)
        targets.append(target)

        gt_transformation = None

        # Perform the registration

        # RPMNet
        rpm_pc_result, _ = l3d_registration_and_evaluation.l3d_reg_and_eval(
            source, target, 'rpmnet', gt_transformation, rpm_args)
        if args.local_reg == 'gicp':
            final_rpm_net_pc_result, final_rpm_net_results = icp_registration_and_evaluation.icp_reg_and_eval(rpm_pc_result, target, 'gicp', 1, np.identity(4), gt_transformation)
        rpm_net_results[key] = final_rpm_net_results
        rpm_net_pc_results[key] = final_rpm_net_pc_result

        # OverlapPredator
        predator_results_pc, _ = predator_registration_and_evaluation.predator_reg_and_eval(
            source,
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
        if args.local_reg == 'gicp':
            final_predator_pc_result, final_predator_results = icp_registration_and_evaluation.icp_reg_and_eval(predator_results_pc, target, 'gicp', 1, np.identity(4), gt_transformation)
        predator_results[key] = final_predator_results
        predator_pc_results[key] = final_predator_pc_result

        # GeoTransformer (ModelNet)
        geotransformer_pc_result, _ = geotransformer_registration_and_evaluation.geotransformer_reg_and_eval(
            source,
            target,
            gt_transformation=gt_transformation,
            geotransformer_root=geotransformer_root,
            exp_subdir=geotransformer_exp_subdir,
            weights_path=geotransformer_weights_path,
        )
        if args.local_reg == 'gicp':
            final_geotransformer_pc_result, final_geotransformer_results = icp_registration_and_evaluation.icp_reg_and_eval(geotransformer_pc_result, target, 'gicp', 1, np.identity(4), gt_transformation)
        geotransformer_results[key] = final_geotransformer_results
        geotransformer_pc_results[key] = final_geotransformer_pc_result

        # LoGDesc
        logdesc_pc_result, _ = logdesc_registration_and_evaluation.logdesc_reg_and_eval(
            source,
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
        if args.local_reg == 'gicp':
            final_logdesc_pc_result, final_logdesc_results = icp_registration_and_evaluation.icp_reg_and_eval(logdesc_pc_result, target, 'gicp', 1, np.identity(4), gt_transformation)
        logdesc_results[key] = final_logdesc_results
        logdesc_pc_results[key] = final_logdesc_pc_result

        # RegTR (ModelNet)
        regtr_pc_result, _ = regtr_registration_and_evaluation.regtr_reg_and_eval(
            source,
            target,
            gt_transformation=gt_transformation,
            regtr_root=regtr_root,
            ckpt_path=regtr_ckpt_path,
            config_path=regtr_config_path,
        )
        if args.local_reg == 'gicp':
            final_regtr_pc_result, final_regtr_results = icp_registration_and_evaluation.icp_reg_and_eval(regtr_pc_result, target, 'gicp', 1, np.identity(4), gt_transformation)
        regtr_results[key] = final_regtr_results
        regtr_pc_results[key] = final_regtr_pc_result

        # R3PM-Net (ours) (ZS)
        r3pm_net_pc_result, _ = l3d_registration_and_evaluation.l3d_reg_and_eval(source, target, 'r3pmnet', gt_transformation, r3pm_net_args)
        if args.local_reg == 'gicp':
            final_r3pm_net_pc_result, final_r3pm_net_results = icp_registration_and_evaluation.icp_reg_and_eval(r3pm_net_pc_result, target, 'gicp', 1, np.identity(4), gt_transformation)
        r3pm_net_results[key] = final_r3pm_net_results
        r3pm_net_pc_results[key] = final_r3pm_net_pc_result

        # R3PM-Net (ours) (FT)
        tuned_r3pm_net_pc_result, _ = l3d_registration_and_evaluation.l3d_reg_and_eval(source, target, 'r3pmnet', gt_transformation, tuned_r3pm_net_args)
        if args.local_reg == 'gicp':
            final_tuned_r3pm_net_pc_result, final_tuned_r3pm_net_results = icp_registration_and_evaluation.icp_reg_and_eval(tuned_r3pm_net_pc_result, target, 'gicp', 1, np.identity(4), gt_transformation)
        tuned_r3pm_net_results[key] = final_tuned_r3pm_net_results
        tuned_r3pm_net_pc_results[key] = final_tuned_r3pm_net_pc_result

        # R3PM-Net (ours) (FT) (Subset)
        subset_tuned_r3pm_net_pc_result, _ = l3d_registration_and_evaluation.l3d_reg_and_eval(source, target, 'r3pmnet', gt_transformation, subset_tuned_r3pm_net_args)
        if args.local_reg == 'gicp':
            final_subset_tuned_r3pm_net_pc_result, final_subset_tuned_r3pm_net_results = icp_registration_and_evaluation.icp_reg_and_eval(subset_tuned_r3pm_net_pc_result, target, 'gicp', 1, np.identity(4), gt_transformation)
        subset_tuned_r3pm_net_results[key] = final_subset_tuned_r3pm_net_results
        subset_tuned_r3pm_net_pc_results[key] = final_subset_tuned_r3pm_net_pc_result

    # Print the results
    print("----- RPMNet: -----")
    rpm_net_table = analyze_results(rpm_net_results, verbose=True)
    show_successful_resutls(rpm_net_table, sources, targets, rpm_net_pc_results, 'RPMNet')

    print("----- Predator: -----")
    predator_table = analyze_results(predator_results, verbose=True)
    show_successful_resutls(predator_table, sources, targets, predator_pc_results, 'Predator')

    print("----- GeoTransformer: -----")
    geotransformer_table = analyze_results(geotransformer_results, verbose=True)
    show_successful_resutls(geotransformer_table, sources, targets, geotransformer_pc_results, 'GeoTransformer')

    print("----- LoGDesc: -----")
    logdesc_table = analyze_results(logdesc_results, verbose=True)
    show_successful_resutls(logdesc_table, sources, targets, logdesc_pc_results, 'LoGDesc')

    print("----- RegTR: -----")
    regtr_table = analyze_results(regtr_results, verbose=True)
    show_successful_resutls(regtr_table, sources, targets, regtr_pc_results, 'RegTR')

    print("----- R3PM-Net (ours) (ZS): -----")
    r3pm_net_table = analyze_results(r3pm_net_results, verbose=True)
    show_successful_resutls(r3pm_net_table, sources, targets, r3pm_net_pc_results, 'R3PM-Net (ours) (ZS)')

    print("----- R3PM-Net (ours) (FT): ----- ")
    tuned_r3pm_net_table = analyze_results(tuned_r3pm_net_results, verbose=True)
    show_successful_resutls(tuned_r3pm_net_table, sources, targets, tuned_r3pm_net_pc_results, 'R3PM-Net (ours) (FT)')

    print("----- R3PM-Net (ours) (FT) (Subset): ----- ")
    subset_tuned_r3pm_net_table = analyze_results(subset_tuned_r3pm_net_results, verbose=True)
    show_successful_resutls(subset_tuned_r3pm_net_table, sources, targets, subset_tuned_r3pm_net_pc_results, 'R3PM-Net (ours) (FT) (Subset)')

if __name__ == "__main__":
    main()