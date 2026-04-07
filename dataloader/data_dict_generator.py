import argparse
import copy
import logging
import os
import shlex
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools import data
from dataloader.dataset_generator import combine_dataset_dict, generate_dataset, generate_dataset_dict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description='Automate dataset generation and processing.')

    # Define arguments (change these as needed)
    parser.add_argument('--pcdPath', type=str, required=True, help='Path to the PCD file')
    parser.add_argument('--cadPath', type=str, required=True, help='Path to the CAD file')
    parser.add_argument('--action', type=str, choices=['generate_dataset', 'generate_dataset_dict', 'combine_dataset_dict'], required=True, help='Action to perform')
    parser.add_argument('--compute_normals', action='store_true', help='Flag to compute normals')
    parser.add_argument('--every_k_points', type=int, default=1, help='Sampling rate for points')
    parser.add_argument('--save', action='store_true', help='Flag to save the generated dataset')
    parser.add_argument(
        '--save_path',
        type=str,
        default='data/simulators',
        help='Directory to save generated datasets (relative to repo root if not absolute)',
    )
    parser.add_argument('--name', type=str, required=True, help='Name identifier for the dataset (e.g., teeth, cube, etc.)')

    # Additional parameters for dataset generation (change these as needed)
    parser.add_argument('--num_transformation', type=int, default=50, help='Number of transformations')
    parser.add_argument('--angles', type=int, nargs='+', default=list(range(0, 360, 10)), help='Rotation angles')
    parser.add_argument('--translation_range', type=float, nargs=2, default=(-1, 1), help='Translation range')
    parser.add_argument('--dataset_size', type=int, default=400, help='Size of the dataset to generate')
    parser.add_argument('--index', type=int, default=0, help='Index for dataset generation')
    parser.add_argument('--noise_level', type=float, default=0, help='Noise level')
    parser.add_argument('--outlier_level', type=float, default=0, help='Outlier level')
    parser.add_argument('--outlier_bounds', type=float, nargs=2, default=(-10, 10), help='Outlier bounds')
    parser.add_argument('--occ_level', type=float, default=0, help='Occlusion level')

    # Parse the arguments

    # Check if an argument file is being used
    if sys.argv[1].startswith('@'):
        args_file = sys.argv[1][1:]  # Strip the '@' from the filename
        with open(args_file, 'r') as file:
            # Read and split arguments from the file
            args = parser.parse_args(shlex.split(file.read()))
    else:
        args = parser.parse_args()

    # Print out the arguments to verify
    print(vars(args))

    # Load the data
    np.random.seed(42)
    if args.compute_normals:
        _, cad, _, cad_normals = data.load_data(args.pcdPath, args.cadPath, every_k_points=args.every_k_points, same_length=True, compute_normals=True)
        suffix = '_with_normals'
    else:
        _, cad = data.load_data(args.pcdPath, args.cadPath, every_k_points=args.every_k_points, same_length=True)
        cad_normals = None
        suffix = ''
    source = copy.deepcopy(cad)

    rp = Path(args.save_path)
    if not rp.is_absolute():
        rp = _REPO_ROOT / args.save_path
    ROOT_DIR = str(rp.resolve())
    if not ROOT_DIR.endswith(os.sep):
        ROOT_DIR += os.sep

    # Perform the selected action
    if args.action == 'generate_dataset':
        logging.info('Generating dataset...')
        generate_dataset(source, args.pcdPath, args.cadPath, args.num_transformation, args.angles, args.translation_range, args.index, args.noise_level, args.outlier_level, args.outlier_bounds, args.occ_level, save_dir=ROOT_DIR)

    elif args.action == 'generate_dataset_dict':
        logging.info('Generating dataset dictionary...')
        output_train_file = f'{ROOT_DIR}data_dict_train_{args.name}{suffix}.pkl'
        output_test_file = f'{ROOT_DIR}data_dict_test_{args.name}{suffix}.pkl'
        generate_dataset_dict(source, args.dataset_size, args.index, output_train_file, output_test_file, cad_normals)

    elif args.action == 'combine_dataset_dict':
        logging.info('Combining dataset dictionaries...')
        train_files = [
            f'{ROOT_DIR}data_dict_train_teeth{suffix}.pkl'
            # f'{ROOT_DIR}data_dict_train_elephant{suffix}.pkl',
            # f'{ROOT_DIR}data_dict_train_house{suffix}.pkl',
            # f'{ROOT_DIR}data_dict_train_shoe{suffix}.pkl'
        ]

        test_files = [
            f'{ROOT_DIR}data_dict_test_teeth{suffix}.pkl'
            # f'{ROOT_DIR}data_dict_test_elephant{suffix}.pkl',
            # f'{ROOT_DIR}data_dict_test_house{suffix}.pkl',
            # f'{ROOT_DIR}data_dict_test_shoe{suffix}.pkl'
        ]

        output_train_file = f'{ROOT_DIR}data_dict_train_{suffix}.pkl'
        output_test_file = f'{ROOT_DIR}data_dict_test_{suffix}.pkl'

        combine_dataset_dict(train_files, test_files, output_train_file, output_test_file)

    else:
        logging.warning('No valid action selected.')

if __name__ == '__main__':
    main()
