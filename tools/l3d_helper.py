import argparse
import os
import sys

import numpy as np
import torch

from r3pm_net.paths import REPO_ROOT

# Default pretrained layout: place learning3d-style checkpoints under this tree (see README).
pretained_base_dir = os.environ.get(
    "R3PM_NET_PRETRAINED_ROOT", str(REPO_ROOT / "checkpoints" / "pretrained")
)


def convert_data(pcd, device):
    pcd_mat = np.asarray(pcd.points)
    pcd_tensor = np.zeros((1, pcd_mat.shape[0], 3))
    pcd_tensor[0, :, :] = pcd_mat
    torch_tensor = torch.from_numpy(pcd_tensor)
    torch_tensor = torch_tensor.to(device=device, dtype=torch.float)
    return torch_tensor


def options(modelName):
    parser = argparse.ArgumentParser(description="Point Cloud Registration")

    if modelName == "DCP":
        parser.add_argument(
            "--pointnet",
            default="tune",
            type=str,
            choices=["fixed", "tune"],
            help="train pointnet (default: tune)",
        )
        parser.add_argument(
            "--emb_dims",
            default=512,
            type=int,
            metavar="K",
            help="dim. of the feature vector (default: 1024)",
        )
        parser.add_argument(
            "--symfn", default="max", choices=["max", "avg"], help="symmetric function (default: max)"
        )
        parser.add_argument(
            "--pretrained",
            default=os.path.join(pretained_base_dir, "exp_dcp/models/best_model.t7"),
            type=str,
            metavar="PATH",
            help="path to pretrained model file (default: null (no-use))",
        )

    elif modelName == "RPMNet":
        parser.add_argument(
            "--pretrained",
            default=os.path.join(pretained_base_dir, "exp_rpmnet/models/clean-trained.pth"),
            type=str,
            metavar="PATH",
            help="path to pretrained model file (default: null (no-use))",
        )
    
    elif modelName == "R3PMNet":
        parser.add_argument(
            "--pretrained",
            default=os.path.join(pretained_base_dir, "exp_rpmnet/models/clean-trained.pth"),
            type=str,
            metavar="PATH",
            help="path to pretrained model file (default: null (no-use))",
        )

    elif modelName == "PCRNet":
        parser.add_argument(
            "--emb_dims",
            default=1024,
            type=int,
            metavar="K",
            help="dim. of the feature vector (default: 1024)",
        )
        parser.add_argument(
            "--symfn", default="max", choices=["max", "avg"], help="symmetric function (default: max)"
        )
        parser.add_argument(
            "--pretrained",
            default=os.path.join(pretained_base_dir, "exp_ipcrnet/models/best_model.t7"),
            type=str,
            metavar="PATH",
            help="path to pretrained model file (default: null (no-use))",
        )

    elif modelName == "PointNetLK":
        parser.add_argument(
            "--emb_dims",
            default=1024,
            type=int,
            metavar="K",
            help="dim. of the feature vector (default: 1024)",
        )
        parser.add_argument(
            "--symfn", default="max", choices=["max", "avg"], help="symmetric function (default: max)"
        )
        parser.add_argument(
            "--pretrained",
            default=os.path.join(pretained_base_dir, "exp_pnlk/models/best_model.t7"),
            type=str,
            metavar="PATH",
            help="path to pretrained model file (default: null (no-use))",
        )

    elif modelName == "PRNet":
        parser.add_argument(
            "--emb_dims",
            default=512,
            type=int,
            metavar="K",
            help="dim. of the feature vector (default: 1024)",
        )
        parser.add_argument("--num_iterations", default=3, type=int, help="Number of Iterations")
        parser.add_argument(
            "-j",
            "--workers",
            default=4,
            type=int,
            metavar="N",
            help="number of data loading workers (default: 4)",
        )
        parser.add_argument(
            "-b",
            "--batch_size",
            default=1,
            type=int,
            metavar="N",
            help="mini-batch size (default: 32)",
        )
        parser.add_argument(
            "--pretrained",
            default=os.path.join(pretained_base_dir, "exp_prnet/models/best_model.t7"),
            type=str,
            metavar="PATH",
            help="path to pretrained model file (default: null (no-use))",
        )

    parser.add_argument(
        "--device", default="cuda:0", type=str, metavar="DEVICE", help="use CUDA if available"
    )

    if "ipykernel" in sys.argv[0]:
        args, _unknown = parser.parse_known_args([])
    else:
        args, _unknown = parser.parse_known_args()

    return args
