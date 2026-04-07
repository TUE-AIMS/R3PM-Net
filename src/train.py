import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

# Repository root on PYTHONPATH (for `python src/train.py` or srun).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from r3pm_net.model import R3PMNet
from r3pm_net.config_loader import parse_train_args, resolve_path_args
from r3pm_net.paths import REPO_ROOT
from thirdparty.learning3d.losses import FrobeniusNormLoss, RMSEFeaturesLoss
from dataloader.user_data import UserData
from r3pm_net.feature_extractor import feature_extractor  # import your feature extractor here

def _init_(args):
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / "checkpoints" / args.exp_name).mkdir(parents=True, exist_ok=True)

    if os.path.isfile("main.py"):
        os.system("cp main.py checkpoints" + "/" + args.exp_name + "/" + "main.py.backup")
    if os.path.isfile("model.py"):
        os.system("cp model.py checkpoints" + "/" + args.exp_name + "/" + "model.py.backup")


class IOStream:
    def __init__(self, path):
        self.f = open(path, "a")

    def cprint(self, text):
        print(text)
        self.f.write(text + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def test_one_epoch(device, model, test_loader):
    model.eval()
    test_loss = 0.0
    count = 0
    for i, data in enumerate(tqdm(test_loader)):
        template, source, igt = data

        template = template.to(device)
        source = source.to(device)
        igt = igt.to(device)

        output = model(template, source)
        loss_val = FrobeniusNormLoss()(output["est_T"], igt) + RMSEFeaturesLoss()(output["r"])

        test_loss += loss_val.item()
        count += 1

    test_loss = float(test_loss) / count
    return test_loss


def test(args, model, test_loader, textio):
    test_loss = test_one_epoch(args.device, model, test_loader)
    textio.cprint("Validation Loss: %f" % (test_loss))


def train_one_epoch(device, model, train_loader, optimizer):
    model.train()
    train_loss = 0.0
    count = 0
    for i, data in enumerate(tqdm(train_loader)):
        template, source, igt = data

        template = template.to(device)
        source = source.to(device)
        igt = igt.to(device)

        output = model(template, source)
        loss_val = FrobeniusNormLoss()(output["est_T"], igt) + RMSEFeaturesLoss()(output["r"])

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        train_loss += loss_val.item()
        count += 1

    train_loss = float(train_loss) / count
    return train_loss


def train(args, model, train_loader, test_loader, boardio, textio, checkpoint):
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    learnable_params = filter(lambda p: p.requires_grad, model.parameters())
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(learnable_params)
    else:
        optimizer = torch.optim.SGD(learnable_params, lr=0.1)

    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    best_test_loss = np.inf

    for epoch in range(args.start_epoch, args.epochs):
        train_loss = train_one_epoch(args.device, model, train_loader, optimizer)
        test_loss = test_one_epoch(args.device, model, test_loader)

        snap = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "min_loss": test_loss,
            "optimizer": optimizer.state_dict(),
        }

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_snap_path = os.path.join(
                args.save_dir, "best_model_snap.t7")
            best_model_path = os.path.join(
                args.save_dir, "best_model.t7")

            torch.save(snap, best_snap_path)
            torch.save(model.state_dict(), best_model_path)

        torch.save(snap, os.path.join(args.save_dir, "model_snap.t7"))
        torch.save(model.state_dict(), os.path.join(args.save_dir, "model.t7"))

        boardio.add_scalar("Train Loss", train_loss, epoch + 1)
        boardio.add_scalar("Test Loss", test_loss, epoch + 1)
        boardio.add_scalar("Best Test Loss", best_test_loss, epoch + 1)

        textio.cprint(
            "EPOCH:: %d, Traininig Loss: %f, Testing Loss: %f, Best Loss: %f"
            % (epoch + 1, train_loss, test_loss, best_test_loss)
        )


def build_parser(default_config_path: str):
    parser = argparse.ArgumentParser(description="Point Cloud Registration")
    parser.add_argument(
        "--config",
        type=str,
        default=default_config_path,
        help="YAML file with defaults (see config/default.yaml); can be overridden on the command line",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="exp_r3pmnet",
        metavar="N",
        help="Name of the experiment",
    )
    parser.add_argument("--eval", action="store_true", help="Run evaluation only (no training).")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="",
        help="Directory to save model checkpoints (default: checkpoints/<exp_name>/models)",
    )

    parser.add_argument(
        "--num_points",
        default=1024,
        type=int,
        metavar="N",
        help="points in point-cloud (default: 1024)",
    )

    parser.add_argument(
        "--fine_tune_feature_extractor",
        default="tune",
        type=str,
        choices=["fixed", "tune"],
        help="train feature extractor (default: tune)",
    )
    parser.add_argument(
        "--transfer_weights",
        default="",
        type=str,
        metavar="PATH",
        help="optional path to feature extractor checkpoint",
    )
    parser.add_argument(
        "--symfn",
        default="max",
        choices=["max", "avg"],
        help="symmetric function (default: max)",
    )

    parser.add_argument("--seed", type=int, default=1234)
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
        default=5,
        type=int,
        metavar="N",
        help="mini-batch size (default: 5)",
    )
    parser.add_argument(
        "--epochs",
        default=50,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--start_epoch",
        default=0,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument(
        "--optimizer",
        default="Adam",
        choices=["Adam", "SGD"],
        metavar="METHOD",
        help="name of an optimizer (default: Adam)",
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--pretrained",
        default="",
        type=str,
        metavar="PATH",
        help="path to pretrained full model (default: none)",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        type=str,
        metavar="DEVICE",
        help="use CUDA if available",
    )

    parser.add_argument(
        "--train_dict_path",
        type=str,
        default="data/simulators/data_dict_train.pkl",
        help="Pickled training data_dict",
    )
    parser.add_argument(
        "--test_dict_path",
        type=str,
        default="data/simulators/data_dict_test.pkl",
        help="Pickled test data_dict",
    )

    return parser


def _torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def main():
    args = parse_train_args(sys.argv[1:], build_parser)

    resolve_path_args(
        args,
        (
            "save_dir",
            "train_dict_path",
            "test_dict_path",
            "resume",
            "pretrained",
            "transfer_weights",
        ),
    )

    if not args.save_dir:
        args.save_dir = str(REPO_ROOT / "checkpoints" / args.exp_name / "models")

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    ckpt_dir = REPO_ROOT / "checkpoints" / args.exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    boardio = SummaryWriter(log_dir=str(ckpt_dir))
    _init_(args)

    textio = IOStream(str(ckpt_dir / "run.log"))
    textio.cprint(str(args))

    if not os.path.isfile(args.train_dict_path):
        raise FileNotFoundError(f"Training dict not found: {args.train_dict_path}")
    if not os.path.isfile(args.test_dict_path):
        raise FileNotFoundError(f"Test dict not found: {args.test_dict_path}")

    with open(args.train_dict_path, "rb") as f:
        data_dict_train = pickle.load(f)
    with open(args.test_dict_path, "rb") as f:
        data_dict_test = pickle.load(f)

    trainset = UserData("registration", data_dict_train)
    testset = UserData("registration", data_dict_test)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=args.workers)
    test_loader = DataLoader(testset, batch_size=5, shuffle=False, drop_last=False, num_workers=args.workers)

    if not torch.cuda.is_available():
        args.device = "cpu"
    args.device = torch.device(args.device)

    # feature extractor model
    FEATURE_MODEL = feature_extractor
    model = R3PMNet(feature_model=FEATURE_MODEL)
    model = model.to(args.device)

    if args.transfer_weights and os.path.isfile(args.transfer_weights):
        feat_model_dict = _torch_load(args.transfer_weights, args.device)
        model.feat_extractor.load_state_dict(feat_model_dict)

    checkpoint = None
    if args.resume:
        assert os.path.isfile(args.resume)
        checkpoint = _torch_load(args.resume, args.device)
        args.start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["model"])

    if args.pretrained:
        assert os.path.isfile(args.pretrained)
        try:
            model.load_state_dict(_torch_load(args.pretrained, "cpu"))
        except RuntimeError:
            model_data = _torch_load(args.pretrained, "cpu")
            state_dict = model_data["state_dict"]
            model.load_state_dict(state_dict)
    model.to(args.device)

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    if args.eval:
        test(args, model, test_loader, textio)
    else:
        train(args, model, train_loader, test_loader, boardio, textio, checkpoint)

if __name__ == "__main__":
    main()
