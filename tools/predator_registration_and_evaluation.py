import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import open3d as o3d
import torch
try:
    from easydict import EasyDict as edict  # type: ignore
except Exception:  # pragma: no cover
    class edict(dict):
        """Minimal EasyDict fallback (dot access)."""

        def __getattr__(self, k):
            try:
                return self[k]
            except KeyError as e:
                raise AttributeError(k) from e

        def __setattr__(self, k, v):
            self[k] = v

from tools import metrics
from r3pm_net.config_loader import get_method_paths


@dataclass
class _PredatorRunner:
    predator_root: Path
    config_path: Path
    weights_path: Path
    device: torch.device
    config: edict
    model: torch.nn.Module
    neighborhood_limits: np.ndarray
    input_num_points: int


_RUNNER: Optional[_PredatorRunner] = None
_METHOD_CFG = get_method_paths().get("predator", {})


def _build_kpconv_architecture(num_layers: int) -> list:
    # Mirrors the logic used in `master_thesis/OverlapPredator/scripts/demo.py`.
    arch = ["simple", "resnetb"]
    for _ in range(num_layers - 1):
        arch += ["resnetb_strided", "resnetb", "resnetb"]
    for _ in range(num_layers - 2):
        arch += ["nearest_upsample", "unary"]
    arch += ["nearest_upsample", "last_unary"]
    return arch


def _get_predator_architecture(cfg_in: edict) -> list:
    """
    OverlapPredator defines dataset-specific architectures in `configs/models.py`.
    We try to use that (it must match the released checkpoints), and fall back to
    the demo-style architecture builder if unavailable.
    """
    try:
        from configs.models import architectures as arch_dict  # type: ignore

        dataset_name = getattr(cfg_in, "dataset", None)
        if dataset_name in arch_dict:
            return arch_dict[dataset_name]
    except Exception:
        pass

    return _build_kpconv_architecture(int(getattr(cfg_in, "num_layers", 3)))


def _resolve_path(predator_root: Path, p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (predator_root / p)


def _maybe_downsample_xyz(xyz: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or xyz.shape[0] <= max_points:
        return xyz
    idx = np.random.permutation(xyz.shape[0])[:max_points]
    return xyz[idx]


def _to_o3d_feature(desc: np.ndarray) -> "o3d.pipelines.registration.Feature":
    feat = o3d.pipelines.registration.Feature()
    feat.data = np.asarray(desc, dtype=np.float32).T  # (C, N)
    return feat


def _ransac_pose_estimation(
    src_xyz: np.ndarray,
    tgt_xyz: np.ndarray,
    src_desc: np.ndarray,
    tgt_desc: np.ndarray,
    *,
    distance_threshold: float = 0.05,
    ransac_n: int = 3,
    mutual: bool = False,
) -> np.ndarray:
    src_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src_xyz))
    tgt_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tgt_xyz))

    src_feat = _to_o3d_feature(src_desc)
    tgt_feat = _to_o3d_feature(tgt_desc)

    estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint(False)
    checkers = [
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
    ]
    criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 1000)

    # Open3D signature varies slightly by version; support both.
    try:
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            src_pcd,
            tgt_pcd,
            src_feat,
            tgt_feat,
            mutual,
            distance_threshold,
            estimation,
            ransac_n,
            checkers,
            criteria,
        )
    except TypeError:
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source=src_pcd,
            target=tgt_pcd,
            source_feature=src_feat,
            target_feature=tgt_feat,
            mutual_filter=mutual,
            max_correspondence_distance=distance_threshold,
            estimation_method=estimation,
            ransac_n=ransac_n,
            checkers=checkers,
            criteria=criteria,
        )

    return np.asarray(result.transformation, dtype=np.float64)


def _init_runner(
    predator_root: Path,
    config_path: Path,
    weights_path: Optional[Path],
    *,
    device: Optional[str | torch.device] = None,
    input_num_points: Optional[int] = None,
    calibrate_neighborhood_limits: bool = True,
) -> _PredatorRunner:
    # Import OverlapPredator modules after adding it to sys.path.
    import sys

    if str(predator_root) not in sys.path:
        sys.path.insert(0, str(predator_root))

    from lib.utils import load_config
    from datasets.my_dataloader import calibrate_neighbors, collate_fn_descriptor
    from models.architectures import KPFCNN

    cfg = edict(load_config(str(config_path)))

    if device is None:
        device_t = torch.device("cuda" if bool(cfg.gpu_mode) and torch.cuda.is_available() else "cpu")
    else:
        device_t = torch.device(device) if not isinstance(device, torch.device) else device

    # Resolve weights path:
    ckpt_path = _resolve_path(predator_root, weights_path) if weights_path else _resolve_path(predator_root, cfg.pretrain)
    state = torch.load(str(ckpt_path), map_location=device_t)
    state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state

    def _try_build_and_load(cfg_in: edict) -> Optional[torch.nn.Module]:
        cfg_in.device = device_t
        cfg_in.architecture = _get_predator_architecture(cfg_in)
        m = KPFCNN(cfg_in).to(device_t)
        m.eval()
        try:
            m.load_state_dict(state_dict, strict=False)
        except RuntimeError:
            return None
        return m

    # First try the config as-is. If it fails (size mismatch), try common reduced widths.
    cfg_candidates: list[edict] = []
    cfg_candidates.append(cfg)

    # Avoid duplicates while exploring smaller widths.
    first_fd = int(getattr(cfg, "first_feats_dim", 0) or 0)
    for cand in [first_fd // 2, 256, 128, 64]:
        if cand and cand != first_fd:
            c = edict(dict(cfg))
            c.first_feats_dim = int(cand)
            cfg_candidates.append(c)

    model = None
    chosen_cfg = None
    for c in cfg_candidates:
        m = _try_build_and_load(c)
        if m is not None:
            model = m
            chosen_cfg = c
            break

    if model is None or chosen_cfg is None:
        # Re-raise with a clear message.
        raise RuntimeError(
            f"Failed to load OverlapPredator weights at '{ckpt_path}'. "
            f"Config '{config_path}' seems incompatible with checkpoint tensor shapes."
        )

    # Decide input sampling count (ModelNet config uses 1024).
    if input_num_points is None:
        input_num_points = int(getattr(cfg, "num_points", 1024))

    if calibrate_neighborhood_limits:
        # Calibrate neighbors once using a minimal one-sample dataset.
        class _SinglePairDataset:
            def __init__(self, config):
                self.config = config

            def __len__(self):
                return 1

            def __getitem__(self, _):
                # Minimal valid sample to satisfy collate_fn_descriptor.
                n = max(64, int(input_num_points))
                src = np.random.randn(n, 3).astype(np.float32)
                tgt = np.random.randn(n, 3).astype(np.float32)
                src_feats = np.ones((n, 1), dtype=np.float32)
                tgt_feats = np.ones((n, 1), dtype=np.float32)
                rot = np.eye(3, dtype=np.float32)
                trans = np.zeros((3, 1), dtype=np.float32)
                matching_inds = torch.ones(1, 2).long()
                sample = torch.ones(1)
                gt = np.eye(4, dtype=np.float32)
                return src, tgt, src_feats, tgt_feats, rot, trans, matching_inds, src, tgt, sample, gt

        dummy_ds = _SinglePairDataset(chosen_cfg)
        neighborhood_limits = calibrate_neighbors(dummy_ds, chosen_cfg, collate_fn=collate_fn_descriptor)
    else:
        # For tasks like parameter counting, we don't need KPConv neighborhood calibration.
        # Pick a conservative default that works for typical KPConv configs.
        n_layers = int(getattr(chosen_cfg, "num_layers", 5) or 5)
        neighborhood_limits = np.asarray([256] * n_layers, dtype=np.int32)

    return _PredatorRunner(
        predator_root=predator_root,
        config_path=config_path,
        weights_path=ckpt_path,
        device=device_t,
        config=chosen_cfg,
        model=model,
        neighborhood_limits=neighborhood_limits,
        input_num_points=int(input_num_points),
    )


def predator_reg_and_eval(
    source: "o3d.geometry.PointCloud",
    target: "o3d.geometry.PointCloud",
    *,
    gt_transformation: Optional[np.ndarray] = None,
    predator_root: str | Path = _METHOD_CFG.get("root", "/home/ykashefbahrami/master_thesis/OverlapPredator"),
    config_path: str | Path = _METHOD_CFG.get("config_path", "/home/ykashefbahrami/master_thesis/OverlapPredator/configs/test/modelnet.yaml"),
    weights_path: Optional[str | Path] = _METHOD_CFG.get("weights_path", None),
    ransac_n_points: int = 1000,
    ransac_distance_threshold: float = 0.05,
    ransac_n: int = 3,
    sampling: str = "prob",
    mutual: bool = False,
    device: Optional[str | torch.device] = None,
    input_num_points: Optional[int] = 1024,
) -> Tuple["o3d.geometry.PointCloud", tuple]:
    """
    Run OverlapPredator on a (source, target) pair and evaluate with the same
    metric outputs as the Learning3D harness in this repo.
    """
    global _RUNNER
    predator_root_p = Path(predator_root).resolve()
    config_path_p = Path(config_path).resolve()
    weights_path_p = Path(weights_path).resolve() if weights_path is not None else None

    if _RUNNER is None:
        _RUNNER = _init_runner(
            predator_root_p,
            config_path_p,
            weights_path_p,
            device=device,
            input_num_points=input_num_points,
        )

    # Import OverlapPredator collate after sys.path is set by _init_runner.
    from datasets.my_dataloader import collate_fn_descriptor

    src_xyz = np.asarray(source.points, dtype=np.float32)
    tgt_xyz = np.asarray(target.points, dtype=np.float32)
    src_xyz = _maybe_downsample_xyz(src_xyz, _RUNNER.input_num_points)
    tgt_xyz = _maybe_downsample_xyz(tgt_xyz, _RUNNER.input_num_points)

    src_feats = np.ones((src_xyz.shape[0], 1), dtype=np.float32)
    tgt_feats = np.ones((tgt_xyz.shape[0], 1), dtype=np.float32)

    rot = np.eye(3, dtype=np.float32)
    trans = np.zeros((3, 1), dtype=np.float32)
    matching_inds = torch.ones(1, 2).long()
    sample = torch.ones(1)
    gt = np.asarray(gt_transformation, dtype=np.float32) if gt_transformation is not None else np.eye(4, dtype=np.float32)

    # Collate into KPConv batch format.
    batch = collate_fn_descriptor(
        [(src_xyz, tgt_xyz, src_feats, tgt_feats, rot, trans, matching_inds, src_xyz, tgt_xyz, sample, gt)],
        config=_RUNNER.config,
        neighborhood_limits=_RUNNER.neighborhood_limits,
    )

    # Move batch tensors to device.
    for k, v in list(batch.items()):
        if isinstance(v, list):
            batch[k] = [t.to(_RUNNER.device) for t in v]
        elif torch.is_tensor(v):
            batch[k] = v.to(_RUNNER.device)

    start = time.time()
    with torch.no_grad():
        feats, scores_overlap, scores_saliency = _RUNNER.model(batch)
    feats = feats.detach().cpu()
    scores_overlap = scores_overlap.detach().cpu()
    scores_saliency = scores_saliency.detach().cpu()

    pcd = batch["points"][0].detach().cpu()
    len_src = int(batch["stack_lengths"][0][0].detach().cpu().item())
    src_pcd = pcd[:len_src]
    tgt_pcd = pcd[len_src:]

    src_desc = feats[:len_src].numpy()
    tgt_desc = feats[len_src:].numpy()
    src_scores = (scores_overlap[:len_src] * scores_saliency[:len_src]).numpy().flatten()
    tgt_scores = (scores_overlap[len_src:] * scores_saliency[len_src:]).numpy().flatten()

    def _sample_idx(scores: np.ndarray, n: int) -> np.ndarray:
        n_all = scores.shape[0]
        if n_all <= n:
            return np.arange(n_all)
        if sampling == "topk":
            return np.argsort(-scores)[:n]
        if sampling == "random":
            return np.random.permutation(n_all)[:n]
        # prob
        s = float(scores.sum())
        if not np.isfinite(s) or s <= 0.0:
            return np.random.permutation(n_all)[:n]
        probs = scores / s
        return np.random.choice(np.arange(n_all), size=n, replace=False, p=probs)

    src_idx = _sample_idx(src_scores, ransac_n_points)
    tgt_idx = _sample_idx(tgt_scores, ransac_n_points)

    tsfm = _ransac_pose_estimation(
        src_pcd[src_idx].numpy(),
        tgt_pcd[tgt_idx].numpy(),
        src_desc[src_idx],
        tgt_desc[tgt_idx],
        distance_threshold=ransac_distance_threshold,
        ransac_n=ransac_n,
        mutual=mutual,
    )
    end = time.time()

    pc_result = copy.deepcopy(source).transform(tsfm)
    eval_results = metrics.all_evaluations(
        source,
        target,
        pc_result,
        end - start,
        gt_transformation=gt_transformation,
        est_transformation=tsfm,
        corres=None,
    )

    return pc_result, eval_results

