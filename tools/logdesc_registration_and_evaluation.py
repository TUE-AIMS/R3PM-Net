import copy
import importlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import open3d as o3d
import torch

from tools import metrics
from r3pm_net.config_loader import get_method_paths


@dataclass
class _LoGDescRunner:
    logdesc_root: Path
    weights_path: Path
    device: torch.device
    model: torch.nn.Module
    sample_radius: float
    max_keypoints: int
    num_points_per_sample: int


_RUNNER: Optional[_LoGDescRunner] = None
_METHOD_CFG = get_method_paths().get("logdesc", {})


def _resolve_path(root: Path, p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (root / p)


def _kabsch_svd(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Estimate rigid transform mapping P -> Q using SVD.
    Returns a 4x4 matrix T where q ≈ R p + t.
    """
    if P.shape != Q.shape or P.ndim != 2 or P.shape[1] != 3:
        raise ValueError(f"Expected P,Q shape (N,3) equal; got {P.shape} and {Q.shape}")

    up = P.mean(axis=0)
    uq = Q.mean(axis=0)
    P_centered = P - up
    Q_centered = Q - uq

    # Same convention as LoGDesc's `solve_icp` in `mvp_test.py`.
    H = Q_centered.T @ P_centered
    U, _s, Vh = np.linalg.svd(H, full_matrices=True, compute_uv=True)
    R = U @ Vh
    if np.linalg.det(R) < 0:
        Vh[-1, :] *= -1.0
        R = U @ Vh
    t = uq - (R @ up)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _init_runner(
    logdesc_root: Path,
    weights_path: Path,
    *,
    device: Optional[str | torch.device] = None,
    sample_radius: float = 0.3,
    max_keypoints: int = 768,
    num_points_per_sample: int = 128,
    sinkhorn_iterations: int = 50,
    descriptor_dim: int = 132,
    L: int = 6,
    use_kpt: bool = False,
) -> _LoGDescRunner:
    # Make LoGDesc importable without installation.
    if str(logdesc_root) not in sys.path:
        sys.path.insert(0, str(logdesc_root))

    # IMPORTANT: other methods (e.g., OverlapPredator) import a top-level `models` package,
    # which collides with LoGDesc's own `models/` package. If `models` is already in
    # `sys.modules`, Python will not re-resolve it from the updated `sys.path`, and
    # importing `models.LoGDesc_reg` will fail.
    #
    # We temporarily clear `models` from `sys.modules` during the import, then restore
    # the previous entries to avoid breaking other runners.
    models_file = logdesc_root / "models" / "LoGDesc_reg.py"
    if not models_file.exists():
        raise FileNotFoundError(f"LoGDesc not found under: {logdesc_root} (missing {models_file})")

    prev_models_modules = {k: v for k, v in sys.modules.items() if k == "models" or k.startswith("models.")}
    for k in list(prev_models_modules.keys()):
        sys.modules.pop(k, None)
    try:
        LoGDesc_reg = importlib.import_module("models.LoGDesc_reg").LoGDesc_reg  # type: ignore[attr-defined]
    finally:
        # Remove LoGDesc-loaded `models.*` entries, then restore previous ones.
        new_models_modules = [k for k in list(sys.modules.keys()) if k == "models" or k.startswith("models.")]
        for k in new_models_modules:
            sys.modules.pop(k, None)
        sys.modules.update(prev_models_modules)

    if device is None:
        device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_t = device if isinstance(device, torch.device) else torch.device(device)

    if not weights_path.exists():
        raise FileNotFoundError(f"LoGDesc weights not found: {weights_path}")

    checkpoint = torch.load(str(weights_path), map_location=device_t)
    if not isinstance(checkpoint, dict) or not checkpoint:
        raise RuntimeError(f"Unexpected LoGDesc checkpoint format at: {weights_path}")

    net_cfg = {
        "sinkhorn_iterations": int(sinkhorn_iterations),
        "descriptor_dim": int(descriptor_dim),
        "L": int(L),
        "GNN_layers": ["self", "cross"],
        "use_kpt": bool(use_kpt),
        "lr": 1e-4,
    }

    model: torch.nn.Module = LoGDesc_reg(net_cfg)

    has_module_prefix = any(str(k).startswith("module.") for k in checkpoint.keys())
    if has_module_prefix:
        model = torch.nn.DataParallel(model)

    try:
        model.load_state_dict(checkpoint, strict=True)
    except RuntimeError:
        # Fallback: try strict=False (some checkpoints differ slightly by wrapper).
        model.load_state_dict(checkpoint, strict=False)

    model = model.to(device_t)
    model.double().eval()

    return _LoGDescRunner(
        logdesc_root=logdesc_root,
        weights_path=weights_path,
        device=device_t,
        model=model,
        sample_radius=float(sample_radius),
        max_keypoints=int(max_keypoints),
        num_points_per_sample=int(num_points_per_sample),
    )


def logdesc_reg_and_eval(
    source: "o3d.geometry.PointCloud",
    target: "o3d.geometry.PointCloud",
    *,
    gt_transformation: Optional[np.ndarray] = None,
    logdesc_root: str | Path = _METHOD_CFG.get("root", "/home/ykashefbahrami/LoGDesc"),
    weights_path: str | Path = _METHOD_CFG.get("weights_path", "/home/ykashefbahrami/LoGDesc/pre-trained/best_model.pth"),
    device: Optional[str | torch.device] = None,
    sample_radius: float = 0.3,
    max_keypoints: int = 768,
    num_points_per_sample: int = 128,
    topk_matches: int = 128,
    sinkhorn_iterations: int = 50,
    descriptor_dim: int = 132,
    L: int = 6,
    use_kpt: bool = False,
) -> Tuple["o3d.geometry.PointCloud", tuple]:
    """
    Run LoGDesc on a (source, target) pair and evaluate using this repo's `common.metrics`.

    This wrapper follows the same output contract as other runners:
      returns (pc_result, eval_results)
    where eval_results matches `metrics.all_evaluations(...)`.
    """
    global _RUNNER

    logdesc_root_p = Path(logdesc_root).resolve()
    weights_path_p = _resolve_path(logdesc_root_p, weights_path).resolve()

    if (
        _RUNNER is None
        or _RUNNER.logdesc_root != logdesc_root_p
        or _RUNNER.weights_path != weights_path_p
        or _RUNNER.max_keypoints != int(max_keypoints)
        or _RUNNER.num_points_per_sample != int(num_points_per_sample)
        or abs(_RUNNER.sample_radius - float(sample_radius)) > 1e-9
    ):
        _RUNNER = _init_runner(
            logdesc_root_p,
            weights_path_p,
            device=device,
            sample_radius=sample_radius,
            max_keypoints=max_keypoints,
            num_points_per_sample=num_points_per_sample,
            sinkhorn_iterations=sinkhorn_iterations,
            descriptor_dim=descriptor_dim,
            L=L,
            use_kpt=use_kpt,
        )

    # Import preprocessing utilities after LoGDesc root is on sys.path.
    from MVP_RG.registration.dataset import get_lrfs, furthest_point_sample  # type: ignore

    src_xyz = np.asarray(source.points, dtype=np.float32)
    tgt_xyz = np.asarray(target.points, dtype=np.float32)

    if src_xyz.shape[0] < 16 or tgt_xyz.shape[0] < 16:
        # Too few points to do anything meaningful.
        est = np.eye(4, dtype=np.float64)
        pc_result = copy.deepcopy(source).transform(est)
        eval_results = metrics.all_evaluations(
            source,
            target,
            pc_result,
            time=0.0,
            gt_transformation=gt_transformation,
            est_transformation=est,
            corres=None,
        )
        return pc_result, eval_results

    # FPS keypoints (same as LoGDesc's dataset).
    if int(max_keypoints) > 0 and src_xyz.shape[0] > int(max_keypoints):
        idx0 = furthest_point_sample(src_xyz, max_points=int(max_keypoints))
    else:
        idx0 = np.arange(src_xyz.shape[0])
    if int(max_keypoints) > 0 and tgt_xyz.shape[0] > int(max_keypoints):
        idx1 = furthest_point_sample(tgt_xyz, max_points=int(max_keypoints))
    else:
        idx1 = np.arange(tgt_xyz.shape[0])

    kpts0 = src_xyz[idx0, :]
    kpts1 = tgt_xyz[idx1, :]

    lrfs0, _patches0, _knn0, plan0, omni0, aniso0 = get_lrfs(
        idx0,
        src_xyz,
        num_points_per_sample=int(num_points_per_sample),
        sample_radius=float(sample_radius),
        with_lrf=True,
    )
    lrfs1, _patches1, _knn1, plan1, omni1, aniso1 = get_lrfs(
        idx1,
        tgt_xyz,
        num_points_per_sample=int(num_points_per_sample),
        sample_radius=float(sample_radius),
        with_lrf=True,
    )

    batch = {
        "pc0": torch.from_numpy(np.asarray(kpts0)).unsqueeze(0),
        "pc1": torch.from_numpy(np.asarray(kpts1)).unsqueeze(0),
        "lrfs_i": torch.from_numpy(np.asarray(lrfs0)).unsqueeze(0),
        "lrfs_j": torch.from_numpy(np.asarray(lrfs1)).unsqueeze(0),
        "planarity0": torch.from_numpy(np.asarray(plan0)).reshape(1, -1, 1),
        "omnivariance0": torch.from_numpy(np.asarray(omni0)).reshape(1, -1, 1),
        "anisotropy0": torch.from_numpy(np.asarray(aniso0)).reshape(1, -1, 1),
        "planarity1": torch.from_numpy(np.asarray(plan1)).reshape(1, -1, 1),
        "omnivariance1": torch.from_numpy(np.asarray(omni1)).reshape(1, -1, 1),
        "anisotropy1": torch.from_numpy(np.asarray(aniso1)).reshape(1, -1, 1),
    }

    # Move to device; LoGDesc internally uses double precision.
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.to(_RUNNER.device)

    start = time.time()
    with torch.no_grad():
        out = _RUNNER.model(batch)
    end = time.time()

    # Extract matches and estimate transform.
    k0 = out["keypoints0"][0].detach().cpu().numpy()
    k1 = out["keypoints1"][0].detach().cpu().numpy()
    matches0 = out["matches0"][0].detach().cpu().numpy().astype(np.int64)
    scores0 = out["matching_scores0"][0].detach().cpu().numpy()

    valid = matches0 > -1
    mkpts0 = k0[valid]
    mkpts1 = k1[matches0[valid]]
    mconf = scores0[valid]

    est = np.eye(4, dtype=np.float64)
    if mkpts0.shape[0] >= 3:
        k = int(min(int(topk_matches), mkpts0.shape[0]))
        if k <= 0:
            k = mkpts0.shape[0]
        if mkpts0.shape[0] > k:
            # Select top-k by confidence (fast).
            top_idx = np.argpartition(-mconf, kth=k - 1)[:k]
            mkpts0_use = mkpts0[top_idx]
            mkpts1_use = mkpts1[top_idx]
        else:
            mkpts0_use = mkpts0
            mkpts1_use = mkpts1
        try:
            est = _kabsch_svd(mkpts0_use.astype(np.float64), mkpts1_use.astype(np.float64))
        except Exception:
            est = np.eye(4, dtype=np.float64)

    pc_result = copy.deepcopy(source).transform(est)
    eval_results = metrics.all_evaluations(
        source,
        target,
        pc_result,
        end - start,
        gt_transformation=gt_transformation,
        est_transformation=est,
        corres=None,
    )
    return pc_result, eval_results

