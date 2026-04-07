import copy
import importlib.util
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import open3d as o3d
import torch

from tools import metrics
from r3pm_net.config_loader import get_method_paths


@dataclass
class _GeoTransformerRunner:
    geotransformer_root: Path
    exp_dir: Path
    weights_path: Path
    device: torch.device
    cfg: Any
    model: torch.nn.Module
    neighbor_limits: list[int]


_RUNNER: Optional[_GeoTransformerRunner] = None


def _to_device(x, device: torch.device):
    """Recursively move tensors to a device (CPU or CUDA)."""
    if isinstance(x, dict):
        return {k: _to_device(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_device(v, device) for v in x]
    if isinstance(x, tuple):
        return tuple(_to_device(v, device) for v in x)
    if torch.is_tensor(x):
        return x.to(device)
    return x


def _init_runner(
    geotransformer_root: Path,
    exp_dir: Path,
    weights_path: Path,
    *,
    device: Optional[str | torch.device] = None,
    neighbor_limits: Optional[list[int]] = None,
) -> _GeoTransformerRunner:
    # Ensure GeoTransformer is importable without installation.
    # IMPORTANT: do NOT use `from config import ...` / `from model import ...` here because
    # other runners (e.g. PARENet) also import `config` and `model`, and Python caches them
    # in `sys.modules`. That can cause accidental cross-imports.
    if str(exp_dir) not in sys.path:
        sys.path.insert(0, str(exp_dir))
    if str(geotransformer_root) not in sys.path:
        sys.path.insert(0, str(geotransformer_root))

    if device is None:
        device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_t = device if isinstance(device, torch.device) else torch.device(device)

    def _load_module(mod_name: str, file_path: Path):
        spec = importlib.util.spec_from_file_location(mod_name, str(file_path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load module spec for {file_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    # Load experiment `config.py` / `model.py` by file path with unique names (avoid collisions).
    cfg_mod = _load_module(f"_geotransformer_cfg_{exp_dir.name}", exp_dir / "config.py")

    prev_backbone = sys.modules.get("backbone")
    try:
        sys.modules["backbone"] = _load_module(f"_geotransformer_backbone_{exp_dir.name}", exp_dir / "backbone.py")
        model_mod = _load_module(f"_geotransformer_model_{exp_dir.name}", exp_dir / "model.py")
    finally:
        if prev_backbone is None:
            sys.modules.pop("backbone", None)
        else:
            sys.modules["backbone"] = prev_backbone

    cfg = cfg_mod.make_cfg()

    if neighbor_limits is None:
        neighbor_limits = [256] * int(cfg.backbone.num_stages)
    if len(neighbor_limits) != int(cfg.backbone.num_stages):
        raise ValueError(
            f"GeoTransformer neighbor_limits must have length {cfg.backbone.num_stages}, got {len(neighbor_limits)}"
        )

    model = model_mod.create_model(cfg).to(device_t)
    state = torch.load(str(weights_path), map_location=device_t)
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        # Be permissive if checkpoint key names differ slightly.
        model.load_state_dict(state_dict, strict=False)
    model.eval()

    return _GeoTransformerRunner(
        geotransformer_root=geotransformer_root,
        exp_dir=exp_dir,
        weights_path=weights_path,
        device=device_t,
        cfg=cfg,
        model=model,
        neighbor_limits=neighbor_limits,
    )


def geotransformer_reg_and_eval(
    source: "o3d.geometry.PointCloud",
    target: "o3d.geometry.PointCloud",
    *,
    gt_transformation: Optional[np.ndarray] = None,
    geotransformer_root: str | Path = "/home/ykashefbahrami/GeoTransformer",
    exp_subdir: str = "experiments/geotransformer.modelnet.rpmnet.stage4.gse.k3.max.oacl.stage2.sinkhorn",
    weights_path: str | Path = "/home/ykashefbahrami/GeoTransformer/weights/geotransformer-modelnet.pth.tar",
    neighbor_limits: Optional[list[int]] = None,
    device: Optional[str | torch.device] = None,
) -> Tuple["o3d.geometry.PointCloud", tuple]:
    """
    Run GeoTransformer on a (source, target) pair and evaluate using this repo's `common.metrics`.

    Notes:
    - GeoTransformer expects `data_dict["transform"]` mapping src -> ref. If `gt_transformation`
      is not provided, we pass identity (this allows no-GT evaluation).
    - The returned `eval_results` matches `metrics.all_evaluations(...)`.
    """
    global _RUNNER

    geotransformer_root_p = Path(geotransformer_root).resolve()
    exp_dir = (geotransformer_root_p / exp_subdir).resolve()
    weights_path_p = Path(weights_path).resolve()

    if not exp_dir.exists():
        raise FileNotFoundError(f"GeoTransformer experiment directory not found: {exp_dir}")
    if not weights_path_p.exists():
        raise FileNotFoundError(f"GeoTransformer weights not found: {weights_path_p}")

    if (
        _RUNNER is None
        or _RUNNER.exp_dir != exp_dir
        or _RUNNER.weights_path != weights_path_p
        or (neighbor_limits is not None and _RUNNER.neighbor_limits != neighbor_limits)
    ):
        _RUNNER = _init_runner(
            geotransformer_root_p,
            exp_dir,
            weights_path_p,
            device=device,
            neighbor_limits=neighbor_limits,
        )

    from geotransformer.utils.data import registration_collate_fn_stack_mode

    src_points = np.asarray(source.points, dtype=np.float32)
    ref_points = np.asarray(target.points, dtype=np.float32)
    src_feats = np.ones((src_points.shape[0], 1), dtype=np.float32)
    ref_feats = np.ones((ref_points.shape[0], 1), dtype=np.float32)

    data_dict = {
        "ref_points": ref_points,
        "src_points": src_points,
        "ref_feats": ref_feats,
        "src_feats": src_feats,
    }
    if gt_transformation is None:
        data_dict["transform"] = np.eye(4, dtype=np.float32)
    else:
        data_dict["transform"] = np.asarray(gt_transformation, dtype=np.float32)

    batch = registration_collate_fn_stack_mode(
        [data_dict],
        _RUNNER.cfg.backbone.num_stages,
        _RUNNER.cfg.backbone.init_voxel_size,
        _RUNNER.cfg.backbone.init_radius,
        _RUNNER.neighbor_limits,
    )
    batch = _to_device(batch, _RUNNER.device)

    # Warm-up (avoid slow first run)
    with torch.no_grad():
        _RUNNER.model(batch)

    start = time.time()
    with torch.no_grad():
        output = _RUNNER.model(batch)
    end = time.time()

    est = output["estimated_transform"].detach().cpu().numpy()
    if est.shape == (1, 4, 4):
        est = est[0]
    if est.shape != (4, 4):
        raise ValueError(f"Unexpected GeoTransformer estimated_transform shape: {est.shape}")
    est = est.astype(np.float64)

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

