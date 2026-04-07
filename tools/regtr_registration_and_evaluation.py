import copy
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


@dataclass
class _RegTRRunner:
    regtr_root: Path
    regtr_src: Path
    ckpt_path: Path
    config_path: Path
    device: torch.device
    cfg: edict
    model: torch.nn.Module
    num_points: int


_RUNNER: Optional[_RegTRRunner] = None
_METHOD_CFG = get_method_paths().get("regtr", {})


class _RegTRImportContext:
    """Temporarily make RegTR's `src/` importable without polluting global imports.

    RegTR uses top-level packages like `models` and `utils`, which can collide with
    other third-party repos loaded into the same Python process (e.g. OverlapPredator).
    We therefore:
      - temporarily add RegTR `src/` to sys.path
      - import the needed symbols
      - then restore sys.path and restore common conflicting sys.modules entries
    """

    _CONFLICT_PREFIXES = (
        "models",
        "utils",
        "cvhelpers",
        "data_loaders",
        "datasets",
        "kernels",
    )

    def __init__(self, regtr_src: Path):
        self.regtr_src = regtr_src
        self._inserted = False
        self._prev_modules: dict[str, object] = {}
        self._cleared_keys: set[str] = set()

    def _iter_conflicting_module_keys(self) -> list[str]:
        keys: list[str] = []
        for prefix in self._CONFLICT_PREFIXES:
            if prefix in sys.modules:
                keys.append(prefix)
            dot = prefix + "."
            for k in list(sys.modules.keys()):
                if k.startswith(dot):
                    keys.append(k)
        # de-dup while preserving order
        seen = set()
        out = []
        for k in keys:
            if k not in seen:
                seen.add(k)
                out.append(k)
        return out

    def __enter__(self):
        if str(self.regtr_src) not in sys.path:
            sys.path.insert(0, str(self.regtr_src))
            self._inserted = True

        # Save & clear potentially-colliding modules so `import models...` resolves
        # to RegTR's `src/models`, not some other repo's `models` package.
        for k in self._iter_conflicting_module_keys():
            if k in sys.modules:
                self._prev_modules[k] = sys.modules[k]
                sys.modules.pop(k, None)
                self._cleared_keys.add(k)
        return self

    def __exit__(self, exc_type, exc, tb):
        # First remove any RegTR-introduced modules under the same prefixes, then restore.
        for prefix in self._CONFLICT_PREFIXES:
            sys.modules.pop(prefix, None)
            dot = prefix + "."
            for k in list(sys.modules.keys()):
                if k.startswith(dot):
                    sys.modules.pop(k, None)

        for k, mod in self._prev_modules.items():
            sys.modules[k] = mod

        # Remove RegTR src path if we inserted it.
        if self._inserted:
            try:
                sys.path.remove(str(self.regtr_src))
            except ValueError:
                pass
        return False


def _maybe_downsample_xyz(xyz: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or xyz.shape[0] <= max_points:
        return xyz
    idx = np.random.permutation(xyz.shape[0])[:max_points]
    return xyz[idx]


def _init_runner(
    regtr_root: Path,
    ckpt_path: Path,
    config_path: Path,
    *,
    device: Optional[str | torch.device] = None,
) -> _RegTRRunner:
    regtr_src = (regtr_root / "src").resolve()
    if not regtr_src.exists():
        raise FileNotFoundError(f"RegTR src directory not found: {regtr_src}")

    if device is None:
        device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_t = device if isinstance(device, torch.device) else torch.device(device)

    with _RegTRImportContext(regtr_src):
        from utils.misc import load_config  # type: ignore
        from models.regtr import RegTR  # type: ignore

        cfg = edict(load_config(str(config_path)))
        model = RegTR(cfg).to(device_t)

        state = torch.load(str(ckpt_path), map_location=device_t)
        state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
        model.load_state_dict(state_dict, strict=False)
        model.eval()

    num_points = int(getattr(cfg, "num_points", 1024) or 1024)
    return _RegTRRunner(
        regtr_root=regtr_root,
        regtr_src=regtr_src,
        ckpt_path=ckpt_path,
        config_path=config_path,
        device=device_t,
        cfg=cfg,
        model=model,
        num_points=num_points,
    )


def regtr_reg_and_eval(
    source: "o3d.geometry.PointCloud",
    target: "o3d.geometry.PointCloud",
    *,
    gt_transformation: Optional[np.ndarray] = None,
    regtr_root: str | Path = _METHOD_CFG.get("root", "/home/ykashefbahrami/RegTR"),
    ckpt_path: str | Path = _METHOD_CFG.get("ckpt_path", "/home/ykashefbahrami/RegTR/trained_models/modelnet/ckpt/model-best.pth"),
    config_path: str | Path = _METHOD_CFG.get("config_path", "/home/ykashefbahrami/RegTR/trained_models/modelnet/config.yaml"),
    device: Optional[str | torch.device] = None,
) -> Tuple["o3d.geometry.PointCloud", tuple]:
    """Run RegTR (ModelNet checkpoint) on a (source, target) pair and evaluate.

    Returns:
      pc_result: transformed copy of `source` (using estimated pose src->tgt)
      eval_results: tuple shaped like `metrics.all_evaluations(...)` with GT provided
    """
    global _RUNNER

    regtr_root_p = Path(regtr_root).resolve()
    ckpt_path_p = Path(ckpt_path).resolve()
    config_path_p = Path(config_path).resolve()

    if not ckpt_path_p.exists():
        raise FileNotFoundError(
            f"RegTR checkpoint not found: {ckpt_path_p}\n"
            f"Expected ModelNet weights at: {regtr_root_p}/trained_models/modelnet/ckpt/model-best.pth"
        )
    if not config_path_p.exists():
        raise FileNotFoundError(
            f"RegTR config not found: {config_path_p}\n"
            f"Expected ModelNet config at: {regtr_root_p}/trained_models/modelnet/config.yaml"
        )

    if device is None:
        requested_device = None
    else:
        requested_device = device if isinstance(device, torch.device) else torch.device(device)

    if (
        _RUNNER is None
        or _RUNNER.regtr_root != regtr_root_p
        or _RUNNER.ckpt_path != ckpt_path_p
        or _RUNNER.config_path != config_path_p
        or (requested_device is not None and _RUNNER.device != requested_device)
    ):
        _RUNNER = _init_runner(regtr_root_p, ckpt_path_p, config_path_p, device=device)

    src_xyz = np.asarray(source.points, dtype=np.float32)
    tgt_xyz = np.asarray(target.points, dtype=np.float32)
    src_xyz = _maybe_downsample_xyz(src_xyz, _RUNNER.num_points)
    tgt_xyz = _maybe_downsample_xyz(tgt_xyz, _RUNNER.num_points)

    # Build batch the way RegTR expects it: list-of-tensors per batch element.
    data_batch = {
        "src_xyz": [torch.from_numpy(src_xyz).float().to(_RUNNER.device)],
        "tgt_xyz": [torch.from_numpy(tgt_xyz).float().to(_RUNNER.device)],
    }

    # Ensure RegTR's internal imports won't be confused by other repos.
    # The forward path itself does not re-import, but its modules reference top-level
    # packages (`models`, `utils`) which we keep isolated during the call.
    with _RegTRImportContext(_RUNNER.regtr_src):
        with torch.no_grad():
            # Warm-up to avoid first-run overhead in timings.
            _RUNNER.model(data_batch)

        start = time.time()
        with torch.no_grad():
            outputs = _RUNNER.model(data_batch)
        end = time.time()

    pose = outputs["pose"][-1, 0].detach().cpu().numpy()
    if pose.shape != (4, 4):
        # pad a row of [0, 0, 0, 1] to the pose because the pose is a 3x4 matrix in the original code
        pose = np.vstack([pose, [0, 0, 0, 1]])
        if pose.shape != (4, 4): # sanity check, should not happen
            raise ValueError(f"Unexpected RegTR pose shape: {pose.shape}")
    pose = pose.astype(np.float64)

    pc_result = copy.deepcopy(source).transform(pose)
    eval_results = metrics.all_evaluations(
        source,
        target,
        pc_result,
        end - start,
        gt_transformation=gt_transformation,
        est_transformation=pose,
        corres=None,
    )
    return pc_result, eval_results

