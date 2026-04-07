"""Load YAML training config and merge with argparse."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Mapping

import yaml

from r3pm_net.paths import REPO_ROOT


def _resolve_maybe_relative(path_str: str | None) -> str | None:
    if path_str is None or path_str == "":
        return path_str
    p = Path(path_str)
    if p.is_absolute():
        return str(p)
    return str(REPO_ROOT / p)


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Config must be a mapping, got {type(data)}")
    return dict(data)


def _extract_config_argv(argv: list[str], default_cfg: str) -> tuple[str, list[str]]:
    """Return (config path for YAML, argv without --config ...)."""
    path = default_cfg
    out: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == "--config" and i + 1 < len(argv):
            path = argv[i + 1]
            i += 2
            continue
        if argv[i].startswith("--config="):
            path = argv[i].split("=", 1)[1]
            i += 1
            continue
        out.append(argv[i])
        i += 1
    return path, out


def parse_train_args(argv: list[str], build_parser) -> argparse.Namespace:
    """Load YAML from --config (default: config/default.yaml), merge as argparse defaults, then parse CLI."""
    default_cfg = str(REPO_ROOT / "config" / "default.yaml")
    cfg_path, argv_rest = _extract_config_argv(list(argv), default_cfg)
    cfg = load_yaml_config(cfg_path) if Path(cfg_path).is_file() else {}
    parser = build_parser(cfg_path)
    if cfg:
        known = {
            a.dest
            for a in parser._actions
            if getattr(a, "dest", None) and a.dest not in ("help", argparse.SUPPRESS)
        }
        filtered = {k: v for k, v in cfg.items() if k in known}
        parser.set_defaults(**filtered)
    return parser.parse_args(argv_rest)


def resolve_path_args(ns: Any, path_keys: tuple[str, ...]) -> None:
    """Mutate namespace: resolve listed keys to absolute paths under REPO_ROOT when relative."""
    for key in path_keys:
        val = getattr(ns, key, None)
        if isinstance(val, str) and val:
            setattr(ns, key, _resolve_maybe_relative(val))


def load_eval_yaml() -> dict[str, Any]:
    """Load ``config/eval.yaml`` if present; otherwise return an empty dict."""
    path = REPO_ROOT / "config" / "eval.yaml"
    if not path.is_file():
        return {}
    return load_yaml_config(path)


def get_pretrained_rpmnet_dir() -> str:
    """Directory containing ``clean-trained.pth``, ``best_model_PointNet*.t7``, etc.

    ``R3PM_NET_PRETRAINED_ROOT`` overrides ``pretrained_rpmnet_dir`` in ``config/eval.yaml``.
    """
    env = os.environ.get("R3PM_NET_PRETRAINED_ROOT")
    if env:
        return str(Path(env).expanduser().resolve())
    cfg = load_eval_yaml()
    rel = (cfg.get("pretrained_rpmnet_dir") or "checkpoints").strip()
    if not rel:
        rel = "checkpoints"
    out = _resolve_maybe_relative(rel)
    return out if out else str(REPO_ROOT / "checkpoints")


def get_sioux_data_root() -> str:
    """Base data directory for Sioux scripts (``data`` / ``sioux_cranfield``, etc.)."""
    cfg = load_eval_yaml()
    sioux = cfg.get("sioux") or {}
    base = sioux.get("base_dir") or cfg.get("data_root") or "data"
    out = _resolve_maybe_relative(str(base).strip())
    return out if out else str(REPO_ROOT / "data")


def get_modelnet40_paths() -> tuple[str, str]:
    """Return ``(dataset_path, cache_dir)`` for ModelNet40 evaluation."""
    cfg = load_eval_yaml()
    m = cfg.get("modelnet40") or {}
    ds = m.get("dataset_path", "data/ModelNet40")
    cache = m.get("cache_dir", "data/down_sampled_modelnet40")
    dsr = _resolve_maybe_relative(ds)
    cr = _resolve_maybe_relative(cache)
    return (
        dsr if dsr else str(REPO_ROOT / "data" / "ModelNet40"),
        cr if cr else str(REPO_ROOT / "data" / "down_sampled_modelnet40"),
    )


def get_method_paths() -> dict[str, Any]:
    """Return resolved path configuration for external registration methods."""
    cfg = load_eval_yaml()
    methods = cfg.get("methods") or {}
    out: dict[str, Any] = {}
    for method_name, method_cfg in methods.items():
        if not isinstance(method_cfg, Mapping):
            continue
        method_out: dict[str, Any] = {}
        for k, v in method_cfg.items():
            if isinstance(v, str) and v.strip():
                rv = _resolve_maybe_relative(v.strip())
                method_out[k] = rv if rv else v
            else:
                method_out[k] = v
        out[str(method_name)] = method_out
    return out


def get_sioux_paths() -> dict[str, Any]:
    """Return Sioux eval paths from config/eval.yaml with absolute paths."""
    cfg = load_eval_yaml()
    sioux = cfg.get("sioux") or {}
    out: dict[str, Any] = {}
    for k, v in sioux.items():
        if isinstance(v, str) and v.strip():
            rv = _resolve_maybe_relative(v.strip())
            out[k] = rv if rv else v
        elif isinstance(v, list):
            vals = []
            for item in v:
                if isinstance(item, str) and item.strip():
                    rv = _resolve_maybe_relative(item.strip())
                    vals.append(rv if rv else item)
                else:
                    vals.append(item)
            out[k] = vals
        else:
            out[k] = v
    return out
