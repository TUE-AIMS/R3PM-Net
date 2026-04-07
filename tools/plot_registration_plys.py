'''
Plot .ply files savedin results, etc.
Run on Google Colab or locally when registration is done on a headless server.
'''

import sys
from pathlib import Path
from common.visualization import plot_point_cloud

def _this_dir() -> Path:
    # __file__ is not defined in notebooks / interactive sessions.
    try:
        return Path(__file__).resolve().parent  # type: ignore[name-defined]
    except NameError:
        return Path.cwd()


def _ensure_imports() -> None:
    # Allow running this file directly from the repo root.
    repo_root = _this_dir()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _group_key_and_role(name: str) -> tuple[str | None, str | None]:
    if name.endswith("_source_transformed.ply"):
        return name[: -len("_source_transformed.ply")], "result"
    if name.endswith("_source.ply"):
        return name[: -len("_source.ply")], "source"
    if name.endswith("_target.ply"):
        return name[: -len("_target.ply")], "target"
    return None, None


def main() -> int:
    _ensure_imports()

    try:
        import open3d as o3d  # type: ignore
    except Exception as e:
        print(f"ERROR: open3d is required to read .ply files: {e}")
        return 2

    default_dir = _this_dir() / "results" / "registration_plys"
    colab_drive_dir = Path("/content/drive/MyDrive/Colab Notebooks/registration_plys")

    # Super-simple CLI: optionally pass the directory as first non-flag argument
    # that actually exists and is a directory.
    # (In notebooks/IPython, argv often contains things like "-f <kernel.json>".)
    dir_args = []
    for a in sys.argv[1:]:
        if a.startswith("-"):
            continue
        p = Path(a).expanduser()
        if p.exists() and p.is_dir():
            dir_args.append(p)

    if dir_args:
        ply_dir = dir_args[0]
    else:
        ply_dir = colab_drive_dir if colab_drive_dir.exists() else default_dir

    ply_paths = sorted(ply_dir.glob("*.ply"))
    if not ply_paths:
        print(f"No .ply files found in: {ply_dir.resolve()}")
        return 0

    groups: dict[str, dict[str, Path]] = {}
    for p in ply_paths[:6]:
        key, role = _group_key_and_role(p.name)
        if key is None or role is None:
            continue
        groups.setdefault(key, {})[role] = p

    for key in sorted(groups.keys()):
        g = groups[key]
        if not ("source" in g and "target" in g and "result" in g):
            continue

        source = o3d.io.read_point_cloud(str(g["source"]))
        target = o3d.io.read_point_cloud(str(g["target"]))
        result = o3d.io.read_point_cloud(str(g["result"]))

        print(f"Plotting group: {key}")
        plot_point_cloud(source, target, result=result)

    return 0


if __name__ == "__main__":
    main()

