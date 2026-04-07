"""Repository-root resolution for portable paths."""

from pathlib import Path

# r3pm_net/paths.py -> parents[1] is the repository root
REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_path(*parts: str) -> str:
    """Join path segments relative to the repository root."""
    return str(REPO_ROOT.joinpath(*parts))
