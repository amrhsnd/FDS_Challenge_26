"""Resolve local and Kaggle paths without hardcoded personal directories."""
import os
from pathlib import Path


def project_root(start=None):
    """Find this checkout from a notebook's cwd or this module's source location."""
    anchors = [Path(start).resolve()] if start is not None else [Path.cwd(), Path(__file__).resolve().parent]
    for anchor in anchors:
        for candidate in (anchor, *anchor.parents):
            if (candidate / "pyproject.toml").is_file() and (candidate / "src/pokemon_challenge").is_dir():
                return candidate
    raise FileNotFoundError("Cannot find the project root; run inside the repository checkout.")


def data_dir():
    """Find official inputs; POKEMON_DATA_DIR overrides automatic detection."""
    override = os.environ.get("POKEMON_DATA_DIR")
    if override:
        result = Path(override).expanduser().resolve()
        if not result.is_dir():
            raise FileNotFoundError(f"POKEMON_DATA_DIR is not a directory: {result}")
        return result
    candidates = [
        Path("/kaggle/input/competitions/fds-pokemon-battles-prediction-2026"),
        Path("/kaggle/input/fds-pokemon-battles-prediction-2026"),
    ]
    try:
        root = project_root()
        candidates.extend([root / "data/raw", root / "data"])
    except FileNotFoundError:
        pass
    for candidate in candidates:
        if (candidate / "train.jsonl").is_file():
            return candidate
    raise FileNotFoundError("Put official files in data/raw/ or set POKEMON_DATA_DIR.")


def output_dir(owner, run):
    """Create an isolated output directory for one owner's run."""
    for value in (owner, run):
        if not isinstance(value, str) or not value or value in {".", ".."} or "/" in value or "\\" in value:
            raise ValueError("Owner and run must each be a single nonempty folder name.")
    base = Path("/kaggle/working/outputs") if Path("/kaggle/working").is_dir() else project_root() / "outputs"
    destination = base / owner / run
    destination.mkdir(parents=True, exist_ok=True)
    return destination
