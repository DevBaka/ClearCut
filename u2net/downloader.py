import hashlib
from pathlib import Path
from typing import Callable, Optional

# Use weights from local directory inside the package: u2net/models/
DEFAULT_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_DIR.mkdir(parents=True, exist_ok=True)

def ensure_weights(model_name: str = "u2netp", models_dir: Path = DEFAULT_DIR, progress: Optional[Callable[[str, int, int], None]] = None) -> Path:
    """
    Return the path to local weights. No downloading is performed.
    Place files as:
      u2net/models/u2netp.pth
      u2net/models/u2net.pth
    """
    if model_name not in {"u2netp", "u2net", "auto"}:
        raise ValueError("model_name must be 'u2netp', 'u2net', or 'auto'")
    if model_name == "auto":
        # prefer small model
        for candidate in ("u2netp.pth", "u2net.pth"):
            p = models_dir / candidate
            if p.exists():
                return p
        for candidate in ("u2netp.pth", "u2net.pth"):
            p = Path.cwd() / "models" / candidate
            if p.exists():
                return p
        raise FileNotFoundError(
            f"Keine Gewichte gefunden. Bitte lege 'u2netp.pth' oder 'u2net.pth' in '{models_dir}' ab.")
    filename = f"{model_name}.pth"
    file_path = models_dir / filename
    if file_path.exists():
        return file_path
    # Also check project root models/ for convenience
    alt = Path.cwd() / "models" / filename
    if alt.exists():
        return alt
    raise FileNotFoundError(
        f"Gewichtsdatei nicht gefunden: {file_path}. Bitte lege '{filename}' in '{models_dir}' ab."
    )


def ensure_default_weights(progress: Optional[Callable[[str, int, int], None]] = None) -> Path:
    """Return available weights, preferring u2netp."""
    return ensure_weights("auto", DEFAULT_DIR, progress)
