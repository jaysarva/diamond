"""
Logging utilities for tracking experiment identity and reproducibility.

This module provides functions to collect system information, git metadata,
and configuration details for experiment tracking and reproducibility.
"""

import datetime
import getpass
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional
import uuid

try:
    import torch
except ImportError:
    torch = None


def get_run_id() -> str:
    """Generate a unique run ID."""
    return str(uuid.uuid4())


def get_timestamp() -> str:
    """Get current timestamp in ISO format."""
    return datetime.datetime.now().isoformat()


def get_host() -> str:
    """Get hostname."""
    return socket.gethostname()


def get_user() -> str:
    """Get current user."""
    return getpass.getuser()


def get_git_commit() -> Optional[str]:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent.parent,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_git_branch() -> Optional[str]:
    """Get current git branch name."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent.parent,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_git_is_dirty() -> bool:
    """Check if git working directory has uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "diff", "--quiet"],
            capture_output=True,
            check=False,
            cwd=Path(__file__).parent.parent,
        )
        return result.returncode != 0
    except FileNotFoundError:
        return False


def get_git_diff(diff_path: Optional[Path] = None) -> Optional[str]:
    """
    Get git diff and optionally save to file.
    
    Args:
        diff_path: Optional path to save git diff. If None, returns diff string.
    
    Returns:
        Git diff string if successful, None otherwise.
    """
    try:
        result = subprocess.run(
            ["git", "diff"],
            capture_output=True,
            text=True,
            check=False,
            cwd=Path(__file__).parent.parent,
        )
        diff_str = result.stdout if result.returncode == 0 else None
        
        if diff_path is not None and diff_str:
            diff_path.parent.mkdir(parents=True, exist_ok=True)
            diff_path.write_text(diff_str)
        
        return diff_str
    except FileNotFoundError:
        return None


def get_python_version() -> str:
    """Get Python version."""
    return sys.version


def get_torch_version() -> Optional[str]:
    """Get PyTorch version."""
    if torch is None:
        return None
    return torch.__version__


def get_cuda_version() -> Optional[str]:
    """Get CUDA version."""
    if torch is None:
        return None
    return torch.version.cuda


def get_cudnn_version() -> Optional[str]:
    """Get cuDNN version."""
    if torch is None:
        return None
    try:
        if torch.backends.cudnn.is_available():
            return str(torch.backends.cudnn.version())
        return None
    except AttributeError:
        return None


def get_gpu_name() -> Optional[str]:
    """Get GPU name."""
    if torch is None or not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.get_device_name(0)
    except Exception:
        return None


def get_gpu_count() -> int:
    """Get number of available GPUs."""
    if torch is None or not torch.cuda.is_available():
        return 0
    try:
        return torch.cuda.device_count()
    except Exception:
        return 0


def get_driver_version() -> Optional[str]:
    """Get NVIDIA driver version."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        # Return first line (first GPU's driver version)
        return result.stdout.strip().split("\n")[0] if result.stdout.strip() else None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_hydra_config_path() -> Optional[str]:
    """
    Get resolved Hydra config path.
    
    Note: This requires Hydra to be initialized. Returns None if not available.
    """
    hydra_cfg = os.environ.get("HYDRA_CONFIG_PATH")
    if hydra_cfg:
        return hydra_cfg
    
    # Try to get from Hydra's output directory structure
    hydra_output_dir = os.environ.get("HYDRA_OUTPUT_DIR")
    if hydra_output_dir:
        # Hydra typically stores config in .hydra/config.yaml
        config_path = Path(hydra_output_dir) / ".hydra" / "config.yaml"
        if config_path.exists():
            return str(config_path.resolve())
    
    return None


def get_cli_overrides() -> Optional[str]:
    """
    Get CLI overrides as a string.
    
    Note: This requires Hydra to be initialized. Returns None if not available.
    """
    # Hydra stores overrides in environment variable
    hydra_overrides = os.environ.get("HYDRA_OVERRIDES")
    if hydra_overrides:
        return hydra_overrides
    
    # Alternative: check for command line args if Hydra isn't used
    return None


def collect_all_metadata(
    seed: Optional[int] = None,
    git_diff_path: Optional[Path] = None,
    hydra_config_path: Optional[str] = None,
    cli_overrides: Optional[str] = None,
) -> Dict[str, any]:
    """
    Collect all metadata for experiment tracking.
    
    Args:
        seed: Random seed used for the experiment.
        git_diff_path: Optional path to save git diff.
        hydra_config_path: Optional resolved Hydra config path (if not auto-detected).
        cli_overrides: Optional CLI overrides string (if not auto-detected).
    
    Returns:
        Dictionary containing all collected metadata.
    """
    metadata = {
        "run_id": get_run_id(),
        "timestamp": get_timestamp(),
        "host": get_host(),
        "user": get_user(),
        "git_commit": get_git_commit(),
        "git_branch": get_git_branch(),
        "git_is_dirty": get_git_is_dirty(),
        "python_version": get_python_version(),
        "torch_version": get_torch_version(),
        "cuda_version": get_cuda_version(),
        "cudnn_version": get_cudnn_version(),
        "gpu_name": get_gpu_name(),
        "gpu_count": get_gpu_count(),
        "driver": get_driver_version(),
        "seed": seed,
    }
    
    # Handle git diff path
    if git_diff_path is not None:
        diff_str = get_git_diff(git_diff_path)
        metadata["git_diff_path"] = str(git_diff_path) if diff_str else None
    else:
        metadata["git_diff_path"] = None
    
    # Handle Hydra config path
    if hydra_config_path is not None:
        metadata["hydra_config_path"] = hydra_config_path
    else:
        metadata["hydra_config_path"] = get_hydra_config_path()
    
    # Handle CLI overrides
    if cli_overrides is not None:
        metadata["cli_overrides"] = cli_overrides
    else:
        metadata["cli_overrides"] = get_cli_overrides()
    
    return metadata


def format_metadata_for_logging(metadata: Dict[str, any]) -> Dict[str, any]:
    """
    Format metadata dictionary for logging (e.g., convert Path objects to strings).
    
    Args:
        metadata: Metadata dictionary from collect_all_metadata.
    
    Returns:
        Formatted metadata dictionary suitable for JSON serialization.
    """
    formatted = {}
    for key, value in metadata.items():
        if isinstance(value, Path):
            formatted[key] = str(value)
        elif value is None:
            formatted[key] = None
        else:
            formatted[key] = value
    return formatted
