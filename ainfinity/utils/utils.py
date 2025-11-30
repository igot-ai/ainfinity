"""
Utility functions
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def generate_job_id(job_name: str) -> str:
    """
    Generate a unique job ID based on job name and timestamp

    Args:
        job_name: Name of the job

    Returns:
        Unique job ID
    """
    timestamp = datetime.now().isoformat()
    content = f"{job_name}-{timestamp}"
    return hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()[:12]  # nosec B324


def ensure_dir(path: Path) -> Path:
    """
    Ensure directory exists, create if not

    Args:
        path: Directory path

    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Dict[str, Any], file_path: Path) -> None:
    """
    Save data to JSON file

    Args:
        data: Data to save
        file_path: Path to save file
    """
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(file_path: Path) -> Dict[str, Any]:
    """
    Load data from JSON file

    Args:
        file_path: Path to JSON file

    Returns:
        Loaded data
    """
    with open(file_path, "r") as f:
        return json.load(f)


def sanitize_name(name: str) -> str:
    """
    Sanitize a name for use in file systems and URLs

    Args:
        name: Name to sanitize

    Returns:
        Sanitized name
    """
    # Replace spaces and special characters
    sanitized = name.lower()
    sanitized = sanitized.replace(" ", "-")

    # Keep only alphanumeric, dash, and underscore
    sanitized = "".join(c for c in sanitized if c.isalnum() or c in ["-", "_"])

    return sanitized


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"
