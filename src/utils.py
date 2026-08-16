"""Shared utilities: seeding, device selection, timing, IO helpers."""

from __future__ import annotations

import json
import os
import random
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch (CPU + CUDA).

    Note on determinism: cuDNN deterministic mode is enabled because this
    project uses only gather/scatter and elementwise kernels, where the cost is
    negligible. ``index_add_`` on CUDA is still atomically ordered at runtime,
    so bit-exact reproducibility across GPU runs is not guaranteed; run-to-run
    losses agree to several decimal places. CPU runs are bit-exact.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(preference: str = "auto") -> torch.device:
    """Resolve a device string ("auto" | "cuda" | "cpu") to a torch.device."""
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        return torch.device("cuda")
    if preference != "auto":
        raise ValueError(f"Unknown device preference: {preference!r}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def device_info(device: torch.device) -> Dict[str, Any]:
    """Collect a description of the compute device actually in use."""
    info: Dict[str, Any] = {"device": str(device), "gpu_name": None, "vram_gb": None}
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        info["gpu_name"] = props.name
        info["vram_gb"] = round(props.total_memory / 1024 ** 3, 2)
    return info


def print_device_banner(device: torch.device) -> Dict[str, Any]:
    """Print and return the device banner required by the project spec."""
    info = device_info(device)
    print(f"Device: {info['device']}")
    if info["gpu_name"]:
        print(f"GPU: {info['gpu_name']}")
        print(f"VRAM: approximately {info['vram_gb']} GB")
    else:
        print("GPU: none detected (CPU fallback)")
    return info


def peak_gpu_memory_mb(device: torch.device) -> Optional[float]:
    """Peak allocated CUDA memory in MB, or None on CPU."""
    if device.type != "cuda":
        return None
    return round(torch.cuda.max_memory_allocated(device) / 1024 ** 2, 2)


def reset_gpu_memory_stats(device: torch.device) -> None:
    """Reset CUDA peak-memory accounting before a measured run."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


@contextmanager
def timer(label: str = "", verbose: bool = False) -> Iterator[Dict[str, float]]:
    """Context manager measuring wall-clock seconds into ``result['seconds']``."""
    result: Dict[str, float] = {"seconds": 0.0}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["seconds"] = time.perf_counter() - start
        if verbose:
            print(f"{label}: {result['seconds']:.2f}s")


def save_json(obj: Any, path: str) -> str:
    """Write ``obj`` as UTF-8 JSON, creating parent directories as needed."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False)
    return path


def load_json(path: str) -> Any:
    """Read a UTF-8 JSON file."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def human_bytes(num_bytes: float) -> str:
    """Format a byte count using binary units."""
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TB"
