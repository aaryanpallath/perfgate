"""Device helpers: availability, synchronization and an environment fingerprint.

GPU work in PyTorch is asynchronous. A timer read right after a kernel launch measures
how long it took to *queue* the work, not to run it. Every timed region must therefore
end with a device synchronize; `synchronize` is the single place that does it.
"""

from __future__ import annotations

import os
import platform
import subprocess
from functools import cache

import torch

KNOWN_DEVICES = ("cpu", "mps", "cuda")


def is_available(device: str) -> bool:
    if device == "cpu":
        return True
    if device == "mps":
        return torch.backends.mps.is_available()
    if device == "cuda":
        return torch.cuda.is_available()
    raise ValueError(f"unknown device {device!r}; expected one of {KNOWN_DEVICES}")


def synchronize(device: str) -> None:
    """Block until all queued work on `device` has finished."""
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()
    # CPU ops run eagerly on the calling thread; nothing to wait for.


@cache
def _sysctl(key: str) -> str | None:
    try:
        out = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def device_name(device: str) -> str:
    if device == "cuda":
        return torch.cuda.get_device_name(0)
    if device == "mps":
        # MPS exposes no device-name API; on Apple silicon the GPU is part of the SoC.
        return _sysctl("machdep.cpu.brand_string") or "Apple GPU (MPS)"
    return _sysctl("machdep.cpu.brand_string") or platform.processor() or platform.machine()


@cache
def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def fingerprint(device: str) -> dict:
    """What a result depends on besides the workload itself.

    Two results are only comparable if these match; `compare` warns when they don't.
    """
    fp = {
        "device": device,
        "device_name": device_name(device),
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cpu_threads": torch.get_num_threads(),
        "logical_cpus": os.cpu_count(),
        "git_sha": _git_sha(),
    }
    if device == "cuda":
        fp["cuda_runtime"] = torch.version.cuda
        fp["cudnn"] = torch.backends.cudnn.version()
    return fp
