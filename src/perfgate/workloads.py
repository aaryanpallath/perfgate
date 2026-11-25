"""Inference workloads.

Each builder returns a zero-argument callable that runs one inference step on inputs
that already live on the device, plus the number of samples (or FLOPs) one step covers.
Weights are randomly initialized: timing does not depend on trained values, and it keeps
the benchmark free of downloads.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


@dataclass
class Workload:
    step: Callable[[], torch.Tensor]
    items_per_step: int  # samples per step, used for throughput
    flops_per_step: float | None  # set only where it is exact (GEMM)


def _gemm(batch: int, dtype: torch.dtype, device: str, size: int = 1024) -> Workload:
    a = torch.randn(batch, size, size, dtype=dtype, device=device)
    b = torch.randn(batch, size, size, dtype=dtype, device=device)
    return Workload(lambda: torch.bmm(a, b), batch, 2.0 * batch * size**3)


def _mlp(batch: int, dtype: torch.dtype, device: str) -> Workload:
    layers: list[nn.Module] = []
    width = 1024
    for _ in range(4):
        layers += [nn.Linear(width, width), nn.GELU()]
    model = nn.Sequential(*layers, nn.Linear(width, 10)).to(device=device, dtype=dtype).eval()
    x = torch.randn(batch, width, dtype=dtype, device=device)
    return Workload(lambda: model(x), batch, None)


def _resnet18(batch: int, dtype: torch.dtype, device: str) -> Workload:
    from torchvision.models import resnet18

    model = resnet18(weights=None).to(device=device, dtype=dtype).eval()
    x = torch.randn(batch, 3, 224, 224, dtype=dtype, device=device)
    return Workload(lambda: model(x), batch, None)


def _transformer(batch: int, dtype: torch.dtype, device: str, seq: int = 128) -> Workload:
    layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048, batch_first=True)
    model = nn.TransformerEncoder(layer, num_layers=6, enable_nested_tensor=False)
    model = model.to(device=device, dtype=dtype).eval()
    x = torch.randn(batch, seq, 512, dtype=dtype, device=device)
    return Workload(lambda: model(x), batch, None)


BUILDERS: dict[str, Callable[[int, torch.dtype, str], Workload]] = {
    "gemm": _gemm,
    "mlp": _mlp,
    "resnet18": _resnet18,
    "transformer": _transformer,
}


def build(name: str, batch: int, dtype: str, device: str, seed: int = 0) -> Workload:
    if name not in BUILDERS:
        raise ValueError(f"unknown workload {name!r}; expected one of {sorted(BUILDERS)}")
    if dtype not in DTYPES:
        raise ValueError(f"unknown dtype {dtype!r}; expected one of {sorted(DTYPES)}")
    if batch < 1:
        raise ValueError("batch must be >= 1")
    torch.manual_seed(seed)
    return BUILDERS[name](batch, DTYPES[dtype], device)
