"""Sweep configuration: a YAML matrix expanded into individual benchmark cases."""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from .workloads import BUILDERS, DTYPES


@dataclass(frozen=True)
class Case:
    """One benchmark configuration. `key` identifies it across runs for comparison."""

    workload: str
    device: str
    dtype: str
    batch: int
    warmup: int = 10
    iters: int = 50
    sync: str = "per_iter"  # "per_iter" (correct) or "none" (to demonstrate the pitfall)
    inject_delay_ms: float = 0.0  # spin inside the timed region; used to test the gate

    @property
    def key(self) -> str:
        return f"{self.workload}/{self.device}/{self.dtype}/bs{self.batch}"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Sweep:
    name: str
    repeats: int
    cases: list[Case] = field(default_factory=list)
    timeout_s: float = 300.0


def _as_list(v) -> list:
    return v if isinstance(v, list) else [v]


def parse(doc: dict) -> Sweep:
    """Expand `matrix` (cartesian product) and drop combinations listed in `exclude`.

    An exclude entry matches a case when every field it names matches, so
    `{device: cpu, dtype: fp16}` removes all CPU fp16 cases.
    """
    for k in ("name", "matrix"):
        if k not in doc:
            raise ValueError(f"sweep config is missing {k!r}")
    m = doc["matrix"]
    axes = {
        "workload": _as_list(m.get("workload", [])),
        "device": _as_list(m.get("device", ["cpu"])),
        "dtype": _as_list(m.get("dtype", ["fp32"])),
        "batch": _as_list(m.get("batch", [1])),
    }
    if not axes["workload"]:
        raise ValueError("matrix.workload must list at least one workload")
    for w in axes["workload"]:
        if w not in BUILDERS:
            raise ValueError(f"unknown workload {w!r}")
    for d in axes["dtype"]:
        if d not in DTYPES:
            raise ValueError(f"unknown dtype {d!r}")
    excludes = doc.get("exclude", []) or []
    for ex in excludes:
        unknown = set(ex) - set(axes)
        if unknown:
            raise ValueError(f"exclude entry uses unknown fields {sorted(unknown)}")

    common = {k: doc[k] for k in ("warmup", "iters") if k in doc}
    cases = []
    for combo in itertools.product(*axes.values()):
        c = dict(zip(axes, combo, strict=True))
        if any(all(c[k] == v for k, v in ex.items()) for ex in excludes):
            continue
        cases.append(Case(**c, **common))
    repeats = int(doc.get("repeats", 3))
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    return Sweep(
        name=doc["name"], repeats=repeats, cases=cases, timeout_s=float(doc.get("timeout_s", 300))
    )


def load(path: str | Path) -> Sweep:
    with open(path) as f:
        return parse(yaml.safe_load(f))
