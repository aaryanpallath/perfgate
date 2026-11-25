"""Measure one case in the current process and print a JSON result to stdout.

The sweep runner starts a fresh interpreter for every (case, repeat). That keeps one
case's allocator state, compiled kernels and thermal history from leaking into the
next, and it means repeat-to-repeat variance includes process start-up effects — the
same variance a nightly CI job sees.
"""

from __future__ import annotations

import json
import sys
import time

import torch

from . import device as dev
from .config import Case
from .workloads import build


def _spin(ms: float) -> None:
    # Busy-wait rather than sleep: time.sleep overshoots by up to a millisecond,
    # which would swamp small injected delays.
    end = time.perf_counter() + ms / 1000.0
    while time.perf_counter() < end:
        pass


def measure(case: Case) -> dict:
    if not dev.is_available(case.device):
        raise RuntimeError(f"device {case.device!r} is not available on this machine")
    if case.sync not in ("per_iter", "none"):
        raise ValueError(f"unknown sync mode {case.sync!r}")
    wl = build(case.workload, case.batch, case.dtype, case.device)
    sync = case.sync == "per_iter"

    t0 = time.perf_counter()
    with torch.inference_mode():
        wl.step()
        dev.synchronize(case.device)
        first_ms = (time.perf_counter() - t0) * 1000.0
        for _ in range(case.warmup):
            wl.step()
        dev.synchronize(case.device)

        samples = []
        loop_t0 = time.perf_counter()
        for _ in range(case.iters):
            start = time.perf_counter()
            wl.step()
            if sync:
                dev.synchronize(case.device)
            if case.inject_delay_ms:
                # After the sync: spinning before it would overlap with GPU execution and
                # the injected slowdown would partly disappear.
                _spin(case.inject_delay_ms)
            samples.append((time.perf_counter() - start) * 1000.0)
        # Wall time for the whole timed loop, ending with a sync. With sync="none" the
        # per-iteration samples only cover kernel launches; this number shows the real cost.
        dev.synchronize(case.device)
        loop_ms = (time.perf_counter() - loop_t0) * 1000.0

    return {
        "case": case.to_dict(),
        "key": case.key,
        "samples_ms": samples,
        "first_call_ms": first_ms,
        "loop_wall_ms": loop_ms,
        "items_per_step": wl.items_per_step,
        "flops_per_step": wl.flops_per_step,
        "fingerprint": dev.fingerprint(case.device),
    }


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    case = Case(**json.loads(argv[0]))
    json.dump(measure(case), sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
