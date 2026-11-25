"""Real measurements on whatever hardware runs the tests (CPU in CI)."""

import pytest
import torch

from perfgate import device, worker, workloads
from perfgate.config import Case


@pytest.mark.parametrize("name", sorted(workloads.BUILDERS))
def test_every_workload_runs_on_cpu(name):
    wl = workloads.build(name, 1, "fp32", "cpu")
    with torch.inference_mode():
        out = wl.step()
    assert torch.isfinite(out).all()


def test_gemm_flops_are_exact():
    wl = workloads.build("gemm", 2, "fp32", "cpu")
    assert wl.flops_per_step == 2 * 2 * 1024**3


def test_measure_returns_requested_samples_and_fingerprint():
    rec = worker.measure(Case("mlp", "cpu", "fp32", 4, warmup=1, iters=5))
    assert len(rec["samples_ms"]) == 5
    assert all(s > 0 for s in rec["samples_ms"])
    assert rec["fingerprint"]["torch"] == torch.__version__
    assert rec["key"] == "mlp/cpu/fp32/bs4"


def test_injected_delay_shows_up_in_samples():
    plain = worker.measure(Case("mlp", "cpu", "fp32", 1, warmup=1, iters=10))
    slow = worker.measure(Case("mlp", "cpu", "fp32", 1, warmup=1, iters=10, inject_delay_ms=5.0))
    assert min(slow["samples_ms"]) >= 5.0
    assert sorted(slow["samples_ms"])[5] > sorted(plain["samples_ms"])[5] + 4.0


def test_unavailable_device_raises(monkeypatch):
    monkeypatch.setattr(device, "is_available", lambda d: False)
    with pytest.raises(RuntimeError):
        worker.measure(Case("gemm", "cuda", "fp32", 1))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs an Apple GPU")
def test_unsynchronized_gpu_timing_underreports():
    synced = worker.measure(Case("gemm", "mps", "fp32", 4, warmup=3, iters=10))
    unsynced = worker.measure(Case("gemm", "mps", "fp32", 4, warmup=3, iters=10, sync="none"))
    s_med = sorted(synced["samples_ms"])[5]
    u_med = sorted(unsynced["samples_ms"])[5]
    assert u_med < s_med / 10
    # The work still happens; it just isn't inside the timer.
    assert unsynced["loop_wall_ms"] > 0.5 * synced["loop_wall_ms"]
