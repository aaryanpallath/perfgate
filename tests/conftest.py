import numpy as np
import pytest


def make_records(key, medians_ms, *, iters=50, jitter=0.01, seed=0, fingerprint=None):
    """Synthetic result records: one per repeat, iterations scattered around that repeat's median."""
    rng = np.random.default_rng(seed)
    fp = fingerprint or {"device_name": "Test CPU", "os": "Linux", "torch": "2.x", "cpu_threads": 4}
    out = []
    for i, m in enumerate(medians_ms):
        samples = (m * (1 + rng.normal(0, jitter, iters))).tolist()
        out.append({
            "key": key, "status": "ok", "repeat": i, "samples_ms": samples,
            "first_call_ms": m * 10, "items_per_step": 8, "flops_per_step": None,
            "fingerprint": dict(fp),
        })
    return out


@pytest.fixture
def records():
    return make_records
