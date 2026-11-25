import pytest

from perfgate import config


def doc(**kw):
    d = {"name": "t", "repeats": 2, "matrix": {"workload": ["gemm", "mlp"], "device": ["cpu", "mps"],
         "dtype": ["fp32", "fp16"], "batch": [1, 8]}}
    d.update(kw)
    return d


def test_matrix_expands_to_cartesian_product():
    sw = config.parse(doc())
    assert len(sw.cases) == 2 * 2 * 2 * 2
    assert len({c.key for c in sw.cases}) == len(sw.cases)


def test_partial_exclude_removes_every_matching_case():
    sw = config.parse(doc(exclude=[{"device": "cpu", "dtype": "fp16"}]))
    assert len(sw.cases) == 12
    assert not any(c.device == "cpu" and c.dtype == "fp16" for c in sw.cases)


def test_scalar_axes_and_common_settings():
    sw = config.parse({"name": "t", "iters": 7, "warmup": 2, "matrix": {"workload": "gemm"}})
    assert [c.key for c in sw.cases] == ["gemm/cpu/fp32/bs1"]
    assert sw.cases[0].iters == 7 and sw.cases[0].warmup == 2


@pytest.mark.parametrize("bad", [
    {"name": "t", "matrix": {"workload": ["nope"]}},
    {"name": "t", "matrix": {"workload": ["gemm"], "dtype": ["fp8"]}},
    {"name": "t", "matrix": {"workload": ["gemm"]}, "exclude": [{"gpu": "x"}]},
    {"name": "t", "matrix": {"workload": ["gemm"]}, "repeats": 0},
    {"matrix": {"workload": ["gemm"]}},
])
def test_rejects_invalid_configs(bad):
    with pytest.raises(ValueError):
        config.parse(bad)


def test_shipped_configs_parse():
    from pathlib import Path
    for p in Path(__file__).parent.parent.joinpath("configs").glob("*.yaml"):
        assert config.load(p).cases, p
