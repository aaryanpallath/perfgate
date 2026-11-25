import json

import pytest

from perfgate import cli, config, runner, slurm


def fake_sweep():
    return config.parse({"name": "t", "repeats": 2,
                         "matrix": {"workload": ["gemm", "mlp"], "device": ["cpu"]}})


def test_repeats_are_interleaved_and_errors_recorded(tmp_path):
    order = []

    def run_one(case, r, timeout):
        order.append((r, case.workload))
        if case.workload == "mlp":
            return {"key": case.key, "case": case.to_dict(), "repeat": r,
                    "status": "error", "error": "boom"}
        return {"key": case.key, "status": "ok", "repeat": r, "samples_ms": [1.0]}

    out = tmp_path / "r.jsonl"
    recs = runner.run_sweep(fake_sweep(), out, run_one=run_one, log=lambda s: None)
    assert order == [(0, "gemm"), (0, "mlp"), (1, "gemm"), (1, "mlp")]
    assert sum(r["status"] == "error" for r in recs) == 2
    assert len(runner.load_results([out])) == 4


def test_unavailable_device_is_skipped(tmp_path, monkeypatch):
    sw = config.parse({"name": "t", "repeats": 1,
                       "matrix": {"workload": ["gemm"], "device": ["cpu", "cuda"]}})
    monkeypatch.setattr(runner.dev, "is_available", lambda d: d == "cpu")
    recs = runner.run_sweep(sw, tmp_path / "r.jsonl", log=lambda s: None,
                            run_one=lambda c, r, t: {"key": c.key, "status": "ok"})
    assert [r["key"] for r in recs] == ["gemm/cpu/fp32/bs1"]


def test_worker_failure_becomes_error_record():
    case = config.Case(workload="gemm", device="cpu", dtype="fp32", batch=1, sync="bogus")
    rec = runner.run_case(case, 0, timeout_s=120)
    assert rec["status"] == "error"
    assert "sync mode" in rec["error"]


def _write(path, recs):
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))


def test_compare_exit_codes(tmp_path, records, capsys):
    base, same, slow = tmp_path / "b.jsonl", tmp_path / "s.jsonl", tmp_path / "x.jsonl"
    _write(base, records("k", [10.0, 10.1, 9.9, 10.0]))
    _write(same, records("k", [10.0, 9.9, 10.1, 10.0], seed=3))
    _write(slow, records("k", [13.0, 13.1, 12.9, 13.0]))
    assert cli.main(["compare", str(base), str(same)]) == 0
    assert cli.main(["compare", str(base), str(slow)]) == 1
    assert "**regression**" in capsys.readouterr().out


def test_slurm_script_has_one_task_per_case():
    sw = fake_sweep()
    s = slurm.sbatch_script(sw, "configs/x.yaml", partition="gpu", container="img.sif")
    assert "#SBATCH --array=0-1" in s
    assert "#SBATCH --partition=gpu" in s
    assert "--case-index \"$SLURM_ARRAY_TASK_ID\"" in s
    assert s.count("singularity exec --nv img.sif") == 1


def test_slurm_rejects_empty_sweep():
    sw = fake_sweep()
    sw.cases = []
    with pytest.raises(ValueError):
        slurm.sbatch_script(sw, "x.yaml")


def test_case_index_out_of_range(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("name: t\nmatrix: {workload: gemm}\n")
    assert cli.main(["sweep", str(cfg), "--out", str(tmp_path / "o.jsonl"), "--case-index", "5"]) == 2
