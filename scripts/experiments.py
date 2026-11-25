"""Reproduce every number in the README. Run on a quiet machine (plugged in, apps closed).

    uv run python scripts/experiments.py [sweep|sync|gate|all]

Outputs go to results/. Raw per-iteration samples are kept so any summary can be recomputed.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

from perfgate import config, report, runner, stats
from perfgate.config import Case

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
MIN_EFFECT = 0.03  # fixed before running; not tuned on these results


def sweep() -> None:
    sw = config.load(ROOT / "configs/m4_sweep.yaml")
    recs = runner.run_sweep(sw, RES / "sweep_m4.jsonl")
    (RES / "sweep_m4.md").write_text(report.sweep_table(recs) + "\n")
    report.speedup_chart(recs, RES / "speedup_m4.png")


SYNC_CASES = [("gemm", "fp32", 8), ("mlp", "fp32", 32), ("resnet18", "fp16", 8), ("transformer", "fp32", 8)]


def sync() -> None:
    """Same GPU work timed with and without a device synchronize."""
    out = []
    for wl, dtype, bs in SYNC_CASES:
        row = {"key": f"{wl}/mps/{dtype}/bs{bs}"}
        for mode in ("per_iter", "none"):
            recs = [runner.run_case(Case(wl, "mps", dtype, bs, sync=mode), r, 300) for r in range(3)]
            ok = [r for r in recs if r["status"] == "ok"]
            row[mode] = {
                "reported_median_ms": float(np.median(stats.repeat_medians(ok))),
                "loop_wall_ms_per_iter": float(np.median([r["loop_wall_ms"] / len(r["samples_ms"])
                                                          for r in ok])),
            }
            print(row["key"], mode, row[mode], file=sys.stderr)
        row["underreport_factor"] = row["per_iter"]["reported_median_ms"] / row["none"]["reported_median_ms"]
        out.append(row)
    (RES / "sync_pitfall.json").write_text(json.dumps(out, indent=2) + "\n")


GATE_CASES = [
    ("gemm", "mps", "fp32", 8), ("resnet18", "mps", "fp16", 8), ("transformer", "mps", "fp32", 1),
    ("mlp", "cpu", "fp32", 32), ("resnet18", "cpu", "fp32", 1), ("transformer", "cpu", "fp32", 8),
]
BASE_REPEATS = 10
CAND_REPEATS = 5
INJECT_PCTS = (0.02, 0.05, 0.10, 0.20)


def gate() -> None:
    """False alarms on A/A splits, and detection of known injected slowdowns.

    A/A: 10 baseline processes of identical code are split every possible way into
    5 vs 5 (126 unordered splits) and gated. Every flag is a false alarm.
    Injected: 5 processes per slowdown level, each iteration spinning for pct x the
    baseline median *after* the device sync, gated against all 10 baseline processes.
    """
    raw = RES / "gate_raw.jsonl"
    rows, all_recs = [], []
    for wl, device, dtype, bs in GATE_CASES:
        base_case = Case(wl, device, dtype, bs)
        base = [runner.run_case(base_case, r, 300) for r in range(BASE_REPEATS)]
        base = [{**r, "role": "baseline"} for r in base if r["status"] == "ok"]
        all_recs += base
        base_med = float(np.median(stats.repeat_medians(base)))

        aa = {"splits": 0, "gate_flags": 0, "naive_flags": 0}
        idx = range(len(base))
        for left in itertools.combinations(idx, len(base) // 2):
            if 0 not in left:  # each unordered split once
                continue
            a = [base[i] for i in left]
            b = [base[i] for i in idx if i not in left]
            # Both directions: a false alarm can be a "regression" either way round.
            for x, y in ((a, b), (b, a)):
                v = stats.gate(base_case.key, x, y, min_effect=MIN_EFFECT)
                aa["splits"] += 1
                aa["gate_flags"] += v.verdict == "regression"
                aa["naive_flags"] += v.naive_verdict == "regression"

        injected = {}
        for pct in INJECT_PCTS:
            c = Case(wl, device, dtype, bs, inject_delay_ms=pct * base_med)
            cand = [runner.run_case(c, r, 300) for r in range(CAND_REPEATS)]
            cand = [{**r, "role": f"inject_{pct}"} for r in cand if r["status"] == "ok"]
            all_recs += cand
            v = stats.gate(base_case.key, base, cand, min_effect=MIN_EFFECT)
            injected[f"{pct:.2f}"] = {"ratio": v.ratio, "ci": [v.ci_low, v.ci_high],
                                      "verdict": v.verdict, "naive_verdict": v.naive_verdict}
        meds = stats.repeat_medians(base)
        row = {"key": base_case.key, "baseline_median_ms": base_med,
               "baseline_repeat_cv": float(np.std(meds, ddof=1) / np.mean(meds)),
               "aa": aa, "injected": injected}
        print(json.dumps(row), file=sys.stderr)
        rows.append(row)

    with raw.open("w") as f:
        for r in all_recs:
            f.write(json.dumps(r) + "\n")
    totals = {
        "aa_comparisons": sum(r["aa"]["splits"] for r in rows),
        "aa_gate_false_alarms": sum(r["aa"]["gate_flags"] for r in rows),
        "aa_naive_false_alarms": sum(r["aa"]["naive_flags"] for r in rows),
        "detected_by_pct": {
            f"{p:.2f}": {
                "gate": sum(r["injected"][f"{p:.2f}"]["verdict"] == "regression" for r in rows),
                "naive": sum(r["injected"][f"{p:.2f}"]["naive_verdict"] == "regression" for r in rows),
                "cases": len(rows),
            } for p in INJECT_PCTS
        },
        "min_effect": MIN_EFFECT,
    }
    (RES / "gate_eval.json").write_text(json.dumps({"totals": totals, "cases": rows}, indent=2) + "\n")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    RES.mkdir(exist_ok=True)
    for name, fn in (("sweep", sweep), ("sync", sync), ("gate", gate)):
        if what in (name, "all"):
            fn()
