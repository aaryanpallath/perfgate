"""Command line: sweep, compare, report, slurm, merge."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from . import config, report, runner, slurm, stats


def _sweep(a) -> int:
    sw = config.load(a.config)
    if a.case_index is not None:
        if not 0 <= a.case_index < len(sw.cases):
            print(f"case index {a.case_index} out of range (0-{len(sw.cases) - 1})", file=sys.stderr)
            return 2
        sw.cases = [sw.cases[a.case_index]]
    if a.repeats:
        sw.repeats = a.repeats
    recs = runner.run_sweep(sw, a.out)
    errors = sum(r["status"] != "ok" for r in recs)
    print(f"{len(recs) - errors} ok, {errors} errors -> {a.out}", file=sys.stderr)
    return 0


def _report(a) -> int:
    recs = runner.load_results(a.results)
    md = ["# perfgate sweep report", "", report.sweep_table(recs)]
    if a.chart:
        p = report.speedup_chart(recs, a.chart)
        if p:
            md += ["", f"![speedup]({Path(p).name})"]
    text = "\n".join(md) + "\n"
    if a.out:
        Path(a.out).write_text(text)
    else:
        print(text)
    return 0


def _compare(a) -> int:
    base = runner.load_results([a.baseline])
    cand = runner.load_results([a.candidate])
    verdicts = stats.compare(base, cand, min_effect=a.min_effect)
    if not verdicts:
        print("no cases in common between baseline and candidate", file=sys.stderr)
        return 2
    if a.json:
        print(json.dumps([asdict(v) for v in verdicts], indent=2))
    else:
        print(report.compare_table(verdicts))
    return 1 if any(v.verdict == "regression" for v in verdicts) else 0


def _slurm(a) -> int:
    sw = config.load(a.config)
    print(slurm.sbatch_script(sw, a.config, partition=a.partition, gpus_per_task=a.gpus,
                              time_limit=a.time, container=a.container), end="")
    return 0


def _merge(a) -> int:
    with open(a.out, "w") as f:
        for r in runner.load_results(a.shards):
            f.write(json.dumps(r) + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="perfgate", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sweep", help="run a benchmark sweep")
    s.add_argument("config")
    s.add_argument("--out", required=True)
    s.add_argument("--repeats", type=int)
    s.add_argument("--case-index", type=int, help="run only this case (used by Slurm arrays)")
    s.set_defaults(fn=_sweep)

    r = sub.add_parser("report", help="summarize results as Markdown")
    r.add_argument("results", nargs="+")
    r.add_argument("--out")
    r.add_argument("--chart", help="write a device speedup chart (PNG)")
    r.set_defaults(fn=_report)

    c = sub.add_parser("compare", help="gate a candidate run against a baseline (exit 1 on regression)")
    c.add_argument("baseline")
    c.add_argument("candidate")
    c.add_argument("--min-effect", type=float, default=0.03)
    c.add_argument("--json", action="store_true")
    c.set_defaults(fn=_compare)

    sl = sub.add_parser("slurm", help="print a Slurm job-array script for a sweep")
    sl.add_argument("config")
    sl.add_argument("--partition")
    sl.add_argument("--gpus", type=int, default=1)
    sl.add_argument("--time", default="00:30:00")
    sl.add_argument("--container", help="Singularity/Apptainer image to run inside")
    sl.set_defaults(fn=_slurm)

    m = sub.add_parser("merge", help="concatenate JSONL result shards")
    m.add_argument("shards", nargs="+")
    m.add_argument("--out", required=True)
    m.set_defaults(fn=_merge)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
