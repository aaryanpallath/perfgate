"""Run a sweep: one subprocess per (case, repeat), results appended to a JSONL file."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections.abc import Callable, Iterable
from pathlib import Path

from . import device as dev
from .config import Case, Sweep


def run_case(case: Case, repeat: int, timeout_s: float) -> dict:
    """Run one repeat of `case` in a fresh interpreter and return its result record.

    Failures (unsupported dtype on a device, out of memory, timeouts) become error
    records instead of aborting the sweep, so one bad case doesn't lose a night of data.
    """
    cmd = [sys.executable, "-m", "perfgate.worker", json.dumps(case.to_dict())]
    base = {"key": case.key, "case": case.to_dict(), "repeat": repeat}
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return {**base, "status": "error", "error": f"timed out after {timeout_s:.0f}s"}
    if proc.returncode != 0:
        lines = [ln for ln in proc.stderr.strip().splitlines() if ln.strip()]
        return {**base, "status": "error", "error": lines[-1] if lines else "worker failed"}
    try:
        rec = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {**base, "status": "error", "error": "worker printed invalid JSON"}
    return {**rec, "repeat": repeat, "status": "ok"}


def run_sweep(
    sweep: Sweep,
    out_path: str | Path,
    *,
    run_one: Callable[[Case, int, float], dict] = run_case,
    log: Callable[[str], None] = lambda s: print(s, file=sys.stderr),
) -> list[dict]:
    """Run every case `sweep.repeats` times.

    Repeats are interleaved (all cases, then all cases again) rather than back to back,
    so slow drift — thermal throttling, a background job starting — spreads across
    cases instead of landing on whichever case happened to run last.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    skipped = set()
    with out_path.open("w") as f:
        for r in range(sweep.repeats):
            for case in sweep.cases:
                if not dev.is_available(case.device):
                    if case.device not in skipped:
                        log(f"skip: device {case.device} not available")
                        skipped.add(case.device)
                    continue
                t = time.perf_counter()
                rec = run_one(case, r, sweep.timeout_s)
                rec["sweep"] = sweep.name
                f.write(json.dumps(rec) + "\n")
                f.flush()
                records.append(rec)
                status = rec["status"] if rec["status"] == "ok" else f"ERROR {rec.get('error')}"
                log(f"[{r + 1}/{sweep.repeats}] {case.key}: {status} ({time.perf_counter() - t:.1f}s)")
    return records


def load_results(paths: Iterable[str | Path]) -> list[dict]:
    out = []
    for p in paths:
        with open(p) as f:
            out.extend(json.loads(line) for line in f if line.strip())
    return out
