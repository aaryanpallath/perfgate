"""Markdown tables and charts for sweep results and comparisons."""

from __future__ import annotations

from pathlib import Path

from .stats import Summary, Verdict, group_ok, summarize


def summaries(records: list[dict]) -> list[Summary]:
    return [summarize(k, recs) for k, recs in sorted(group_ok(records).items())]


def sweep_table(records: list[dict]) -> str:
    rows = [
        "| case | repeats | median ms | p95 ms | repeat CV | items/s | TFLOPS | first call ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries(records):
        tf = f"{s.tflops:.2f}" if s.tflops is not None else ""
        rows.append(
            f"| {s.key} | {s.repeats} | {s.median_ms:.3f} | {s.p95_ms:.3f} | "
            f"{s.repeat_cv * 100:.1f}% | {s.items_per_s:,.0f} | {tf} | {s.first_call_ms:.1f} |"
        )
    errors = [r for r in records if r.get("status") == "error"]
    seen = set()
    for r in errors:
        if r["key"] not in seen:
            seen.add(r["key"])
            rows.append(f"| {r['key']} | error | {r['error'][:60]} | | | | | |")
    return "\n".join(rows)


def compare_table(verdicts: list[Verdict]) -> str:
    rows = [
        "| case | latency ratio | 95% CI | verdict | naive verdict |",
        "|---|---:|---|---|---|",
    ]
    for v in verdicts:
        ci = f"{v.ci_low:.3f}–{v.ci_high:.3f}" if v.ci_low == v.ci_low else "n/a"
        flag = "**regression**" if v.verdict == "regression" else v.verdict
        rows.append(f"| {v.key} | {v.ratio:.3f} | {ci} | {flag} | {v.naive_verdict} |")
    notes = [f"- {v.key}: environment differs ({'; '.join(v.fingerprint_diffs)})"
             for v in verdicts if v.fingerprint_diffs]
    if notes:
        rows += ["", "Environment differences (results may not be comparable):", *notes]
    return "\n".join(rows)


def _sort_key(label: str) -> tuple:
    wl, dtype, bs, _ = label.split(" ")
    return (wl, dtype, int(bs.removeprefix("bs")))


def speedup_chart(records: list[dict], out: str | Path, baseline_device: str = "cpu") -> Path | None:
    """Bar chart of per-case speedup of each non-baseline device over `baseline_device`."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    by_key = {s.key: s for s in summaries(records)}
    labels, speedups = [], []
    for key, s in by_key.items():
        wl, device, dtype, bs = key.split("/")
        if device == baseline_device:
            continue
        base = by_key.get(f"{wl}/{baseline_device}/{dtype}/{bs}")
        if base:
            labels.append(f"{wl} {dtype} {bs} ({device})")
            speedups.append(base.median_ms / s.median_ms)
    if not labels:
        return None
    # Bars grow from 1x so "slower than baseline" points left and "faster" points right.
    order = sorted(range(len(labels)), key=lambda i: _sort_key(labels[i]))
    labels = [labels[i] for i in order]
    speedups = [speedups[i] for i in order]
    colors = ["#4C78A8" if s >= 1 else "#E45756" for s in speedups]
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(labels) + 1.4))
    ax.barh(labels, [s - 1 for s in speedups], left=1, color=colors)
    ax.axvline(1.0, color="#555", lw=1)
    ax.set_xscale("log")
    ticks = [t for t in (0.125, 0.25, 0.5, 1, 2, 4, 8, 16)
             if min(speedups) / 2 <= t <= max(speedups) * 2]
    ax.set_xticks(ticks, [f"{t:g}x" for t in ticks])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    for i, s in enumerate(speedups):
        ax.text(s * (1.04 if s >= 1 else 0.96), i, f"{s:.2f}x", va="center",
                ha="left" if s >= 1 else "right", fontsize=8)
    ax.set_xlim(min(min(speedups), 1) / 1.6, max(speedups) * 1.6)
    ax.set_xlabel(f"speedup over {baseline_device} (median latency, log scale)")
    ax.invert_yaxis()
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
