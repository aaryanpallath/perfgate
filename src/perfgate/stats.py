"""Summaries and the regression gate.

The unit of evidence is a *repeat*: one fresh process, summarized by its median
iteration latency. Iterations inside one process are not independent — they share a
warm cache, one allocator state and one clock-frequency history — so pooling them
treats 250 correlated samples as if they were 250 independent ones, and makes tiny
process-to-process differences look significant. The gate therefore resamples repeats,
never individual iterations.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

FINGERPRINT_KEYS = ("device_name", "os", "torch", "cpu_threads")


def group_ok(records: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        if r.get("status") == "ok":
            groups[r["key"]].append(r)
    return dict(groups)


def repeat_medians(recs: list[dict]) -> np.ndarray:
    return np.array([float(np.median(r["samples_ms"])) for r in recs])


@dataclass
class Summary:
    key: str
    repeats: int
    median_ms: float  # median of per-repeat medians
    p95_ms: float  # pooled across repeats; describes the tail, not used by the gate
    repeat_cv: float  # coefficient of variation of the per-repeat medians
    items_per_s: float
    tflops: float | None
    first_call_ms: float


def summarize(key: str, recs: list[dict]) -> Summary:
    meds = repeat_medians(recs)
    pooled = np.concatenate([np.asarray(r["samples_ms"], dtype=float) for r in recs])
    med = float(np.median(meds))
    items = recs[0]["items_per_step"]
    flops = recs[0].get("flops_per_step")
    cv = float(np.std(meds, ddof=1) / np.mean(meds)) if len(meds) > 1 else 0.0
    return Summary(
        key=key,
        repeats=len(recs),
        median_ms=med,
        p95_ms=float(np.percentile(pooled, 95)),
        repeat_cv=cv,
        items_per_s=items / (med / 1000.0),
        tflops=(flops / (med / 1000.0) / 1e12) if flops else None,
        first_call_ms=float(np.median([r["first_call_ms"] for r in recs])),
    )


def bootstrap_ratio_ci(
    base: np.ndarray, cand: np.ndarray, *, n_boot: int = 2000, level: float = 0.95, seed: int = 0
) -> tuple[float, float, float]:
    """Point estimate and percentile-bootstrap CI for median(cand) / median(base)."""
    if len(base) == 0 or len(cand) == 0:
        raise ValueError("need at least one repeat on each side")
    rng = np.random.default_rng(seed)
    b = rng.choice(base, size=(n_boot, len(base)), replace=True)
    c = rng.choice(cand, size=(n_boot, len(cand)), replace=True)
    ratios = np.median(c, axis=1) / np.median(b, axis=1)
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(ratios, [alpha, 1.0 - alpha])
    return float(np.median(cand) / np.median(base)), float(lo), float(hi)


@dataclass
class Verdict:
    key: str
    ratio: float  # candidate / baseline latency; > 1 means slower
    ci_low: float
    ci_high: float
    verdict: str  # "regression", "improvement", "no change", "insufficient data"
    naive_verdict: str  # pooled-mean threshold check, kept for comparison
    base_repeats: int
    cand_repeats: int
    fingerprint_diffs: list[str]


def gate(
    key: str,
    base: list[dict],
    cand: list[dict],
    *,
    min_effect: float = 0.03,
    min_repeats: int = 3,
    seed: int = 0,
) -> Verdict:
    """Flag a regression only when it is both real and large enough to matter.

    Regression: the whole 95% CI of the latency ratio lies above 1.0 *and* the point
    estimate is at least `min_effect` slower. The CI condition guards against noise;
    the threshold guards against statistically real but irrelevant shifts.
    """
    bm, cm = repeat_medians(base), repeat_medians(cand)
    naive = naive_gate(base, cand, min_effect=min_effect)
    diffs = fingerprint_diffs(base, cand)
    if len(bm) < min_repeats or len(cm) < min_repeats:
        ratio = float(np.median(cm) / np.median(bm)) if len(bm) and len(cm) else float("nan")
        return Verdict(key, ratio, float("nan"), float("nan"), "insufficient data", naive,
                       len(bm), len(cm), diffs)
    ratio, lo, hi = bootstrap_ratio_ci(bm, cm, seed=seed)
    if lo > 1.0 and ratio >= 1.0 + min_effect:
        v = "regression"
    elif hi < 1.0 and ratio <= 1.0 - min_effect:
        v = "improvement"
    else:
        v = "no change"
    return Verdict(key, ratio, lo, hi, v, naive, len(bm), len(cm), diffs)


def naive_gate(base: list[dict], cand: list[dict], *, min_effect: float = 0.03) -> str:
    """What many CI scripts do: compare pooled mean latency against a fixed threshold."""
    bmean = float(np.mean(np.concatenate([r["samples_ms"] for r in base])))
    cmean = float(np.mean(np.concatenate([r["samples_ms"] for r in cand])))
    r = cmean / bmean
    if r >= 1.0 + min_effect:
        return "regression"
    if r <= 1.0 - min_effect:
        return "improvement"
    return "no change"


def fingerprint_diffs(base: list[dict], cand: list[dict]) -> list[str]:
    out = []
    for k in FINGERPRINT_KEYS:
        bv = {str(r.get("fingerprint", {}).get(k)) for r in base}
        cv = {str(r.get("fingerprint", {}).get(k)) for r in cand}
        if bv != cv:
            out.append(f"{k}: {sorted(bv)} -> {sorted(cv)}")
    return out


def compare(
    base_records: list[dict], cand_records: list[dict], *, min_effect: float = 0.03
) -> list[Verdict]:
    b, c = group_ok(base_records), group_ok(cand_records)
    return [gate(k, b[k], c[k], min_effect=min_effect) for k in sorted(set(b) & set(c))]
