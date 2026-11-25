import numpy as np
import pytest

from perfgate import stats


def test_summary_uses_median_of_repeat_medians(records):
    recs = records("k", [10.0, 11.0, 30.0], jitter=0.0)
    s = stats.summarize("k", recs)
    assert s.median_ms == pytest.approx(11.0)
    assert s.items_per_s == pytest.approx(8 / 0.011)
    assert s.repeats == 3


def test_bootstrap_ci_brackets_the_true_ratio():
    rng = np.random.default_rng(1)
    base = 10 * (1 + rng.normal(0, 0.02, 8))
    cand = 12 * (1 + rng.normal(0, 0.02, 8))
    ratio, lo, hi = stats.bootstrap_ratio_ci(base, cand)
    assert lo < 1.2 < hi
    assert lo < ratio < hi


def test_identical_runs_are_no_change(records):
    base = records("k", [10.0, 10.1, 9.9, 10.05, 9.95], seed=1)
    cand = records("k", [10.02, 9.98, 10.1, 9.9, 10.0], seed=2)
    assert stats.gate("k", base, cand).verdict == "no change"


def test_large_clean_slowdown_is_a_regression(records):
    base = records("k", [10.0, 10.1, 9.9, 10.05, 9.95])
    cand = records("k", [12.0, 12.1, 11.9, 12.05, 11.95])
    v = stats.gate("k", base, cand)
    assert v.verdict == "regression"
    assert v.ci_low > 1.0


def test_faster_candidate_is_an_improvement(records):
    base = records("k", [10.0, 10.1, 9.9, 10.05, 9.95])
    cand = records("k", [8.0, 8.1, 7.9, 8.05, 7.95])
    assert stats.gate("k", base, cand).verdict == "improvement"


def test_real_but_tiny_shift_is_not_flagged(records):
    # 1% slower with almost no noise: statistically clear, below the 3% threshold.
    base = records("k", [10.0, 10.001, 9.999, 10.0, 10.0], jitter=0.0001)
    cand = records("k", [10.1, 10.101, 10.099, 10.1, 10.1], jitter=0.0001)
    v = stats.gate("k", base, cand)
    assert v.ci_low > 1.0
    assert v.verdict == "no change"


def test_one_slow_process_fools_naive_gate_but_not_repeat_gate(records):
    # A/A comparison where one candidate process landed on a slow core / hot chip.
    base = records("k", [10.0, 10.1, 9.9, 10.05, 9.95])
    cand = records("k", [10.0, 9.95, 10.1, 9.9, 13.0])
    v = stats.gate("k", base, cand)
    assert v.naive_verdict == "regression"
    assert v.verdict == "no change"


def test_too_few_repeats_is_reported_not_guessed(records):
    v = stats.gate("k", records("k", [10.0, 10.0]), records("k", [20.0, 20.0]))
    assert v.verdict == "insufficient data"


def test_fingerprint_mismatch_is_reported(records):
    base = records("k", [10.0] * 3)
    cand = records("k", [10.0] * 3, fingerprint={"device_name": "Other GPU", "os": "Linux",
                                                  "torch": "2.x", "cpu_threads": 4})
    v = stats.gate("k", base, cand)
    assert any(d.startswith("device_name") for d in v.fingerprint_diffs)


def test_compare_skips_errors_and_unmatched_keys(records):
    base = records("a", [10.0] * 3) + records("b", [5.0] * 3)
    cand = records("a", [10.0] * 3) + [{"key": "b", "status": "error", "error": "oom"}]
    verdicts = stats.compare(base, cand)
    assert [v.key for v in verdicts] == ["a"]
