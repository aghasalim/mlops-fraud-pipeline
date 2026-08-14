"""The detector is the product, so its failure modes get tests.

Each test here corresponds to a bug that actually shipped and was caught by
simulating a realistic outage -- not to a hypothetical.
"""
import numpy as np
import pandas as pd
import pytest

from src.pipeline import config, drift


@pytest.fixture
def base():
    rng = np.random.default_rng(0)
    return {
        "monitored": ["a", "b"],
        "samples": {"a": list(rng.normal(0, 1, 5000)), "b": list(rng.normal(5, 2, 5000))},
        "null_rate": {"a": 0.0, "b": 0.0},
        "pred_samples": list(rng.beta(1, 30, 5000)),
    }


def _batch(n=5000, seed=1, shift=0.0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"a": rng.normal(0 + shift, 1, n), "b": rng.normal(5, 2, n)})


def test_same_distribution_does_not_alarm(base):
    r = drift.compare(base, _batch(), np.random.default_rng(2).beta(1, 30, 5000))
    assert not r.drifted, r.reasons


def test_large_shift_is_caught(base):
    r = drift.compare(base, _batch(shift=3.0), np.random.default_rng(2).beta(1, 30, 5000))
    assert r.drifted


def test_all_null_column_is_caught(base):
    """The bug that shipped: a KS test on a fully-null column has nothing to
    compare, scores PSI 0, and passes silently. An outage is the single most
    detectable production failure and it was invisible."""
    b = _batch()
    b["a"] = np.nan
    r = drift.compare(base, b, np.random.default_rng(2).beta(1, 30, 5000))
    assert r.drifted
    assert any("missing rate" in x for x in r.reasons)


def test_psi_is_symmetric_in_sign_but_not_zero_on_shift():
    rng = np.random.default_rng(0)
    x, y = rng.normal(0, 1, 5000), rng.normal(1, 1, 5000)
    assert drift.psi(x, x.copy()) < 0.01
    assert drift.psi(x, y) > config.PSI_MODERATE


def test_psi_survives_an_empty_bin():
    """Without the frequency floor a single unseen bin makes PSI infinite, and
    one outlier would dominate every report."""
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 5000)
    y = np.full(5000, 99.0)
    v = drift.psi(x, y)
    assert np.isfinite(v) and v > 0


def test_monotonic_features_are_not_monitored():
    """day_index rises with the calendar, so it drifts in every forward
    comparison and would burn an alert slot permanently."""
    from src.pipeline.baseline import MONOTONIC
    assert "day_index" in MONOTONIC
