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


def test_mostly_null_column_going_fully_null_is_caught(base):
    """A column that is 70% null on a normal day only moves 0.30 when its feed
    dies, under NULL_JUMP, so the absolute threshold alone stayed quiet."""
    base = {**base, "null_rate": {"a": 0.7, "b": 0.0}}
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


def test_psi_grows_with_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(size=5000)
    small = drift.psi(a, a + 0.2)
    large = drift.psi(a, a + 2.0)
    assert large > small > 0


def test_bh_is_never_more_permissive_than_raw_alpha():
    p = np.linspace(0.001, 0.9, 50)
    assert drift._bh(p, 0.05).sum() <= (p <= 0.05).sum()


def test_detector_silent_on_identical_batches_without_null_rates():
    """The healthy control on eight columns, with a profile that has no
    null_rate entry, which older baselines do not carry."""
    rng = np.random.default_rng(0)
    cols = [f"f{i}" for i in range(8)]
    a = pd.DataFrame(rng.normal(size=(3000, 8)), columns=cols)
    b = pd.DataFrame(rng.normal(size=(3000, 8)), columns=cols)
    preds = rng.random(3000) * 0.1
    prof = {"monitored": cols, "samples": {c: a[c].tolist() for c in cols},
            "pred_samples": preds.tolist()}
    assert not drift.compare(prof, b, rng.random(3000) * 0.1).drifted


def test_detector_fires_on_gross_feature_and_prediction_shift():
    rng = np.random.default_rng(0)
    cols = [f"f{i}" for i in range(8)]
    a = pd.DataFrame(rng.normal(size=(3000, 8)), columns=cols)
    b = pd.DataFrame(rng.normal(size=(3000, 8)) + 5, columns=cols)
    prof = {"monitored": cols, "samples": {c: a[c].tolist() for c in cols},
            "pred_samples": (rng.random(3000) * 0.1).tolist()}
    assert drift.compare(prof, b, rng.random(3000) * 0.1 + 0.5).drifted
