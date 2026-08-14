"""Gate tests. CI blocks the deployable marker on these, so each one asserts
something that would actually make the service wrong -- not that imports work.
"""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.pipeline import config, drift, serving


@pytest.fixture(scope="module")
def client():
    with TestClient(serving.app) as c:
        yield c


def test_health_reports_model_loaded(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_returns_calibrated_probability(client):
    r = client.post("/predict", json={"TransactionAmt": 120.0})
    assert r.status_code == 200
    p = r.json()["fraud_probability"]
    assert 0.0 <= p <= 1.0


def test_predict_rejects_impossible_input(client):
    """A negative amount is not a low-risk transaction, it is a broken caller."""
    assert client.post("/predict", json={"TransactionAmt": -5}).status_code == 422
    assert client.post("/predict", json={"TransactionAmt": 1e9}).status_code == 422


def test_amount_moves_the_score(client):
    """Guards against the model silently receiving defaults instead of input --
    a serving bug that returns plausible constants and looks fine."""
    lo = client.post("/predict", json={"TransactionAmt": 10.0}).json()["fraud_probability"]
    hi = client.post("/predict", json={"TransactionAmt": 5000.0}).json()["fraud_probability"]
    assert lo != hi


# --- drift maths ----------------------------------------------------------

def test_psi_zero_on_identical_distributions():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    assert drift.psi(x, x.copy()) < 0.01


def test_psi_grows_with_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(size=5000)
    small = drift.psi(a, a + 0.2)
    large = drift.psi(a, a + 2.0)
    assert large > small > 0


def test_psi_survives_empty_bins():
    """An unseen value must not send PSI to infinity."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=1000)
    assert np.isfinite(drift.psi(a, np.full(1000, 99.0)))


def test_bh_is_never_more_permissive_than_raw_alpha():
    p = np.linspace(0.001, 0.9, 50)
    assert drift._bh(p, 0.05).sum() <= (p <= 0.05).sum()


def test_detector_silent_on_identical_batches():
    """The healthy control, in miniature: same distribution must not alarm."""
    rng = np.random.default_rng(0)
    cols = [f"f{i}" for i in range(8)]
    a = pd.DataFrame(rng.normal(size=(3000, 8)), columns=cols)
    b = pd.DataFrame(rng.normal(size=(3000, 8)), columns=cols)
    preds = rng.random(3000) * 0.1
    prof = {"monitored": cols, "samples": {c: a[c].tolist() for c in cols},
            "pred_samples": preds.tolist()}
    assert not drift.compare(prof, b, rng.random(3000) * 0.1).drifted


def test_detector_fires_on_gross_shift():
    rng = np.random.default_rng(0)
    cols = [f"f{i}" for i in range(8)]
    a = pd.DataFrame(rng.normal(size=(3000, 8)), columns=cols)
    b = pd.DataFrame(rng.normal(size=(3000, 8)) + 5, columns=cols)
    prof = {"monitored": cols, "samples": {c: a[c].tolist() for c in cols},
            "pred_samples": (rng.random(3000) * 0.1).tolist()}
    assert drift.compare(prof, b, rng.random(3000) * 0.1 + 0.5).drifted
