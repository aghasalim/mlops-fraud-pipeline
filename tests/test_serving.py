"""Serving contract tests. These are the CI deploy gate."""
from fastapi.testclient import TestClient

from src.pipeline.serving import app


def test_health_reports_model_version():
    with TestClient(app) as c:
        r = c.get("/health")
        assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_returns_calibrated_probability():
    with TestClient(app) as c:
        r = c.post("/predict", json={"TransactionAmt": 120.0})
        assert r.status_code == 200
        b = r.json()
        assert 0.0 <= b["fraud_probability"] <= 1.0
        assert b["decision"] in {"approve", "review"}


def test_predict_rejects_out_of_range_input():
    """A negative amount is not a prediction problem, it is a bad request."""
    with TestClient(app) as c:
        assert c.post("/predict", json={"TransactionAmt": -5.0}).status_code == 422


def test_large_amount_scores_higher_than_typical():
    with TestClient(app) as c:
        lo = c.post("/predict", json={"TransactionAmt": 50.0}).json()["fraud_probability"]
        hi = c.post("/predict", json={"TransactionAmt": 5000.0}).json()["fraud_probability"]
        assert hi != lo, "amount must actually reach the model"
