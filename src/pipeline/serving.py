"""FastAPI serving layer with request logging for monitoring.

Every scored request is appended to a JSONL log holding the monitored features
and the prediction. That log is what the drift job and the dashboard read, so
monitoring observes what the model *actually saw*, not a re-derivation of it
from raw inputs. Re-deriving is the standard shortcut and it hides exactly the
bug you most want to catch: the serving pipeline and the training pipeline
disagreeing.
"""
from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import config

STATE: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["bundle"] = joblib.load(config.MODEL_PATH)
    try:
        from . import baseline as bl

        STATE["baseline"] = bl.load()
    except FileNotFoundError:
        STATE["baseline"] = None  # serving works without it; monitoring does not
    STATE["n_served"] = 0
    STATE["started"] = time.time()
    yield
    STATE.clear()


app = FastAPI(
    title="Fraud scoring service",
    version=config.MODEL_VERSION,
    lifespan=lifespan,
)


class Transaction(BaseModel):
    """Only the fields a caller realistically has. Everything else falls back to
    the training medians carried in the model bundle."""

    TransactionAmt: float = Field(..., ge=0, le=100_000)
    ProductCD: int = Field(4, ge=0, le=4)
    card1: int = Field(7919, ge=0)
    card4: int = Field(3, ge=0, le=3)
    card6: int = Field(1, ge=0, le=3)
    addr1: float = 299.0
    hour: float = Field(14.0, ge=0, lt=24)
    dayofweek: float = Field(2.0, ge=0, lt=7)
    C1: float = 1.0
    C13: float = 1.0
    D1: float = 0.0
    has_identity: int = Field(0, ge=0, le=1)


class Prediction(BaseModel):
    fraud_probability: float
    decision: str
    model_version: str
    baseline_rate: float
    lift_over_base: float


def _row(tx: Transaction) -> pd.DataFrame:
    b = STATE["bundle"]
    row = pd.DataFrame([b["defaults"]])[b["columns"]]
    d = tx.model_dump()
    for k, v in d.items():
        if k in row.columns:
            row.loc[0, k] = v
    amt = d["TransactionAmt"]
    for k, v in {
        "amt_log": float(np.log1p(amt)),
        "amt_cents": round((amt - np.floor(amt)) * 100),
        "amt_is_round": int(round((amt - np.floor(amt)) * 100) == 0),
    }.items():
        if k in row.columns:
            row.loc[0, k] = v
    return row.astype("float32")


def _log(row: pd.DataFrame, p: float) -> None:
    base = STATE.get("baseline")
    if base is None:
        return
    rec = {"ts": time.time(), "pred": p, "model_version": config.MODEL_VERSION}
    rec.update({c: float(row.iloc[0][c]) for c in base["monitored"] if c in row.columns})
    config.LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.LOG_PATH, "a") as f:
        f.write(json.dumps(rec) + "\n")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if "bundle" in STATE else "loading",
        "model_version": config.MODEL_VERSION,
        "baseline_loaded": STATE.get("baseline") is not None,
        "uptime_s": round(time.time() - STATE.get("started", time.time()), 1),
    }


@app.get("/metrics")
def metrics() -> dict:
    return {
        "requests_served": STATE.get("n_served", 0),
        "model_version": config.MODEL_VERSION,
        "n_features": len(STATE["bundle"]["columns"]) if "bundle" in STATE else 0,
        "monitored_features": len(STATE["baseline"]["monitored"]) if STATE.get("baseline") else 0,
    }


@app.post("/predict", response_model=Prediction)
def predict(tx: Transaction) -> Prediction:
    if "bundle" not in STATE:
        raise HTTPException(503, "model not loaded")
    row = _row(tx)
    p = float(STATE["bundle"]["model"].predict_proba(row)[0, 1])
    STATE["n_served"] = STATE.get("n_served", 0) + 1
    _log(row, p)
    return Prediction(
        fraud_probability=round(p, 6),
        decision="review" if p >= config.DECISION_THRESHOLD else "approve",
        model_version=config.MODEL_VERSION,
        baseline_rate=config.BASE_RATE,
        lift_over_base=round(p / config.BASE_RATE, 3),
    )
