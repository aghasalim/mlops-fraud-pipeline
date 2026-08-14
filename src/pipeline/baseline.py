"""Build the reference profile that incoming traffic is compared against.

The baseline stores a *reservoir sample* of each monitored feature rather than
summary statistics, because a KS test needs the empirical distribution and PSI
needs stable bin edges. 20k values per feature is enough for both and keeps the
artifact small enough to commit.

Which features get monitored matters. Monitoring all 443 sounds thorough and is
actively harmful: most carry almost no weight, so drift in them is noise that
dilutes the batch verdict, and every extra feature is another chance to fire on
healthy data. We take the top-N by the model's own gain importance.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from . import config

SAMPLE_N = 20_000


def load_model() -> dict:
    return joblib.load(config.MODEL_PATH)


# Features that advance with the calendar drift by construction: every batch is
# later than the baseline, so they fire forever and can never indicate a problem.
# Found by testing rather than by reasoning -- `day_index` scored PSI 12.4 in
# every scenario including the healthy control, permanently consuming one slot
# of the alert budget and training whoever is on call to ignore the dashboard.
MONOTONIC = {"day_index", "TransactionDT", "D1", "D2", "D10", "D15"}


def monitored_features(bundle: dict, n: int | None = None) -> list[str]:
    n = config.N_MONITORED if n is None else n
    m = bundle["model"]
    imp = pd.Series(m.booster_.feature_importance(importance_type="gain"),
                    index=bundle["columns"])
    imp = imp.drop(labels=[c for c in MONOTONIC if c in imp.index])
    return list(imp.nlargest(n).index)


def build(df: pd.DataFrame, bundle: dict, preds: np.ndarray) -> dict:
    rng = np.random.default_rng(config.SEED)
    feats = monitored_features(bundle)
    samples, null_rate = {}, {}
    for c in feats:
        raw = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        # Stored separately because it is invisible to a distribution test: a
        # column that arrives entirely null has no values left to compare.
        null_rate[c] = float(np.mean(~np.isfinite(raw)))
        v = raw[np.isfinite(raw)]
        if len(v) > SAMPLE_N:
            v = rng.choice(v, SAMPLE_N, replace=False)
        samples[c] = [round(float(x), 6) for x in v]

    p = preds if len(preds) <= SAMPLE_N else rng.choice(preds, SAMPLE_N, replace=False)
    return {
        "model_version": config.MODEL_VERSION,
        "n_train_rows": int(len(df)),
        "monitored": feats,
        "samples": samples,
        "null_rate": null_rate,
        "pred_samples": [round(float(x), 6) for x in p],
        "pred_mean": float(np.mean(preds)),
    }


def save(profile: dict) -> None:
    config.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with open(config.BASELINE_PATH, "w") as f:
        json.dump(profile, f)


def load() -> dict:
    with open(config.BASELINE_PATH) as f:
        return json.load(f)
