"""Deliberate production failures, and whether the monitor catches them.

Four scenarios, chosen so the set is not rigged. Two are the kind of break a
monitor is supposed to catch, one is a real-world shift taken from the data
rather than invented, and one is a failure that is genuinely invisible to
input-distribution monitoring — included precisely because a scenario list where
everything gets caught tells you nothing about the detector's limits.

Run with `make simulate`.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(__file__.rsplit("/experiments/", 1)[0]))
from src.pipeline import baseline as bl  # noqa: E402
from src.pipeline import config, drift, featurize  # noqa: E402

BASELINE_FRAC = 0.4
BATCH = 40_000


def scenario_currency(X: pd.DataFrame, rng) -> pd.DataFrame:
    """Upstream service starts sending amounts in cents, not dollars.

    A units bug in a feed. Nothing errors -- every value is still a valid
    positive float -- and the model happily scores all of it.
    """
    d = X.copy()
    d["TransactionAmt"] = d["TransactionAmt"] * 100
    d["amt_log"] = np.log1p(d["TransactionAmt"])
    return d


def scenario_new_segment(X: pd.DataFrame, rng) -> pd.DataFrame:
    """A marketing push brings in a new cohort: younger cards, no history.

    Realistic, gradual-looking, and the sort of thing that is nobody's bug.
    """
    d = X.copy()
    n = len(d)
    take = rng.random(n) < 0.35
    d.loc[take, "C1"] = 1.0
    d.loc[take, "C13"] = 1.0
    d.loc[take, "D1"] = 0.0
    d.loc[take, "has_identity"] = 0
    if "card1_freq" in d.columns:
        d.loc[take, "card1_freq"] = 1.0
    return d


def scenario_missing_feed(X: pd.DataFrame, rng) -> pd.DataFrame:
    """The identity provider goes down: its columns arrive as nulls.

    The classic partial outage. LightGBM treats NaN as a routable value, so
    predictions keep flowing and nothing raises.
    """
    d = X.copy()
    for c in [c for c in d.columns if c.startswith("id_")]:
        d[c] = np.nan
    if "has_identity" in d.columns:
        d["has_identity"] = 0
    return d


def scenario_label_shift(X: pd.DataFrame, rng) -> pd.DataFrame:
    """Fraud tactics change but the inputs look identical.

    The honest negative control. Features are untouched, so an
    input-distribution monitor cannot see this by construction -- only labels
    or a proxy for them would. Included so the results show what this design
    does NOT cover.
    """
    return X.copy()


SCENARIOS = {
    "healthy (control)": None,
    "currency units bug": scenario_currency,
    "new customer segment": scenario_new_segment,
    "identity feed outage": scenario_missing_feed,
    "label shift only": scenario_label_shift,
}


def main() -> None:
    print("loading ...")
    raw = featurize.load_split("train")
    bundle = bl.load_model()
    X = featurize.prepare(raw, bundle["columns"])
    t = raw["TransactionDT"].to_numpy()
    X = X.iloc[np.argsort(t, kind="stable")].reset_index(drop=True)

    cut = int(len(X) * BASELINE_FRAC)
    Xa = X.iloc[:cut]
    pa = bundle["model"].predict_proba(Xa)[:, 1]
    prof = bl.build(Xa, bundle, pa)
    bl.save(prof)
    print(f"baseline: {cut:,} rows, {len(prof['monitored'])} monitored features")

    rng = np.random.default_rng(config.SEED)
    live = X.iloc[cut : cut + BATCH].reset_index(drop=True)

    rows = []
    for name, fn in SCENARIOS.items():
        batch = live if fn is None else fn(live, rng)
        preds = bundle["model"].predict_proba(batch)[:, 1]
        r = drift.compare(prof, batch, preds)
        rows.append({
            "scenario": name,
            "caught": r.drifted,
            "features_flagged": f"{r.n_flagged}/{len(r.features)}",
            "share": round(r.share_flagged, 3),
            "pred_psi": round(r.pred_psi, 3),
            "pred_mean": round(r.pred_mean_batch, 4),
            "top_feature": r.top(1).index[0] if len(r.features) else "",
            "top_psi": round(float(r.top(1)["psi"].iloc[0]), 3) if len(r.features) else 0.0,
        })
        mark = "CAUGHT" if r.drifted else "not flagged"
        print(f"  {name:24s} {mark:12s} {r.n_flagged}/{len(r.features)} features, "
              f"pred PSI {r.pred_psi:.3f}")
        for reason in r.reasons:
            print(f"      -> {reason}")

    out = pd.DataFrame(rows)
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.REPORTS / "simulated_failures.csv", index=False)
    print()
    print(out.to_string(index=False))
    print(f"\n-> {config.REPORTS / 'simulated_failures.csv'}")


if __name__ == "__main__":
    main()
