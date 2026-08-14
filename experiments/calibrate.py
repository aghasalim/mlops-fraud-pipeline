"""Measure what the alert rules do on traffic with no injected fault.

Originally written to set thresholds so that these windows stay silent. That
turned out to be the wrong objective: scored out-of-sample the same windows lose
0.06-0.137 AUC, so they are fault-free but not healthy, and calibrating until
they go quiet would have tuned away a correct warning. The output therefore
reports rates without labelling them false -- deciding that needs
drift_vs_degradation.csv, not this file.

The healthy control in `detector_validation.py` splits one period at random, so
it holds everything constant including the calendar. Production never looks like
that: live traffic is always *later* than the baseline, and things like D-column
missing rates move with time on their own, with nothing wrong.

Calibrating against the random-split null therefore sets thresholds far too
tight and the detector fires on ordinary Tuesdays -- which is exactly what
happened here: after adding the missing-rate rule, the healthy control alarmed
in 5 of 5 scenarios, including the untouched one.

So this measures every rule against consecutive later windows with no injected
fault, and thresholds are set above the observed healthy maximum rather than by
eye. Run with `make calibrate`.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(__file__.rsplit("/experiments/", 1)[0]))
from src.pipeline import baseline as bl  # noqa: E402
from src.pipeline import config, drift, featurize  # noqa: E402

BASELINE_FRAC = 0.4
N_WINDOWS = 8


def main() -> None:
    print("loading ...")
    raw = featurize.load_split("train")
    bundle = bl.load_model()
    X = featurize.prepare(raw, bundle["columns"])
    X = X.iloc[np.argsort(raw["TransactionDT"].to_numpy(), kind="stable")].reset_index(drop=True)

    cut = int(len(X) * BASELINE_FRAC)
    Xa = X.iloc[:cut]
    prof = bl.build(Xa, bundle, bundle["model"].predict_proba(Xa)[:, 1])
    print(f"baseline {cut:,} rows, {len(prof['monitored'])} monitored features")

    bounds = np.linspace(cut, len(X), N_WINDOWS + 1).astype(int)
    rows = []
    for i in range(N_WINDOWS):
        w = X.iloc[bounds[i] : bounds[i + 1]]
        r = drift.compare(prof, w, bundle["model"].predict_proba(w)[:, 1])
        rows.append({
            "window": i + 1,
            "share_flagged": round(r.share_flagged, 3),
            "n_null_shift": int(r.features["null_shift"].sum()),
            "max_null_delta": round(float(r.features["null_delta"].max()), 3),
            "pred_psi": round(r.pred_psi, 3),
            "would_alert": r.drifted,
        })

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    print("\nfault-free maxima. NOTE: 'fault-free' means nothing was injected --")
    print("it does NOT mean these windows are healthy. Scored out-of-sample they")
    print("lose 0.06-0.137 AUC, so an alarm here is not necessarily a false one,")
    print("and tuning the threshold until they go quiet tunes away a true signal.")
    print(f"  share_flagged  max {out['share_flagged'].max():.3f}"
          f"   (current threshold {config.BATCH_DRIFT_SHARE})")
    print(f"  null_delta     max {out['max_null_delta'].max():.3f}"
          f"   (current threshold {config.NULL_JUMP})")
    print(f"  pred_psi       max {out['pred_psi'].max():.3f}"
          f"   (current threshold {config.PSI_MAJOR})")
    print(f"\nalarms at current settings: "
          f"{int(out['would_alert'].sum())}/{len(out)} fault-free windows "
          f"(cross-reference reports/drift_vs_degradation.csv before calling "
          f"any of them false)")

    config.REPORTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.REPORTS / "calibration.csv", index=False)
    print(f"-> {config.REPORTS / 'calibration.csv'}")


if __name__ == "__main__":
    main()
