"""Does the drift detector work? Three questions, in order of how much they matter.

1. **How often does it fire when nothing is wrong?** A detector is only evidence
   if its false-alarm rate is known. This is the drift-monitoring equivalent of
   scoring a saliency map against a random heatmap: without it, "it flagged
   something" means nothing.

2. **Does flagged drift predict the model actually getting worse?** The whole
   justification for drift monitoring is that it warns you before performance
   drops. That is testable here, because the training period carries labels, so
   both drift and AUC can be measured on the same windows.

3. **Only then: does it catch injected failures?** (in simulated_failure.py)

Run with `make validate`.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(__file__.rsplit("/experiments/", 1)[0]))
from src.pipeline import baseline as bl  # noqa: E402
from src.pipeline import config, drift, featurize  # noqa: E402

BASELINE_FRAC = 0.4     # first 40% of the period defines "normal"
N_WINDOWS = 8           # later period cut into equal windows
N_TRIALS = 20           # repetitions of the healthy control


def _load():
    print("loading training period ...")
    raw = featurize.load_split("train")
    bundle = bl.load_model()
    X = featurize.prepare(raw, bundle["columns"])
    y = raw["isFraud"].to_numpy()
    t = raw["TransactionDT"].to_numpy()
    order = np.argsort(t, kind="stable")
    return X.iloc[order].reset_index(drop=True), y[order], t[order], bundle


def healthy_control(X, bundle, cut, n_trials=N_TRIALS):
    """Null experiment: two disjoint random halves of the SAME period.

    No drift exists by construction, so anything flagged is a false alarm.
    """
    rng = np.random.default_rng(config.SEED)
    idx = np.arange(cut)
    rows = []
    for trial in range(n_trials):
        perm = rng.permutation(idx)
        a, b = perm[: cut // 2], perm[cut // 2 :]
        Xa, Xb = X.iloc[a], X.iloc[b]
        pa = bundle["model"].predict_proba(Xa)[:, 1]
        pb = bundle["model"].predict_proba(Xb)[:, 1]
        prof = bl.build(Xa, bundle, pa)
        for corr in ("none", "bonferroni", "bh"):
            r = drift.compare(prof, Xb, pb, correction=corr)
            # Significance alone, before the PSI effect-size gate. Reported
            # separately because it turns out to be the gate -- not the
            # multiple-testing correction -- that does the real work, and
            # claiming credit for the wrong mechanism would be its own error.
            ks_only = int((r.features["p_raw"] <= config.ALPHA).sum()) if corr == "none" \
                else int(r.features["flagged"].sum())
            rows.append({"trial": trial, "correction": corr,
                         "n_flagged": r.n_flagged, "ks_only_flagged": ks_only,
                         "n_monitored": len(r.features),
                         "batch_drifted": r.drifted})
    return pd.DataFrame(rows)


def drift_vs_degradation(X, y, bundle, cut):
    """Do the windows flagged as drifted actually score worse?

    The shipped model was trained on this entire period, so scoring it here
    would be in-sample: ~0.98 AUC that reflects memorisation and cannot degrade,
    which would make the correlation meaningless. So a fresh model is fitted on
    the baseline window only and the later windows are genuinely out-of-sample --
    the same relationship a deployed model has with future traffic.
    """
    import lightgbm as lgb

    Xa, ya = X.iloc[:cut], y[:cut]
    probe = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=63,
        colsample_bytree=0.7, subsample=0.8, subsample_freq=1,
        min_child_samples=50, random_state=config.SEED, verbose=-1, n_jobs=-1,
    )
    probe.fit(Xa, ya)
    pa = probe.predict_proba(Xa)[:, 1]
    prof = bl.build(Xa, bundle, pa)
    # In-sample by construction; kept only as the reference point the windows
    # are compared against, and labelled as such in the output.
    base_auc = roc_auc_score(ya, pa)

    bounds = np.linspace(cut, len(X), N_WINDOWS + 1).astype(int)
    rows = []
    for i in range(N_WINDOWS):
        s, e = bounds[i], bounds[i + 1]
        Xw, yw = X.iloc[s:e], y[s:e]
        pw = probe.predict_proba(Xw)[:, 1]
        r = drift.compare(prof, Xw, pw)
        rows.append({
            "window": i + 1, "n": len(Xw),
            "share_flagged": round(r.share_flagged, 3),
            "pred_psi": round(r.pred_psi, 3),
            "drifted": r.drifted,
            "auc": round(float(roc_auc_score(yw, pw)), 4),
            "auc_drop": round(float(base_auc - roc_auc_score(yw, pw)), 4),
            "fraud_rate": round(float(yw.mean()), 4),
        })
    return base_auc, pd.DataFrame(rows)


def main() -> None:
    X, y, t, bundle = _load()
    cut = int(len(X) * BASELINE_FRAC)
    print(f"rows={len(X):,}  baseline={cut:,}  monitored={config.N_MONITORED} features")

    print("\n=== 1. healthy control: how often does it fire on no drift? ===")
    hc = healthy_control(X, bundle, cut)
    summary = hc.groupby("correction").agg(
        mean_flagged=("n_flagged", "mean"),
        max_flagged=("n_flagged", "max"),
        mean_ks_only=("ks_only_flagged", "mean"),
        batch_false_alarm_rate=("batch_drifted", "mean"),
    ).round(3)
    print(summary.to_string())

    print("\n=== 2. does flagged drift predict AUC loss? ===")
    base_auc, wins = drift_vs_degradation(X, y, bundle, cut)
    print(f"baseline-period AUC {base_auc:.4f} (IN-SAMPLE reference; windows are out-of-sample)")
    print(wins.to_string(index=False))
    if wins["share_flagged"].std() > 0:
        c = np.corrcoef(wins["share_flagged"], wins["auc_drop"])[0, 1]
        print(f"\ncorr(share of features flagged, AUC drop) = {c:+.3f}")
        cp = np.corrcoef(wins["pred_psi"], wins["auc_drop"])[0, 1]
        print(f"corr(prediction PSI, AUC drop)            = {cp:+.3f}")

    config.REPORTS.mkdir(parents=True, exist_ok=True)
    summary.to_csv(config.REPORTS / "healthy_control.csv")
    wins.to_csv(config.REPORTS / "drift_vs_degradation.csv", index=False)
    print(f"\n-> {config.REPORTS}")


if __name__ == "__main__":
    main()
