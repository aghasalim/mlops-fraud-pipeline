"""Drift detection: per-feature tests plus a batch-level verdict.

Two things this module takes seriously that most drift dashboards do not.

**Multiple testing.** A KS test at alpha=0.05 fires on 5% of features *when
nothing has changed*. Monitor 443 features and roughly 22 light up on healthy
data, every batch, forever. That is not a detector, it is a random number
generator with a dashboard. `experiments/detector_validation.py` measures this
directly, and the correction here is the response.

**Statistical significance is not operational significance.** With 50,000 rows
per batch, a KS test detects shifts far too small to move a prediction. Sample
size is a property of your traffic volume, not of whether the model is in
trouble. So every feature also gets a PSI, which is an effect size and does not
inflate with n, and the batch verdict requires both.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from . import config


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = config.PSI_BINS) -> float:
    """Population Stability Index against fixed baseline quantile edges.

    Edges come from the baseline, never recomputed on the incoming batch --
    recomputing them would rescale the comparison and hide the very shift being
    looked for.
    """
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) < 2 or len(actual) < 2:
        return 0.0
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, bins=edges)[0].astype(float)
    a = np.histogram(actual, bins=edges)[0].astype(float)
    # Laplace-style floor: an empty bin makes the log term infinite, which would
    # let a single unseen value dominate the whole score.
    e = np.maximum(e / e.sum(), 1e-6)
    a = np.maximum(a / a.sum(), 1e-6)
    return float(np.sum((a - e) * np.log(a / e)))


def _bh(pvals: np.ndarray, alpha: float) -> np.ndarray:
    """Benjamini-Hochberg: controls the expected *proportion* of false alarms
    among the features flagged, which is the quantity an on-call engineer
    actually cares about."""
    n = len(pvals)
    order = np.argsort(pvals)
    thresh = alpha * (np.arange(1, n + 1) / n)
    passed = pvals[order] <= thresh
    out = np.zeros(n, dtype=bool)
    if passed.any():
        cutoff = np.max(np.where(passed)[0])
        out[order[: cutoff + 1]] = True
    return out


@dataclass
class DriftReport:
    n_rows: int
    features: pd.DataFrame          # per-feature ks_stat, p_raw, flagged, psi
    n_flagged: int
    share_flagged: float
    pred_psi: float
    pred_mean_baseline: float
    pred_mean_batch: float
    drifted: bool
    reasons: list[str] = field(default_factory=list)

    def top(self, k: int = 10) -> pd.DataFrame:
        return self.features.nlargest(k, "psi")[["psi", "ks_stat", "p_raw", "flagged"]]


def compare(
    baseline: dict,
    batch: pd.DataFrame,
    preds: np.ndarray | None = None,
    correction: str | None = None,
    alpha: float | None = None,
) -> DriftReport:
    """Compare an incoming batch against the stored baseline profile."""
    correction = config.CORRECTION if correction is None else correction
    alpha = config.ALPHA if alpha is None else alpha

    cols = [c for c in baseline["monitored"] if c in batch.columns]
    base_null = baseline.get("null_rate", {})
    rows = []
    for c in cols:
        ref = np.asarray(baseline["samples"][c], dtype=float)
        raw = pd.to_numeric(batch[c], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(raw)
        cur = raw[finite]

        # Missing-rate shift, tracked independently of the distribution test.
        # A feed outage sends a column to 100% null, which leaves the KS test
        # with nothing to compare and scores PSI 0 -- the most conspicuous
        # failure in production is exactly the one a distribution test cannot
        # see. Found by simulating an identity-provider outage and watching it
        # sail through.
        null_now = float(np.mean(~finite))
        null_delta = abs(null_now - float(base_null.get(c, null_now)))

        if len(cur) < 20 or len(ref) < 20:
            rows.append({"feature": c, "ks_stat": 0.0, "p_raw": 1.0, "psi": 0.0,
                         "null_rate": null_now, "null_delta": null_delta})
            continue
        ks = stats.ks_2samp(ref, cur)
        rows.append({"feature": c, "ks_stat": float(ks.statistic),
                     "p_raw": float(ks.pvalue), "psi": psi(ref, cur),
                     "null_rate": null_now, "null_delta": null_delta})

    f = pd.DataFrame(rows).set_index("feature")
    p = f["p_raw"].to_numpy()
    if correction == "bonferroni":
        f["flagged"] = p <= alpha / max(len(p), 1)
    elif correction == "bh":
        f["flagged"] = _bh(p, alpha)
    else:
        f["flagged"] = p <= alpha

    # A feature only counts toward the batch verdict if it is both statistically
    # detectable and large enough to matter. Either alone is noise at this n.
    f["flagged"] = f["flagged"] & (f["psi"] >= config.PSI_MODERATE)
    # ...or if its missing rate moved, which the distribution test cannot see.
    f["null_shift"] = f["null_delta"] >= config.NULL_JUMP
    f["flagged"] = f["flagged"] | f["null_shift"]

    n_flagged = int(f["flagged"].sum())
    share = n_flagged / max(len(f), 1)

    pred_psi, pred_mean = 0.0, float("nan")
    if preds is not None and len(preds):
        pred_psi = psi(np.asarray(baseline["pred_samples"], dtype=float),
                       np.asarray(preds, dtype=float))
        pred_mean = float(np.mean(preds))

    reasons = []
    n_null = int(f["null_shift"].sum())
    if n_null:
        worst = f.nlargest(1, "null_delta")
        reasons.append(f"{n_null} feature(s) changed missing rate, worst "
                       f"{worst.index[0]} by {float(worst['null_delta'].iloc[0]):.1%}")
    if share >= config.BATCH_DRIFT_SHARE:
        reasons.append(f"{n_flagged}/{len(f)} monitored features drifted "
                       f"({share:.1%} >= {config.BATCH_DRIFT_SHARE:.0%})")
    if pred_psi >= config.PSI_MAJOR:
        reasons.append(f"prediction distribution PSI {pred_psi:.3f} "
                       f">= {config.PSI_MAJOR}")

    return DriftReport(
        n_rows=len(batch), features=f, n_flagged=n_flagged, share_flagged=share,
        pred_psi=pred_psi,
        pred_mean_baseline=float(np.mean(baseline["pred_samples"])),
        pred_mean_batch=pred_mean, drifted=bool(reasons), reasons=reasons,
    )
