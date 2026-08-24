"""Draw the README figures from reports/*.csv.

Reads the saved monitoring output only -- no retraining, no data download.

    python scripts/make_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"


def scenarios(out: Path) -> Path:
    """Which injected failures the monitor catches, and which it does not.

    The two it misses are the informative ones. A pure label shift changes the
    fraud rate without touching any input distribution, so a drift monitor
    watching features is structurally blind to it -- no threshold fixes that.
    """
    table = pd.read_csv(REPORTS / "simulated_failures.csv")
    table = table.sort_values("pred_psi")
    positions = np.arange(len(table))

    figure, (left, right) = plt.subplots(
        1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1.4, 1]}
    )
    colours = ["#1a9850" if c else "#b2182b" for c in table.caught]
    left.barh(positions, table.pred_psi, color=colours, edgecolor="0.3", lw=0.4)
    left.set_yticks(positions)
    left.set_yticklabels(table.scenario, fontsize=8.5)
    left.set_xlabel("prediction PSI")
    left.set_title("green = caught by the monitor", fontsize=10)
    left.spines[["top", "right"]].set_visible(False)
    for index, row in enumerate(table.itertuples()):
        left.text(row.pred_psi + 0.008, index, f"top: {row.top_feature}",
                  va="center", fontsize=7, color="0.4")

    right.barh(positions, table.share * 100, color=colours, edgecolor="0.3", lw=0.4)
    right.set_xlabel("% of monitored features flagged")
    right.set_title("features flagged", fontsize=10)
    right.spines[["top", "right"]].set_visible(False)

    figure.suptitle(
        "Label shift is invisible to a feature-drift monitor by construction, "
        "not by mis-tuning.",
        fontsize=10, y=0.02, color="0.35",
    )
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    figure.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return out


def drift_vs_degradation(out: Path) -> Path:
    """Does the drift signal track the thing you actually care about?

    Windows flagged as drifted are not the windows where AUC fell furthest. Over
    these eight windows the correlation between prediction PSI and AUC drop is
    *negative*, around -0.71: window 6 has the largest degradation at 0.137 and one
    of the lowest PSI values, while window 1 has the highest PSI and the smallest
    drop. On this data the drift signal points the wrong way.
    """
    table = pd.read_csv(REPORTS / "drift_vs_degradation.csv")

    figure, (left, right) = plt.subplots(1, 2, figsize=(13, 4.4))
    colours = ["#b2182b" if d else "#bdbdbd" for d in table.drifted]
    left.bar(table.window, table.auc_drop, 0.6, color=colours,
             edgecolor="0.3", lw=0.4)
    left.set_xlabel("window")
    left.set_ylabel("AUC drop from baseline")
    left.set_title("red = the monitor flagged this window as drifted", fontsize=10)
    left.spines[["top", "right"]].set_visible(False)

    right.scatter(table.pred_psi, table.auc_drop, s=110,
                  c=["#b2182b" if d else "#bdbdbd" for d in table.drifted],
                  edgecolor="0.3", lw=0.5)
    for _, row in table.iterrows():
        right.annotate(f"w{int(row.window)}", (row.pred_psi, row.auc_drop),
                       textcoords="offset points", xytext=(6, 4), fontsize=7,
                       color="0.4")
    right.set_xlabel("prediction PSI (the drift signal)")
    right.set_ylabel("AUC drop (the thing you care about)")
    corr = table.pred_psi.corr(table.auc_drop)
    right.set_title(
        f"correlation between them: {corr:+.2f}\n"
        "the drift signal points the wrong way here",
        fontsize=10,
    )
    right.spines[["top", "right"]].set_visible(False)

    figure.tight_layout()
    figure.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return out


def false_alarms(out: Path) -> Path:
    """The healthy control: how often the monitor cries wolf on clean data.

    Forty features tested per batch means forty chances to be unlucky. Without a
    multiplicity correction the raw KS test flags 3.35 features per batch on data
    with nothing wrong with it; both corrections take that to zero.
    """
    table = pd.read_csv(REPORTS / "healthy_control.csv")
    positions = np.arange(len(table))

    figure, ax = plt.subplots(figsize=(9, 4.4))
    ax.bar(positions - 0.2, table.mean_ks_only, 0.4,
           label="raw KS, no correction", color="#b2182b", edgecolor="0.3", lw=0.5)
    ax.bar(positions + 0.2, table.mean_flagged, 0.4,
           label="after the correction", color="#1a9850", edgecolor="0.3", lw=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(table.correction)
    ax.set_ylabel("features flagged per batch, on healthy data")
    ax.set_title(
        "Forty features means forty chances to be unlucky. "
        "Uncorrected, the monitor\nflags 3.35 of them on data with nothing wrong.",
        fontsize=10,
    )
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return out


def calibration_windows(out: Path) -> Path:
    """The alerting rule applied window by window, with its inputs."""
    table = pd.read_csv(REPORTS / "calibration.csv")

    figure, ax = plt.subplots(figsize=(10, 4.4))
    ax.bar(table.window, table.share_flagged * 100, 0.55,
           color=["#b2182b" if a else "#bdbdbd" for a in table.would_alert],
           edgecolor="0.3", lw=0.4, label="% features flagged")
    twin = ax.twinx()
    twin.plot(table.window, table.pred_psi, "o-", color="#2166ac", lw=2,
              label="prediction PSI")
    ax.set_xlabel("calibration window")
    ax.set_ylabel("% of monitored features flagged")
    twin.set_ylabel("prediction PSI", color="#2166ac")
    twin.tick_params(axis="y", labelcolor="#2166ac")
    alerts = int(table.would_alert.sum())
    ax.set_title(
        f"Red bars are windows that would page someone: {alerts} of "
        f"{len(table)} on calibration data.",
        fontsize=10,
    )
    ax.spines[["top"]].set_visible(False)
    twin.spines[["top"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return out


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for path in (
        scenarios(FIGURES / "scenarios.png"),
        drift_vs_degradation(FIGURES / "drift-vs-degradation.png"),
        false_alarms(FIGURES / "false-alarms.png"),
        calibration_windows(FIGURES / "calibration-windows.png"),
    ):
        print(f"-> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
