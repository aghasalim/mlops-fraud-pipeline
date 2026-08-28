"""Draw the README figures from reports/*.csv.

Reads the saved monitoring output only, no retraining and no data download. A
figure can therefore never disagree with a number quoted in the README.

    python scripts/make_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.animation import FuncAnimation, PillowWriter

from style import PALETTE, titled

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

# One colour rule across every figure: red means the monitor raised an alert,
# grey means it stayed silent. Whether firing was the right call is said in
# words, not smuggled into the colour.
ALERT, SILENT = "#b2182b", "#8a8a8a"
LINE = PALETTE[0]

# The shipped thresholds, from src/pipeline/config.py. Repeated here rather than
# imported so this script never needs the model or its dependencies.
PSI_MAJOR = 0.25          # prediction PSI that alerts on its own
BATCH_SHARE = 0.05        # share of monitored features that alerts on its own
N_MONITORED = 40


def _dots(ax, x, y, alerted, size=90) -> None:
    """Lollipop row: a stem to the axis and a dot at the value.

    Dots rather than bars because several of these values are zero or nearly
    zero, and a bar of length zero reads as missing data instead of as a
    measurement of zero.
    """
    ax.hlines(y, 0, x, color="#cccccc", lw=1.2, zorder=1)
    ax.scatter(x, y, s=size, c=[ALERT if a else SILENT for a in alerted],
               edgecolor="white", lw=0.8, zorder=3)


def scenarios(out: Path) -> Path:
    """Which injected failure trips which rule.

    The interesting thing is that no single signal covers the three real
    failures: the units bug is the only one that moves the prediction
    distribution, the new segment is the only one that trips the feature-share
    rule, and the outage trips neither. It is caught by the missing-rate rule,
    which is not a distribution test at all.
    """
    t = pd.read_csv(REPORTS / "simulated_failures.csv").sort_values("pred_psi")
    y = np.arange(len(t))
    caught = t.caught.tolist()

    fig, (a, b) = plt.subplots(1, 2, figsize=(13.4, 4.6), sharey=True)

    _dots(a, t.pred_psi, y, caught)
    a.axvline(PSI_MAJOR, color="#555555", ls="--", lw=1.1, zorder=2)
    a.text(PSI_MAJOR + 0.012, 1.9, f"alerts on its own\nat PSI {PSI_MAJOR}",
           fontsize=8.6, color="#555555", va="center")
    a.set_yticks(y)
    a.set_yticklabels(t.scenario)
    a.set_xlim(-0.02, 0.58)
    a.set_xlabel("prediction PSI (index, 0 = identical to baseline)")
    titled(a, "Only the units bug moves the prediction distribution",
           "five scenarios replayed through the shipped monitor, red = it alerted")

    _dots(b, t.share * 100, y, caught)
    b.axvline(BATCH_SHARE * 100, color="#555555", ls="--", lw=1.1, zorder=2)
    b.text(BATCH_SHARE * 100 + 0.25, 0.55,
           f"alerts on its own at\n{BATCH_SHARE:.0%} of {N_MONITORED} features",
           fontsize=8.6, color="#555555", va="center")
    b.set_xlim(-0.4, 11)
    b.set_xlabel(f"monitored features flagged (% of {N_MONITORED})")
    titled(b, "Only the new segment trips the feature-share rule",
           "the outage clears neither line, so a third rule has to catch it")

    outage = int(np.where(t.scenario == "identity feed outage")[0][0])
    b.annotate("caught only by the missing-rate rule:\nevery id_ column arrived null",
               xy=(t.share.iloc[outage] * 100, outage), xytext=(3.3, outage + 0.02),
               fontsize=8.6, color="#333333", va="center")
    for i, row in enumerate(t.itertuples()):
        if "control" in row.scenario or row.scenario == "label shift only":
            a.text(row.pred_psi + 0.018, i, "control, silence is correct",
                   fontsize=8.4, color="#666666", va="center")

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def drift_vs_degradation(out: Path) -> Path:
    """Does the drift signal track the thing you actually care about?

    No. Over these eight windows the correlation between prediction PSI and the
    measured AUC drop is negative, about -0.71: window 6 loses the most AUC on
    one of the lowest PSI values, window 1 has the highest PSI and the smallest
    drop. n=8, so suggestive rather than settled, but the direction alone kills
    "the predictions look normal, so we are fine".
    """
    t = pd.read_csv(REPORTS / "drift_vs_degradation.csv")
    corr = t.pred_psi.corr(t.auc_drop)

    fig, (a, b) = plt.subplots(1, 2, figsize=(13.4, 4.8))

    a.bar(t.window, t.auc_drop, 0.62, edgecolor="white", lw=0.8,
          color=[ALERT if d else SILENT for d in t.drifted])
    a.set_xticks(t.window)
    a.set_xlabel("window (consecutive, all later than the baseline period)")
    a.set_ylabel("AUC lost against baseline (AUC points)")
    a.set_ylim(0, t.auc_drop.max() * 1.22)
    titled(a, "Every window has lost AUC, alert or no alert",
           f"red = the monitor called it drifted ({int(t.drifted.sum())} of {len(t)}), "
           "grey = it stayed silent")
    a.annotate("the silent window is not\na healthy window", xy=(1, t.auc_drop[0]),
               xytext=(1.35, t.auc_drop.max() * 1.05), fontsize=8.8, color="#333333",
               arrowprops=dict(arrowstyle="-", color="#999999", lw=0.9))

    fit = np.poly1d(np.polyfit(t.pred_psi, t.auc_drop, 1))
    xs = np.linspace(t.pred_psi.min(), t.pred_psi.max(), 2)
    b.plot(xs, fit(xs), ls="--", lw=1.2, color="#999999", zorder=1)
    b.scatter(t.pred_psi, t.auc_drop, s=120, zorder=3, edgecolor="white", lw=0.8,
              c=[ALERT if d else SILENT for d in t.drifted])
    for row in t.itertuples():
        b.annotate(f"w{row.window}", (row.pred_psi, row.auc_drop), fontsize=8.4,
                   color="#555555", textcoords="offset points", xytext=(8, 4))
    b.set_xlim(0.038, 0.096)
    b.set_xlabel("prediction PSI (index, the signal the monitor watches)")
    b.set_ylabel("AUC lost against baseline (AUC points)")
    titled(b, "The signal is lowest where the damage is worst",
           f"Pearson r = {corr:+.2f} over {len(t)} windows, dashed line is the least squares fit")

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def false_alarms(out: Path) -> Path:
    """The healthy control, and which mechanism actually silences it.

    Forty features tested per batch is forty chances to be unlucky, and a raw KS
    test flags 3.35 of them on two random halves of the same data. I added
    Benjamini-Hochberg believing that was the fix. It is not: the PSI effect-size
    gate takes the count to zero on its own, with the correction set to none, so
    both corrections arrive with nothing left to remove.
    """
    t = pd.read_csv(REPORTS / "healthy_control.csv").set_index("correction")

    # The CSV records ks_only separately only for correction=none; for the other
    # two rows that column holds the fully corrected count, so the honest read is
    # one raw-KS number and three final numbers.
    rows = [
        ("KS test alone, alpha 0.05", float(t.loc["none", "mean_ks_only"])),
        ("KS + PSI effect-size gate", float(t.loc["none", "mean_flagged"])),
        ("KS + PSI gate + Bonferroni", float(t.loc["bonferroni", "mean_flagged"])),
        ("KS + PSI gate + Benjamini-Hochberg", float(t.loc["bh", "mean_flagged"])),
    ]
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    y = np.arange(len(rows))[::-1]

    fig, ax = plt.subplots(figsize=(9.6, 3.7))
    _dots(ax, vals, y, [v > 0 for v in vals], size=110)
    for value, pos in zip(vals, y):
        ax.text(value + 0.09, pos, f"{value:.2f}", fontsize=9.2, va="center",
                color=ALERT if value else "#444444")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(-0.12, 4.3)
    ax.set_xlabel(f"features flagged per batch on healthy data (of {N_MONITORED})")
    titled(ax, "The effect-size gate removes the false alarms, the correction does not",
           "20 trials on two disjoint random halves of the same period, so any flag is a false one")
    ax.text(1.05, 1.5, "all three zero rows: 0 flagged in every one of the 20\n"
            "trials, not an average with a few misses hidden in it",
            fontsize=8.8, color="#555555", va="center")

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def calibration_windows(out: Path) -> Path:
    """The alerting rule window by window on traffic with nothing injected.

    Every alert here comes from the feature-share rule. Prediction PSI never
    reaches a tenth of its own threshold, which is the same weakness the
    drift-vs-degradation figure measures, seen from the alerting side.
    """
    t = pd.read_csv(REPORTS / "calibration.csv")
    alerts = int(t.would_alert.sum())
    gutter = 2.3

    fig, (a, b) = plt.subplots(1, 2, figsize=(13.4, 4.6))

    a.bar(t.window, t.share_flagged * 100, 0.62, edgecolor="white", lw=0.8,
          color=[ALERT if w else SILENT for w in t.would_alert])
    a.axhline(BATCH_SHARE * 100, color="#555555", ls="--", lw=1.1)
    a.set_ylim(0, t.share_flagged.max() * 100 * 1.16)
    # Both panels keep an empty strip to the right of the last window and the
    # threshold label lives there. Anywhere over the data a tall bar or a high
    # PSI reading can grow into the text, and one did.
    a.set_xlim(0.4, len(t) + gutter)
    a.text(len(t) + 0.6, BATCH_SHARE * 100 + 0.15, "alert threshold", fontsize=8.6,
           color="#555555", va="bottom", ha="left")
    a.set_xticks(t.window)
    a.set_xlabel("calibration window (consecutive, nothing injected)")
    a.set_ylabel(f"monitored features flagged (% of {N_MONITORED})")
    titled(a, f"{alerts} of {len(t)} fault-free windows would page someone",
           "red = would alert, and these windows are fault-free rather than healthy")

    b.plot(t.window, t.pred_psi, marker="o", color=LINE, zorder=3)
    b.axhline(PSI_MAJOR, color="#555555", ls="--", lw=1.1)
    b.set_ylim(0, PSI_MAJOR * 1.12)
    b.set_xlim(0.4, len(t) + gutter)
    b.text(len(t) + 0.6, PSI_MAJOR - 0.004, f"alert threshold\nPSI {PSI_MAJOR}",
           fontsize=8.6, color="#555555", va="top", ha="left")
    b.set_xticks(t.window)
    b.set_xlabel("calibration window (consecutive, nothing injected)")
    b.set_ylabel("prediction PSI (index)")
    titled(b, "Prediction PSI stays an order of magnitude below its threshold",
           f"highest window is {t.pred_psi.max():.3f}, so every alert on the left came "
           "from the feature rule")

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def _shrink(path: Path) -> Path:
    """Requantise every frame onto one shared palette. Roughly halves the file."""
    src = Image.open(path)
    frames, durations = [], []
    try:
        while True:
            frames.append(src.convert("RGB"))
            durations.append(src.info.get("duration", 62))
            src.seek(src.tell() + 1)
    except EOFError:
        pass
    shared = frames[len(frames) // 2].quantize(64, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=shared, dither=Image.Dither.NONE) for f in frames]
    q[0].save(path, save_all=True, append_images=q[1:], loop=0, duration=durations,
              optimize=True)
    return path


def anim_threshold(out: Path, fps: int = 14, frames: int = 58, hold: int = 16) -> Path:
    """Sweep the batch alert threshold and watch degraded windows go quiet.

    Every one of the eight windows lost AUC, and on these windows the
    feature-share rule is the only rule that fires: no missing rate moved and no
    prediction PSI came near 0.25. So the sweep is the whole alerting decision.
    Raising the threshold to make the dashboard quiet also walks past the share
    the injected new-segment fault produces, which is the argument in config.py
    for leaving it at 5%.
    """
    t = pd.read_csv(REPORTS / "drift_vs_degradation.csv")
    faults = pd.read_csv(REPORTS / "simulated_failures.csv").set_index("scenario")
    fault_share = float(faults.loc["new customer segment", "share"]) * 100
    share = t.share_flagged.to_numpy() * 100
    y = np.arange(len(t))[::-1]
    cuts = np.linspace(0, 15.5, frames)

    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    # Room on the left for the row labels and above the top row for the readout.
    # anim.save writes the figure at its own size, so the tight bounding box the
    # style applies to savefig does not rescue a cramped margin here.
    fig.subplots_adjust(left=0.225, right=0.975, top=0.855, bottom=0.115)
    ax.set_xlim(0, 15.5)
    ax.set_ylim(-2.7, len(t) + 1.6)
    ax.set_xlabel(f"batch alert threshold (% of the {N_MONITORED} monitored features)")
    ax.set_yticks(list(y) + [-1.8])
    ax.set_yticklabels([f"w{w}   AUC -{d:.3f}" for w, d in zip(t.window, t.auc_drop)]
                       + ["new segment fault"])
    titled(ax, "Quieting the dashboard means ignoring windows that are degrading",
           "8 consecutive out-of-sample windows, labelled with the AUC each one lost")

    # Vertical rules stop below the readout band rather than running through it.
    span = (-2.7, len(t) + 0.05)
    ax.axhline(-1.0, color="#dddddd", lw=1.0)
    ax.plot([BATCH_SHARE * 100] * 2, span, color="#555555", ls=":", lw=1.2, zorder=2)
    ax.text(BATCH_SHARE * 100 + 0.2, -2.55, "shipped threshold", fontsize=8.4,
            color="#555555", va="bottom")

    low, high = ax.get_ylim()
    quiet = ax.axvspan(0, 0.001, ymax=(span[1] - low) / (high - low), color="#f0f0f0",
                       zorder=0)
    ax.hlines(y, 0, share, color="#dddddd", lw=1.2, zorder=1)
    ax.hlines([-1.8], 0, fault_share, color="#dddddd", lw=1.2, zorder=1)
    dots = ax.scatter(np.append(share, fault_share), np.append(y, -1.8), s=110,
                      edgecolor="white", lw=0.8, zorder=3)
    sweep, = ax.plot([0, 0], span, color="#333333", lw=1.4, zorder=4)
    readout = ax.text(0.2, len(t) + 1.15, "", fontsize=9.6, va="center", color="#333333")
    verdict = ax.text(0.2, len(t) + 0.4, "", fontsize=9.6, va="center", color=ALERT)

    values = np.append(share, fault_share)

    def draw(i):
        cut = cuts[min(i, frames - 1)]
        live = values >= cut
        dots.set_facecolor([ALERT if a else SILENT for a in live])
        sweep.set_xdata([cut, cut])
        quiet.set_width(cut)
        readout.set_text(f"threshold {cut:.1f}%:  {int(live[:len(share)].sum())} of "
                         f"{len(share)} degraded windows still alert")
        verdict.set_text("" if live[-1] else "the injected fault no longer alerts either")
        return [dots, sweep, quiet, readout, verdict]

    anim = FuncAnimation(fig, draw, frames=frames + hold, interval=1000 // fps, blit=False)
    anim.save(out, writer=PillowWriter(fps=fps), dpi=100)
    plt.close(fig)
    return _shrink(out)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for path in (
        scenarios(FIGURES / "scenarios.png"),
        drift_vs_degradation(FIGURES / "drift-vs-degradation.png"),
        false_alarms(FIGURES / "false-alarms.png"),
        calibration_windows(FIGURES / "calibration-windows.png"),
        anim_threshold(REPORTS / "threshold-sweep.gif"),
    ):
        print(f"-> {path.relative_to(ROOT)} ({path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
