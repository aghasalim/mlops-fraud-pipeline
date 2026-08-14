"""Configuration for the serving and monitoring pipeline."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
MODEL_PATH = ARTIFACTS / "model.pkl"
BASELINE_PATH = ARTIFACTS / "baseline.json"
LOG_PATH = DATA / "predictions.jsonl"

# Source data for building the baseline and replaying traffic. Not vendored --
# it is 1.3 GB of competition data; see README for how to point at your own copy.
SOURCE = Path(os.getenv("IEEE_DATA", Path.home() / "ieee-fraud-ml" / "data" / "raw"))

SEED = 42

# --- Drift detection ------------------------------------------------------
# Per-feature significance level, BEFORE multiple-testing correction. The whole
# point of experiments/detector_validation.py is that this number is not the
# false-alarm rate of the *system*: running it across hundreds of features means
# a fixed fraction fire on healthy data by construction.
ALPHA = float(os.getenv("DRIFT_ALPHA", "0.05"))

# Correction applied across features: "none" | "bonferroni" | "bh".
# Default is Benjamini-Hochberg. `none` is kept because it is what most drift
# dashboards actually do, and the validation experiment reports what that costs.
CORRECTION = os.getenv("DRIFT_CORRECTION", "bh")

# Population Stability Index thresholds. These are industry convention rather
# than anything derived: <0.1 stable, 0.1-0.25 moderate, >0.25 significant.
PSI_MODERATE = 0.10
PSI_MAJOR = 0.25
PSI_BINS = 10

# A batch is declared drifted when this share of monitored features fires.
#
# Set against the measured null, which is the same-period random split in
# detector_validation: 0 of 40 features fire there across 20 trials. Anything
# above zero is therefore signal, and 0.05 (2 of 40) keeps a margin.
#
# I briefly raised this to 0.20 on the grounds that the eight later windows were
# "clean traffic" being alarmed on. That was wrong twice over. The 0.25-0.30
# figures it was set against came from a run where the null-rate rule was
# mis-thresholded and flagging everything; with that fixed, injected faults
# score 0.075, so 0.20 would sleep through them. And those windows are not clean
# -- they carry 0.060 to 0.137 of measured AUC loss. Firing on them is the
# detector working, not crying wolf. Tuning a monitor until it goes quiet
# optimises for the wrong thing: silence is not the same as health.
BATCH_DRIFT_SHARE = float(os.getenv("BATCH_DRIFT_SHARE", "0.05"))

# A monitored feature's missing rate moving by more than this is an alert on its
# own. Distribution tests are blind to it: an all-null column has no values to
# compare, so an outage scores PSI 0.
#
# Calibrated, not guessed -- and my first guess (0.10) was wrong in the opposite
# direction to my first BATCH_DRIFT_SHARE guess. Missing rates move on their own
# across time: on healthy forward traffic D4 shifts 14.2% with nothing broken,
# so a 10% trigger fired on every batch including both controls. 0.35 sits above
# the natural drift measured across the eight healthy windows and still catches
# an outage, which takes a column to 100%.
NULL_JUMP = float(os.getenv("DRIFT_NULL_JUMP", "0.35"))

# Monitoring every one of the 443 features is possible but not informative --
# most carry almost no weight in the model, so drift in them says little about
# whether predictions will suffer. We monitor the top-N by gain importance plus
# the prediction distribution itself.
N_MONITORED = int(os.getenv("N_MONITORED", "40"))

# --- Serving --------------------------------------------------------------
MODEL_VERSION = os.getenv("MODEL_VERSION", "fraud-lgbm-v1")
DECISION_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.5"))
BASE_RATE = 0.03499  # measured on the training period
