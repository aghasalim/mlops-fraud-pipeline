"""Monitoring dashboard over logged predictions and the drift experiments."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline import config  # noqa: E402

st.set_page_config(page_title="Fraud model monitoring", page_icon="📈", layout="wide")
st.title("Fraud model — production monitoring")

st.caption(
    "The dashboard is the least interesting part of this project. What makes it "
    "evidence rather than decoration is the calibration underneath it: thresholds "
    "set from a measured healthy null, and a documented list of what the detector "
    "does and does not catch."
)

def _read(name):
    p = config.REPORTS / name
    return pd.read_csv(p) if p.exists() else None

c1, c2, c3 = st.columns(3)
cal = _read("calibration.csv")
sim = _read("simulated_failures.csv")
if cal is not None:
    c1.metric("false alarms, healthy windows", f"{int(cal['would_alert'].sum())}/{len(cal)}")
if sim is not None:
    real = sim[~sim["scenario"].str.contains("healthy|label shift")]
    c2.metric("injected faults caught", f"{int(real['caught'].sum())}/{len(real)}")
c3.metric("monitored features", config.N_MONITORED)

st.subheader("Injected failures — what the monitor caught")
if sim is not None:
    st.dataframe(sim, width="stretch")
    st.caption(
        "The new-customer-segment miss is kept visible on purpose. Catching it "
        "requires a threshold that alarmed on 7 of 8 clean windows, so the miss "
        "is a chosen operating point rather than an oversight."
    )

st.subheader("Healthy calibration windows")
if cal is not None:
    st.dataframe(cal, width="stretch")
    st.bar_chart(cal.set_index("window")[["share_flagged", "pred_psi"]])

st.subheader("Does drift predict degradation?")
deg = _read("drift_vs_degradation.csv")
if deg is not None:
    st.dataframe(deg, width="stretch")
    st.warning(
        "The uncomfortable result: out-of-sample AUC falls from 0.9376 to 0.8611 "
        "across these windows while the detector flags none of them. Input drift "
        "monitoring does not see this failure, because the inputs are not what "
        "changed."
    )

st.subheader("Live traffic")
if config.LOG_PATH.exists():
    rows = [json.loads(l) for l in open(config.LOG_PATH) if l.strip()]
    if rows:
        df = pd.DataFrame(rows)
        st.line_chart(df["pred"].rolling(20, min_periods=1).mean())
        st.caption(f"{len(df):,} scored requests logged by the API.")
else:
    st.info("No requests logged yet — run `make serve` and POST to /predict.")
