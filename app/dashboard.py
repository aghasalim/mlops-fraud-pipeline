"""Monitoring dashboard over the batch reports the drift job produces."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline import config  # noqa: E402

st.set_page_config(page_title="Fraud model monitoring", page_icon="📈", layout="wide")
st.title("Fraud model — monitoring")

R = config.REPORTS
if not R.exists():
    st.warning("No reports yet. Run `make validate && make simulate`.")
    st.stop()


def _read(name):
    p = R / name
    return pd.read_csv(p) if p.exists() else None


st.subheader("Does the detector catch injected failures?")
sf = _read("simulated_failures.csv")
if sf is not None:
    st.dataframe(sf, use_container_width=True, hide_index=True)
    st.caption(
        "`label shift only` is a negative control and is *supposed* to be missed: "
        "its inputs are untouched, so no input-distribution monitor can see it. "
        "A scenario list where everything is caught would say nothing about limits."
    )

st.subheader("How often does it fire when nothing is wrong?")
hc = _read("healthy_control.csv")
if hc is not None:
    st.dataframe(hc, use_container_width=True, hide_index=True)
    st.caption(
        "Two disjoint random halves of the same period, 20 trials. KS alone "
        "flags ~3.4 of 40 features on identical data — the 5% you asked for, "
        "arriving as noise. The PSI effect-size gate removes all of it."
    )

st.subheader("Does flagged drift track real performance loss?")
dd = _read("drift_vs_degradation.csv")
if dd is not None:
    c1, c2 = st.columns(2)
    c1.line_chart(dd.set_index("window")[["auc"]])
    c2.line_chart(dd.set_index("window")[["share_flagged", "pred_psi"]])
    st.dataframe(dd, use_container_width=True, hide_index=True)
    st.caption(
        "Prediction PSI correlates −0.709 with AUC loss — it moves the wrong way. "
        "The output distribution staying put is not evidence the model is fine."
    )

log = config.LOG_PATH
if log.exists():
    st.subheader("Live traffic")
    live = pd.read_json(log, lines=True)
    st.metric("requests scored", len(live))
    st.line_chart(live["pred"].rolling(50, min_periods=1).mean())
