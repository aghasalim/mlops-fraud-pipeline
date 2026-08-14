"""Reproduce the training-time feature pipeline for incoming traffic.

This file is a liability by design and worth naming as such: it re-implements
transformations that live in another repo, so the two can silently diverge.
That is training/serving skew, and it is one of the most common ways a
production model degrades without anything appearing to break. The honest fixes
are a shared library or a feature store; the honest mitigation here is a test
that asserts serving features reproduce the training ones on known rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def load_split(split: str, nrows: int | None = None) -> pd.DataFrame:
    tx = pd.read_csv(config.SOURCE / f"{split}_transaction.csv", nrows=nrows)
    idf = pd.read_csv(config.SOURCE / f"{split}_identity.csv")
    idf.columns = [c.replace("-", "_") for c in idf.columns]
    df = tx.merge(idf, on="TransactionID", how="left")
    df["has_identity"] = df["TransactionID"].isin(idf["TransactionID"]).astype("int8")
    return df


def base(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    t = out["TransactionDT"]
    out["hour"] = (t / 3600) % 24
    out["dayofweek"] = (t / 86400) % 7
    out["day_index"] = t / 86400
    amt = out["TransactionAmt"]
    out["amt_log"] = np.log1p(amt)
    out["amt_cents"] = ((amt - np.floor(amt)) * 100).round().astype("float32")
    out["amt_is_round"] = (out["amt_cents"] == 0).astype("int8")
    return out


def encode(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if not pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].astype("category").cat.codes.astype("int32")
    return out


def frequency(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Frequency encodings, computed on the batch itself.

    At training time these were fitted fold-locally. At serving time there is no
    fold, so they are computed over the incoming batch -- which means the value
    for an identical transaction depends on what else arrived alongside it. That
    is a real weakness of frequency features in streaming inference, and it is
    part of why the drift study below tracks them.
    """
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[f"{c}_freq"] = out[c].map(out[c].value_counts()).fillna(0).astype("float32")
    return out


def prepare(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    d = encode(base(df))
    d["_uid"] = (d["card1"].astype("string") + "_" + d["addr1"].astype("string")
                 + "_" + (d["day_index"] - d["D1"]).round().astype("string"))
    d = frequency(d, ["card1", "card2", "addr1", "P_emaildomain", "_uid"])
    for c in columns:
        if c not in d.columns:
            d[c] = np.nan
    return d[columns].astype("float32")
