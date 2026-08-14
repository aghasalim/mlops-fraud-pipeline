"""MLflow registration with an explicit deploy gate.

The point of a registry here is not that a version number exists somewhere. It
is that a version carries the evidence used to decide whether it may serve
traffic, so "which model is running and why was it allowed to" has an answer
that does not depend on anyone's memory.

A run is tagged `deployable=true` only when every gate passes. CI reads that
tag, so a model failing its gates cannot be promoted by re-running the job.
"""
from __future__ import annotations

import json
import subprocess
import sys

import joblib
import mlflow

from . import config

# Gates. Deliberately conservative: a model that cannot beat these should not be
# reachable from a network port.
GATES = {
    "min_val_auc": 0.85,          # the honest embargoed number from ieee-fraud-ml
    "max_n_features": 500,        # a jump here means the feature pipeline changed
    "min_train_rows": 500_000,
}


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def evaluate_gates(bundle: dict) -> tuple[bool, dict]:
    checks = {
        "val_auc_ok": float(bundle.get("val_auc", 0)) >= GATES["min_val_auc"],
        "n_features_ok": len(bundle["columns"]) <= GATES["max_n_features"],
        "train_rows_ok": int(bundle.get("n_train", 0)) >= GATES["min_train_rows"],
        "model_loads": hasattr(bundle.get("model"), "predict_proba"),
    }
    return all(checks.values()), checks


def main() -> None:
    bundle = joblib.load(config.MODEL_PATH)
    passed, checks = evaluate_gates(bundle)

    # SQLite rather than the ./mlruns file store: MLflow put the filesystem
    # backend into maintenance mode and now raises on it outright.
    mlflow.set_tracking_uri(f"sqlite:///{config.ROOT / 'mlflow.db'}")
    mlflow.set_experiment("fraud-serving")
    with mlflow.start_run(run_name=config.MODEL_VERSION) as run:
        mlflow.log_params({
            "model_version": config.MODEL_VERSION,
            "n_features": len(bundle["columns"]),
            "n_train_rows": bundle.get("n_train"),
            "git_sha": _git_sha(),
        })
        mlflow.log_metric("val_auc", float(bundle.get("val_auc", 0)))
        for k, v in checks.items():
            mlflow.log_metric(f"gate_{k}", int(v))
        mlflow.set_tag("deployable", str(passed).lower())
        mlflow.log_dict(GATES, "gates.json")
        print(f"run {run.info.run_id}  deployable={passed}")
        for k, v in checks.items():
            print(f"  {'PASS' if v else 'FAIL'}  {k}")

    # CI reads this file rather than parsing logs.
    (config.ARTIFACTS / "deploy_decision.json").write_text(
        json.dumps({"deployable": passed, "checks": checks,
                    "model_version": config.MODEL_VERSION}, indent=2))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
