"""The deploy gate must reject, not just accept.

A gate that has only ever returned True is indistinguishable from no gate, and
that is precisely the failure this project exists to argue against.
"""
import joblib

from src.pipeline import config, registry


def _bundle():
    return dict(joblib.load(config.MODEL_PATH))


def test_good_model_passes():
    assert registry.evaluate_gates(_bundle())[0]


def test_low_auc_is_blocked():
    b = _bundle(); b["val_auc"] = 0.60
    ok, checks = registry.evaluate_gates(b)
    assert not ok and not checks["val_auc_ok"]


def test_undertrained_model_is_blocked():
    b = _bundle(); b["n_train"] = 1_000
    assert not registry.evaluate_gates(b)[0]


def test_feature_count_explosion_is_blocked():
    """A jump in feature count means the pipeline changed shape; serving a model
    whose inputs no longer match the caller is worse than not deploying."""
    b = _bundle(); b["columns"] = list(b["columns"]) * 2
    assert not registry.evaluate_gates(b)[0]


def test_unloadable_model_is_blocked():
    b = _bundle(); b["model"] = object()
    assert not registry.evaluate_gates(b)[0]
